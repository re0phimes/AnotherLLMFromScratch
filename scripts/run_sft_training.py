# file: scripts/run_sft_training.py
# Description: 读取 YAML 配置并使用 SFTTrainer 启动 GPT-2 结构模型的训练流程。
#              支持多卡 torchrun 启动、余弦退火调度器及本地 JSONL 数据集。
#
# Call Logic:
# 1. `main()` 解析命令行参数并加载配置，随后初始化分布式与日志。
# 2. `prepare_components()` 构建 tokenizer、模型、数据模块、优化器与调度器。
# 3. `run_training()` 负责启动 SFTTrainer，执行完整训练循环并处理善后。

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer
import yaml

from src.dataset import BaseDatasetModule, PretrainDatasetModule
from src.models.modeling_auto import AutoConfig, AutoModelForCausalLM
from src.trainer import SFTTrainer
from src.trainer.optimizer import configure_optimizer
from src.utils import (
    cleanup_distributed,
    get_world_size,
    is_distributed,
    is_main_process,
    logger,
    setup_distributed,
    setup_logger,
)


@dataclass
class TrainingArtifacts:
    """汇总训练所需的核心组件。"""

    model: torch.nn.Module
    tokenizer: Any
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    optimizer: Optimizer
    scheduler: Optional[LambdaLR]
    max_epochs: int
    grad_accum_steps: int
    use_amp: bool
    save_dir: Path
    log_interval: int
    steps_per_epoch: Optional[int] = None


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="Run GPT-2 training with SFTTrainer")
    parser.add_argument(
        "--config",
        required=True,
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="DataLoader worker 数量覆盖值，可选",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="梯度累积步数覆盖值，可选",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="训练轮数覆盖值，可选",
    )
    parser.add_argument(
        "--inspect-data",
        action="store_true",
        default=True,
        help="在训练前检查第一个batch的数据（默认启用）",
    )
    parser.add_argument(
        "--no-inspect-data",
        action="store_false",
        dest="inspect_data",
        help="禁用数据检查",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """加载 YAML 配置文件。"""

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_random_seed(seed: int) -> None:
    """设置 Python、NumPy 与 PyTorch 的随机种子。"""

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        logger.warning("NumPy 未安装，跳过 NumPy 随机种子设置。")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_tokenizer(model_cfg: Dict[str, Any]):
    """根据模型配置初始化 tokenizer。"""

    tokenizer_name = model_cfg.get("tokenizer_name_or_path")
    if tokenizer_name is None:
        raise ValueError("`model.tokenizer_name_or_path` 未在配置中指定。")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return tokenizer


def load_model(model_cfg: Dict[str, Any]) -> torch.nn.Module:
    """加载 GPT-2 模型配置并实例化模型。"""

    config_path = model_cfg.get("model_config_path")
    if config_path is None:
        raise ValueError("`model.model_config_path` 未在配置中指定。")
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        model_dict = yaml.safe_load(handle)
    auto_config = AutoConfig.from_dict(model_dict)
    model = AutoModelForCausalLM.from_config(auto_config)
    return model


def build_dataset_module(
    data_cfg: Dict[str, Any],
    tokenizer,
    seed: int,
) -> PretrainDatasetModule:
    """构建预训练数据模块。"""

    return PretrainDatasetModule.from_config(data_cfg, tokenizer=tokenizer, seed=seed)


def create_dataloader(
    module: BaseDatasetModule,
    *,
    batch_size: int,
    rank: int,
    world_size: int,
    is_train: bool,
    override_workers: Optional[int],
) -> DataLoader:
    """构建支持分布式采样的数据加载器。"""

    dataset = module.build_dataset()
    sampler: Optional[DistributedSampler] = None
    
    is_iterable = isinstance(dataset, torch.utils.data.IterableDataset)
    
    if is_distributed() and not is_iterable:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=is_train,
            drop_last=is_train and module.config.drop_last,
        )
        shuffle = False
    else:
        if is_iterable:
            shuffle = False
        else:
            shuffle = is_train and any(src.shuffle for src in module.config.sources)

    num_workers = override_workers if override_workers is not None else module.config.num_workers
    prefetch_factor = module.config.prefetch_factor if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=module.collate_fn,
        num_workers=num_workers,
        pin_memory=module.config.pin_memory,
        drop_last=is_train and module.config.drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor,
    )


def create_scheduler(
    optimizer: Optimizer,
    scheduler_cfg: Dict[str, Any],
    total_steps: int,
) -> Optional[LambdaLR]:
    """创建学习率调度器。"""

    if not scheduler_cfg:
        return None
    name = scheduler_cfg.get("name", "").lower()
    if name != "cosine":
        logger.warning("当前脚本仅实现 cosine 调度，忽略其他调度器配置。")
        return None

    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    min_lr = float(scheduler_cfg.get("min_lr", 0.0))
    base_lr = optimizer.param_groups[0]["lr"]
    min_lr_ratio = min_lr / base_lr if base_lr > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if total_steps <= 0:
            return 1.0
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def prepare_components(
    config: Dict[str, Any],
    args: argparse.Namespace,
    rank: int,
    world_size: int,
) -> TrainingArtifacts:
    """根据配置构建训练所需组件。"""

    seed = int(config.get("seed", 42)) + rank
    set_random_seed(seed)

    model_cfg = config.get("model", {})
    tokenizer = build_tokenizer(model_cfg)
    model = load_model(model_cfg)

    data_cfg = config.get("data", {})
    dataset_module = build_dataset_module(data_cfg, tokenizer, seed)

    training_cfg = config.get("training", {})
    micro_batch_size = int(training_cfg.get("micro_batch_size", 1))
    grad_accum_steps = args.gradient_accumulation or int(training_cfg.get("gradient_accumulation", 1))
    max_epochs = args.max_epochs or int(training_cfg.get("max_epochs", 1))

    train_loader = create_dataloader(
        dataset_module,
        batch_size=micro_batch_size,
        rank=rank,
        world_size=world_size,
        is_train=True,
        override_workers=args.num_workers,
    )
    if not isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
        if len(train_loader) == 0:
            raise ValueError("训练数据加载器为空，请检查 data.path 或批大小设置。")
    val_loader = None

    optimizer_cfg = config.get("optimizer", {})
    optimizer = configure_optimizer(
        model,
        optimizer_type=optimizer_cfg.get("name", "adamw"),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.01)),
        betas=tuple(optimizer_cfg.get("betas", (0.9, 0.95))),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )

    if isinstance(train_loader.dataset, torch.utils.data.IterableDataset):
        max_steps = training_cfg.get("max_steps")
        if max_steps is None:
            steps_per_epoch = 1000
            logger.warning("⚠ IterableDataset 无法确定长度，使用默认 steps_per_epoch={}", steps_per_epoch)
        else:
            steps_per_epoch = max_steps // max_epochs if max_epochs > 0 else max_steps
    else:
        steps_per_epoch = len(train_loader)
    
    scheduler_cfg = config.get("scheduler", {})
    total_steps = max_epochs * steps_per_epoch
    scheduler = create_scheduler(optimizer, scheduler_cfg, total_steps)

    use_amp = bool(training_cfg.get("use_amp", True)) and torch.cuda.is_available()
    save_dir = Path(training_cfg.get("save_dir", "./checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)
    log_interval = int(training_cfg.get("log_interval", 10))

    return TrainingArtifacts(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        max_epochs=max_epochs,
        grad_accum_steps=grad_accum_steps,
        use_amp=use_amp,
        save_dir=save_dir,
        log_interval=log_interval,
        steps_per_epoch=steps_per_epoch if isinstance(train_loader.dataset, torch.utils.data.IterableDataset) else None,
    )


def run_training(config_path: Path, args: argparse.Namespace) -> None:
    """主训练入口。"""

    config = load_yaml_config(config_path)
    rank, local_rank, world_size = setup_distributed()

    log_dir = Path(config.get("training", {}).get("save_dir", "./checkpoints")) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"{config.get('run_name', 'train')}_{timestamp}.log"
    setup_logger(str(log_file), rank=rank, world_size=world_size)

    logger.info("加载配置完成: {}", config_path)
    logger.info("世界大小: {}, 当前 rank/local_rank: {}/{}", world_size, rank, local_rank)

    artifacts = prepare_components(config, args, rank, world_size)

    if not isinstance(artifacts.train_loader.dataset, torch.utils.data.IterableDataset):
        logger.info("数据加载器每轮步数: {}", len(artifacts.train_loader))
    else:
        logger.info("数据加载器: IterableDataset (未知长度)")
    logger.info("训练最大轮数: {}", artifacts.max_epochs)
    logger.info("梯度累积步数: {}", artifacts.grad_accum_steps)

    # 提取生成评估配置
    training_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    
    eval_interval = training_cfg.get("eval_interval", None)
    eval_prompts = eval_cfg.get("prompts", None)
    eval_max_tokens = eval_cfg.get("max_new_tokens", 100)
    eval_temperature = eval_cfg.get("temperature", 0.8)
    eval_top_k = eval_cfg.get("top_k", None)
    eval_top_p = eval_cfg.get("top_p", 0.9)
    eval_repetition_penalty = eval_cfg.get("repetition_penalty", 1.0)
    
    if eval_interval is not None and eval_prompts:
        logger.info("启用生成评估: 每 {} 步生成样本", eval_interval)
        logger.info("评估 prompts 数量: {}", len(eval_prompts))
    
    # 数据检查（仅在 rank 0 执行）
    if is_main_process() and args.inspect_data:
        from src.utils import inspect_first_batch
        logger.info("=" * 80)
        logger.info("开始检查训练数据...")
        logger.info("=" * 80)
        inspect_first_batch(artifacts.train_loader, artifacts.tokenizer, num_samples=2)
        logger.info("数据检查完成，准备开始训练")
        logger.info("=" * 80)

    trainer = SFTTrainer(
        model=artifacts.model,
        optimizer=artifacts.optimizer,
        train_loader=artifacts.train_loader,
        val_loader=artifacts.val_loader,
        scheduler=artifacts.scheduler,
        max_epochs=artifacts.max_epochs,
        grad_accum_steps=artifacts.grad_accum_steps,
        use_amp=artifacts.use_amp,
        save_dir=str(artifacts.save_dir),
        log_interval=artifacts.log_interval,
        estimated_steps_per_epoch=artifacts.steps_per_epoch,
        # 生成评估参数
        tokenizer=artifacts.tokenizer,
        eval_interval=eval_interval,
        eval_prompts=eval_prompts,
        eval_max_tokens=eval_max_tokens,
        eval_temperature=eval_temperature,
        eval_top_k=eval_top_k,
        eval_top_p=eval_top_p,
        eval_repetition_penalty=eval_repetition_penalty,
    )

    try:
        trainer.train()
    finally:
        cleanup_distributed()


def main() -> None:
    """脚本入口。"""

    args = parse_args()
    try:
        run_training(Path(args.config), args)
    except Exception as exc:  # pylint: disable=broad-except
        logger.exception("训练流程失败: {}", exc)
        raise


if __name__ == "__main__":
    main()


"""

    Command

单卡可直接执行：

    python scripts/run_sft_training.py --config configs/train/gpt2_sft_chinanews.yaml

多卡可通过：

    torchrun --nproc_per_node=NUM_GPUS scripts/run_sft_training.py --config configs/train/gpt2_sft_chinanews.yaml
"""
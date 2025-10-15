# file: src/trainer/tutorial/train_with_custom_data.py
# Description: 运行教学版训练器的示例脚本，演示如何使用 JSONL 文本数据训练语言模型。
#              依赖项目内的 TrainerFromScratch 与 AdamWFromScratch，并要求用户提供已准备的数据集。
#
# Call Logic:
# 1. `main()` 解析命令行参数，校验数据路径并加载分词器与基础模型。
# 2. `build_dataloader()` 构建基于 JSONL 文本的 DataLoader，输出 input_ids 与 labels。
# 3. `run_training()` 使用 TrainerFromScratch 训练并保存检查点。

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from loguru import logger

from src.trainer.tutorial.optimizer_from_scratch import AdamWFromScratch
from src.trainer.tutorial.trainer_from_scratch import TrainerFromScratch


class JsonlCausalDataset(Dataset):
    """基于 JSONL 文本数据构建的语言模型数据集。"""

    def __init__(self, data_path: Path, tokenizer: AutoTokenizer, block_size: int) -> None:
        self._samples: List[Dict[str, torch.Tensor]] = []

        try:
            with data_path.open("r", encoding="utf-8") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    text = payload.get("text")
                    if not text:
                        continue
                    tokens = tokenizer(
                        text,
                        max_length=block_size,
                        truncation=True,
                        padding="max_length",
                        return_tensors="pt",
                    )
                    input_ids = tokens["input_ids"].squeeze(0)
                    self._samples.append(
                        {
                            "input_ids": input_ids,
                            "labels": input_ids.clone(),
                        }
                    )
        except FileNotFoundError as exc:
            logger.error("未找到数据文件: {}", exc)
            raise
        except json.JSONDecodeError as exc:
            logger.error("JSONL 数据解析失败: {}", exc)
            raise
        except OSError as exc:
            logger.error("读取数据文件时出错: {}", exc)
            raise

        if not self._samples:
            raise ValueError("数据集中没有可用样本，请确保 JSONL 文件包含 'text' 字段。")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self._samples[index]


class CausalLMWrapper(nn.Module):
    """包装 AutoModelForCausalLM 以输出 logits。"""

    def __init__(self, base_model: AutoModelForCausalLM) -> None:
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        outputs = self.base_model(input_ids=input_ids)
        return outputs.logits


def build_dataloader(args: argparse.Namespace, tokenizer: AutoTokenizer) -> DataLoader:
    """构建训练与验证数据加载器。"""

    dataset = JsonlCausalDataset(Path(args.data_path), tokenizer, args.block_size)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True)


def run_training(args: argparse.Namespace) -> None:
    """执行训练流程。"""

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = CausalLMWrapper(base_model)

    train_loader = build_dataloader(args, tokenizer)
    val_loader = None
    if args.eval_split:
        val_loader = build_dataloader(args, tokenizer)

    optimizer = AdamWFromScratch(
        params=list(model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    trainer = TrainerFromScratch(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        max_epochs=args.max_epochs,
        gradient_accumulation_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=args.use_amp,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
        save_dir=args.save_dir,
    )

    trainer.train()


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description=(
            "使用教学版 TrainerFromScratch 训练语言模型。"
            "请提供包含 'text' 字段的 JSONL 数据集。"
        )
    )
    parser.add_argument("--data-path", required=True, help="JSONL 数据文件路径，必须包含 'text' 字段。")
    parser.add_argument("--model-name", default="gpt2", help="Hugging Face 预训练模型名称。")
    parser.add_argument("--block-size", type=int, default=256, help="每个样本的最大序列长度。")
    parser.add_argument("--batch-size", type=int, default=2, help="训练批次大小。")
    parser.add_argument("--max-epochs", type=int, default=1, help="训练轮数。")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="学习率。")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="权重衰减。")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="梯度累积步数。")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="梯度裁剪阈值。")
    parser.add_argument("--warmup-steps", type=int, default=10, help="学习率预热步数。")
    parser.add_argument("--use-amp", action="store_true", help="启用混合精度训练。")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备。",
    )
    parser.add_argument("--log-interval", type=int, default=10, help="日志打印间隔（按步）。")
    parser.add_argument("--save-dir", default="./tutorial_checkpoints", help="检查点保存路径。")
    parser.add_argument("--eval-split", action="store_true", help="使用同一数据集构建验证集（演示用途）。")
    return parser.parse_args()


def main() -> None:
    """入口函数，校验数据并运行训练。"""

    args = parse_args()
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error("数据路径不存在，请准备包含 'text' 字段的 JSONL 文件: {}", data_path)
        raise FileNotFoundError(f"data path not found: {data_path}")

    logger.info("开始教学版训练，数据文件: {}", data_path)
    logger.info("请确保数据已完成预处理并包含可训练的文本字段。")

    run_training(args)


if __name__ == "__main__":
    main()

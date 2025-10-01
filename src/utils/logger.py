"""
日志记录工具模块 - 基于loguru

提供统一的日志记录功能，支持分布式训练环境。
主要功能：
- 控制台和文件双重输出
- 分布式训练中只有主进程输出到控制台
- 所有进程都写入各自的日志文件
- 自动创建日志目录
- 彩色输出和结构化日志

使用方式：
    from utils.logger import logger, setup_logger
    
    # 设置日志器
    setup_logger(log_file="logs/train.log", rank=0, world_size=1)
    
    # 使用日志器
    logger.info("这是一条信息")
    logger.warning("这是一条警告")
"""

import sys
from pathlib import Path
from typing import Optional
from loguru import logger


def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    rank: int = 0,
    world_size: int = 1,
    console_output: bool = True,
) -> None:
    """
    设置loguru日志记录器
    
    Args:
        log_file: 日志文件路径，None则不写入文件
        log_level: 日志级别 (TRACE, DEBUG, INFO, SUCCESS, WARNING, ERROR, CRITICAL)
        rank: 当前进程rank（分布式训练用）
        world_size: 总进程数（分布式训练用）
        console_output: 是否输出到控制台
    
    日志输出逻辑：
    1. 控制台输出：只有主进程(rank=0)输出到控制台，避免重复日志
    2. 文件输出：所有进程都写入各自的日志文件，便于调试分布式问题
    """
    # 移除默认的控制台handler
    logger.remove()
    
    # 控制台输出逻辑：只有主进程输出到控制台
    if console_output and rank == 0:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>Rank {extra[rank]}/{extra[world_size]}</cyan> | "
                   "<level>{message}</level>",
            level=log_level,
            colorize=True,
            enqueue=True,  # 线程安全
        )
    
    # 文件输出逻辑：所有进程都写入各自的日志文件
    if log_file is not None:
        # 为不同rank创建不同的日志文件
        if world_size > 1:
            log_path = Path(log_file)
            log_file = log_path.parent / f"{log_path.stem}_rank{rank}{log_path.suffix}"
        
        # 创建日志目录
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            format="{time:YYYY-MM-DD HH:mm:ss} | "
                   "{level: <8} | "
                   "Rank {extra[rank]}/{extra[world_size]} | "
                   "{name}:{function}:{line} | "
                   "{message}",
            level=log_level,
            rotation="100 MB",  # 文件大小超过100MB时轮转
            retention="7 days",  # 保留7天的日志
            compression="zip",   # 压缩旧日志
            enqueue=True,       # 线程安全
            encoding="utf-8",
        )
    
    # 配置额外信息
    logger.configure(extra={"rank": rank, "world_size": world_size})


def get_logger():
    """
    获取全局logger实例
    
    Returns:
        loguru logger实例
        
    注意：其他模块只需要调用这个函数获取logger，无需额外配置
    """
    return logger


def log_model_info(model, config: dict):
    """
    记录模型信息
    
    Args:
        model: 模型实例
        config: 模型配置
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info("=" * 60)
        logger.info("模型信息")
        logger.info("=" * 60)
        logger.info(f"模型名称: {config.get('model_name', 'Unknown')}")
        logger.info(f"模型类型: {config.get('model_type', 'Unknown')}")
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数: {trainable_params:,}")
        logger.info(f"参数量 (MB): {total_params * 4 / 1024 / 1024:.2f}")
        logger.info(f"架构: {config.get('n_layer', 0)}层 × {config.get('n_head', 0)}头 × {config.get('n_embd', 0)}维")
        logger.info(f"词汇表大小: {config.get('vocab_size', 0):,}")
        logger.info(f"序列长度: {config.get('block_size', 0)}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"记录模型信息时出错: {e}")


def log_training_info(config: dict, dataset_size: int):
    """
    记录训练信息
    
    Args:
        config: 训练配置
        dataset_size: 数据集大小
    """
    logger.info("=" * 60)
    logger.info("训练信息")
    logger.info("=" * 60)
    logger.info(f"实验名称: {config.get('experiment_name', 'Unknown')}")
    logger.info(f"数据集大小: {dataset_size:,}")
    logger.info(f"批次大小: {config.get('batch_size', 0)}")
    logger.info(f"学习率: {config.get('learning_rate', 0)}")
    logger.info(f"训练轮数: {config.get('num_epochs', 0)}")
    logger.info(f"预热步数: {config.get('warmup_steps', 0)}")
    logger.info("=" * 60)


def log_system_info():
    """
    记录系统信息
    """
    try:
        import platform
        
        logger.info("=" * 60)
        logger.info("系统信息")
        logger.info("=" * 60)
        logger.info(f"Python版本: {platform.python_version()}")
        
        try:
            import torch
            logger.info(f"PyTorch版本: {torch.__version__}")
            logger.info(f"CUDA可用: {torch.cuda.is_available()}")
            
            if torch.cuda.is_available():
                logger.info(f"CUDA版本: {torch.version.cuda}")
                logger.info(f"GPU数量: {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        except ImportError:
            logger.warning("PyTorch未安装，跳过GPU信息")
        
        logger.info(f"操作系统: {platform.system()} {platform.release()}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"记录系统信息时出错: {e}")


# 导出全局logger实例，其他模块直接导入使用
__all__ = ["logger", "setup_logger", "get_logger", "log_model_info", "log_training_info", "log_system_info"]

"""
Utils模块 - 提供项目通用工具函数

主要组件:
- logger: 日志记录工具 (基于loguru)
- distributed: 分布式训练工具  
- misc: 杂项工具函数

使用方式:
    from utils.logger import logger, setup_logger
    
    # 设置日志器
    setup_logger(log_file="logs/train.log", rank=0, world_size=1)
    
    # 使用日志器
    logger.info("训练开始")
"""

from .logger import logger, setup_logger, get_logger, log_model_info, log_training_info, log_system_info
from .distributed import (
    setup_distributed, cleanup_distributed, is_distributed, is_main_process,
    get_rank, get_local_rank, get_world_size, get_device, barrier,
    reduce_tensor, gather_tensor, broadcast_tensor, print_distributed_info
)

__all__ = [
    # Logger模块
    "logger",           # 全局logger实例，其他模块直接使用
    "setup_logger",     # 配置日志器
    "get_logger",       # 获取logger实例
    "log_model_info",   # 记录模型信息
    "log_training_info", # 记录训练信息
    "log_system_info",  # 记录系统信息
    
    # Distributed模块
    "setup_distributed",    # 初始化分布式环境
    "cleanup_distributed",  # 清理分布式环境
    "is_distributed",       # 检测是否分布式模式
    "is_main_process",      # 是否主进程
    "get_rank",            # 获取全局rank
    "get_local_rank",      # 获取本地rank
    "get_world_size",      # 获取总进程数
    "get_device",          # 获取设备
    "barrier",             # 进程同步
    "reduce_tensor",       # 张量归约
    "gather_tensor",       # 张量收集
    "broadcast_tensor",    # 张量广播
    "print_distributed_info", # 打印分布式信息
]

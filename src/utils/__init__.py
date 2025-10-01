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

__all__ = [
    "logger",           # 全局logger实例，其他模块直接使用
    "setup_logger",     # 配置日志器
    "get_logger",       # 获取logger实例
    "log_model_info",   # 记录模型信息
    "log_training_info", # 记录训练信息
    "log_system_info",  # 记录系统信息
]

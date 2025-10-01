"""
分布式训练工具模块

提供分布式训练的基础功能：
- 自动检测torchrun环境
- 初始化分布式进程组
- 获取分布式训练参数
- 进程同步和通信工具
"""

import os
from typing import Tuple, Optional
from utils.logger import logger


def is_distributed() -> bool:
    """
    检测是否在分布式环境中运行
    
    Returns:
        bool: 是否为分布式环境
    """
    return (
        "RANK" in os.environ and 
        "WORLD_SIZE" in os.environ and 
        "LOCAL_RANK" in os.environ
    )


def get_rank() -> int:
    """
    获取当前进程的全局rank
    
    Returns:
        int: 全局rank，单进程时返回0
    """
    if is_distributed():
        return int(os.environ["RANK"])
    return 0


def get_local_rank() -> int:
    """
    获取当前进程的本地rank
    
    Returns:
        int: 本地rank，单进程时返回0
    """
    if is_distributed():
        return int(os.environ["LOCAL_RANK"])
    return 0


def get_world_size() -> int:
    """
    获取总进程数
    
    Returns:
        int: 总进程数，单进程时返回1
    """
    if is_distributed():
        return int(os.environ["WORLD_SIZE"])
    return 1


def get_local_world_size() -> int:
    """
    获取本地进程数（单机GPU数量）
    
    Returns:
        int: 本地进程数
    """
    if is_distributed():
        return int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    return 1


def setup_distributed() -> Tuple[int, int, int]:
    """
    初始化分布式训练环境
    
    Returns:
        Tuple[int, int, int]: (rank, local_rank, world_size)
    """
    rank = get_rank()
    local_rank = get_local_rank()
    world_size = get_world_size()
    
    if is_distributed():
        try:
            import torch
            import torch.distributed as dist
            
            # 设置CUDA设备
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
                device = f"cuda:{local_rank}"
            else:
                device = "cpu"
                logger.warning("CUDA不可用，使用CPU进行分布式训练")
            
            # 初始化进程组
            if not dist.is_initialized():
                # 使用NCCL后端进行GPU通信，GLOO后端进行CPU通信
                backend = "nccl" if torch.cuda.is_available() else "gloo"
                
                logger.info(f"初始化分布式训练: backend={backend}, rank={rank}, world_size={world_size}")
                dist.init_process_group(
                    backend=backend,
                    rank=rank,
                    world_size=world_size
                )
                
                logger.info(f"分布式训练初始化成功: rank={rank}/{world_size}, device={device}")
            
            return rank, local_rank, world_size
            
        except ImportError:
            logger.error("PyTorch未安装，无法进行分布式训练")
            raise
        except Exception as e:
            logger.error(f"分布式训练初始化失败: {e}")
            raise
    else:
        logger.info("单进程模式，跳过分布式初始化")
        return rank, local_rank, world_size


def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if is_distributed():
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.destroy_process_group()
                logger.info("分布式训练环境清理完成")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"清理分布式环境时出错: {e}")


def is_main_process() -> bool:
    """
    判断是否为主进程
    
    Returns:
        bool: 是否为主进程(rank=0)
    """
    return get_rank() == 0


def barrier():
    """
    同步所有进程
    """
    if is_distributed():
        try:
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"进程同步失败: {e}")


def reduce_tensor(tensor, op="mean"):
    """
    跨进程归约张量
    
    Args:
        tensor: 要归约的张量
        op: 归约操作 ("sum", "mean")
    
    Returns:
        归约后的张量
    """
    if not is_distributed():
        return tensor
    
    try:
        import torch
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return tensor
        
        # 克隆张量避免修改原始数据
        tensor = tensor.clone()
        
        # 执行all_reduce操作
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # 如果是求平均，除以进程数
        if op == "mean":
            tensor = tensor / get_world_size()
        
        return tensor
        
    except ImportError:
        return tensor
    except Exception as e:
        logger.warning(f"张量归约失败: {e}")
        return tensor


def gather_tensor(tensor, dst=0):
    """
    收集所有进程的张量到指定进程
    
    Args:
        tensor: 要收集的张量
        dst: 目标进程rank
    
    Returns:
        收集后的张量列表（只在目标进程有效）
    """
    if not is_distributed():
        return [tensor]
    
    try:
        import torch
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return [tensor]
        
        world_size = get_world_size()
        
        if get_rank() == dst:
            # 目标进程准备接收缓冲区
            tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            dist.gather(tensor, tensor_list, dst=dst)
            return tensor_list
        else:
            # 其他进程发送数据
            dist.gather(tensor, dst=dst)
            return None
            
    except ImportError:
        return [tensor]
    except Exception as e:
        logger.warning(f"张量收集失败: {e}")
        return [tensor]


def broadcast_tensor(tensor, src=0):
    """
    从指定进程广播张量到所有进程
    
    Args:
        tensor: 要广播的张量
        src: 源进程rank
    
    Returns:
        广播后的张量
    """
    if not is_distributed():
        return tensor
    
    try:
        import torch.distributed as dist
        
        if not dist.is_initialized():
            return tensor
        
        dist.broadcast(tensor, src=src)
        return tensor
        
    except ImportError:
        return tensor
    except Exception as e:
        logger.warning(f"张量广播失败: {e}")
        return tensor


def get_device() -> str:
    """
    获取当前进程应该使用的设备
    
    Returns:
        str: 设备名称 ("cuda:0", "cpu"等)
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            if is_distributed():
                return f"cuda:{get_local_rank()}"
            else:
                return "cuda:0"
        else:
            return "cpu"
            
    except ImportError:
        return "cpu"


def print_distributed_info():
    """
    打印分布式训练信息
    """
    logger.info("=" * 60)
    logger.info("分布式训练信息")
    logger.info("=" * 60)
    logger.info(f"分布式模式: {is_distributed()}")
    logger.info(f"全局Rank: {get_rank()}")
    logger.info(f"本地Rank: {get_local_rank()}")
    logger.info(f"总进程数: {get_world_size()}")
    logger.info(f"本地进程数: {get_local_world_size()}")
    logger.info(f"主进程: {is_main_process()}")
    logger.info(f"设备: {get_device()}")
    
    # 环境变量信息
    env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    for var in env_vars:
        value = os.environ.get(var, "未设置")
        logger.info(f"{var}: {value}")
    
    logger.info("=" * 60)

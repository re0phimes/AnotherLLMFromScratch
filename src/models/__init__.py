from __future__ import annotations

"""模型子系统公共入口。

按照 docs/05_project_structure.md 要求，导出常用模型与配置类，
并提供 Auto 工厂入口以便上层脚本统一访问。
"""

from .gpt2.config import GPT2Config
from .gpt2.model import GPT2Block, GPT2Model, GPT2ModelOutput
from .modeling_auto import AutoConfig, AutoModelForCausalLM

__all__ = [
    "AutoConfig",
    "AutoModelForCausalLM",
    "GPT2Block",
    "GPT2Config",
    "GPT2Model",
    "GPT2ModelOutput",
]


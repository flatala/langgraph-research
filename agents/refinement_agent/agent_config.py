from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from dataclasses import fields
from agents.planning_agent import prompts
from agents.shared.main_config import MainConfiguration

@dataclass(kw_only=True)
class RefinementAgentConfiguration(MainConfiguration):
    """Full configuration for the planning agent, extending MainConfiguration."""


    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "RefinementAgentConfiguration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
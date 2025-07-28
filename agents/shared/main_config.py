from __future__ import annotations
from dataclasses import dataclass, field
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from dataclasses import fields
import os

@dataclass(kw_only=True)
class MainConfiguration:
    """The main configuration for LLM providers and API keys."""

    # API KEYS
    
    openai_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )
    
    google_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("GOOGLE_API_KEY")
    )

    # MODELS

    orchestrator_provider: str = field(
        default="openai",
        metadata={"description": "Provider for orchestrator model: 'openai', 'anthropic', or 'google'"},
    )

    orchestrator_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="o3",
        metadata={"description": "The model name for planning the literature review process."},
    )

    text_provider: str = field(
        default="openai", 
        metadata={"description": "Provider for text model: 'openai', 'anthropic', or 'google'"},
    )

    text_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="gpt-4.1-2025-04-14",
        metadata={"description": "The model name for text processing tasks."},
    )

    # HELPERS

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MainConfiguration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
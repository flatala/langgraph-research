from __future__ import annotations
from dataclasses import dataclass, field
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from dataclasses import fields
import os


@dataclass(kw_only=True)
class MainConfiguration:
    """The main configuration for OpenRouter API."""

    # API Configuration

    openrouter_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY"),
        metadata={"description": "OpenRouter API key"}
    )

    openrouter_base_url: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        metadata={"description": "OpenRouter API base URL"}
    )

    # Model Configuration

    orchestrator_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: os.getenv("ORCHESTRATOR_MODEL"),
        metadata={"description": "The model name for planning the literature review process (OpenRouter format: provider/model)."},
    )

    text_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: os.getenv("TEXT_MODEL"),
        metadata={"description": "The model name for text processing tasks (OpenRouter format: provider/model)."},
    )

    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL"),
        metadata={"description": "The model name for embeddings (OpenRouter format: provider/model)."},
    )

    # HELPERS

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "MainConfiguration":
        cfg = ensure_config(config or {})
        data = cfg.get("configurable", {})
        return cls(**{k: v for k, v in data.items() if k in {f.name for f in fields(cls)}})
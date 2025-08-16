from __future__ import annotations
from dataclasses import dataclass, field
from typing import Annotated, Optional
from langchain_core.runnables import RunnableConfig, ensure_config
from dataclasses import fields
from pathlib import Path
import os
import json


def _load_model_config() -> dict:
    """Load model configuration from model_config.json file."""
    config_path = Path(__file__).resolve().parent.parent.parent / "model_config.json"
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Validate the configuration structure
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a JSON object")
            
        # Extract and validate model configurations
        model_config = {}
        
        if "orchestrator" in config:
            orch = config["orchestrator"]
            if not isinstance(orch, dict) or "provider" not in orch or "model" not in orch:
                raise ValueError("orchestrator configuration must have 'provider' and 'model' fields")
            model_config["orchestrator_provider"] = orch["provider"]
            model_config["orchestrator_model"] = orch["model"]
            
        if "text" in config:
            text = config["text"]
            if not isinstance(text, dict) or "provider" not in text or "model" not in text:
                raise ValueError("text configuration must have 'provider' and 'model' fields")
            model_config["text_provider"] = text["provider"]
            model_config["text_model"] = text["model"]
            
        # Validate providers
        supported_providers = {"openai", "anthropic", "google"}
        for key in ["orchestrator_provider", "text_provider"]:
            if key in model_config and model_config[key] not in supported_providers:
                raise ValueError(f"Unsupported provider: {model_config[key]}. Supported: {supported_providers}")
                
        return model_config
        
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in model_config.json: {e}")
    except Exception as e:
        raise ValueError(f"Error loading model_config.json: {e}")


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
        default_factory=lambda: _load_model_config().get("orchestrator_provider"),
        metadata={"description": "Provider for orchestrator model: 'openai', 'anthropic', or 'google'"},
    )

    orchestrator_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: _load_model_config().get("orchestrator_model"),
        metadata={"description": "The model name for planning the literature review process."},
    )

    text_provider: str = field(
        default_factory=lambda: _load_model_config().get("text_provider"),
        metadata={"description": "Provider for text model: 'openai', 'anthropic', or 'google'"},
    )

    text_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=lambda: _load_model_config().get("text_model"),
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
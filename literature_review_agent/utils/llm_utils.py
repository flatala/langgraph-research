from literature_review_agent.configuration import Configuration
from dotenv import load_dotenv
from pathlib import Path
from typing import Union
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv(                
    Path(__file__).resolve().parent.parent.parent / ".env",
    override=False,         
)    

LLMType = Union[ChatOpenAI, ChatAnthropic, ChatGoogleGenerativeAI]

def _create_llm(provider: str, model: str, api_key: str) -> LLMType:
    """Factory function to create LLM instances based on provider."""
    if provider == "openai":
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            streaming=True,
        )
    elif provider == "anthropic":
        return ChatAnthropic(
            model=model,
            api_key=api_key,
            streaming=True,
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            streaming=True,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}. Supported providers are: openai, anthropic, google")

def get_orchestrator_llm(cfg: Configuration) -> LLMType:
    """Get the instance of the strongest orchestrating LLM."""
    api_key = getattr(cfg, f"{cfg.orchestrator_provider}_api_key")
    if not api_key:
        raise ValueError(f"API key not found for provider: {cfg.orchestrator_provider}")
    
    return _create_llm(cfg.orchestrator_provider, cfg.orchestrator_model, api_key)

def get_text_llm(cfg: Configuration) -> LLMType:
    """Get the instance of the text processing LLM."""
    api_key = getattr(cfg, f"{cfg.text_provider}_api_key")
    if not api_key:
        raise ValueError(f"API key not found for provider: {cfg.text_provider}")
    
    return _create_llm(cfg.text_provider, cfg.text_model, api_key)
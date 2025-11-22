from agents.shared.main_config import MainConfiguration
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv(
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,
)

LLMType = ChatOpenAI


def get_orchestrator_llm(cfg: MainConfiguration) -> LLMType:
    """Get the instance of the strongest orchestrating LLM via OpenRouter."""
    if not cfg.openrouter_api_key:
        raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env")

    return ChatOpenAI(
        model=cfg.orchestrator_model,
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        streaming=True,
    )


def get_text_llm(cfg: MainConfiguration) -> LLMType:
    """Get the instance of the text processing LLM via OpenRouter."""
    if not cfg.openrouter_api_key:
        raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env")

    return ChatOpenAI(
        model=cfg.text_model,
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
        streaming=True,
    )


def get_embedding_model(cfg: MainConfiguration) -> OpenAIEmbeddings:
    """Get the instance of the embeddings model via OpenRouter."""
    if not cfg.openrouter_api_key:
        raise ValueError("OpenRouter API key not found. Please set OPENROUTER_API_KEY in .env")

    return OpenAIEmbeddings(
        model=cfg.embedding_model,
        api_key=cfg.openrouter_api_key,
        base_url=cfg.openrouter_base_url,
    )
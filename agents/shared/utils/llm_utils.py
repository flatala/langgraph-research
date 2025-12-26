from agents.shared.main_config import MainConfiguration
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from agents.shared.utils.json_utils import clean_and_parse_json
from typing import List, Tuple, Dict, Any, Union
import logging

load_dotenv(
    Path(__file__).resolve().parent.parent.parent.parent / ".env",
    override=False,
)

logger = logging.getLogger(__name__)

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


async def invoke_llm_with_json_retry(
    llm,
    messages: List[BaseMessage],
    max_retries: int = 2,
) -> Union[Dict[str, Any], List[Any]]:
    """Invokes LLM and parses JSON response with retry logic on parse failures."""
    last_error = None
    current_messages = messages.copy()

    for attempt in range(max_retries + 1):
        try:
            ai_msg = await llm.ainvoke(current_messages)
            parsed_data = clean_and_parse_json(ai_msg.content.strip())

            if attempt > 0:
                logger.info(f"JSON parsing succeeded on attempt {attempt + 1}")
            return parsed_data

        except ValueError as e:
            last_error = e
            logger.warning(f"JSON parsing failed on attempt {attempt + 1}/{max_retries + 1}. Error: {str(e)[:200]}")
            if attempt < max_retries:
                retry_feedback = (
                    "Your previous response was not valid JSON. Respond with ONLY a single valid JSON value "
                    "(object or array) that matches the expected schema—no extra text, no markdown fences. "
                    "Ensure all strings are valid JSON: escape backslashes (\\\\) and quotes (\\\"), and do not "
                    "include raw LaTeX backslashes unless properly escaped."
                )
                current_messages = messages + [HumanMessage(content=retry_feedback)]
            else:
                logger.error(f"All {max_retries + 1} attempts failed to produce valid JSON")
                raise ValueError(
                    f"Failed to get valid JSON from LLM after {max_retries + 1} attempts. Last error: {last_error}"
                ) from last_error

    raise ValueError(f"Unexpected error in retry logic. Last error: {last_error}")
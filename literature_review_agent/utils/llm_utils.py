from langchain_openai import ChatOpenAI
from literature_review_agent.configuration import Configuration
from dotenv import load_dotenv
from pathlib import Path
import os

load_dotenv(                
    Path(__file__).resolve().parent.parent.parent / ".env",
    override=False,         
)    

def get_orchestrator_llm(cfg: Configuration) -> ChatOpenAI:
    ''' Get the insatance of the strongest orchestrating LLM. '''
    return ChatOpenAI(
        model=cfg.orchestrator_model,
        api_key=cfg.openai_api_key,
        streaming=True,
    )

def get_text_llm(cfg: Configuration) -> ChatOpenAI:
    ''' Get the instance of the weaker text processing LLM. '''
    return ChatOpenAI(
        model=cfg.text_model,
        api_key=cfg.openai_api_key,
        streaming=True,
    )
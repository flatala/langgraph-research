from dotenv import load_dotenv
load_dotenv()        

from langchain_openai import ChatOpenAI

def get_llm():
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.2,
    )

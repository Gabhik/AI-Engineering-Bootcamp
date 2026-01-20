from fastapi import FastAPI, Request
from pydantic import BaseModel

from openai import OpenAI
from google import genai
from groq import Groq

from api.core.config import config

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_llm(provider: str, modelname : str, messages, max_tokens: int = 500):
    if provider == "openai":
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    elif provider == "groq":
        client = Groq(api_key=config.GROQ_API_KEY)
    elif provider == "google":
        client = genai.Client(api_key=config.GOOGLE_API_KEY)
    else:
        raise ValueError(f"Invalid provider: {provider}")

    if provider == "google":
        return client.models.generate_content(
            model=modelname,
            contents=[message["content"] for message in messages]
        ).text

    elif provider == "groq":
        return client.chat.completions.create(
            model=modelname,
            messages=messages,
            max_completion_tokens=max_tokens
        ).choices[0].message.content

    else:
        return client.chat.completions.create(
            model=modelname,
            messages=messages,
            max_completion_tokens=max_tokens
        ).choices[0].message.content


class ChatRequest(BaseModel):
    provider: str
    modelname: str
    messages: list[dict]

class ChatResponse(BaseModel):
    message: str

app = FastAPI()

@app.post("/chat")
def chat(
    request: Request,
    payload: ChatRequest
) -> ChatResponse:

    result = run_llm(payload.provider, payload.modelname, payload.messages)
    return ChatResponse(message=result)
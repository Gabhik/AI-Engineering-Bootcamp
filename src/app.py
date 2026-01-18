import streamlit as st
from typing import cast
from openai import OpenAI
from google import genai
from groq import Groq
from core.config import config

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

with st.sidebar:
    st.title("settings")

    provider = cast(str, st.selectbox("provider", ["openai", "google", "groq"]))
    if provider == "openai":
        model_name = cast(str, st.selectbox("model", ["gpt-5-nano", "gpt-5-mini"]))
    elif provider == "groq":
        model_name = cast(str, st.selectbox("model", ["llama-3.3-70b-versatile"]))
    else:
        model_name = cast(str, st.selectbox("model", ["gemini-2.5-flash"]))

    st.session_state.provider = provider
    st.session_state.model_name = model_name
 
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Enter a message:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            output=run_llm(st.session_state.provider, st.session_state.model_name, st.session_state.messages)
            response_data = output
            answer = response_data
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
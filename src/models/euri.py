
import requests
from dotenv import load_dotenv
import numpy as np
import aiohttp
import os
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
from typing import Optional, List


load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")

def euri_chat(messages, temperature=0.6, max_tokens=500):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": EURI_API_KEY
    }
    payload = {
        "model": "gemini-2.5-pro-preview-06-05",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']

async def euri_chat_async(messages, temperature=0.6, max_tokens=500):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": EURI_API_KEY
    }
    payload = {
        "model": "gemini-2.5-pro-preview-06-05",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data['choices'][0]['message']['content']


class EuriLLM(LLM):
    def _call(self, prompt, stop=None, **kwargs) -> str:
        """Single prompt usage (e.g., LLMChain)"""
        return euri_chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])
    
        # Async call (for .ainvoke() / agents)
    async def _acall(self, prompt: str, stop: Optional[List[str]] = None, run_manager=None, **kwargs) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        return await euri_chat_async(messages)

    def _generate(self, prompts, stop=None, **kwargs) -> LLMResult:
        """Batch prompt usage (e.g., Agents)"""
        generations = []
        for prompt in prompts:
            output = self._call(prompt)
            generations.append([Generation(text=output)])
        return LLMResult(generations=generations)

    @property
    def _identifying_params(self):
        return {}

    @property
    def _llm_type(self):
        return "euri-llm"

import requests
from dotenv import load_dotenv
import numpy as np
import os
from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation
load_dotenv()
EURI_API_KEY = os.getenv("EURI_API_KEY")

def euri_chat(messages, temperature=0.6, max_tokens=500):
    url = "https://api.euron.one/api/v1/euri/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": EURI_API_KEY
    }
    payload = {
        "model": "openai/gpt-oss-120b",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()['choices'][0]['message']['content']



from langchain.llms.base import LLM
from langchain.schema import LLMResult, Generation

class EuriLLM(LLM):
    def _call(self, prompt, stop=None, **kwargs) -> str:
        """Single prompt usage (e.g., LLMChain)"""
        return euri_chat([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ])

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
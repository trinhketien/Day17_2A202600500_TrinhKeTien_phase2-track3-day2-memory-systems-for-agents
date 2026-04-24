"""
No-Memory Agent — Stateless baseline for benchmark comparison.
One fixed system prompt, no history, no recall.
"""
import logging
import tiktoken
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

_SYSTEM = "You are a helpful assistant. Answer the user's question accurately and concisely."


class NoMemoryAgent:
    """Stateless agent with zero memory — benchmark baseline."""

    def __init__(self, llm_model: str = "gpt-4o-mini"):
        self.llm     = ChatOpenAI(model=llm_model, temperature=0.7)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        logger.info("NoMemoryAgent initialized")

    def chat(self, query: str) -> dict:
        prompt = f"SYSTEM: {_SYSTEM}\n\nUser: {query}\nAssistant:"
        resp   = self.llm.invoke([HumanMessage(content=prompt)])
        return {
            "response":      resp.content,
            "input_tokens":  len(self.encoder.encode(prompt)),
            "output_tokens": len(self.encoder.encode(resp.content)),
        }

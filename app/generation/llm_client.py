"""
LLM client (Section 1.2, Section 6.1): talks to our self-hosted Qwen 3
model via Ollama's OpenAI-compatible API.

NOTE on architecture deviation: the doc specifies vLLM as the serving
engine. vLLM requires an NVIDIA GPU (CUDA), which is not available on
this Mac (M1, no NVIDIA GPU). Switched to Ollama instead - it runs
Qwen locally using Apple's Metal GPU support, and exposes the same
OpenAI-compatible API shape vLLM would have. This keeps the model
(Qwen) and the "self-hosted, not a cloud API" principle intact; only
the serving software changed. Flagged here explicitly, not silently.
"""

import logging

from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)

_client = OpenAI(
    base_url=settings.llm_base_url,
    api_key="not-needed-for-local-ollama",  # Ollama doesn't check this, but the library requires a value
)



from typing import Any, TypeVar, Optional, Type
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

def generate_response(
    system_prompt: str, 
    user_message: str, 
    temperature: float = 0.3,
    response_format: Optional[Type[T]] = None
) -> str | T:
    """
    Sends a chat request to our locally-running Qwen model.

    Args:
        system_prompt: instructions for how the model should behave (Section 6.1).
        user_message: the actual query + context to respond to.
        temperature: randomness of output (0 = deterministic, 1 = creative).
                     Kept low (0.3) since we want consistent, grounded answers,
                     not creative writing.
        response_format: Optional Pydantic model class for structured outputs.

    Returns:
        The model's text response, or a parsed Pydantic object if response_format is provided.

    Raises:
        Exception: bubbled up if Ollama is unreachable or errors out - caller
        should handle this (e.g. show the user a "generation failed" message).
    """
    logger.info(f"Sending request to {settings.llm_model_name}")

    if response_format is not None:
        response = _client.beta.chat.completions.parse(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            response_format=response_format,
        )
        return response.choices[0].message.parsed
    else:
        response = _client.chat.completions.create(
            model=settings.llm_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content
import json
from pydantic import BaseModel, Field
from openai import OpenAI
from app.config import settings

client = OpenAI(
    base_url=settings.llm_base_url,
    api_key="not-needed"
)

class MathResponse(BaseModel):
    answer: int = Field(description="The numeric answer")
    explanation: str = Field(description="How you got the answer")

try:
    response = client.beta.chat.completions.parse(
        model=settings.llm_model_name,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        response_format=MathResponse,
    )
    print("beta.parse success:")
    print(response.choices[0].message.parsed)
except Exception as e:
    print(f"beta.parse failed: {e}")

try:
    schema = MathResponse.model_json_schema()
    response = client.chat.completions.create(
        model=settings.llm_model_name,
        messages=[{"role": "user", "content": "What is 3 + 3?"}],
        response_format={
            "type": "json_object"
        }
    )
    print("json_object success:")
    print(response.choices[0].message.content)
except Exception as e:
    print(f"json_object failed: {e}")

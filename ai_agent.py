
from openai import OpenAI
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_KEY"))
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_KEY"))

SYSTEM_PROMPT = "You are a helpful assistant. You gives the answer in the same language as the question. "

# response = response = openai_client.responses.create(
#     model="gpt-4.1-nano-2025-04-14",
#     input=[
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": "What is the capital of the Odisha?"}
#     ]
# )

llm = ChatOpenAI(
  openai_api_key=os.getenv("OPENROUTER_KEY"),
  openai_api_base="https://openrouter.ai/api/v1",
  model_name="deepseek/deepseek-chat-v3-0324:free",

)
messages = [
    (
        "system",
        "You are a helpful translator. Translate the user sentence to Odia.",
    ),
    ("human", "I love programming."),
    ]

response = llm.invoke(messages)
print(response.content)










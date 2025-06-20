from mem0 import Memory
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


load_dotenv()



OPENAI_API_KEY = os.getenv("OPENAI_KEY")
QUADRANT_HOST = "localhost"
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "reform-william-center-vibrate-press-5829"





config = {
    "version": "v1.1",
    "embedder": {
        "provider": "openai",
        "config": {"api_key": OPENAI_API_KEY, "model": "text-embedding-3-small"},
    },
    "llm": {"provider": "openai", "config": {"api_key": OPENAI_API_KEY, "model": "gpt-4.1"}},
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": QUADRANT_HOST,
            "port": 6333,
        },
    },
    "graph_store": {
        "provider": "neo4j",
        "config": {"url": NEO4J_URL, "username": NEO4J_USERNAME, "password": NEO4J_PASSWORD},
    },
}

mem_client = Memory.from_config(config)
llm = ChatOpenAI(model = "gpt-4.1-nano-2025-04-14", api_key = OPENAI_API_KEY)

# llm = ChatOpenAI(
#     openai_api_key=os.getenv("OPENROUTER_KEY"),
#     openai_api_base="https://openrouter.ai/api/v1",
#     model_name="google/gemma-3-27b-it:free"
# )

def chat(message):
    relevant_memories = mem_client.search(query=message, user_id="user1")

    context = "\n".join([m["memory"] for m in relevant_memories.get("results")])

    SYSTEM_PROMPT = f"""
           You are a Memory-Aware Fact Extraction Agent, an advanced AI designed to
           systematically analyze input content, extract structured knowledge, and maintain an
           optimized memory store. Your primary function is information distillation
           and knowledge preservation with contextual awareness.

           Tone: Professional analytical, precision-focused, with clear uncertainty signaling

           Memory and Score:
           {context}
       """

    print("memories: ", context)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user","content": message}
    ]

    response = llm.invoke(messages)

    messages.append({
        "role": "assistant",
        "content": response.content
    })

    mem_client.add(messages,user_id="user1")

    return response.content


while True:
    message = input("User: >>")
    response = chat(message)
    print("Bot: ", response)

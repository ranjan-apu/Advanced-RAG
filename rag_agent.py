from langchain_qdrant import QdrantVectorStore
from langchain_mistralai import MistralAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
You are an expert machine learning assistant. You must answer the user's question strictly using the given context.
If the answer cannot be found in the context, respond with "I don't know based on the provided context."
Always be accurate and cite page numbers or document titles if mentioned in the context.
"""

def build_context_with_metadata(docs):
    chunks = []
    for doc in docs:
        meta = doc.metadata
        page = meta.get("page_label", meta.get("page", "Unknown"))
        title = meta.get("title", "Unknown Title")
        chunk = f"[Page {page} of {title}]\n{doc.page_content}"
        chunks.append(chunk)
    return "\n\n".join(chunks)

def format_sources(docs: List) -> str:
    sources = []
    for doc in docs:
        meta = doc.metadata
        page = meta.get("page_label", meta.get("page", "Unknown"))
        title = meta.get("title", "Unknown Title")
        sources.append(f"- Page {page} of *{title}*")
    return "\n".join(set(sources)) 

def rag_qa(question, retriever, llm, history):
    retrived_docs = retriever.invoke(question)
    context = build_context_with_metadata(retrived_docs)
    sources = format_sources(retrived_docs)

    if not history:
        history = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    history.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}\n\nSources:\n{sources}"
    })

    response = llm.invoke(history)
    history.append({"role": "assistant", "content": response.content})
    answer = f"{response.content}\n\n\ud83d\udcda Sources:\n{sources}"
    return response.content, history

# ---- Initialize RAG Assistant ----

embeddings = MistralAIEmbeddings(
    api_key=os.getenv("MISTRALAI_API_KEY"),
    model="mistral-embed"
)

vector_store = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name="learning - rag",
    url="http://localhost:6333"
)

retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENROUTER_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="qwen/qwen3-32b:free"
)

# ---- Usage Loop ----

if __name__ == "__main__":
    chat_history = []
    print("\n🤖 Welcome to the Machine Learning RAG Assistant! Type 'exit' to quit.\n")
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye! 👋")
            break
        response, chat_history = rag_qa(query, retriever, llm, chat_history)
        print("\n💬 Answer:\n", response)
        print("\n" + "="*50 + "\n")

from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

# Debug statement to check if HF_TOKEN is loaded from .env
print("HF_TOKEN from environment:", os.getenv("HF_TOKEN"))

FILE_PATH = "/Users/apurba/Desktop/projects/poc/Advanced-RAG/resources/Machine Learning Yearning.pdf"

loader = PyPDFLoader(FILE_PATH)

async def load_pages():
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

pages = asyncio.run(load_pages())


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

texts = text_splitter.split_documents(pages)


embeddings = MistralAIEmbeddings(api_key=os.getenv("MISTRALAI_API_KEY"), model="mistral-embed")

# embedding_dict = {}

# for i, text in enumerate(texts):
#     embedded = embeddings.embed_documents([text.page_content])[0]
#     embedding_dict[i] = {
#         "content": text.page_content,
#         "embedding": embedded,
#         "metadata": text.metadata
#     }
#     print(f"Embedding {i} done, stored with content and metadata.")


# client.create_collection(
#     collection_name="demo_collection",
#     vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
# )

# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="demo_collection",
#     embedding=embeddings,
# )

vector_store = QdrantVectorStore.from_documents(
    documents=[],
    url = "http://localhost:6333",
    collection_name="learning - rag",
    embedding=embeddings
)

vector_store.add_documents(documents=texts)
print("Documents added to the vector store.")
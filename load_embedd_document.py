from langchain_community.document_loaders import PyPDFLoader
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import asyncio
import os
from dotenv import load_dotenv
load_dotenv()

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

embeddings = MistralAIEmbeddings(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-embed")


print (texts[57])

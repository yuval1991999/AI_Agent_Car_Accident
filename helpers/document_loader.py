from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .config import FILE_PATH, CHUNK_SIZE, CHUNK_OVERLAP

loader = TextLoader(FILE_PATH, encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
split_docs = splitter.split_documents(docs)

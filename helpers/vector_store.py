from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import sys
import os
from .config import TOP_K
from .config import OPENAI_API_KEY, CHROMA_DIR
from .document_loader import split_docs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
vectorstore = Chroma.from_documents(split_docs, embedding=embeddings, persist_directory=CHROMA_DIR)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "lambda_mult": 0.7}
)

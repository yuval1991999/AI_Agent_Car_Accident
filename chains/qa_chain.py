from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.config import OPENAI_API_KEY
from helpers.vector_store import retriever
from helpers.config import TOP_K, RETRIEVED_DOCS_PATH
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-4.1-nano")


qa_prompt_template = """
You are a helpful assistant. 
Use the following context to answer the user's question as accurately and concisely as possible.

Context:
{context}

Question:
{question}

Answer:
"""


qa_prompt = PromptTemplate.from_template(qa_prompt_template)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
    chain_type="stuff",  # Default chain type, good for short contexts
    chain_type_kwargs={"prompt": qa_prompt}
)


def run_qa_chain(prompt: str):


    result = qa_chain.invoke({"query": prompt})



    # Extract retrieved documents
    retrieved_docs = result["source_documents"]

    print(f"[Retriever] Using top_k = {TOP_K}")
    print(f"\n[INFO] Retrieved Top-{len(retrieved_docs)} Chunks for query: '{prompt}'")

    # Optional: save to file
    with open(RETRIEVED_DOCS_PATH, "w", encoding="utf-8") as f:
        f.write(f"[Retriever] Using top_k = {TOP_K}")
        f.write(f"\n[INFO] Retrieved Top-{len(retrieved_docs)} Chunks for query: '{prompt}'")
        for i, doc in enumerate(retrieved_docs, 1):
            f.write(f"\n--- Chunk {i} ---\n")
            f.write(doc.page_content + "\n")

    return result["result"]



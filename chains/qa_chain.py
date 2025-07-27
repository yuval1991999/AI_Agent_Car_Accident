from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import sys
import os
from helpers.config import OPENAI_API_KEY
from helpers.vector_store import retriever
from helpers.config import TOP_K
# from helpers.vector_store import vectorstore
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-4.1-nano")
# retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)


# def run_qa_chain(prompt: str):
#     return qa_chain.run(prompt)

def run_qa_chain(prompt: str):
    result = qa_chain(prompt)

    # Extract retrieved documents
    retrieved_docs = result["source_documents"]

    print(f"[Retriever] Using top_k = {TOP_K}")
    print(f"\n[INFO] Retrieved Top-{len(retrieved_docs)} Chunks for query: '{prompt}'")

    # Optional: save to file
    with open("./results/retrieved_chunks.txt", "w", encoding="utf-8") as f:
        f.write(f"[Retriever] Using top_k = {TOP_K}")
        f.write(f"\n[INFO] Retrieved Top-{len(retrieved_docs)} Chunks for query: '{prompt}'")
        for i, doc in enumerate(retrieved_docs, 1):
            f.write(f"\n--- Chunk {i} ---\n")
            f.write(doc.page_content + "\n")

    return result["result"]

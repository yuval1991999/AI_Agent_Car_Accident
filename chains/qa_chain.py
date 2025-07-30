from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from helpers.config import OPENAI_API_KEY
from helpers.vector_store import retriever
from helpers.config import TOP_K, RETRIEVED_DOCS_PATH
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-4o-mini", max_tokens=400)


qa_prompt_template = """
You are a helpful assistant. 
Provide clear, concise, and accurate answers based only on the provided context.
Your response should be the answer to the question, without any irrelevant misleading content.
Not more than 3-4 sentences unless necessary. Do not repeat information.

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


def run_qa_chain(prompt: str, custom_retriever=None):
    """
    Run QA chain with optional custom retriever for evaluation purposes
    """
    # Use custom retriever if provided, otherwise use default
    if custom_retriever:
        # Create a new chain with custom retriever
        custom_qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=custom_retriever,
            return_source_documents=True,
            chain_type="stuff",
            chain_type_kwargs={"prompt": qa_prompt}
        )
        result = custom_qa_chain.invoke({"query": prompt})
        current_top_k = custom_retriever.search_kwargs.get("k", TOP_K)
    else:
        result = qa_chain.invoke({"query": prompt})
        current_top_k = TOP_K

    # Extract retrieved documents
    retrieved_docs = result["source_documents"]

    print(f"[Retriever] Using top_k = {current_top_k}")
    print(f"\n[INFO] Retrieved Top-{len(retrieved_docs)} Chunks for query: '{prompt}'")

    # Optional: save to file (only if using default retriever)
    if not custom_retriever:
        with open(RETRIEVED_DOCS_PATH, "w", encoding="utf-8") as f:
            f.write(f"[Retriever] Using top_k = {current_top_k}")
            f.write(f"\n[INFO] Retrieved Top-{len(retrieved_docs)} Chunks for query: '{prompt}'")
            for i, doc in enumerate(retrieved_docs, 1):
                f.write(f"\n--- Chunk {i} ---\n")
                f.write(doc.page_content + "\n")

    return result["result"]



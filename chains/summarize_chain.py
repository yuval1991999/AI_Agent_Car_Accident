import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_community.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from helpers.config import OPENAI_API_KEY
from helpers.document_loader import split_docs
from langchain.prompts import PromptTemplate
import textwrap

# === LLM Setup ===
llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-4o-mini")

# === Summarization Prompt ===
summarization_prompt = """You are an expert document summarizer specializing in creating comprehensive, well-structured summaries with chronological timelines.

SYSTEM INSTRUCTIONS:
1. **Concise but Comprehensive**: Create summaries that capture all key information while remaining concise and readable.
2. **Chronological Timeline**: Always organize events, actions, and developments in chronological order.
3. **Structured Format**: Include sections: Executive Summary, Key Events Timeline, Main Parties, Critical Details, Outcomes.
4. **Detail Preservation**: Include names, dates, locations, and specific details.
5. **Clarity and Readability**: Use professional language and explain technical terms.
6. **Factual Accuracy**: Maintain complete accuracy with no assumptions.
7. **Logical Flow**: Ensure clear logical and causal flow."""

def refine_summary(_: str) -> str:
    question_template = summarization_prompt + """

DOCUMENT CHUNK TO SUMMARIZE:
{text}

Please create a comprehensive summary of this document chunk following the system instructions above.

SUMMARY:
"""
    question_prompt = PromptTemplate(
        input_variables=["text"],
        template=question_template
    )

    refine_template ="""

NEW DOCUMENT CHUNK:
{text}

EXISTING SUMMARY:
{existing_answer}

Please refine the existing summary by incorporating information from the new document chunk. Follow the system instructions above.

REFINED SUMMARY:
"""
    refine_prompt = PromptTemplate(
        input_variables=["text", "existing_answer"],
        template=refine_template
    )

    chain = load_summarize_chain(
        llm, 
        chain_type="refine",
        question_prompt=question_prompt,
        refine_prompt=refine_prompt
    )
    return chain.invoke({"input_documents": split_docs})

tools = [
    Tool(
        name="RefineSummarizer",
        func=refine_summary,
        description="Use this tool to create comprehensive summaries of documents with chronological timelines and structured format"
    )
]

summarize_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=50,  # default is 5
    max_execution_time=300,  # seconds
    agent_kwargs={
        "system_message": "You are a document summarization assistant. When asked to summarize a document, use the RefineSummarizer tool."
    }
)

def run_summary_chain():
    document_text = "\n\n".join([doc.page_content for doc in split_docs])
    result = summarize_chain.invoke({"input": f"summarize the following document:\n{document_text}"})
    return result["output"]




if __name__ == "__main__":
    print("Summarize Chain loaded successfully!")
    print("Starting document summarization using agent...")

    try:
        result = run_summary_chain()

        print("\n" + "="*50)
        print("SUMMARY RESULT:")
        print("="*50)
        print(result)
        print("="*50)

        os.makedirs("results", exist_ok=True)

        wrapped = "\n\n".join(
            textwrap.fill(p, width=100) for p in result.split("\n\n")
        )  # or 80 if you prefer narrower
        with open("results/summary_result.txt", "w", encoding="utf-8") as f:
            f.write(wrapped)
        print("\nSummary saved to results/summary_result.txt")

    except Exception as e:
        print(f"Error during summarization: {e}")

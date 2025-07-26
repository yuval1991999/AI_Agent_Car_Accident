from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import sys
import os
from helpers.config import OPENAI_API_KEY
from helpers.document_loader import split_docs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# === LLM Setup ===
llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-4.1-nano")


# === Summarization Functions ===

def refine_summary(_):
    chain = load_summarize_chain(llm, chain_type="refine")
    return chain.run(split_docs)


# === Tools Setup ===
tools = [
    Tool(
        name="RefineSummarizer",
        func=refine_summary,
        description="Useful for summarizing when more detailed iterative refinement is preferred"
    )
]

# === Initialize Summarization Agent ===
summarize_chain = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)


def run_summary_chain(prompt: str):
    return summarize_chain.run(prompt)

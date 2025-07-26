from chains.qa_chain import run_qa_chain
from ragas_evaluations.ragas_eval_summarize import get_summary
from ragas_evaluations.ragas_eval_qna import make_qna_ragas_evaluation
from ragas_evaluations.ragas_eval_summarize import make_summarize_evaluation
import gradio as gr
from langchain_openai import ChatOpenAI
from helpers.config import OPENAI_API_KEY

llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY, model="gpt-4.1-nano")

# 📦 Router prompt
router_prompt = """
You are a smart AI router that decides what to do with a user's input.
If the user wants a summary, respond only with 'summarization'.
If the user is asking a question about the content, respond only with 'qa'.

Examples:
Input: What is the main idea of this document?
Action: summarization

Input: Who is mentioned in the document?
Action: qa

Now analyze:
Input: {input}
Action:
"""


def route_agent(prompt: str):
    """LLM-based routing to decide between Q&A and summarization."""
    decision = llm.predict(router_prompt.format(input=prompt)).strip().lower()

    if decision == "qa":
        print("🤖 Routing to Q&A Agent...")
        make_qna_ragas_evaluation(prompt)
        return run_qa_chain(prompt)
    elif decision == "summarization":  # The keywords have indicated summarization intent
        print("📝 Routing to Summarization Agent...")
        make_summarize_evaluation()
        return get_summary()
    else:  # We can add and detect explicit instructions to avoid summarization
        return "❌ Could not determine if your input was a question or a summary request."


def main():
    interface = gr.Interface(
        fn=route_agent,
        inputs=gr.Textbox(label="Prompt", placeholder="Ask a question or request a summary"),
        outputs="text",
        title="Main Agent (LLM-Routed)",
    )
    interface.launch()


if __name__ == "__main__":
    main()

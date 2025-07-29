from ragas.metrics import answer_correctness
from ragas.evaluation import evaluate
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from helpers.config import SUMMARY_RESULT_PATH, SUMMARY_GROUND_TRUTH_PATH, RAGAS_SUMMARIZE_EVALUATION_PATH, FILE_PATH, \
    OPENAI_API_KEY
from langchain_openai import ChatOpenAI
from datasets import Dataset


def get_summary():
    return open(SUMMARY_RESULT_PATH, 'r', encoding='utf-8').read()


def make_summarize_evaluation():
    print("Making summarize evaluation...")
    llm = ChatOpenAI(temperature=0, api_key=OPENAI_API_KEY)

    question = "Summarize the document in the best way possible. Focus on creating a clear and structured summary"
    context = open(FILE_PATH, 'r', encoding='utf-8').read()
    summary = get_summary()
    ground_truth = open(SUMMARY_GROUND_TRUTH_PATH, 'r', encoding='utf-8').read()

    eval_dataset = Dataset.from_dict({
        "question": [question],
        "context": [context],
        "answer": [summary],
        "ground_truth": [ground_truth]
    })

    try:
        results = evaluate(
            eval_dataset,
            metrics=[answer_correctness],
            llm=llm
        )

        answer_correctness_score = results["answer_correctness"][0]
        print("\n=== RAGAS Answer Correctness Score ===")
        print(f"Answer Correctness: {answer_correctness_score:.4f}")

        with open(RAGAS_SUMMARIZE_EVALUATION_PATH, "w", encoding="utf-8") as f:
            f.write("=== RAGAS ANSWER CORRECTNESS EVALUATION FOR SUMMARY ===\n")
            f.write(f"Answer Correctness Score: {answer_correctness_score:.4f}\n")
            f.write("\n--- SUMMARY ---\n")
            f.write(summary)
            f.write("\n\n--- CONTEXT (truncated) ---\n")
            f.write(context[:1000])

    except Exception as e:
        print(f"❌ Error evaluating answer correctness: {e}")

make_summarize_evaluation()

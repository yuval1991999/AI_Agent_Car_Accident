import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chains.qa_chain import run_qa_chain, retriever
from helpers.questions import questions, ground_truths
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.evaluation import evaluate
from datasets import Dataset
from helpers.config import RAGAS_QNA_EVALUATION_PATH
from difflib import get_close_matches


def find_ground_truth(prompt: str):
    """
    Find the closest question and corresponding ground truth.
    """
    match = get_close_matches(prompt, questions, n=1, cutoff=0.5)
    if not match:
        raise ValueError("No close question found for the given prompt.")
    index = questions.index(match[0])
    return match[0], ground_truths[index]


def make_qna_ragas_evaluation(prompt: str):
    print(f"🔍 Evaluating QnA for: {prompt}")
    question, ground_truth = find_ground_truth(prompt)

    # Get answer and context
    answer = run_qa_chain(prompt)
    context = [doc.page_content for doc in retriever.get_relevant_documents(prompt)]

    # Print for debug
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print(f"Ground Truth: {ground_truth}\n")

    dataset = Dataset.from_dict({
        "question": [prompt],
        "contexts": [context],
        "response": [answer],
        "ground_truth": [ground_truth]
    })

    results = evaluate(
        dataset,
        metrics=[faithfulness, context_precision, context_recall]
    )

    results_dict = {
        "faithfulness": float(results["faithfulness"][0]),
        "context_precision": float(results["context_precision"][0]),
        "context_recall": float(results["context_recall"][0])
    }
    
    print("=== RAGAS EVALUATION RESULTS FOR Q&A ===\n")
    for key, val in results_dict.items():
        print(f"{key.replace('_', ' ').title()}: {val:.4f}\n")

    with open(RAGAS_QNA_EVALUATION_PATH, "w", encoding="utf-8") as f:
        f.write("=== RAGAS EVALUATION FOR SINGLE Q&A ===\n")
        f.write(f"Question: {prompt}\n")
        f.write(f"Answer: {answer}\n")
        f.write(f"Ground Truth: {ground_truth}\n\n")
        f.write("=== METRICS ===\n")
        for key, val in results_dict.items():
            f.write(f"{key.replace('_', ' ').title()}: {val:.4f}\n")

    print("✅ Evaluation complete. Check the output file.")

    return answer

make_qna_ragas_evaluation("Where was the accident?")
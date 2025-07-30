import os
from dotenv import load_dotenv

# === CONFIGURABLE VARIABLES ===
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
FILE_PATH = "./Intersection_at_Maple_Street.txt"
CHROMA_DIR = "./chroma_store"
RAGAS_QNA_EVALUATION_PATH = "./results/ragas_qna_evaluation_result.txt"
RAGAS_SUMMARIZE_EVALUATION_PATH = "./results/ragas_summarize_evaluation_result.txt"
SUMMARY_RESULT_PATH = "./results/summary_result.txt"
SUMMARY_GROUND_TRUTH_PATH = "./results/refinement_summary_ground_truth.txt"
RETRIEVED_DOCS_PATH = "./results/retrieved_chunks.txt"
TOP_K = 5 # Number of documents to retrieve
CHUNK_SIZE = 500   # Size of text chunks in characters
CHUNK_OVERLAP = 100


# CHUNK_SIZE = [500, 800, 1000, 1500]
# CHUNK_OVERLAP = [50, 100, 200]
# TOP_K_VALUES = [3, 5]
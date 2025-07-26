# Project Setup and Usage

This repository provides a simple RAG example that works with a sample car-accident document. The main components are:

- `Car_Accident_Insurance_Case.txt` – the source document used to generate answers or summaries.
- `main_agent.py` – an interactive entry point that routes prompts to a Q&A or summarization agent.
- The `ragas_evaluations/` directory – scripts that evaluate either Q&A or summarization output.
- The `results/` directory – contains example evaluation files and the summary text produced from the document.

# Car Accident Insurance RAG

This project demonstrates retrieval-augmented generation (RAG) for both summarization and question answering over a sample case study. The case study text is provided in `Car_Accident_Insurance_Case.txt`.

The `helpers/` package stores configuration, document loading utilities, and example prompts. Evaluation scripts and results are in the `ragas_evaluations/` and `results/` directories.

The system indexes the content of **Car_Accident_Insurance_Case.txt** and allows interactive queries or summary generation via `main_agent.py`.

The files demonstrate how to build question answering and summarization pipelines with LangChain and OpenAI as well as how to evaluate them using the **ragas** metrics framework.

The main entry point is `main_agent.py`, which routes user prompts either to the QA chain or the summarization chain. RAGAS evaluation scripts can be found under `ragas_evaluations/`.


## Dependencies

Install Python packages from `requirements.txt`:

## Requirements

Dependencies are listed in `requirements.txt` and include LangChain, OpenAI, ChromaDB, and ragas. Install them with:

Install dependencies with:


```bash
pip install -r requirements.txt
```

`requirements.txt` lists packages like `langchain`, `openai`, `ragas`, `datasets`, and `chromadb`.

## API Key and Environment Setup

This project uses the OpenAI API via the `OPENAI_API_KEY` environment variable. You can provide it in one of two ways:

**Option 1 – Export it in your shell:**

```bash
export OPENAI_API_KEY="sk-..."

**Option 2 – Create a .env file at the repository root:**
OPENAI_API_KEY=your-key-here

The key will be loaded using python-dotenv in helpers/config.py:

## Running the Agent

To start the interactive agent, run:

```bash
python main_agent.py
```

You will be prompted for text. Questions are answered using the document, and summarization prompts trigger the summarize evaluation.
`main_agent.py` decides whether the prompt is a question or a request for a summary and calls the corresponding chain.

## Evaluation

The folder `ragas_evaluations/` contains scripts for RAGAS-based evaluation:

- `ragas_eval_qna.py` evaluates question answering by comparing answers and context with ground truths.
- `ragas_eval_summarize.py` checks summarization correctness against a reference summary.

Run these modules directly to produce evaluation scores in `results/`.

## Running the Evaluation Scripts

- **Q&A evaluation**:

  ```bash
  python ragas_evaluations/ragas_eval_qna.py
  ```

- **Summarization evaluation**:

  ```bash
  python ragas_evaluations/ragas_eval_summarize.py
  ```

Both scripts write their results to the `results/` directory.

## Sample Document and Expected Outputs

The file `Car_Accident_Insurance_Case.txt` contains the car-accident case study that the agents operate on. Running the provided scripts will create/update text files in `results/`:

- `summary_result.txt` – the generated summary of the case.
- `ragas_qna_evaluation_result.txt` – metrics for one Q&A example.
- `ragas_summarize_evaluation_result.txt` – metrics for the summary.
- `refinement_summary_ground_truth.txt` – ground-truth summary used for comparison.

## Building Embeddings

Generate and persist embeddings using the helper module:

```bash
python -m helpers.vector_store
```

This uses `OpenAIEmbeddings` and stores vectors in the directory defined by `CHROMA_DIR` in `helpers/config.py`.


This project uses:

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/docs)
- [RAGAS (RAG Evaluation)](https://github.com/explodinggradients/ragas)
- [ChromaDB (Vector Store)](https://www.trychroma.com/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [Gradio (Optional UI)](https://www.gradio.app/)

LLM Evaluation Pipeline
Overview

This repository contains a Python-based evaluation pipeline to automatically assess the reliability of LLM (Large Language Model) responses. The pipeline evaluates AI answers to user queries against the following parameters in real-time:

Response Relevance & Completeness

Hallucination / Factual Accuracy

Latency & Costs

The evaluation uses two inputs:

Chat JSON – contains the conversation history between the user and the AI.

Context JSON – contains context vectors from a knowledge base, which are used to verify factual accuracy of the AI response.

Local Setup Instructions

Clone the repository

git clone <your_repo_url>
cd <repo_folder>


Create and activate a virtual environment (optional but recommended)

python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows


Install dependencies

pip install -r requirements.txt


Run the evaluation pipeline

python working.py


By default, the pipeline evaluates the AI response using the sample JSONs in sample_data/.

For individual testing, you can modify working.py in the Individual Testing Block to input custom queries, context, and AI responses.

Architecture of the Pipeline

Data Loading – JSON files containing chat and context vectors are loaded.

Latest AI Response Extraction – Only the latest AI turn is evaluated. The last user message is used to compute relevance.

Relevance & Completeness – Compares the AI response to the user query and measures token coverage against source context.

Hallucination Detection – Each sentence in the AI response is checked against all provided context sources. Memoization is applied to speed up repeated similarity computations.

Latency & Cost Estimation – Tracks token usage and computes an estimated cost for generating the response.

Report Generation – Generates a structured report including summary metrics, detailed evaluation, and metadata.

Design Decisions

Latest-turn evaluation: Since the AI response is intended for the last user query, evaluating only the latest turn is both logical and efficient.

Context vector grounding: To verify factual accuracy, we rely on the pre-fetched context vectors, which are assumed to contain relevant information for that query.

Optimizations:

Memoization for repeated similarity computations.

Efficient selection of best-supporting sources per sentence.

Individual Testing: Optional testing block allows debugging with custom queries and responses without modifying the main pipeline.

Scalability Considerations

Memoization drastically reduces repeated computations across multiple responses using the same sources.

Vector-based similarity checks can be parallelized or batch-processed for large datasets.

Processing only the latest turn ensures minimal overhead per conversation, even when chat histories contain hundreds of messages.

Token-based cost estimation allows real-time cost monitoring and budgeting when evaluating millions of conversations.

JSON structure separates chat and context data, enabling distributed or incremental evaluation without reprocessing the entire dataset.

With these measures, the pipeline can efficiently scale to millions of daily conversations with minimal latency and cost.

Folder Structure
.
├── functions.py          # Core evaluation functions
├── working.py            # Main script to run evaluations
├── sample_data/          # Sample chat and context JSONs
│   ├── sample-chat-conversation-01.json
│   └── sample_context_vectors-01.json
├── requirements.txt      # Python dependencies
└── README.md             # This file

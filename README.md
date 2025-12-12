# LLM Evaluation Pipeline
## Local Setup Instructions
Ensure the following libraries are installed:

~ pip install json tabulate sentence-transformers

## Architecture of the Evaluation Pipeline
Git repo content:
1) Main.py: This is the main python file to run. 
  Enter the following command in the terminal:
  python main.py
2) Functions.py: This file contains all functions which will be used by main.py
3) Dataset: Contains sample conversation and context json files used in main.py
4) Images: Contains images of terminal outputs 

Input:                           Accepts two JSONs — chat conversation and context vectors.

Latest Interaction Extraction:   Identifies the latest user query and corresponding AI response.

Relevance & Completeness:        Scores the AI response against the user query and context sources using embedding similarity and token coverage.

Hallucination Detection:         Labels unsupported sentences in the response using context-based support scores.

Latency & Cost:                  Measures execution time, token usage, and estimated cost.

Report Generation:               Combines all metrics into a structured JSON report with metadata.

## Why This Solution

Focused Evaluation: Targets the latest user–AI interaction, which aligns with typical real-time evaluation needs.

Context-Aware: Uses provided context vectors to validate factual accuracy and detect hallucinations.

Modular & Maintainable: Separate functions for relevance, completeness, hallucination detection, and cost calculation make the pipeline easy to extend or modify.

Optimized Performance: Memoization and precomputed similarity checks reduce redundant computations, ensuring faster evaluations even on larger datasets.

## Scalability & Real-Time Performance

Efficient Targeting: Only the latest user–AI turn is scored, avoiding unnecessary computations on older messages.

Optimized Hallucination Checks: Memoization and precomputed embeddings reduce repeated similarity calculations.

Token & Cost Awareness: The pipeline tracks tokens and estimates costs, helping manage compute resources efficiently.

Parallelizable: Each conversation can be evaluated independently, making it easy to scale horizontally for millions of chats.

Lightweight Design: Minimal dependencies and modular functions keep latency low, suitable for real-time evaluation scenarios.

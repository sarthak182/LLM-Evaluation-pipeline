# LLM Evaluation Pipeline
## Local Setup Instructions
Install the following libraries

`pip install json tabulate sentence-transformers`

## Architecture of the Evaluation Pipeline
Git repo content:
1) Main.py: This is the main python file to run. 
  Enter the following command in the terminal:
  `python main.py`
2) Functions.py: This file contains all functions which will be used by main.py
3) Sample_data: Contains sample conversation and context json files used in main.py
4) Output: Contains images of terminal outputs 

## Pipeline Steps:

- Input: Two JSONs – chat conversation and context vectors.

- Latest Interaction Extraction: Finds the latest user query and AI response.

- Relevance & Completeness: Scores response via embedding similarity and token coverage.

- Hallucination Detection: Labels unsupported sentences using context vectors.

- Latency & Cost: Tracks execution time, token usage, and estimated cost.

- Report Generation: Produces a structured JSON report with metrics and metadata.

## Why This Solution

- Uses pre-trained “all-mpnet-base-v2” from Sentence-Transformers for relevance scoring.

- Caching & memoization significantly reduce computation time, enabling scalability.

- Provides the following metrics:

  - Relevance: Alignment of response with the query topic.

  - Completeness: Coverage of relevant context information.

  - Support Ratio: Fraction of sentences supported by context.

  - Hallucination Sentences: Sentences lacking context support.

## Scalability & Real-Time Performance

- Efficient Targeting: Only evaluates the latest user–AI turn.

- Optimized Hallucination Checks: Memoization and precomputed embeddings reduce redundant calculations.

- Token & Cost Awareness: Tracks tokens and estimates costs to manage compute efficiently.

- Parallelizable: Each conversation can be processed independently for horizontal scaling.

- Lightweight Design: Minimal dependencies and modular functions keep latency low for real-time use.

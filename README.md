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

1. Input Processing

The pipeline takes two JSON files:

- chat JSON → full conversation history

- context JSON → vector-database results used by the LLM during generation

2. Relevance & Completeness Scoring

We use the all-mpnet-base-v2 sentence-transformer model:

Compute embeddings for the user query, AI response, and context sources.

Measure Relevance → semantic similarity between query and answer.

Measure Completeness → how much of the context the answer actually uses.

Both metrics use cosine similarity + token coverage for accuracy.

3. Hallucination Detection

The AI response is broken into sentences.
For each sentence:

Compare it with all context vectors.

Identify its best support score.

If the score is below a threshold → mark as hallucinated.

Output:

Unsupported sentences

Support ratios

Sentence-level details

Memoization speeds this up significantly.

4. Latency & Cost Tracking

The script measures:

Execution time

Token usage (query, response, context)

Estimated operational cost

This allows the system to report not just quality, but efficiency.

5. Final Report Generation

All metrics are packaged into a clean JSON output containing:

Relevance

Completeness

Support ratio

Hallucinated sentences

Latency & cost breakdown

Metadata (timestamp, source count)

## Why This Solution

- Uses pre-trained “all-mpnet-base-v2” from Sentence-Transformers for relevance scoring.

- Caching & memoization significantly reduce computation time, enabling scalability.

- Provides the following metrics:

  - Relevance: Alignment of response with the query topic.

  - Completeness: Coverage of relevant context information.

  - Support Ratio: Fraction of sentences supported by context.

  - Hallucination Sentences: Sentences lacking context support.

## Scalability & Real-Time Performance

The script remains efficient even at millions of conversations per day because:

- Only the latest turn is evaluated — constant time per conversation.

- Memoization removes redundant embedding computations.

- Embedding models, not LLMs, are used for scoring — reducing compute cost drastically.

- Lightweight design — minimal dependencies, no external API calls, very low latency.

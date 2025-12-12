# LLM Evaluation Pipeline
## Local Setup Instructions
1. Install the following libraries

    `pip install tabulate sentence-transformers`

2. Run the main.py file in the terminal using:
   
    `python main.py`


## Architecture of the Evaluation Pipeline
Git repo content:
1) Main.py: This is the main python file to run. 
  Enter the following command in the terminal:
  `python main.py`
2) Functions.py: This file contains all functions which will be used by main.py
3) Sample_data: Contains sample conversation and context json files used in main.py
4) Output: Contains images of terminal outputs 

## Pipeline Steps:

1. Inputs

    Takes two JSONs:

    - chat JSON → conversation history

    - context JSON → vector DB results used for generating the answer

    Extracts only the latest user query and latest AI response for scoring.

2. Relevance & Completeness

    Uses all-mpnet-base-v2 embeddings to:

    - Measure how relevant the answer is to the query

    - Check how much of the provided context is covered

3. Hallucination Detection

    Breaks the response into sentences and checks each one against context vectors.

    Sentences with low support scores are flagged as hallucinations.
    Memoization is used to speed this up.

5. Latency & Cost

    Tracks:

    - Execution time

    - Token counts

    - Estimated compute cost

6. Output

    Produces a compact JSON report containing:

    - Relevance

    - Completeness

    - Support ratio

    - Hallucinated sentences

    - Latency & cost metrics

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

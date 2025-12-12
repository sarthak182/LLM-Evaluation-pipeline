## Architecture of the Evaluation Pipeline

Input:                           Accepts two JSONs — chat conversation and context vectors.

Latest Interaction Extraction:   Identifies the latest user query and corresponding AI response.

Relevance & Completeness:        Scores the AI response against the user query and context sources using embedding similarity and token coverage.

Hallucination Detection:         Labels unsupported sentences in the response using context-based support scores.

Latency & Cost:                  Measures execution time, token usage, and estimated cost.

Report Generation:               Combines all metrics into a structured JSON report with metadata.

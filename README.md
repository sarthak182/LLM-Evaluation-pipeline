# Architecture of the Evaluation Pipeline

The evaluation pipeline is designed to automatically assess the reliability of AI responses in real-time using two JSON inputs: a chat conversation and context vectors. The architecture can be summarized in the following steps:

Input

chat-conversation.json: Contains all conversation turns between the user and the AI.

context-vectors.json: Contains source text, vector IDs, and other metadata used to verify the AI’s response.

Extract Latest Interaction

The pipeline identifies the latest AI response and the corresponding user query from the conversation JSON.

This ensures the evaluation focuses on the most recent AI answer, which is the relevant output to verify.

Relevance & Completeness Scoring

Compares the AI response with the user query using embedding similarity.

Computes coverage by checking which portions of the AI response are supported by the source context.

Calculates completeness using token overlap between the response and aggregated context keywords.

Outputs a combined score reflecting both relevance and completeness.

Hallucination / Factual Accuracy Detection

Splits the AI response into sentences and computes a support score for each sentence against all context sources.

Sentences below a defined threshold are labeled unsupported, identifying potential hallucinations.

Returns a detailed report of unsupported sentences and their best matching source.

Latency & Cost Estimation

Computes token usage and estimated costs for generating the AI response.

Measures execution time for analysis to track pipeline performance.

Report Generation

Consolidates relevance, completeness, hallucination analysis, and latency/cost metrics into a structured JSON report.

Includes metadata such as generation timestamp and sources used.

Design Notes:

Modular design separates evaluation functions (relevance_and_completeness, detect_hallucinations_optimized, etc.) for easy maintenance.

Memoization and optimized similarity computations improve runtime efficiency.

The pipeline is flexible to accept new JSON inputs or additional context sources without modifying core logic.

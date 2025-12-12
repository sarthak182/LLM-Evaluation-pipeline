import json
import re
import time
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer, util
from tabulate import tabulate

# Initialize embedding model
# model = SentenceTransformer('all-MiniLM-L6-v2') # Model too small
model = SentenceTransformer('all-mpnet-base-v2')

# -----------------------------
# Configuration / thresholds
# -----------------------------
MODEL_CONFIG = {
    "token_estimate_per_word": 1.3,  # rough tokens per word
    "cost_per_1k_tokens_usd": 0.002,  # default estimate (user should change to real model cost)
    "support_score_threshold": 0.6,   # used to label a sentence supported vs unsupported
    "relevance_threshold": 0.4,       # similarity threshold for relevance
}

# -----------------------------
# Utility functions
# -----------------------------

def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def tokenize_text(text: str) -> List[str]:
    # simple word tokenizer
    text = text.lower()
    # remove punctuation except / and : because urls may exist
    text = re.sub(r"[\.,;:\"'()\[\]{}!?\\]", " ", text)
    tokens = [t for t in text.split() if t]
    return tokens

def embedding_similarity(a: str, b: str) -> float:
    emb_a = model.encode(a, convert_to_tensor=True)
    emb_b = model.encode(b, convert_to_tensor=True)
    return util.cos_sim(emb_a, emb_b).item()  # cosine similarity between 0 and 1

# -----------------------------
# Source extraction / merging
# -----------------------------

def extract_source_texts(context_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return list of sources with fields: {'id','text','url','score','tokens'}"""
    sources = []
    data = context_json.get("data", {})

    # Many vector stores format data differently. We'll try to robustly extract "vector_data", "sources", "vectors_info" etc.
    # 1) vector_data (often a list of dicts or strings)
    vec_data = data.get("vector_data")
    if isinstance(vec_data, list):
        for item in vec_data:
            if isinstance(item, dict):
                sources.append({
                    "id": item.get("id"),
                    "text": item.get("text", ""),
                    "url": item.get("source_url") or item.get("url"),
                    "score": None,
                    "tokens": item.get("tokens") or item.get("tokens_count")
                })

    # 2) vectors_info + mapping by vector_id
    vectors_info = None
    sources_section = data.get("sources") or {}
    vectors_info = sources_section.get("vectors_info") or data.get("vectors_info")
    if vectors_info and isinstance(vectors_info, list):
        # We need to find matching texts for each vector_id if available in vector_data
        # build dict from existing sources by id
        by_id = {s.get("id"): s for s in sources}
        for vi in vectors_info:
            vid = vi.get("vector_id") or vi.get("vectorId")
            if vid is None:
                continue
            s = by_id.get(vid)
            if s:
                s["score"] = vi.get("score")
                s["tokens"] = s.get("tokens") or vi.get("tokens_count")
            else:
                sources.append({
                    "id": vid,
                    "text": "",
                    "url": None,
                    "score": vi.get("score"),
                    "tokens": vi.get("tokens_count")
                })

    # final_response sometimes contains model answer parts; we'll not treat them as sources
    # Clean up: ensure unique by id
    unique = {}
    for s in sources:
        key = s.get("id") or hash(s.get("text") or "")
        if key in unique:
            # merge
            existing = unique[key]
            if not existing.get("text") and s.get("text"):
                existing["text"] = s["text"]
            if not existing.get("url") and s.get("url"):
                existing["url"] = s["url"]
            if existing.get("score") is None and s.get("score") is not None:
                existing["score"] = s.get("score")
        else:
            unique[key] = s

    return list(unique.values())

# -----------------------------
# Evaluation primitives
# -----------------------------

def sentence_split(text: str) -> List[str]:
    if not text:
        return []

    # list of abbreviations that are allowed to have periods but NOT sentence breaks
    abbreviations = [
        "Mr.", "Ms.", "Mrs.", "Dr.", "Prof.",
        "St.", "Sr.", "Jr.",
        "etc.", "e.g.", "i.e."
    ]

    # Temporarily replace periods inside abbreviations with a placeholder
    placeholder = "§§§"
    protected_text = text

    for abbr in abbreviations:
        protected_text = protected_text.replace(abbr, abbr.replace(".", placeholder))

    # Now split normally on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+", protected_text.strip())

    # Restore the periods
    restored = [p.replace(placeholder, ".").strip() for p in parts]

    return [p for p in restored if p]



def compute_support_score(sentence: str, sources: List[Dict[str, Any]]) -> Tuple[float, Dict[str, float]]:
    """Return a support score [0..1] for a sentence using best-match similarity to any source text.
    Also return per-source similarity map.
    """
    sims = {}
    best = 0.0
    for i, s in enumerate(sources):
        text = s.get("text") or ""
        sim = embedding_similarity(sentence, text)
        sims[str(s.get("id") or i)] = sim
        if sim > best:
            best = sim
    return best, sims

def detect_hallucinations(response: str, sources: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Label sentences in response as supported / unsupported by the sources.
    Returns {total_sentences, unsupported_count, unsupported_sentences:[...], details:[{sentence, best_score, best_source}]}
    """
    sentences = sentence_split(response)
    details = []
    unsupported = []
    for sent in sentences:
        best, sims = compute_support_score(sent, sources)
        best_src = max(sources, key=lambda x: sims.get(str(x.get("id")) if x.get("id") is not None else "", 0.0), default=None) if sources else None
        details.append({
            "sentence": sent,
            "best_score": best,
            "best_source_id": best_src.get("id") if best_src else None,
        })
        if best < threshold:
            unsupported.append(sent)
    total = len(sentences)
    unsupported_count = len(unsupported)
    support_ratio = 1.0 - (unsupported_count / total) if total > 0 else 0.0
    return {
        "total_sentences": total,
        "unsupported_count": unsupported_count,
        "unsupported_sentences": unsupported,
        "support_ratio": support_ratio,
        "details": details
    }

def detect_hallucinations_optimized(response: str, sources: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    """Label sentences in response as supported / unsupported by the sources.
    Returns {total_sentences, unsupported_count, unsupported_sentences:[...], details:[{sentence, best_score, best_source}]}
    """
    sentences = sentence_split(response)
    details = []
    unsupported = []

    # Precompute source IDs as strings for quick lookup
    source_ids_str = [str(s.get("id")) if s.get("id") is not None else "" for s in sources]

    for sent in sentences:
        best, sims = compute_support_score(sent, sources)

        # Find the source with the maximum similarity efficiently
        max_score = -1.0
        best_src_id = None
        for src, src_id_str in zip(sources, source_ids_str):
            score = sims.get(src_id_str, 0.0)
            if score > max_score:
                max_score = score
                best_src_id = src.get("id")

        details.append({
            "sentence": sent,
            "best_score": best,
            "best_source_id": best_src_id,
        })

        if best < threshold:
            unsupported.append(sent)

    total = len(sentences)
    unsupported_count = len(unsupported)
    support_ratio = 1.0 - (unsupported_count / total) if total > 0 else 0.0

    return {
        "total_sentences": total,
        "unsupported_count": unsupported_count,
        "unsupported_sentences": unsupported,
        "support_ratio": support_ratio,
        "details": details
    }


def relevance_and_completeness(user_message: str, response: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return relevance and completeness heuristics.
    - relevance: similarity between user message and response
    - coverage: how much of the response is supported by sources (support_ratio)
    - combined score: weighted
    """
    rel = embedding_similarity(user_message, response)
    # Use hallucination detector to compute coverage
    halluc = detect_hallucinations_optimized(response, sources, MODEL_CONFIG["support_score_threshold"]) if sources else {"support_ratio": 0.0}
    coverage = halluc.get("support_ratio", 0.0)
    # completeness: does response mention important tokens from sources? We'll compute token overlap between sources and response
    # build aggregated source keywords
    all_source_text = " \n ".join([s.get("text") or "" for s in sources])
    src_tokens = set(tokenize_text(all_source_text))
    resp_tokens = set(tokenize_text(response))
    if src_tokens:
        coverage_tokens = len(src_tokens & resp_tokens) / len(src_tokens)
    else:
        coverage_tokens = 0.0

    completeness = (coverage + coverage_tokens) / 2.0
    combined = (rel * 0.5) + (completeness * 0.5)
    return {
        "relevance": rel,
        "coverage": coverage,
        "coverage_tokens": coverage_tokens,
        "completeness": completeness,
        "combined_score": combined
    }

# -----------------------------
# Latency & cost estimation
# -----------------------------

def estimate_tokens(text: str) -> int:
    words = len(tokenize_text(text))
    return max(1, int(words * MODEL_CONFIG["token_estimate_per_word"]))


def estimate_cost_usd(tokens: int) -> float:
    return tokens / 1000.0 * MODEL_CONFIG["cost_per_1k_tokens_usd"]


def latency_and_costs(chat_json: Dict[str, Any], response_text: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    # latency: try to extract metadata from chat_json for latest AI turn
    latency_ms = None
    # Common fields that might exist: 'latency_ms', 'response_time_ms', 'duration_ms'
    turns = chat_json.get("conversation_turns") or []
    latest_ai = None
    for t in reversed(turns):
        if t.get("role", "").lower().startswith("ai") or t.get("sender_id") == 1:
            latest_ai = t
            break
    if latest_ai:
        latency_ms = latest_ai.get("latency_ms") or latest_ai.get("response_time_ms") or latest_ai.get("duration_ms")

    # cost: estimate on prompt+context+response tokens. We'll estimate prompt as user message tokens and context as tokens in vectors_info
    user_msg = None
    # find last user message before this AI turn
    if latest_ai:
        ai_index = latest_ai.get("turn")
        # naive search for previous turn by turn number
        user_msg = None
        for t in reversed(turns):
            if t.get("turn") and t.get("turn") < ai_index and t.get("role", "").lower().startswith("user"):
                user_msg = t.get("message")
                break
    if not user_msg:
        # fallback to the last user turn anywhere
        for t in reversed(turns):
            if t.get("role", "").lower().startswith("user"):
                user_msg = t.get("message")
                break

    prompt_tokens = estimate_tokens(user_msg or "")
    response_tokens = estimate_tokens(response_text or "")

    context_tokens = 0
    # try to extract tokens from sources
    for s in sources:
        t = s.get("tokens")
        if isinstance(t, int):
            context_tokens += t
        else:
            # estimate from text
            context_tokens += estimate_tokens(s.get("text") or "")

    total_tokens = prompt_tokens + response_tokens + context_tokens
    estimated_cost = estimate_cost_usd(total_tokens)

    return {
        "latency_ms": latency_ms,
        "tokens": {
            "prompt": prompt_tokens,
            "response": response_tokens,
            "context": context_tokens,
            "total": total_tokens
        },
        "estimated_cost_usd": estimated_cost,
        "cost_model": {
            "token_estimate_per_word": MODEL_CONFIG["token_estimate_per_word"],
            "cost_per_1k_tokens_usd": MODEL_CONFIG["cost_per_1k_tokens_usd"]
        }
    }


def print_full_report(report: dict):
    summary = report.get("summary", {})
    details = report.get("details", {})
    halluc = details.get("hallucination_details", {})
    lat_cost = details.get("latency_and_costs", {})

    # === Summary Table ===
    summary_table = [
        ["relevance", summary.get("relevance")],
        ["completeness", summary.get("completeness")],
        ["combined_score", summary.get("combined_score")],
        ["support_ratio", summary.get("support_ratio")],
        ["coverage_tokens", details.get("relevance_details", {}).get("coverage_tokens")],
        ["sources_count", details.get("sources_count")]
    ]
    print("\n=== Summary ===")
    print(tabulate(summary_table, headers=["Metric", "Value"], tablefmt="grid", floatfmt=".6f"))

    # === Hallucination Details ===
    if halluc and halluc.get("details"):
        hall_table = []
        for d in halluc["details"]:
            hall_table.append([d.get("sentence"), d.get("best_score"), d.get("best_source_id")])
        print("\n=== Hallucination Details ===")
        print(tabulate(hall_table, headers=["Sentence", "Best Score", "Source ID"], tablefmt="grid", floatfmt=".6f"))

    # === Latency & Costs ===
    if lat_cost:
        tokens = lat_cost.get("tokens", {})
        cost_model = lat_cost.get("cost_model", {})
        lat_table = [
            ["latency_ms", lat_cost.get("latency_ms")],
            ["prompt_tokens", tokens.get("prompt")],
            ["response_tokens", tokens.get("response")],
            ["context_tokens", tokens.get("context")],
            ["total_tokens", tokens.get("total")],
            ["estimated_cost_usd", lat_cost.get("estimated_cost_usd")],
            ["token_estimate_per_word", cost_model.get("token_estimate_per_word")],
            ["cost_per_1k_tokens_usd", cost_model.get("cost_per_1k_tokens_usd")]
        ]
        print("\n=== Latency & Cost Details ===")
        print(tabulate(lat_table, headers=["Metric", "Value"], tablefmt="grid", floatfmt=".6f"))

    # === Metadata ===
    metadata = report.get("metadata", {})
    if metadata:
        print("\n=== Metadata ===")
        for k, v in metadata.items():
            print(f"{k}: {v}")

# -----------------------------
# Top-level evaluation
# -----------------------------

def evaluate(chat_json: Dict[str, Any], context_json: Dict[str, Any]) -> Dict[str, Any]:
    # extract latest AI response
    turns = chat_json.get("conversation_turns") or []
    latest_ai = None
    latest_user = None
    for t in reversed(turns):
        role = t.get("role", "").lower()
        sender = t.get("sender_id")

        # Identify AI message
        if latest_ai is None and (role.startswith("ai") or role.startswith("assistant") or sender == 2):
            latest_ai = t

        # Identify User message
        if latest_user is None and (role.startswith("user") or sender == 1):
            latest_user = t

        if latest_ai and latest_user:
            break


    if latest_ai is None:
        raise ValueError("No AI/Chatbot turn found in chat JSON")

    response_text = latest_ai.get("message", "")
    user_message = (latest_user or {}).get("message", "")

    # extract sources
    sources = extract_source_texts(context_json)

    # Relevance & Completeness
    rel = relevance_and_completeness(user_message, response_text, sources)

    # Hallucination / factual accuracy
    halluc = detect_hallucinations_optimized(response_text, sources, MODEL_CONFIG["support_score_threshold"]) if sources else {}

    # Latency & Costs
    lat_cost = latency_and_costs(chat_json, response_text, sources)

    # Build final report
    report = {
        "summary": {
            "relevance": rel.get("relevance"),
            "completeness": rel.get("completeness"),
            "combined_score": rel.get("combined_score"),
            "support_ratio": halluc.get("support_ratio")
        },
        "details": {
            "relevance_details": rel,
            "hallucination_details": halluc,
            "latency_and_costs": lat_cost,
            "sources_count": len(sources)
        },
        "metadata": {
            "generated_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        }
    }

    return report
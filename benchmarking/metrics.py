# -*- coding: utf-8 -*-
"""
Centralized metric computation for Moroccan RAG benchmarking.

Metric categories
─────────────────
1.  RAGAS (full suite)        — faithfulness, answer_relevancy, context_precision,
                                context_recall, answer_correctness, answer_similarity
2.  ARES-style LLM judge      — context_relevance, answer_faithfulness, answer_relevance
                                (implemented via Ollama so no Stanford ARES install needed)
3.  Lexical                   — ROUGE-L, Token-F1, Exact Match
4.  Semantic                  — BERTScore (multilingual xlm-roberta-large)
5.  Domain                    — keyword_hit_rate, abstain_accuracy (TP/FP/TN/FN/F1)
6.  Efficiency                — avg/p50/p95/p99 latency, abstain_rate, avg_contexts
7.  V12-specific              — CFI, claim_grounded_ratio, is_grounded_rate
8.  Breakdown helpers         — per_category_scores(), per_language_scores()
9.  Win-rate matrix           — win_rate_matrix()
10. Comparison table          — print_comparison_table()
"""

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("benchmarking")


def _openai_client(base_url: str, api_key: Optional[str] = None):
    """
    Return an OpenAI-compatible client for any provider.

    Provider detection is URL-based so the same helper works for:
      • Local Ollama / LM Studio  → http://localhost:11434/v1         (no key needed)
      • OpenAI                    → https://api.openai.com/v1          OPENAI_API_KEY
      • Google Gemini             → https://generativelanguage.          GEMINI_API_KEY
                                     googleapis.com/v1beta/openai/
      • Groq                      → https://api.groq.com/openai/v1     GROQ_API_KEY
      • Anthropic (via proxy)     → custom                              ANTHROPIC_API_KEY
      • Any other remote          → reads JUDGE_API_KEY (universal)

    Key resolution order (first non-empty wins):
      1. Explicit api_key argument
      2. JUDGE_API_KEY  env var  (universal override for any provider)
      3. Provider-specific env var (OPENAI_API_KEY / GEMINI_API_KEY / GROQ_API_KEY …)
      4. "ollama"  (for localhost / 127.0.0.1 / 0.0.0.0 endpoints)
    """
    from openai import OpenAI

    if api_key is not None:
        return OpenAI(base_url=base_url, api_key=api_key)

    # Universal override
    api_key = os.environ.get("JUDGE_API_KEY", "")
    if api_key:
        return OpenAI(base_url=base_url, api_key=api_key)

    # Detect local vs remote
    _is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))
    if _is_local:
        api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        return OpenAI(base_url=base_url, api_key=api_key)

    # Remote endpoint — pick provider-specific key
    _PROVIDER_KEYS = [
        ("openrouter.ai",   "OPENROUTER_API_KEY"),
        ("googleapis.com",  "GEMINI_API_KEY"),
        ("openai.com",      "OPENAI_API_KEY"),
        ("groq.com",        "GROQ_API_KEY"),
        ("anthropic.com",   "ANTHROPIC_API_KEY"),
        ("azure",           "AZURE_OPENAI_API_KEY"),
        ("together.ai",     "TOGETHER_API_KEY"),
        ("mistral.ai",      "MISTRAL_API_KEY"),
    ]
    for domain, env_var in _PROVIDER_KEYS:
        if domain in base_url:
            api_key = os.environ.get(env_var, "")
            if api_key:
                return OpenAI(base_url=base_url, api_key=api_key)
            raise EnvironmentError(
                f"Remote judge endpoint detected ({domain}) but {env_var} env var is not set.\n"
                f"  Set it with:  set {env_var}=your-key-here\n"
                f"  Or use the universal override: set JUDGE_API_KEY=your-key-here"
            )

    # Unknown remote — try generic OPENAI_API_KEY, then fail gracefully
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            f"Remote judge endpoint '{base_url}' requires an API key.\n"
            f"Set JUDGE_API_KEY=your-key  (works for any provider)"
        )
    return OpenAI(base_url=base_url, api_key=api_key)


# ══════════════════════════════════════════════════════════════════════════════
# Judge completion helper — thinking-safe across Ollama and cloud providers
# ══════════════════════════════════════════════════════════════════════════════

def _is_local_endpoint(base_url: str) -> bool:
    return any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))


def _ollama_native_url(base_url: str) -> str:
    """Convert an OpenAI-style Ollama URL (…/v1) to the native /api/chat URL."""
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3]
    return root.rstrip("/") + "/api/chat"


def judge_complete(
    client,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """
    Run a single judge completion and return the text content.

    CRITICAL FIX — Ollama thinking models (gemma4:e4b, qwen3, …):
    On the OpenAI-compatible /v1 endpoint Ollama dumps the model's *thinking*
    inline into `content`, consuming the max_tokens budget. With a small budget
    (the judge used max_tokens=5) `content` comes back EMPTY → every score = 0.0.

    `think:false` is NOT honored on the /v1 endpoint, but IS honored on the
    native /api/chat endpoint (thinking goes to a separate `thinking` field,
    `content` stays clean). So for local Ollama we call the native endpoint with
    think:false; for cloud providers we use the normal OpenAI SDK path.

    Returns the assistant message content (may be empty string on failure).
    """
    # Local Ollama → native /api/chat with think disabled (fast + clean)
    if _is_local_endpoint(base_url):
        try:
            import requests
            resp = requests.post(
                _ollama_native_url(base_url),
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "think": False,                       # ← the actual fix
                    "options": {"num_predict": max_tokens, "temperature": temperature},
                },
                timeout=120,
            )
            resp.raise_for_status()
            return (resp.json().get("message", {}).get("content") or "").strip()
        except Exception as exc:
            log.debug("[judge] native /api/chat failed (%s) — falling back to /v1", exc)
            # fall through to the OpenAI SDK path

    # Cloud providers (OpenAI / Gemini / Groq …) or fallback
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content or ""
        # Strip any inline <think>…</think> the model may emit
        if "</think>" in text:
            text = text.rsplit("</think>", 1)[-1]
        return text.strip()
    except Exception as exc:
        log.debug("[judge] completion failed: %s", exc)
        return ""


# ══════════════════════════════════════════════════════════════════════════════
# 1. RAGAS — full metric suite
# ══════════════════════════════════════════════════════════════════════════════

# All RAGAS metrics available in ragas >= 0.1.0
_RAGAS_CORE = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]
_RAGAS_EXTENDED = [
    "answer_correctness",   # combines semantic similarity + factual correctness vs gold
    "answer_similarity",    # pure semantic similarity between answer and gold_answer
]


def _resolve_judge_key(base_url: str) -> str:
    """Resolve the API key string for a judge endpoint (mirrors _openai_client)."""
    key = os.environ.get("JUDGE_API_KEY", "")
    if key:
        return key
    if _is_local_endpoint(base_url):
        return os.environ.get("OPENAI_API_KEY", "ollama")
    for domain, env_var in [
        ("openrouter.ai", "OPENROUTER_API_KEY"),
        ("googleapis.com", "GEMINI_API_KEY"), ("openai.com", "OPENAI_API_KEY"),
        ("groq.com", "GROQ_API_KEY"), ("anthropic.com", "ANTHROPIC_API_KEY"),
        ("azure", "AZURE_OPENAI_API_KEY"), ("together.ai", "TOGETHER_API_KEY"),
        ("mistral.ai", "MISTRAL_API_KEY"),
    ]:
        if domain in base_url:
            k = os.environ.get(env_var, "")
            if k:
                return k
            raise EnvironmentError(f"{env_var} not set for judge endpoint {base_url}")
    k = os.environ.get("OPENAI_API_KEY", "")
    if not k:
        raise EnvironmentError(f"Judge endpoint {base_url} requires a key (set JUDGE_API_KEY).")
    return k


# Embedding-based RAGAS metrics (bge-m3 / xlm-roberta family). Validated to be
# unreliable for Arabizi (romanized Latin Arabic): matched-vs-mismatched
# discrimination ~4x weaker than Darija/MSA/French. Arabizi rows are excluded
# from THESE metrics only — the LLM-based RAGAS metrics (faithfulness,
# context_precision, context_recall) keep all languages.
_RAGAS_EMBED_SUBSTRINGS = ("similarity", "relevancy", "correctness")


def _ragas_result_to_dict(result, languages: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extract aggregate metric scores from a RAGAS result across versions.

    ragas <0.2  : evaluate() returned a dict (had .items()).
    ragas 0.2+  : returns an EvaluationResult whose `.scores` is a list of
                  per-row {metric: value} dicts. Aggregate = mean over rows,
                  skipping NaN (failed/timed-out rows) so partial runs still
                  yield a number.

    languages : optional per-row language labels (aligned 1:1 with the dataset
                rows). When provided, Arabizi rows are dropped from the
                embedding-based metrics (_RAGAS_EMBED_SUBSTRINGS) because the
                multilingual embedder can't reliably embed romanized Arabic.
    """
    import math

    def _is_embed(metric_name: str) -> bool:
        ml = metric_name.lower()
        return any(s in ml for s in _RAGAS_EMBED_SUBSTRINGS)

    # Old dict-like API (no per-row info → can't language-filter; return as-is)
    if hasattr(result, "items"):
        try:
            return {str(k): float(v) for k, v in result.items()}
        except Exception:
            pass
    # EvaluationResult.scores = list[dict], aligned 1:1 with dataset rows
    scores = getattr(result, "scores", None)
    if scores:
        rows = [dict(r) for r in scores]
        langs = languages if (languages and len(languages) == len(rows)) else None
        agg: Dict[str, list] = {}
        for i, row in enumerate(rows):
            is_arabizi = bool(langs) and langs[i] == "Arabizi"
            for k, v in row.items():
                if not isinstance(v, (int, float)) or math.isnan(float(v)):
                    continue
                if is_arabizi and _is_embed(str(k)):
                    continue  # bge-m3 unreliable on Arabizi → skip for embed metrics
                agg.setdefault(str(k), []).append(float(v))
        if agg:
            return {k: sum(vs) / len(vs) for k, vs in agg.items() if vs}
    # Last resort: to_pandas numeric columns
    try:
        df = result.to_pandas()
        skip = {"user_input", "question", "retrieved_contexts", "contexts", "response",
                "answer", "reference", "ground_truth", "reference_contexts"}
        out: Dict[str, float] = {}
        for col in df.columns:
            if col in skip:
                continue
            if str(df[col].dtype).startswith(("float", "int")):
                out[str(col)] = float(df[col].mean())
        return out
    except Exception:
        return {}


def build_ragas_llm(judge_url: str, judge_model: str, timeout: int = 180):
    """
    Wrap the configured judge model as a RAGAS LLM so RAGAS metrics use the SAME
    model as ARES/G-Eval/FActScore — instead of RAGAS's version-dependent OpenAI
    default. Returns a LangchainLLMWrapper, or None if RAGAS/langchain missing.

    NOTE: a LOCAL Ollama judge goes through the /v1 endpoint here (langchain
    ChatOpenAI is OpenAI-protocol only) and is subject to the think-mode
    empty-content issue — run RAGAS with a CLOUD judge (e.g. gpt-4o-mini) for
    reliable scores.
    """
    try:
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
    except ImportError:
        log.warning("[RAGAS] langchain_openai/ragas wrapper unavailable — using RAGAS default LLM")
        return None
    key = _resolve_judge_key(judge_url)
    chat = ChatOpenAI(model=judge_model, base_url=judge_url, api_key=key,
                      temperature=0.0, timeout=timeout, max_retries=2)
    log.info("[RAGAS] judge LLM wired: model=%s endpoint=%s", judge_model, judge_url)
    return LangchainLLMWrapper(chat)


def build_ragas_embeddings(model: str = "BAAI/bge-m3", device: str = "cpu"):
    """
    Local embeddings for RAGAS's embedding-based metrics (answer_relevancy,
    answer_similarity, answer_correctness).

    REQUIRED for chat-only judge endpoints that expose NO /embeddings API —
    e.g. OpenRouter, Groq, Gemini's OpenAI-compat proxy. Without it RAGAS falls
    back to OpenAI embeddings (needs a real OPENAI_API_KEY) and those metrics
    fail. bge-m3 is multilingual → better for Arabic/Darija than OpenAI ada,
    and matches the embedder the RAG system itself uses.

    Runs on CPU by default to avoid VRAM contention with the running API's
    models. Override with RAGAS_EMBED_DEVICE=cuda if the GPU has headroom.

    Returns a LangchainEmbeddingsWrapper, or None if unavailable.
    """
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        log.warning("[RAGAS] HF embeddings unavailable — embedding-based RAGAS "
                    "metrics (answer_relevancy/similarity/correctness) will fail")
        return None
    emb = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    log.info("[RAGAS] local embeddings wired: %s on %s", model, device)
    return LangchainEmbeddingsWrapper(emb)


def compute_ragas_scores(
    ragas_dataset,
    metrics: Optional[List] = None,
    extended: bool = True,
    llm=None,
    embeddings=None,
    languages: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Run the full RAGAS suite on a ragas.Dataset.

    Parameters
    ----------
    ragas_dataset : datasets.Dataset
        Built via ragas_adapter.build_ragas_dataset().
    metrics : list, optional
        Override metric list. None → all core + extended metrics.
    extended : bool
        If True, include answer_correctness + answer_similarity in addition
        to the 4 core metrics. Ignored when metrics is provided explicitly.

    Returns
    -------
    Dict[str, float]  metric_name → score (0–1)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
    except ImportError:
        raise ImportError("pip install ragas datasets")

    if metrics is None:
        metrics = [faithfulness, answer_relevancy, context_precision, context_recall]

        if extended:
            # These exist in ragas >= 0.1.7 — soft-import each
            for metric_name, import_name in [
                ("answer_correctness", "answer_correctness"),
                ("answer_similarity",  "answer_similarity"),
            ]:
                try:
                    import importlib
                    mod = importlib.import_module("ragas.metrics")
                    metric_obj = getattr(mod, import_name, None)
                    if metric_obj is not None:
                        metrics.append(metric_obj)
                        log.info("[RAGAS] Added extended metric: %s", metric_name)
                    else:
                        log.debug("[RAGAS] %s not available in this ragas version", metric_name)
                except Exception as exc:
                    log.debug("[RAGAS] Could not import %s: %s", metric_name, exc)

    log.info("[RAGAS] Evaluating %d rows with %d metrics…", len(ragas_dataset), len(metrics))
    t0 = time.time()
    _kw = {}
    if llm is not None:
        _kw["llm"] = llm
    if embeddings is not None:
        _kw["embeddings"] = embeddings
    result = evaluate(ragas_dataset, metrics=metrics, **_kw)
    log.info("[RAGAS] Done in %.1fs", time.time() - t0)

    return _ragas_result_to_dict(result, languages=languages)


# ══════════════════════════════════════════════════════════════════════════════
# 1b. RAGAS — additional SOTA metrics (ragas >= 0.2.x)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ragas_extended(ragas_dataset, llm=None, embeddings=None,
                           languages: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Extra RAGAS metrics available in ragas >= 0.2.x:
      - context_entity_recall  : entities from gold answer present in retrieved contexts
      - noise_sensitivity      : faithfulness drop when irrelevant context is injected
      - factual_correctness    : NLI-based fact checking vs gold (replaces answer_correctness)

    Silently skips any metric not available in the installed ragas version.
    """
    out: Dict[str, float] = {}
    candidates = [
        ("context_entity_recall", "context_entity_recall"),
        ("noise_sensitivity",     "noise_sensitivity"),
        ("factual_correctness",   "factual_correctness"),
    ]
    for metric_name, import_name in candidates:
        try:
            import importlib
            from ragas import evaluate
            mod = importlib.import_module("ragas.metrics")
            metric_obj = getattr(mod, import_name, None)
            if metric_obj is None:
                continue
            _kw = {}
            if llm is not None:
                _kw["llm"] = llm
            if embeddings is not None:
                _kw["embeddings"] = embeddings
            result = evaluate(ragas_dataset, metrics=[metric_obj], **_kw)
            _vals = list(_ragas_result_to_dict(result, languages=languages).values())
            out[metric_name] = _vals[0] if _vals else float("nan")
            log.info("[RAGAS-ext] %s = %.4f", metric_name, out[metric_name])
        except Exception as exc:
            log.debug("[RAGAS-ext] %s unavailable: %s", metric_name, exc)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2. ARES-style LLM judge (Ollama-backed — no Stanford ARES required)
# ══════════════════════════════════════════════════════════════════════════════

_ARES_CONTEXT_RELEVANCE_PROMPT = """\
You are an evaluation assistant. Given a question and a retrieved document passage, rate how RELEVANT the document is to answering the question.

Question: {question}
Document: {document}

Rate on a scale of 0 to 1 (0 = completely irrelevant, 1 = highly relevant).
Respond with ONLY a single number between 0 and 1, no explanation.
Score:"""

_ARES_ANSWER_FAITHFULNESS_PROMPT = """\
You are an evaluation assistant. Given a document passage and an answer, rate how FAITHFUL the answer is to the document — i.e., is the answer supported by and grounded in the document?

Document: {document}
Answer: {answer}

Rate on a scale of 0 to 1 (0 = completely unsupported / hallucinated, 1 = fully grounded in document).
Respond with ONLY a single number between 0 and 1, no explanation.
Score:"""

_ARES_ANSWER_RELEVANCE_PROMPT = """\
You are an evaluation assistant. Given a question and an answer, rate how RELEVANT the answer is to the question.

Question: {question}
Answer: {answer}

Rate on a scale of 0 to 1 (0 = completely irrelevant, 1 = directly answers the question).
Respond with ONLY a single number between 0 and 1, no explanation.
Score:"""


def compute_ares_scores(
    results: List[Dict],
    gold_items: List[Dict],
    ollama_base_url: str = "http://localhost:11434/v1",
    model: str = "gemma4:e4b",
    max_contexts: int = 3,
) -> Dict[str, float]:
    """
    ARES-style 3-dimension evaluation via Ollama LLM judge.

    Scores three dimensions per (question, document, answer) triple:
      - ares_context_relevance   : Is the retrieved document relevant to the query?
      - ares_answer_faithfulness : Is the answer grounded in the document?
      - ares_answer_relevance    : Does the answer address the question?

    Parameters
    ----------
    results       : pipeline/baseline result dicts
    gold_items    : gold testset items
    ollama_base_url, model : Ollama connection settings
    max_contexts  : max chunks evaluated per question (for speed)

    Returns
    -------
    Dict[str, float]  averaged scores across all (question, doc) pairs
    """
    try:
        client = _openai_client(ollama_base_url)
    except ImportError:
        raise ImportError("pip install openai")

    def _llm_score(prompt: str) -> float:
        # max_tokens=16: enough for a "0.85"-style number; native endpoint
        # disables thinking so the budget isn't eaten by reasoning tokens.
        text = judge_complete(client, ollama_base_url, model, prompt, max_tokens=16)
        match = re.search(r"0?\.\d+|[01]", text)
        return float(match.group()) if match else 0.0

    cr_scores, af_scores, ar_scores = [], [], []

    for result, gold in zip(results, gold_items):
        question = gold.get("question", result.get("question", ""))
        answer = result.get("answer", "")
        contexts = _extract_contexts(result)[:max_contexts]
        if not contexts:
            contexts = [""]

        for doc in contexts:
            cr_scores.append(_llm_score(
                _ARES_CONTEXT_RELEVANCE_PROMPT.format(question=question, document=doc[:800])
            ))
            af_scores.append(_llm_score(
                _ARES_ANSWER_FAITHFULNESS_PROMPT.format(document=doc[:800], answer=answer[:600])
            ))
            ar_scores.append(_llm_score(
                _ARES_ANSWER_RELEVANCE_PROMPT.format(question=question, answer=answer[:600])
            ))

    n = max(len(cr_scores), 1)
    return {
        "ares_context_relevance":   sum(cr_scores) / n,
        "ares_answer_faithfulness": sum(af_scores) / n,
        "ares_answer_relevance":    sum(ar_scores) / n,
    }


def compute_ares_extended(
    results: List[Dict],
    gold_items: List[Dict],
    ollama_base_url: str = "http://localhost:11434/v1",
    model: str = "gemma4:e4b",
) -> Dict[str, float]:
    """
    5 additional ARES-style judge dimensions tailored to this dataset:
      - ares_completeness      : does the answer cover all aspects of the question?
      - ares_dialect_coherence : Darija/Arabizi Q → answer in correct language?
      - ares_legal_accuracy    : LEGAL items — is the cited law/decree correct?
      - ares_multihop_coverage : MULTIHOP items — are facts from both procedures present?
      - ares_abstain_quality   : OUTSCOPE items — is the refusal helpful and informative?
    """
    client = _openai_client(ollama_base_url)

    def _score(prompt: str) -> float:
        text = judge_complete(client, ollama_base_url, model, prompt, max_tokens=16)
        m = re.search(r"0?\.\d+|[01]", text)
        return float(m.group()) if m else 0.0

    completeness, dialect, legal, multihop, abstain = [], [], [], [], []

    for result, gold in zip(results, gold_items):
        question = gold.get("question", "")
        answer   = result.get("answer", "")
        category = gold.get("category", "")
        language = gold.get("language", "")
        contexts = _extract_contexts(result)
        doc      = contexts[0][:800] if contexts else ""

        # Completeness (all items)
        completeness.append(_score(
            f"Does this answer completely address ALL aspects of the question?\n"
            f"Question: {question}\nAnswer: {answer[:600]}\n"
            f"Rate 0–1 (1=fully complete). Respond with only a number.\nScore:"
        ))

        # Dialect coherence (Darija / Arabizi only)
        if language in ("Darija", "Arabizi"):
            dialect.append(_score(
                f"A user asked a question in Moroccan Darija or Arabizi (romanized Arabic). "
                f"Does the answer respond appropriately in Arabic or Darija (NOT in French or English only)?\n"
                f"Question: {question}\nAnswer: {answer[:400]}\n"
                f"Rate 0–1 (1=correct language). Respond with only a number.\nScore:"
            ))

        # Legal accuracy (LEGAL items only)
        if category == "LEGAL":
            legal.append(_score(
                f"This question asks about a specific Moroccan law, decree, or article number. "
                f"Does the answer correctly cite the EXACT legal reference from the document?\n"
                f"Document: {doc}\nQuestion: {question}\nAnswer: {answer[:600]}\n"
                f"Rate 0–1 (1=exact citation present). Respond with only a number.\nScore:"
            ))

        # Multi-hop coverage (MULTIHOP items only)
        if category == "MULTIHOP":
            multihop.append(_score(
                f"This question requires combining facts from TWO different procedures. "
                f"Does the answer successfully combine information from both procedures?\n"
                f"Question: {question}\nAnswer: {answer[:600]}\n"
                f"Rate 0–1 (1=both procedures' facts present). Respond with only a number.\nScore:"
            ))

        # Abstain quality (OUTSCOPE items only)
        if category == "OUTSCOPE":
            abstain.append(_score(
                f"The system should have refused to answer this out-of-scope question. "
                f"If it refused, is the refusal helpful — does it explain WHY it cannot answer?\n"
                f"Question: {question}\nAnswer: {answer[:400]}\n"
                f"Rate 0–1 (1=correctly refused AND explanation given). Respond with only a number.\nScore:"
            ))

    out = {}
    if completeness: out["ares_completeness"]      = round(sum(completeness) / len(completeness), 4)
    if dialect:      out["ares_dialect_coherence"] = round(sum(dialect)      / len(dialect),      4)
    if legal:        out["ares_legal_accuracy"]    = round(sum(legal)        / len(legal),         4)
    if multihop:     out["ares_multihop_coverage"] = round(sum(multihop)     / len(multihop),      4)
    if abstain:      out["ares_abstain_quality"]   = round(sum(abstain)      / len(abstain),       4)
    return out


def _extract_contexts(result: Dict) -> List[str]:
    """Extract context list from any result dict (v12 or baseline)."""
    if result.get("ragas_contexts"):
        return [c for c in result["ragas_contexts"] if c]
    if result.get("execution_trace", {}).get("all_contexts"):
        return [c for c in result["execution_trace"]["all_contexts"] if c]
    if result.get("retrieval", {}).get("contexts"):
        return [c for c in result["retrieval"]["contexts"] if c]
    return [c for c in result.get("contexts", []) if c]


# ══════════════════════════════════════════════════════════════════════════════
# 2b. G-Eval (Wang et al. 2023) — CoT LLM judge with explicit criteria
# ══════════════════════════════════════════════════════════════════════════════

_GEVAL_PROMPT = """\
You are an expert evaluator for a Moroccan government document question-answering system.
Evaluate the answer on the criterion: {criterion}

Definition: {definition}

Question: {question}
Retrieved context: {context}
Answer: {answer}

Output format (follow EXACTLY):
Score: <a single integer from 1 to 5>
Reason: <one short sentence>

Begin your response with the Score line now."""

_GEVAL_CRITERIA = {
    "coherence": (
        "Coherence",
        "The answer is well-structured, logically organized, and easy to follow. "
        "It presents information in a clear sequence (e.g., documents → cost → deadline)."
    ),
    "consistency": (
        "Consistency",
        "The answer is factually consistent with the retrieved context. "
        "It does not contradict, hallucinate, or add information not present in the context."
    ),
    "fluency": (
        "Fluency",
        "The answer is written in natural, grammatically correct language appropriate "
        "for the question's language (Arabic MSA, French, Darija, or Arabizi)."
    ),
    "relevance": (
        "Relevance",
        "The answer directly addresses the question asked and stays on topic. "
        "It does not include irrelevant tangents or generic filler."
    ),
}


def compute_geval_scores(
    results: List[Dict],
    gold_items: List[Dict],
    ollama_base_url: str = "http://localhost:11434/v1",
    model: str = "gemma4:e4b",
    max_contexts: int = 1,
) -> Dict[str, float]:
    """
    G-Eval: LLM-as-judge with chain-of-thought scoring on 4 NLG dimensions.
    Reference: Wang et al. 2023 — https://arxiv.org/abs/2303.16634

    Returns geval_coherence, geval_consistency, geval_fluency, geval_relevance (1–5 normalized to 0–1).
    """
    client = _openai_client(ollama_base_url)

    def _score(prompt: str) -> float:
        # Prompt demands "Score: N" on the FIRST line, so a small budget is
        # enough (think disabled — no hidden reasoning eats the tokens).
        text = judge_complete(client, ollama_base_url, model, prompt, max_tokens=96)
        m = re.search(r"Score:\s*([1-5](?:\.\d+)?)", text, re.IGNORECASE)
        if not m:
            m = re.search(r"([1-5])", text)  # first digit in range as fallback
        return (float(m.group(1)) - 1) / 4 if m else 0.0  # normalize to 0–1

    scores: Dict[str, List[float]] = {k: [] for k in _GEVAL_CRITERIA}

    for result, gold in zip(results, gold_items):
        question = gold.get("question", "")
        answer   = result.get("answer", "")
        contexts = _extract_contexts(result)
        context  = " ".join(contexts[:max_contexts])[:1200]

        for key, (criterion, definition) in _GEVAL_CRITERIA.items():
            prompt = _GEVAL_PROMPT.format(
                criterion=criterion, definition=definition,
                question=question, context=context, answer=answer[:600],
            )
            scores[key].append(_score(prompt))

    return {
        f"geval_{k}": round(sum(v) / len(v), 4) if v else 0.0
        for k, v in scores.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2c. FActScore-style atomic claim verification (Min et al. 2023)
# ══════════════════════════════════════════════════════════════════════════════

_FACTSCORE_DECOMPOSE = """\
Break the following answer into a list of atomic factual claims.
Each claim must be a single, standalone statement that can be independently verified.
Output ONLY a JSON array of strings, no explanation.
Answer: {answer}
Claims:"""

_FACTSCORE_VERIFY = """\
Does the following document support this claim? Answer only "yes" or "no".
Document: {document}
Claim: {claim}
Answer:"""


def compute_factscore(
    results: List[Dict],
    gold_items: List[Dict],
    ollama_base_url: str = "http://localhost:11434/v1",
    model: str = "gemma4:e4b",
    max_claims: int = 5,
) -> Dict[str, float]:
    """
    FActScore: break each answer into atomic claims, verify each against context.
    Reference: Min et al. 2023 — https://arxiv.org/abs/2305.14251

    Returns:
      factscore            : % of claims supported by retrieved context (0–1)
      unsupported_claim_rate: % of claims NOT supported (hallucination proxy)
      avg_claims_per_answer : avg number of atomic claims extracted
    """
    client = _openai_client(ollama_base_url)

    def _decompose(answer: str) -> List[str]:
        raw = judge_complete(
            client, ollama_base_url, model,
            _FACTSCORE_DECOMPOSE.format(answer=answer[:800]), max_tokens=400,
        )
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            try:
                claims = json.loads(raw[start:end+1])
                return [c for c in claims if isinstance(c, str) and len(c.strip()) > 5][:max_claims]
            except Exception:
                pass
        return []

    def _verify(claim: str, document: str) -> bool:
        text = judge_complete(
            client, ollama_base_url, model,
            _FACTSCORE_VERIFY.format(document=document[:800], claim=claim), max_tokens=8,
        )
        t = (text or "").lower()
        return "yes" in t or "oui" in t or "نعم" in (text or "")

    supported_fracs, claim_counts = [], []

    for result, gold in zip(results, gold_items):
        answer   = result.get("answer", "")
        contexts = _extract_contexts(result)
        document = " ".join(contexts[:2])[:1200]

        if not answer.strip() or not document.strip():
            continue

        claims = _decompose(answer)
        if not claims:
            continue

        claim_counts.append(len(claims))
        supported = sum(1 for c in claims if _verify(c, document))
        supported_fracs.append(supported / len(claims))

    if not supported_fracs:
        return {}

    avg_supported = sum(supported_fracs) / len(supported_fracs)
    return {
        "factscore":              round(avg_supported, 4),
        "unsupported_claim_rate": round(1 - avg_supported, 4),
        "avg_claims_per_answer":  round(sum(claim_counts) / len(claim_counts), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2d. RGB-style robustness metrics (Chen et al. 2023)
# ══════════════════════════════════════════════════════════════════════════════

def compute_rgb_robustness(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, float]:
    """
    RGB (Retrieval-augmented Generation Benchmark) robustness dimensions.
    Reference: Chen et al. 2023 — https://arxiv.org/abs/2309.01431

    Computed without an LLM — uses proxy signals from existing result fields:

    noise_robustness       : avg token_f1 on items where multiple contexts were retrieved
                             (more contexts = more noise risk; high score = robust)
    negative_rejection_rate: fraction of OUTSCOPE items where the system correctly abstained
    information_integration: avg token_f1 on MULTIHOP items only
                             (requires combining info across chunks)
    counterfactual_robustness: fraction of LEGAL items where the exact law number
                               from gold_keywords appears in the answer
    """
    noise_f1, neg_rej, integration_f1, counter = [], [], [], []

    for result, gold in zip(results, gold_items):
        answer   = result.get("answer", "")
        category = gold.get("category", "")
        ref      = gold.get("gold_answer", "")
        keywords = gold.get("gold_keywords", [])
        contexts = _extract_contexts(result)

        # Noise robustness — items where system retrieved ≥ 3 contexts
        if len(contexts) >= 3:
            noise_f1.append(token_f1(answer, ref))

        # Negative rejection — OUTSCOPE items
        if category == "OUTSCOPE":
            abstained = result.get("is_outscope", False) or _has_abstain_marker(answer)
            neg_rej.append(1.0 if abstained else 0.0)

        # Information integration — MULTIHOP items
        if category == "MULTIHOP":
            integration_f1.append(token_f1(answer, ref))

        # Counterfactual robustness — LEGAL items (exact keyword in answer)
        if category == "LEGAL" and keywords:
            answer_norm = _normalize(answer)
            found = any(_normalize(kw) in answer_norm for kw in keywords)
            counter.append(1.0 if found else 0.0)

    out = {}
    if noise_f1:      out["rgb_noise_robustness"]        = round(sum(noise_f1)      / len(noise_f1),      4)
    if neg_rej:       out["rgb_negative_rejection_rate"] = round(sum(neg_rej)       / len(neg_rej),       4)
    if integration_f1:out["rgb_information_integration"] = round(sum(integration_f1)/ len(integration_f1),4)
    if counter:       out["rgb_counterfactual_robustness"]= round(sum(counter)      / len(counter),       4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2e. Retrieval quality metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_retrieval_quality(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, float]:
    """
    Retrieval-focused metrics (no LLM needed):

    context_coverage    : fraction of gold_keywords found in retrieved contexts
                          (measures if relevant facts were actually retrieved)
    context_redundancy  : avg token overlap between consecutive retrieved chunks
                          (high = redundant retrieval, low = diverse coverage)
    mean_reciprocal_rank: proxy MRR — 1/rank of the first chunk containing any gold keyword
    context_utilization : fraction of retrieved context tokens that appear in the answer
                          (measures how well the generator uses what was retrieved)
    """
    coverage, redundancy, mrr, utilization = [], [], [], []

    for result, gold in zip(results, gold_items):
        answer   = result.get("answer", "")
        keywords = gold.get("gold_keywords", [])
        contexts = _extract_contexts(result)

        if not contexts:
            continue

        # Context coverage — gold keywords in contexts
        if keywords:
            all_ctx = " ".join(contexts)
            ctx_norm = _normalize(all_ctx)
            found = sum(1 for kw in keywords if _normalize(kw) in ctx_norm)
            coverage.append(found / len(keywords))

        # Context redundancy — avg pairwise token overlap between chunks
        if len(contexts) >= 2:
            overlaps = []
            for i in range(len(contexts) - 1):
                t1 = set(_normalize(contexts[i]).split())
                t2 = set(_normalize(contexts[i+1]).split())
                if t1 or t2:
                    overlaps.append(len(t1 & t2) / max(len(t1 | t2), 1))
            if overlaps:
                redundancy.append(sum(overlaps) / len(overlaps))

        # MRR — first chunk containing any gold keyword
        if keywords:
            rank = None
            for i, ctx in enumerate(contexts, 1):
                ctx_norm = _normalize(ctx)
                if any(_normalize(kw) in ctx_norm for kw in keywords):
                    rank = i
                    break
            mrr.append(1.0 / rank if rank else 0.0)

        # Context utilization — how much of retrieved info appears in answer
        if answer:
            ctx_tokens = set(_normalize(" ".join(contexts)).split())
            ans_tokens = set(_normalize(answer).split())
            if ctx_tokens:
                utilization.append(len(ans_tokens & ctx_tokens) / len(ctx_tokens))

    out = {}
    if coverage:     out["retrieval_context_coverage"]    = round(sum(coverage)     / len(coverage),     4)
    if redundancy:   out["retrieval_context_redundancy"]  = round(sum(redundancy)   / len(redundancy),   4)
    if mrr:          out["retrieval_mrr"]                 = round(sum(mrr)          / len(mrr),          4)
    if utilization:  out["retrieval_context_utilization"] = round(sum(utilization)  / len(utilization),  4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2f. Domain precision metrics (Morocco-specific)
# ══════════════════════════════════════════════════════════════════════════════

_CITATION_RE = re.compile(r"\[Source:[^\]]*\]", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")


def _strip_citations(text: str) -> str:
    """
    Remove [Source:…] tags and bare URLs before scoring. v12 injects citation
    URLs into answers, whose long path/UUID fragments otherwise (a) pollute
    token_f1/rouge as junk tokens and (b) make URL digits count as hallucinated
    numbers. Citations are metadata, not answer content — strip them for a fair
    comparison across all systems.
    """
    return _URL_RE.sub("", _CITATION_RE.sub("", text or ""))


# Shared abstention markers — the phrases systems actually emit when they refuse
# ("غير متوفر"/"not available"), used by BOTH the abstain metric and RGB. Tuned for
# precision: excludes generic "لا توجد/لا يوجد" which appear in real answers
# ("no fees required"). Matched case-insensitively (lower() leaves Arabic intact).
ABSTAIN_MARKERS = [
    "غير متوفر", "غير متاح", "غير موجودة في", "غير موجود في",
    "ماكاينةش فالوثائق", "ماكاينةش ف لوثائق", "لم أجد", "لم نجد",
    "خارج نطاق", "خارج اختصاص", "خارج النطاق",
    "non disponible", "n'est pas disponible", "ne figure pas",
    "informations insuffisantes", "je ne peux pas", "hors de",
    "not available", "[error", "makaynach f",
]


def _has_abstain_marker(answer: str) -> bool:
    a = (answer or "").lower()
    return any(m in a for m in ABSTAIN_MARKERS)


def compute_domain_precision(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, float]:
    """
    Morocco government RAG-specific precision metrics (no LLM needed):

    legal_citation_hit    : LEGAL items — exact law/decree number from gold_keywords in answer
    cost_deadline_hit     : answers containing numeric cost or deadline values from gold_keywords
    dialect_response_match: Darija/Arabizi questions answered in Arabic script (not French/Latin only)
    multihop_keyword_hit  : MULTIHOP items — fraction of gold_keywords found in answer
    hallucination_number_rate: numbers in answer NOT present in any retrieved context
    """
    legal_hits, cost_hits, dialect_matches, multihop_kw, halluc_nums = [], [], [], [], []

    _NUMBER_RE = re.compile(r'\b\d[\d,\.]*\b')

    for result, gold in zip(results, gold_items):
        answer   = _strip_citations(result.get("answer", ""))
        category = gold.get("category", "")
        language = gold.get("language", "")
        keywords = gold.get("gold_keywords", [])
        contexts = _extract_contexts(result)
        ans_norm = _normalize(answer)

        # Legal citation hit
        if category == "LEGAL" and keywords:
            hit = any(_normalize(kw) in ans_norm for kw in keywords)
            legal_hits.append(1.0 if hit else 0.0)

        # Cost/deadline hit — look for numeric gold_keywords in answer
        numeric_kws = [kw for kw in keywords if any(c.isdigit() for c in kw)]
        if numeric_kws:
            hit = any(_normalize(kw) in ans_norm for kw in numeric_kws)
            cost_hits.append(1.0 if hit else 0.0)

        # Dialect response match — the dialect Q should be answered in the dialect,
        # NOT French. Darija → Arabic script; Arabizi → romanized (digit-letters)
        # OR Arabic script (both are Moroccan colloquial); pure-French = miss.
        if language == "Darija":
            arabic_chars = sum(1 for c in answer if "؀" <= c <= "ۿ")
            dialect_matches.append(1.0 if arabic_chars >= 10 else 0.0)
        elif language == "Arabizi":
            arabic_chars = sum(1 for c in answer if "؀" <= c <= "ۿ")
            has_arabizi = bool(re.search(r"[a-zA-Z][2379]|[2379][a-zA-Z]", answer))
            dialect_matches.append(1.0 if (arabic_chars >= 10 or has_arabizi) else 0.0)

        # MULTIHOP keyword hit
        if category == "MULTIHOP" and keywords:
            found = sum(1 for kw in keywords if _normalize(kw) in ans_norm)
            multihop_kw.append(found / len(keywords))

        # Hallucination number rate — numbers in answer not in any context
        ans_nums = set(_NUMBER_RE.findall(answer))
        if ans_nums and contexts:
            ctx_text = " ".join(contexts)
            ctx_nums = set(_NUMBER_RE.findall(ctx_text))
            halluc = len(ans_nums - ctx_nums) / len(ans_nums)
            halluc_nums.append(halluc)

    out = {}
    if legal_hits:     out["domain_legal_citation_hit"]      = round(sum(legal_hits)     / len(legal_hits),     4)
    if cost_hits:      out["domain_cost_deadline_hit"]       = round(sum(cost_hits)      / len(cost_hits),      4)
    if dialect_matches:out["domain_dialect_response_match"]  = round(sum(dialect_matches)/ len(dialect_matches),4)
    if multihop_kw:    out["domain_multihop_keyword_hit"]    = round(sum(multihop_kw)    / len(multihop_kw),    4)
    if halluc_nums:    out["domain_hallucination_number_rate"]= round(sum(halluc_nums)   / len(halluc_nums),    4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 2g. Cross-lingual consistency
# ══════════════════════════════════════════════════════════════════════════════

def compute_cross_lingual_consistency(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, float]:
    """
    Cross-lingual consistency: for a procedure that appears in MULTIPLE languages
    (same source), the system should answer each language version equally well.

    For every source with ≥2 language versions, each version's answer is scored
    against its own gold (token_f1). Consistency for a language pair = 1 - |f1_a - f1_b|;
    the metric averages this over ALL same-source language pairs (any languages —
    e.g. french↔Darija, french↔Arabizi, Darija↔Arabizi, arabic_msa↔french).

    1.0 = the system is equally accurate regardless of question language.
    """
    from collections import defaultdict
    from itertools import combinations

    by_source: Dict[str, Dict[str, float]] = defaultdict(dict)
    for result, gold in zip(results, gold_items):
        source = gold.get("source", "")
        lang   = gold.get("language", "")
        if source and lang:
            by_source[source][lang] = token_f1(_strip_citations(result.get("answer", "")),
                                               gold.get("gold_answer", ""))

    consistencies = []
    for source, lang_f1 in by_source.items():
        if len(lang_f1) >= 2:
            for la, lb in combinations(sorted(lang_f1), 2):
                consistencies.append(1 - abs(lang_f1[la] - lang_f1[lb]))

    if not consistencies:
        return {}
    return {"cross_lingual_consistency": round(sum(consistencies) / len(consistencies), 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 3. Lexical metrics (no LLM needed)
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^\w\s؀-ۿ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(_normalize(prediction) == _normalize(reference))


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(_normalize(prediction).split())
    ref_tokens  = set(_normalize(reference).split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common    = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L (LCS-based) — no external dependencies."""
    pred = _normalize(prediction).split()
    ref  = _normalize(reference).split()
    if not pred or not ref:
        return 0.0
    m, n = len(pred), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if pred[i-1] == ref[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_lexical_scores(
    results: List[Dict],
    gold_items: List[Dict],
    gold_answer_key: str = "gold_answer",
) -> Dict[str, float]:
    """Average EM, Token-F1, ROUGE-L across the dataset."""
    em, f1, rl = [], [], []
    for result, gold in zip(results, gold_items):
        pred = _strip_citations(result.get("answer", ""))
        ref  = gold.get(gold_answer_key, "")
        em.append(exact_match(pred, ref))
        f1.append(token_f1(pred, ref))
        rl.append(rouge_l(pred, ref))
    n = max(len(em), 1)
    return {
        "exact_match": sum(em) / n,
        "token_f1":    sum(f1) / n,
        "rouge_l":     sum(rl) / n,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4. BERTScore — multilingual semantic similarity
# ══════════════════════════════════════════════════════════════════════════════

def compute_bertscore(
    results: List[Dict],
    gold_items: List[Dict],
    gold_answer_key: str = "gold_answer",
    model_type: str = "xlm-roberta-large",
    device: Optional[str] = None,
) -> Dict[str, float]:
    """
    BERTScore using a multilingual encoder (xlm-roberta-large).

    Covers Arabic MSA, French, and Darija better than monolingual BERT.
    Falls back to bert-base-multilingual-cased if xlm-roberta-large fails.

    device : "cpu" / "cuda". Defaults to env BERTSCORE_DEVICE, else "cpu" — so
    the metrics phase adds NO VRAM on top of the running API (xlm-roberta-large
    would otherwise grab ~2.2 GB on CUDA). Set BERTSCORE_DEVICE=cuda for speed
    if the GPU has headroom. Scores are device-independent.

    pip install bert-score
    """
    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        log.warning("[BERTScore] bert-score not installed — skipping. pip install bert-score")
        return {}

    device = device or os.environ.get("BERTSCORE_DEVICE", "cpu")

    # BERTScore's multilingual encoder does NOT reliably embed Arabizi (romanized
    # Latin-script Arabic) — validated: matched-vs-mismatched discrimination is
    # ~3-4x weaker for Arabizi than for Darija/MSA/French. Exclude Arabizi here;
    # it's scored by arabizi_normalized_f1 + human evaluation instead.
    pairs = [(_strip_citations(r.get("answer", "")), g.get(gold_answer_key, ""))
             for r, g in zip(results, gold_items)
             if (g.get("language") or "") != "Arabizi"]
    predictions = [p for p, _ in pairs]
    references  = [ref for _, ref in pairs]

    if not any(predictions) or not any(references):
        return {}

    try:
        log.info("[BERTScore] Computing with model=%s on %d pairs (device=%s)…",
                 model_type, len(predictions), device)
        P, R, F = bert_score_fn(
            predictions, references,
            model_type=model_type,
            lang="other",        # multilingual — no language-specific tokenizer
            verbose=False,
            rescale_with_baseline=False,
            device=device,
        )
        return {
            "bertscore_precision": float(P.mean()),
            "bertscore_recall":    float(R.mean()),
            "bertscore_f1":        float(F.mean()),
        }
    except Exception as exc:
        # Fallback to smaller multilingual model
        log.warning("[BERTScore] %s failed (%s) — retrying with bert-base-multilingual-cased", model_type, exc)
        try:
            P, R, F = bert_score_fn(
                predictions, references,
                model_type="bert-base-multilingual-cased",
                lang="other",
                verbose=False,
                device=device,
            )
            return {
                "bertscore_precision": float(P.mean()),
                "bertscore_recall":    float(R.mean()),
                "bertscore_f1":        float(F.mean()),
            }
        except Exception as exc2:
            log.warning("[BERTScore] Both models failed: %s", exc2)
            return {}


# ══════════════════════════════════════════════════════════════════════════════
# 4b. Romanization-normalized Arabizi F1 (no standard spelling -> token_f1 unfair)
# ══════════════════════════════════════════════════════════════════════════════

def _norm_arabizi(text: str) -> set:
    """
    Canonicalize Moroccan Arabizi so spelling variants collide: map digit-letters
    to rough phonetic letters, drop residual digits/punct, collapse doubled
    letters, and reduce to a consonant skeleton (vowels vary most across writers).
    """
    t = (text or "").lower()
    t = t.translate(str.maketrans("372985", "ahqakh"))  # 3=3ayn,7=7a,2=hamza,9=qaf,8=gh,5=kha
    t = re.sub(r"[0-9]", "", t)
    t = re.sub(r"[^a-z\s]", " ", t)          # keep Latin letters only
    t = re.sub(r"(.)\1+", r"\1", t)          # collapse doubled letters
    t = re.sub(r"[aeiou]", "", t)            # consonant skeleton
    return set(w for w in t.split() if len(w) >= 2)


def compute_arabizi_normalized(results: List[Dict], gold_items: List[Dict],
                               gold_answer_key: str = "gold_answer") -> Dict[str, float]:
    """
    Romanization-normalized token-F1 over Arabizi items only. Moroccan Arabizi has
    no standard orthography, so two correct answers can share almost no surface
    tokens; raw token_f1 therefore *understates* Arabizi quality. This collapses
    spelling variants before comparison for a fairer dialect score.
    """
    f1s = []
    for r, g in zip(results, gold_items):
        if (g.get("language") or "") != "Arabizi":
            continue
        pred = _norm_arabizi(_strip_citations(r.get("answer", "")))
        ref = _norm_arabizi(g.get(gold_answer_key, ""))
        if not pred or not ref:
            continue
        c = len(pred & ref)
        pr, rc = c / len(pred), c / len(ref)
        f1s.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    if not f1s:
        return {}
    return {"arabizi_normalized_f1": round(sum(f1s) / len(f1s), 4)}


# ══════════════════════════════════════════════════════════════════════════════
# 5. Domain-specific metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_keyword_hit_rate(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, float]:
    """
    Keyword hit rate: fraction of gold_keywords found in the answer.

    gold_keywords are key facts/terms that a correct answer must contain.
    Computed per-question then averaged.
    """
    hits = []
    for result, gold in zip(results, gold_items):
        keywords = gold.get("gold_keywords", [])
        if not keywords:
            continue
        answer_norm = _normalize(result.get("answer", ""))
        found = sum(1 for kw in keywords if _normalize(kw) in answer_norm)
        hits.append(found / len(keywords))

    if not hits:
        return {}
    return {"keyword_hit_rate": sum(hits) / len(hits)}


def compute_abstain_accuracy(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, float]:
    """
    Abstain accuracy — did the system correctly abstain on OOS questions?

    Uses gold_items[i]["should_abstain"] as ground truth.
    Classifies each result as abstained if is_outscope=True or answer
    contains abstain markers (لم أجد / Informations insuffisantes / [ERROR).

    Returns: TP, FP, TN, FN, precision, recall, f1, accuracy
    """
    def _is_abstained(result: Dict) -> bool:
        if result.get("is_outscope") or result.get("is_abstained"):
            return True
        return _has_abstain_marker(result.get("answer", ""))

    TP = FP = TN = FN = 0
    for result, gold in zip(results, gold_items):
        should = gold.get("should_abstain", False)
        did    = _is_abstained(result)
        if should and did:     TP += 1
        elif not should and did: FP += 1
        elif not should and not did: TN += 1
        else:                   FN += 1

    total = TP + FP + TN + FN
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (TP + TN) / total if total else 0.0

    return {
        "abstain_tp": TP, "abstain_fp": FP, "abstain_tn": TN, "abstain_fn": FN,
        "abstain_precision": round(precision, 4),
        "abstain_recall":    round(recall, 4),
        "abstain_f1":        round(f1, 4),
        "abstain_accuracy":  round(accuracy, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. Efficiency / aggregate stats
# ══════════════════════════════════════════════════════════════════════════════

def _percentile(values: List[float], p: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = int(len(s) * p / 100)
    return s[min(idx, len(s) - 1)]


def aggregate_baseline_results(baseline_results: List[Any]) -> Dict[str, float]:
    """Timing, abstain rate, context count stats for a list of results."""
    if not baseline_results:
        return {}
    latencies = [r.latency_sec for r in baseline_results]
    abstain_count = sum(
        1 for r in baseline_results
        if r.is_outscope or "ERROR" in r.answer
    )
    avg_contexts = sum(len(r.contexts) for r in baseline_results) / len(baseline_results)
    return {
        "count":                len(baseline_results),
        "avg_latency_sec":      round(sum(latencies) / len(latencies), 3),
        "p50_latency_sec":      round(_percentile(latencies, 50), 3),
        "p95_latency_sec":      round(_percentile(latencies, 95), 3),
        "p99_latency_sec":      round(_percentile(latencies, 99), 3),
        "min_latency_sec":      round(min(latencies), 3),
        "max_latency_sec":      round(max(latencies), 3),
        "abstain_rate":         round(abstain_count / len(baseline_results), 4),
        "avg_contexts_retrieved": round(avg_contexts, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. V12-specific metrics (extracted from PipelineResult audit trail)
# ══════════════════════════════════════════════════════════════════════════════

def compute_v12_specific(results: List[Dict]) -> Dict[str, float]:
    """
    Extract v12 pipeline-specific metrics from /api/ask responses.

    These are only meaningful for the v12_pipeline system — baselines don't
    have audit trails.

    Returns
    -------
    cfi                  : avg Composite Fidelity Index (0–1)
    claim_grounded_ratio : avg fraction of claims grounded by NLI
    is_grounded_rate     : fraction of answers marked is_grounded=True
    entity_match_ratio   : avg entity match ratio from verification
    """
    cfi_vals, cgr_vals, ig_vals, emr_vals = [], [], [], []

    for r in results:
        audit = r.get("audit_trail") or {}
        if isinstance(audit, dict):
            if "composite_fidelity_index" in audit:
                cfi_vals.append(float(audit["composite_fidelity_index"]))
            if "claim_grounded_ratio" in audit:
                cgr_vals.append(float(audit["claim_grounded_ratio"]))
            if "entity_match_ratio" in audit:
                emr_vals.append(float(audit["entity_match_ratio"]))
        ig = r.get("is_grounded")
        if ig is not None:
            ig_vals.append(1.0 if ig else 0.0)

    out = {}
    if cfi_vals:
        out["v12_cfi"] = round(sum(cfi_vals) / len(cfi_vals), 4)
    if cgr_vals:
        out["v12_claim_grounded_ratio"] = round(sum(cgr_vals) / len(cgr_vals), 4)
    if ig_vals:
        out["v12_is_grounded_rate"] = round(sum(ig_vals) / len(ig_vals), 4)
    if emr_vals:
        out["v12_entity_match_ratio"] = round(sum(emr_vals) / len(emr_vals), 4)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 8. Per-category and per-language breakdowns
# ══════════════════════════════════════════════════════════════════════════════

def per_category_scores(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute Token-F1, ROUGE-L, keyword_hit_rate broken down by category.

    Returns {category: {metric: score}}
    """
    from collections import defaultdict
    groups: Dict[str, List[Tuple]] = defaultdict(list)
    for r, g in zip(results, gold_items):
        cat = g.get("category", "UNKNOWN")
        groups[cat].append((r, g))

    out = {}
    for cat, pairs in groups.items():
        rs, gs = zip(*pairs)
        scores = compute_lexical_scores(list(rs), list(gs))
        kw = compute_keyword_hit_rate(list(rs), list(gs))
        scores.update(kw)
        scores["n"] = len(pairs)
        out[cat] = scores
    return out


def per_language_scores(
    results: List[Dict],
    gold_items: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute Token-F1, ROUGE-L, keyword_hit_rate broken down by language.

    Returns {language: {metric: score}}
    """
    from collections import defaultdict
    groups: Dict[str, List[Tuple]] = defaultdict(list)
    for r, g in zip(results, gold_items):
        lang = g.get("language", "unknown")
        groups[lang].append((r, g))

    out = {}
    for lang, pairs in groups.items():
        rs, gs = zip(*pairs)
        scores = compute_lexical_scores(list(rs), list(gs))
        kw = compute_keyword_hit_rate(list(rs), list(gs))
        scores.update(kw)
        scores["n"] = len(pairs)
        out[lang] = scores
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 9. Win-rate matrix
# ══════════════════════════════════════════════════════════════════════════════

def win_rate_matrix(
    all_results: Dict[str, List[Dict]],
    gold_items: List[Dict],
    metric: str = "token_f1",
) -> Dict[str, Dict[str, float]]:
    """
    Pairwise win-rate matrix.

    win_rate_matrix[A][B] = fraction of questions where system A scores higher
    than system B on `metric`. 0.5 means tie, >0.5 means A beats B.

    Parameters
    ----------
    all_results : {system_name: [result_dict, ...]}
    gold_items  : gold test items (aligned with results)
    metric      : per-question metric to compare ("token_f1" or "rouge_l")

    Returns
    -------
    {system_A: {system_B: win_rate}}
    """
    systems = list(all_results.keys())

    # Compute per-question score for each system
    def _per_q_scores(results):
        scores = []
        for r, g in zip(results, gold_items):
            pred = r.get("answer", "")
            ref  = g.get("gold_answer", "")
            if metric == "rouge_l":
                scores.append(rouge_l(pred, ref))
            else:
                scores.append(token_f1(pred, ref))
        return scores

    per_q = {s: _per_q_scores(all_results[s]) for s in systems}

    matrix: Dict[str, Dict[str, float]] = {}
    for a in systems:
        matrix[a] = {}
        for b in systems:
            if a == b:
                matrix[a][b] = 0.5
                continue
            wins = sum(
                1 if qa > qb else (0.5 if qa == qb else 0)
                for qa, qb in zip(per_q[a], per_q[b])
            )
            matrix[a][b] = round(wins / max(len(gold_items), 1), 4)

    return matrix


# ══════════════════════════════════════════════════════════════════════════════
# 10. Comparison table
# ══════════════════════════════════════════════════════════════════════════════

# Metric display order for the comparison table
_METRIC_ORDER = [
    # ── Lexical ───────────────────────────────────────────────────────────────
    "exact_match", "token_f1", "rouge_l",
    # ── Semantic ──────────────────────────────────────────────────────────────
    "bertscore_f1", "bertscore_precision", "bertscore_recall",
    "arabizi_normalized_f1",
    # ── RAGAS core ────────────────────────────────────────────────────────────
    "faithfulness", "answer_relevancy", "context_precision", "context_recall",
    # ── RAGAS extended (ragas >= 0.2.x) ──────────────────────────────────────
    "answer_correctness", "answer_similarity",
    "context_entity_recall", "noise_sensitivity", "factual_correctness",
    # ── ARES core ─────────────────────────────────────────────────────────────
    "ares_context_relevance", "ares_answer_faithfulness", "ares_answer_relevance",
    # ── ARES extended (5 domain-specific dimensions) ──────────────────────────
    "ares_completeness", "ares_dialect_coherence", "ares_legal_accuracy",
    "ares_multihop_coverage", "ares_abstain_quality",
    # ── G-Eval (Wang et al. 2023) — CoT LLM judge ────────────────────────────
    "geval_coherence", "geval_consistency", "geval_fluency", "geval_relevance",
    # ── FActScore (Min et al. 2023) — atomic claim verification ──────────────
    "factscore", "unsupported_claim_rate", "avg_claims_per_answer",
    # ── RGB robustness (Chen et al. 2023) ─────────────────────────────────────
    "rgb_noise_robustness", "rgb_negative_rejection_rate",
    "rgb_information_integration", "rgb_counterfactual_robustness",
    # ── Retrieval quality (reference-free) ────────────────────────────────────
    "retrieval_context_coverage", "retrieval_context_redundancy",
    "retrieval_mrr", "retrieval_context_utilization",
    # ── Domain precision (Morocco-specific, reference-free) ───────────────────
    "domain_legal_citation_hit", "domain_cost_deadline_hit",
    "domain_dialect_response_match", "domain_multihop_keyword_hit",
    "domain_hallucination_number_rate",
    # ── Cross-lingual consistency ─────────────────────────────────────────────
    "cross_lingual_consistency",
    # ── Domain (gold_keywords — needs --with-gt) ──────────────────────────────
    "keyword_hit_rate",
    "abstain_accuracy", "abstain_f1", "abstain_precision", "abstain_recall",
    # ── V12 specific ──────────────────────────────────────────────────────────
    "v12_cfi", "v12_claim_grounded_ratio", "v12_is_grounded_rate", "v12_entity_match_ratio",
    # ── Efficiency ────────────────────────────────────────────────────────────
    "avg_latency_sec", "p50_latency_sec", "p95_latency_sec", "p99_latency_sec",
    "abstain_rate", "avg_contexts_retrieved",
]


def print_comparison_table(
    scores_by_baseline: Dict[str, Dict[str, float]],
    highlight_best: bool = True,
) -> None:
    """
    Print a formatted comparison table.
    Best score per metric is marked with *.
    """
    if not scores_by_baseline:
        log.warning("[metrics] No scores to display")
        return

    all_metrics = []
    seen = set()
    for m in _METRIC_ORDER:
        if m not in seen and any(m in s for s in scores_by_baseline.values()):
            all_metrics.append(m)
            seen.add(m)
    # Append any remaining metrics not in _METRIC_ORDER
    for s in scores_by_baseline.values():
        for m in s:
            if m not in seen:
                all_metrics.append(m)
                seen.add(m)

    baselines = list(scores_by_baseline.keys())

    # Metrics where LOWER is better
    lower_is_better = {
        "avg_latency_sec", "p50_latency_sec", "p95_latency_sec", "p99_latency_sec",
        "abstain_rate", "abstain_fp", "abstain_fn",
        # SOTA metrics where lower = better
        "unsupported_claim_rate",            # fewer hallucinated claims = better
        "retrieval_context_redundancy",      # less redundancy = more diverse retrieval
        "domain_hallucination_number_rate",  # fewer hallucinated numbers = better
    }

    # Best value per metric
    best: Dict[str, float] = {}
    if highlight_best:
        for m in all_metrics:
            vals = [(b, scores_by_baseline[b][m]) for b in baselines if m in scores_by_baseline[b]]
            if vals:
                best[m] = min(v for _, v in vals) if m in lower_is_better else max(v for _, v in vals)

    col_w    = max(24, max(len(b) for b in baselines) + 2)
    metric_w = 22

    header = f"{'Metric':<{metric_w}}" + "".join(f"{b:<{col_w}}" for b in baselines)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("  MOROCCAN RAG — BENCHMARKING COMPARISON TABLE")
    print(sep)
    print(header)
    print("-" * len(header))

    for m in all_metrics:
        row = f"{m:<{metric_w}}"
        for b in baselines:
            val = scores_by_baseline[b].get(m)
            if val is None:
                cell = "—"
            else:
                is_best = highlight_best and best.get(m) is not None and abs(val - best[m]) < 1e-6
                cell = f"{'*' if is_best else ' '}{val:.4f}"
            row += f"{cell:<{col_w}}"
        print(row)

    print(sep)
    print("  * = best value for that metric")
    print(sep)


def print_category_table(
    category_scores: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """
    Print per-category breakdown.

    category_scores: {system_name: {category: {metric: score}}}
    """
    systems = list(category_scores.keys())
    if not systems:
        return
    all_cats = sorted({c for s in category_scores.values() for c in s})

    print("\n" + "=" * 80)
    print("  PER-CATEGORY BREAKDOWN (token_f1 / rouge_l / keyword_hit_rate)")
    print("=" * 80)
    print(f"  {'Category':<16}" + "".join(f"{s[:18]:<20}" for s in systems))
    print("-" * 80)
    for cat in all_cats:
        row = f"  {cat:<16}"
        for s in systems:
            sc = category_scores.get(s, {}).get(cat, {})
            f1 = sc.get("token_f1", 0.0)
            kw = sc.get("keyword_hit_rate", 0.0)
            n  = int(sc.get("n", 0))
            row += f"F1={f1:.2f} KW={kw:.2f}(n={n}) "
        print(row)
    print("=" * 80)


def print_language_table(
    language_scores: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """
    Print per-language breakdown.

    language_scores: {system_name: {language: {metric: score}}}
    """
    systems = list(language_scores.keys())
    if not systems:
        return
    all_langs = sorted({l for s in language_scores.values() for l in s})

    print("\n" + "=" * 80)
    print("  PER-LANGUAGE BREAKDOWN (token_f1 / keyword_hit_rate)")
    print("=" * 80)
    print(f"  {'Language':<16}" + "".join(f"{s[:18]:<20}" for s in systems))
    print("-" * 80)
    for lang in all_langs:
        row = f"  {lang:<16}"
        for s in systems:
            sc = language_scores.get(s, {}).get(lang, {})
            f1 = sc.get("token_f1", 0.0)
            kw = sc.get("keyword_hit_rate", 0.0)
            n  = int(sc.get("n", 0))
            row += f"F1={f1:.2f} KW={kw:.2f}(n={n}) "
        print(row)
    print("=" * 80)


def print_win_rate_matrix(matrix: Dict[str, Dict[str, float]]) -> None:
    """Print the pairwise win-rate matrix."""
    systems = list(matrix.keys())
    if not systems:
        return
    col_w = max(20, max(len(s) for s in systems) + 2)
    print("\n" + "=" * (col_w * (len(systems) + 1)))
    print("  WIN-RATE MATRIX (row beats column) — based on token_f1 per question")
    print("=" * (col_w * (len(systems) + 1)))
    header = f"{'':>{col_w}}" + "".join(f"{s[:col_w-1]:<{col_w}}" for s in systems)
    print(header)
    print("-" * len(header))
    for a in systems:
        row = f"{a[:col_w-1]:>{col_w}}"
        for b in systems:
            val = matrix[a].get(b, 0.5)
            row += f"{val:.2f}{'':>{col_w - 4}}"
        print(row)
    print("=" * len(header))


# ══════════════════════════════════════════════════════════════════════════════
# Save helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_scores(scores_by_baseline: Dict[str, Dict], output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores_by_baseline, f, ensure_ascii=False, indent=2)
    log.info("[metrics] Saved scores → %s", output_path)

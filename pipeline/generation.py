# -*- coding: utf-8 -*-
"""
Steps 8-9 — LLM generation: OllamaClient, AgentMemory, ToolRegistry, PlannerAgent.

Key changes vs the monolith:
  - All print() replaced with log.debug() / log.info()
  - All CONFIG.* replaced with settings.*
  - ToolRegistry tools return ScoredChunk lists (unchanged) but also record
    each call as a ToolCall in AgentState.execution_trace (new — enables
    RAGAS multi-hop context capture).
  - FIX 67 (content=None for Gemma4 thinking mode) is preserved.
"""

import copy
import json
import logging
import re
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

from openai import APIError, APITimeoutError, OpenAI, RateLimitError

from pipeline.config import settings
from pipeline.models import (
    AgentPlan,
    AgentState,
    Chunk,
    ExecutionTrace,
    PlannedStep,
    QuestionFlags,
    ScoredChunk,
    ToolCall,
    VALID_INTENTS,
    short_source,
)
from pipeline.retrieval import HybridRetriever

log = logging.getLogger("MoroccanRAG")


# ── OllamaClient ──────────────────────────────────────────────────────────────


class OllamaClient:
    """
    Thin wrapper around the OpenAI SDK pointing at an Ollama endpoint.
    Adds retry logic with exponential backoff and the FIX 67 content-extraction
    chain for Gemma4 / thinking models.
    """

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
    ):
        self._base_url = base_url or settings.ollama_base_url
        self._client = OpenAI(
            base_url=self._base_url,
            api_key=api_key or settings.ollama_api_key,
            timeout=settings.api_timeout,
        )
        self.model = model or settings.generator_model
        self._calls = 0
        self._fails = 0

        # For LOCAL Ollama, route through the native /api/chat endpoint with
        # think:false. The OpenAI-compatible /v1 endpoint ignores the think flag
        # and lets gemma4:e4b emit reasoning that consumes the token budget →
        # empty content (finish_reason=length) on low-max_tokens calls
        # (translation, planner). The native endpoint returns clean content at
        # any budget. Cloud endpoints keep the OpenAI SDK path.
        self._use_native = any(h in self._base_url
                               for h in ("localhost", "127.0.0.1", "0.0.0.0", "::1"))
        _root = self._base_url.rstrip("/")
        if _root.endswith("/v1"):
            _root = _root[:-3]
        self._native_url = _root.rstrip("/") + "/api/chat"

    def _extract_content(self, resp, attempt: int) -> Optional[str]:
        """FIX 67: extract text from response, handling thinking-model content=None."""
        choice = resp.choices[0]
        message = choice.message
        content = message.content

        if not content:
            reasoning = getattr(message, "reasoning_content", None)
            if reasoning and reasoning.strip():
                log.debug("  FIX67: used reasoning_content field")
                return reasoning.strip()

        if not content:
            raw = getattr(message, "text", None) or str(message)
            think_end = raw.rfind("</think>")
            if think_end != -1:
                log.debug("  FIX67: extracted content after </think>")
                return raw[think_end + 8:].strip()

        if not content:
            log.warning(
                f"  Ollama 200 but empty content (attempt {attempt + 1}) "
                f"finish_reason={getattr(choice, 'finish_reason', '?')} model={self.model}"
            )
            return None

        return content.strip()

    def _call_native(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: Optional[float],
        stop: Optional[List[str]],
        seed: Optional[int],
        fmt: Any,
        extra: Dict[str, Any],
    ) -> Optional[str]:
        """
        Call Ollama's native /api/chat with think:false (clean content at any
        max_tokens). Returns the assistant content, or None on empty/failure.
        Raises on HTTP/connection errors so the retry loop can handle them.
        """
        import requests

        options: Dict[str, Any] = dict(extra.get("options") or {})
        options["temperature"] = temperature
        options["num_predict"] = max_tokens
        if top_p is not None:
            options["top_p"] = top_p
        if stop is not None:
            options["stop"] = stop
        if seed is not None:
            options["seed"] = seed

        # Respect the per-call think flag. On the NATIVE endpoint, think:true puts
        # reasoning in a separate field and keeps `content` clean — so the final
        # answer can reason without polluting/truncating its content; utility
        # calls (translation, planner) pass think:false for fast clean output.
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "think": bool(extra.get("think", False)),
            "options": options,
            "keep_alive": extra.get("keep_alive", settings.ollama_keep_alive),
        }
        # Structured output: Ollama's native `format` wants "json" OR a *raw* JSON
        # schema — NOT the OpenAI wrapper {"type":"json_schema","json_schema":{...}}.
        # Translate so claims/planner/judge get schema-enforced output (correct keys).
        if fmt == "json":
            body["format"] = "json"
        elif isinstance(fmt, dict):
            if fmt.get("type") == "json_schema":
                schema = (fmt.get("json_schema") or {}).get("schema")
                if schema:
                    body["format"] = schema
            elif fmt.get("type") == "json_object":
                body["format"] = "json"
            else:
                body["format"] = fmt   # already a raw schema

        resp = requests.post(self._native_url, json=body, timeout=settings.api_timeout)
        resp.raise_for_status()
        msg = resp.json().get("message", {}) or {}
        content = (msg.get("content") or "").strip()
        if not content:
            log.warning("  Ollama native: empty content model=%s", self.model)
            return None
        return content

    def _call_with_retry(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        top_p: Optional[float],
        stop: Optional[List[str]],
        seed: Optional[int],
        fmt: Any,
        extra: Dict[str, Any],
    ) -> Optional[str]:
        """Retry loop with exponential backoff."""
        for attempt in range(settings.api_max_retries):
            self._calls += 1
            try:
                # Local Ollama → native /api/chat (think:false, clean content).
                if self._use_native:
                    content = self._call_native(
                        messages, temperature, max_tokens, top_p, stop, seed, fmt, extra
                    )
                    if content is not None:
                        return content
                    self._fails += 1
                    return None

                create_kwargs: Dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                )
                if top_p is not None:
                    create_kwargs["top_p"] = top_p
                if stop is not None:
                    create_kwargs["stop"] = stop
                if seed is not None:
                    create_kwargs["seed"] = seed
                if fmt is not None:
                    create_kwargs["response_format"] = (
                        {"type": "json_object"} if fmt == "json" else fmt
                    )
                if extra:
                    create_kwargs["extra_body"] = extra

                resp = self._client.chat.completions.create(**create_kwargs)
                content = self._extract_content(resp, attempt)
                if content is not None:
                    return content
                self._fails += 1
                return None

            except (APITimeoutError, RateLimitError) as exc:
                wait = settings.api_retry_delay * (2 ** attempt)
                log.warning(f"  Ollama {type(exc).__name__} (attempt {attempt + 1}) — retry in {wait:.1f}s")
                if attempt < settings.api_max_retries - 1:
                    time.sleep(wait)
            except APIError as exc:
                log.error(f"  Ollama APIError: {exc}")
                self._fails += 1
                return None
            except Exception as exc:
                log.error(f"  Ollama unexpected error: {exc}")
                self._fails += 1
                return None

        self._fails += 1
        log.error("  Ollama: max retries exceeded")
        return None

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None,
        max_tokens: int = None,
        top_p: float = None,
        stop: List[str] = None,
        seed: int = None,
        fmt: Any = None,
        top_k: int = None,
        repeat_penalty: float = None,
        num_ctx: int = None,
        num_predict: int = None,
        think: bool = None,
        keep_alive: str = None,
    ) -> Optional[str]:
        temperature = temperature if temperature is not None else settings.temperature
        max_tokens = max_tokens if max_tokens is not None else settings.max_new_tokens
        _num_ctx = num_ctx if num_ctx is not None else settings.ollama_num_ctx
        _think = think if think is not None else settings.ollama_think
        _repeat_penalty = repeat_penalty if repeat_penalty is not None else settings.ollama_repeat_penalty
        _keep_alive = keep_alive if keep_alive is not None else settings.ollama_keep_alive

        options: Dict[str, Any] = {"num_ctx": _num_ctx, "repeat_penalty": _repeat_penalty}
        if top_k is not None:
            options["top_k"] = top_k
        if num_predict is not None:
            options["num_predict"] = num_predict

        extra: Dict[str, Any] = {"options": options, "think": _think, "keep_alive": _keep_alive}

        return self._call_with_retry(messages, temperature, max_tokens, top_p, stop, seed, fmt, extra)

    def diagnose(self) -> Dict:
        """Call when Ollama returns HTTP 200 but generate() returns None."""
        log.info(f"  OLLAMA DIAGNOSTIC | URL={self._client.base_url} | Model={self.model}")
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Reply with the single word: HELLO"}],
                temperature=0.0,
                max_tokens=50,
                stream=False,
            )
            choice = resp.choices[0]
            message = choice.message
            log.info(f"  finish_reason={choice.finish_reason} content={repr(message.content)}")
            rc = getattr(message, "reasoning_content", "FIELD_NOT_PRESENT")
            log.info(f"  reasoning_content={repr(rc)}")
            if message.content and message.content.strip():
                return {"ok": True, "path": "content"}
            if isinstance(rc, str) and rc.strip():
                return {"ok": False, "cause": "thinking_only"}
            raw = str(message)
            think_end = raw.rfind("</think>")
            if think_end != -1:
                return {"ok": False, "cause": "think_block_in_str"}
            return {"ok": False, "cause": "empty_response"}
        except Exception as exc:
            return {"ok": False, "cause": str(exc)}

    def stats(self) -> Dict:
        return {
            "total": self._calls,
            "failed": self._fails,
            "success_rate": (self._calls - self._fails) / max(self._calls, 1),
        }


# ── AgentMemory ───────────────────────────────────────────────────────────────


class AgentMemory:
    """Bounded chunk store with deduplication and per-source limits."""

    def __init__(self):
        self._store: Dict[str, ScoredChunk] = {}
        self._content_hashes: Dict[int, str] = {}
        self._source_counts: Dict[str, int] = defaultdict(int)
        self.stats = {
            "total_added": 0,
            "rejected_duplicate": 0,
            "rejected_low_score": 0,
            "rejected_source_limit": 0,
            "evicted": 0,
        }

    def add(self, sc: ScoredChunk) -> bool:
        chunk, score = sc.chunk, sc.rrf_score
        if score < settings.memory_min_score:
            self.stats["rejected_low_score"] += 1
            return False
        content_hash = hash(chunk.text.strip())
        if content_hash in self._content_hashes:
            existing_id = self._content_hashes[content_hash]
            if existing_id in self._store and score > self._store[existing_id].rrf_score:
                self._store[existing_id].rrf_score = score
            self.stats["rejected_duplicate"] += 1
            return False
        if self._source_counts[chunk.source] >= settings.memory_max_per_src:
            src_chunks = [s for s in self._store.values() if s.chunk.source == chunk.source]
            if src_chunks:
                worst = min(src_chunks, key=lambda x: x.rrf_score)
                if score > worst.rrf_score:
                    self._evict(worst.chunk.chunk_id)
                else:
                    self.stats["rejected_source_limit"] += 1
                    return False
        self._store[chunk.chunk_id] = sc
        self._content_hashes[content_hash] = chunk.chunk_id
        self._source_counts[chunk.source] += 1
        self.stats["total_added"] += 1
        if len(self._store) > settings.memory_max_chunks:
            self._evict_lowest()
        return True

    def add_all(self, scored_chunks: List[ScoredChunk], step: int) -> None:
        added = 0
        for sc in scored_chunks:
            sc.step_found = step
            if self.add(sc):
                added += 1
        log.info(f"  Memory: added {added}/{len(scored_chunks)} from step {step}")

    def _evict(self, chunk_id: str) -> None:
        if chunk_id not in self._store:
            return
        sc = self._store.pop(chunk_id)
        self._source_counts[sc.chunk.source] -= 1
        self._content_hashes.pop(hash(sc.chunk.text.strip()), None)
        self.stats["evicted"] += 1

    def _evict_lowest(self) -> None:
        if not self._store:
            return
        lowest_id = min(self._store, key=lambda k: self._store[k].rrf_score)
        self._evict(lowest_id)

    def get_top_chunks(self, n: int = None) -> List[Chunk]:
        n = n or settings.compress_top_n
        return [
            sc.chunk
            for sc in sorted(self._store.values(), key=lambda sc: sc.rrf_score, reverse=True)[:n]
        ]

    def build_context(self, max_chunks: int = None) -> str:
        top = self.get_top_chunks(max_chunks or settings.compress_top_n)
        parts = [f"[Source: {short_source(c.source)} | Page: {c.page}]\n{c.text}" for c in top]
        return "\n\n".join(parts)

    def size(self) -> int:
        return len(self._store)

    def source_counts(self) -> Dict[str, int]:
        return dict(self._source_counts)

    def score_range(self) -> Tuple[float, float]:
        if not self._store:
            return 0.0, 0.0
        scores = [sc.rrf_score for sc in self._store.values()]
        return min(scores), max(scores)


# ── ToolRegistry ──────────────────────────────────────────────────────────────


class ToolRegistry:
    """
    Registry of deterministic tools available to the PlannerAgent.

    Tools retrieve from the knowledge base — they do NOT call the LLM.
    Each call returns List[ScoredChunk] and records itself as a ToolCall on
    AgentState.execution_trace so the benchmarking layer sees ALL contexts.
    """

    _INTENT_TOOL_MAP = {
        "DOCUMENTS": "retrieve_kb",
        "COST": "search_by_amount",
        "DEADLINE": "calculate_deadline",
        "PROCEDURE": "retrieve_kb",
        "ELIGIBILITY": "check_eligibility",
        "LEGAL": "retrieve_kb",
        "COMPARISON": "retrieve_kb",
    }

    def __init__(self, retriever: HybridRetriever):
        self._retriever = retriever
        self._registry = {
            "retrieve_kb": self._retrieve_kb,
            "lookup_article": self._lookup_article,
            "calculate_deadline": self._calculate_deadline,
            "check_eligibility": self._check_eligibility,
            "search_by_amount": self._search_by_amount,
        }

    @property
    def available_tools(self) -> List[str]:
        return list(self._registry.keys())

    @staticmethod
    def best_tool_for(intent: str) -> str:
        return ToolRegistry._INTENT_TOOL_MAP.get(intent, "retrieve_kb")

    def execute(
        self,
        tool: str,
        args: Dict[str, Any],
        flags: QuestionFlags,
        step_index: int,
        state: AgentState,
    ) -> List[ScoredChunk]:
        fn = self._registry.get(tool)
        if fn is None:
            log.warning(f"  [ToolRegistry] Unknown tool '{tool}' — falling back to retrieve_kb")
            fn = self._retrieve_kb
            tool = "retrieve_kb"
        try:
            t0 = time.time()
            results = fn(args, flags)
            latency_ms = round((time.time() - t0) * 1000)
            tc = ToolCall(
                step_index=step_index,
                tool_name=tool,
                query=args.get("query", ""),
                intent=args.get("_intent", "UNKNOWN"),
                contexts=[sc.chunk.text for sc in results],
                scores=[sc.rrf_score for sc in results],
                chunks_returned=len(results),
                latency_ms=latency_ms,
            )
            state.execution_trace.add_tool_call(tc)
            return results
        except Exception as exc:
            log.warning(f"  [ToolRegistry] Tool '{tool}' failed: {exc} — returning []")
            return []

    def _retrieve_kb(self, args: Dict, flags: QuestionFlags) -> List[ScoredChunk]:
        query = args.get("query", "")
        if not query:
            return []
        result = self._retriever.retrieve(query, flags)
        return result.chunks

    def _lookup_article(self, args: Dict, flags: QuestionFlags) -> List[ScoredChunk]:
        art_num = str(args.get("article_number", "")).strip()
        law_name = str(args.get("law_name", "")).strip().lower()
        kb = self._retriever.kb

        matched: List[ScoredChunk] = []
        for chunk in kb.all_chunks:
            art_match = art_num and art_num in chunk.article_number
            law_match = law_name and law_name in chunk.law_name.lower()
            if art_match or law_match:
                matched.append(ScoredChunk(chunk=chunk, rrf_score=0.9, bm25_score=1.0, dense_score=0.9))

        if matched:
            log.info(f"  [lookup_article] {len(matched)} chunks for art={art_num} law='{law_name}'")
            return matched[: settings.retrieve_top_k]

        fallback_parts = " ".join(filter(None, [law_name, f"المادة {art_num}" if art_num else ""]))
        fallback_query = fallback_parts.strip() or args.get("query", law_name or art_num)
        return self._retrieve_kb({"query": fallback_query}, flags)

    def _calculate_deadline(self, args: Dict, flags: QuestionFlags) -> List[ScoredChunk]:
        query = args.get("query", "مدة الإنجاز délai traitement")
        results = self._retrieve_kb({"query": query}, flags)

        n_days = args.get("working_days", None)
        if n_days is None:
            for sc in results[:3]:
                m = re.search(r"(\d+)\s+(?:يوم|jour)", sc.chunk.text)
                if m:
                    n_days = int(m.group(1))
                    break

        if n_days and isinstance(n_days, int):
            today = date.today()
            deadline = today
            days_added = 0
            while days_added < n_days:
                deadline += timedelta(days=1)
                if deadline.weekday() < 5:
                    days_added += 1
            deadline_str = deadline.strftime("%d/%m/%Y")
            if results:
                top = copy.copy(results[0])
                top.chunk = copy.copy(results[0].chunk)
                top.chunk.text = (
                    f"[Délai calculé / المدة المحسوبة: {n_days} jours ouvrables → {deadline_str}] "
                    + top.chunk.text
                )
                results = [top] + results[1:]
        return results

    def _check_eligibility(self, args: Dict, flags: QuestionFlags) -> List[ScoredChunk]:
        query = args.get("query", "شروط الأهلية conditions éligibilité")
        results = self._retrieve_kb({"query": query}, flags)
        ELIGIBILITY_KW = ["سن", "سنة", "مواطن", "أجنبي", "قاصر", "âge", "ans", "citoyen", "étranger", "mineur"]
        for sc in results:
            if any(kw in sc.chunk.text for kw in ELIGIBILITY_KW):
                sc.rrf_score = min(sc.rrf_score * 1.3, 1.0)
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        return results[: settings.retrieve_top_k]

    def _search_by_amount(self, args: Dict, flags: QuestionFlags) -> List[ScoredChunk]:
        min_amt = args.get("min_amount", 0)
        max_amt = args.get("max_amount", 100_000)
        query = args.get("query", "رسوم درهم frais dirhams")
        results = self._retrieve_kb({"query": query}, flags)
        AMOUNT_RE = re.compile(r"(\d[\d\s]*)\s*(?:درهم|DH|MAD|dirhams?)", re.IGNORECASE)
        for sc in results:
            amounts = [
                int(re.sub(r"\s", "", m))
                for m in AMOUNT_RE.findall(sc.chunk.text)
                if re.sub(r"\s", "", m).isdigit()
            ]
            if any(min_amt <= a <= max_amt for a in amounts):
                sc.rrf_score = min(sc.rrf_score * 1.4, 1.0)
        results.sort(key=lambda x: x.rrf_score, reverse=True)
        return results[: settings.retrieve_top_k]


# ── PlannerAgent ──────────────────────────────────────────────────────────────


class PlannerAgent:
    """
    Agentic RAG with explicit planning, structured state, and tool orchestration.

    Plan → Execute → Reflect cycle:
      1. PLAN      — LLM generates an explicit AgentPlan (JSON with steps, tools, rationale)
      2. EXECUTE   — ToolRegistry dispatches to retrieve_kb / lookup_article / etc.
                     Each call is logged in AgentState.execution_trace for RAGAS.
      3. REFLECT   — lightweight LLM check: did the retrieved info answer the sub-question?
      4. SYNTHESISE — assembles the structured final answer from AgentState.facts.
    """

    def __init__(self, ollama: OllamaClient, retriever: HybridRetriever):
        self.ollama = ollama
        self.retriever = retriever
        self.memory = AgentMemory()
        self.tools = ToolRegistry(retriever)
        self.steps = 0
        self.disclaimer_added = False

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        question: str,
        flags: QuestionFlags,
        initial_scored: List[ScoredChunk],
    ) -> Tuple[str, int, List[str], AgentState]:
        """Delegates to _run_simple or _run_multihop after shared guard checks."""
        log.info("  GENERATE — PlannerAgent")

        state = AgentState(question=question, language=flags.language, flags=flags)

        if not initial_scored:
            answer = self._get_abstain(flags.language)
            state.final_answer = answer
            return answer, 0, [], state

        self.memory.add_all(initial_scored, step=0)
        if self.memory.size() == 0:
            answer = self._get_abstain(flags.language)
            state.final_answer = answer
            return answer, 0, [], state

        if flags.OUTSCOPE:
            answer = self._get_refusal(flags.language)
            state.final_answer = answer
            return answer, 0, [], state

        if flags.SIMPLE and not flags.LEGAL:
            return self._run_simple(question, flags, state)
        return self._run_multihop(question, flags, state)

    def _run_simple(
        self,
        question: str,
        flags: QuestionFlags,
        state: AgentState,
    ) -> Tuple[str, int, List[str], AgentState]:
        """Direct generation path for SIMPLE questions (no planning)."""
        log.info("  SIMPLE → Direct generation (no planning)")
        state.log("THOUGHT: Simple question — direct generation without planning.")
        state.execution_trace.agent_path = "simple"
        t_gen = time.time()
        answer = self._generate_direct(question, flags, state)
        state.execution_trace.generation_latency_ms = round((time.time() - t_gen) * 1000)
        state.final_answer = answer
        state.is_complete = True
        state.execution_trace.synthesis_context = self.memory.build_context()
        return answer, 1, [answer], state

    def _run_multihop(
        self,
        question: str,
        flags: QuestionFlags,
        state: AgentState,
    ) -> Tuple[str, int, List[str], AgentState]:
        """Plan → Execute → Reflect loop for MULTIHOP / LEGAL questions."""
        log.info("  MULTIHOP/LEGAL → Plan → Execute → Reflect")
        state.execution_trace.agent_path = "multihop"

        t_plan = time.time()
        state.plan = self._plan(question, flags, state)
        state.execution_trace.plan_latency_ms = round((time.time() - t_plan) * 1000)
        state.log(f"PLAN generated ({state.plan.plan_source}): {len(state.plan.steps)} steps")
        state.execution_trace.plan = [
            {"step_id": s.step_id, "intent": s.intent, "tool": s.tool,
             "sub_question": s.sub_question, "rationale": s.rationale}
            for s in state.plan.steps
        ]

        intermediate_texts: List[str] = []

        for planned_step in state.plan.steps:
            if self.steps >= settings.max_agent_steps - 1:
                state.log(f"STOP: reached MAX_AGENT_STEPS ({settings.max_agent_steps})")
                break

            self.steps += 1
            state.step = self.steps
            intent = planned_step.intent
            sub_q = planned_step.sub_question

            state.log(f"\nSTEP {self.steps} | intent={intent} | tool={planned_step.tool}")
            log.info(f"  Step {self.steps}: [{intent}] {planned_step.tool}('{sub_q[:60]}')")

            tool_args = dict(planned_step.tool_args)
            tool_args["_intent"] = intent

            tool_results = self.tools.execute(
                planned_step.tool, tool_args, flags, self.steps, state
            )

            if tool_results:
                self.memory.add_all(tool_results, step=self.steps)
                step_chunks = [sc.chunk for sc in tool_results[: settings.compress_top_n]]
            else:
                step_chunks = self.memory.get_top_chunks(settings.compress_top_n)

            state.evidence[intent] = step_chunks
            log.info(f"  → {len(step_chunks)} chunks retrieved")

            prior = [(i, state.facts.get(i, "")) for i in list(state.facts.keys())]
            intermediate = self._generate_intermediate(sub_q, intent, step_chunks, prior, flags.language)
            state.facts[intent] = intermediate
            intermediate_texts.append(intermediate)
            state.log(f"  ANSWER: {intermediate[:120]}…")

            reflection = self._reflect(sub_q, intent, intermediate, step_chunks, flags.language, state)
            state.reflections.append(reflection)
            state.log(f"  REFLECT: {reflection[:100]}")

            if state.execution_trace.tool_calls:
                last_tc = state.execution_trace.tool_calls[-1]
                last_tc.intermediate_answer = intermediate
                last_tc.reflection = reflection

            # Reflection trigger: use STRUCTURED status from _reflect (returns one of
            # "complete" | "partial" | "not_found" via JSON schema). Trigger adaptive
            # re-retrieval ONLY on "partial" or "not_found". Removed the keyword
            # fallback (kw in reflection.lower() for partial/insufficient/incomplet/
            # manque) since the LLM always returns a structured status now.
            if reflection.strip().lower() in ("partial", "not_found"):
                if self.steps < settings.max_agent_steps - 1:
                    state.log(f"  → Reflection triggered adaptive retrieval for {intent}")
                    rephrased = self._rephrase_for_retry(sub_q, intermediate, flags.language)
                    extra_args = {"query": rephrased, "_intent": intent}
                    extra = self.tools.execute("retrieve_kb", extra_args, flags, self.steps, state)
                    if extra:
                        self.memory.add_all(extra, step=self.steps)
                        extra_chunks = [sc.chunk for sc in extra[:3]]
                        state.evidence[intent] = state.evidence.get(intent, []) + extra_chunks
                        state.log(f"  → Added {len(extra_chunks)} extra chunks")

        t_synth = time.time()
        answer = self._synthesise(question, state, flags) if state.facts else self._generate_direct(question, flags, state)
        state.execution_trace.generation_latency_ms = round((time.time() - t_synth) * 1000)

        if flags.LEGAL and not self.disclaimer_added:
            answer = self._add_disclaimer(answer, flags.language)

        state.final_answer = answer
        state.is_complete = True
        state.execution_trace.synthesis_context = self.memory.build_context()
        state.log(f"\nFINAL ANSWER ({len(answer)} chars)")
        return answer, self.steps, intermediate_texts, state

    # ── PLAN ──────────────────────────────────────────────────────────────────

    def _call_plan_llm(self, prompt: str) -> Optional[str]:
        """Single LLM call that returns the raw JSON plan string."""
        return self.ollama.generate(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
            fmt={
                "type": "json_schema",
                "json_schema": {
                    "name": "execution_plan",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_id": {"type": "integer"},
                                        "intent": {"type": "string"},
                                        "sub_question": {"type": "string"},
                                        "tool": {"type": "string"},
                                        "tool_args": {"type": "object", "additionalProperties": True},
                                        "rationale": {"type": "string"},
                                    },
                                    "required": ["step_id", "intent", "sub_question", "tool", "tool_args", "rationale"],
                                },
                            }
                        },
                        "required": ["steps"],
                    },
                },
            },
        )

    def _parse_plan_json(self, raw: str, question: str) -> List[PlannedStep]:
        """Extract JSON from raw response, validate steps, and normalize intents/tools."""
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in plan")
        parsed = json.loads(match.group(0))
        raw_steps = parsed.get("steps", [])
        if not raw_steps:
            raise ValueError("Empty steps list")

        steps: List[PlannedStep] = []
        for s in raw_steps:
            intent = s.get("intent", "DOCUMENTS")
            if intent not in VALID_INTENTS:
                intent = "DOCUMENTS"
            tool = s.get("tool", "retrieve_kb")
            if tool not in self.tools.available_tools:
                tool = "retrieve_kb"
            tool_args = s.get("tool_args", {})
            if "query" not in tool_args:
                tool_args["query"] = s.get("sub_question", question)
            steps.append(PlannedStep(
                step_id=int(s.get("step_id", len(steps) + 1)),
                intent=intent,
                sub_question=s.get("sub_question", question),
                tool=tool,
                tool_args=tool_args,
                rationale=s.get("rationale", ""),
                depends_on=int(s.get("depends_on", 0)),
            ))
            if len(steps) >= settings.max_agent_steps:
                break
        return steps

    def _plan(self, question: str, flags: QuestionFlags, state: AgentState) -> AgentPlan:
        """Build prompt, call LLM, parse result, fallback if needed."""
        from pipeline.prompts import plan_prompt
        prompt = plan_prompt(question, flags.intents or ["DOCUMENTS"], flags.language)

        try:
            raw = self._call_plan_llm(prompt)
            log.debug(f"  [Planner] raw plan: {(raw or '')[:200]}")
            if not raw:
                raise ValueError("Empty plan response")

            steps = self._parse_plan_json(raw, question)
            steps = self._ensure_intent_coverage(steps, question, flags)
            state.log(f"PLAN (LLM): {len(steps)} steps")
            return AgentPlan(steps=steps, raw_plan_text=raw, plan_source="llm")

        except Exception as exc:
            log.warning(f"  [Planner] LLM plan failed: {exc} — fallback to intent decomposition")
            state.log(f"PLAN (fallback): {exc}")
            return self._fallback_plan(question, flags)

    def _ensure_intent_coverage(
        self, steps: List[PlannedStep], question: str, flags: QuestionFlags
    ) -> List[PlannedStep]:
        """
        Guarantee the plan covers every detected intent. The LLM planner often
        drops intents (e.g. a DOCUMENTS+PROCEDURE question becomes a single
        PROCEDURE step), which then makes synthesis answer only that narrow
        slice. Inject a retrieval step for any missing intent.
        """
        if not flags.intents:
            return steps
        covered = {s.intent for s in steps}
        missing = [
            i for i in flags.intents
            if i in VALID_INTENTS and i != "OUT_OF_SCOPE" and i not in covered
        ]
        if not missing:
            return steps
        sub_qs = self._decompose_by_intents(question, missing, flags.language)
        next_id = max((s.step_id for s in steps), default=0) + 1
        for sub_q, intent in zip(sub_qs, missing):
            if len(steps) >= settings.max_agent_steps:
                break
            steps.append(PlannedStep(
                step_id=next_id,
                intent=intent,
                sub_question=sub_q,
                tool=ToolRegistry.best_tool_for(intent),
                tool_args={"query": sub_q},
                rationale=f"Added for intent coverage: {intent}",
            ))
            next_id += 1
        return steps

    def _fallback_plan(self, question: str, flags: QuestionFlags) -> AgentPlan:
        intents = flags.intents if flags.intents else ["DOCUMENTS"]
        sub_questions = self._decompose_by_intents(question, intents, flags.language)
        steps = [
            PlannedStep(
                step_id=i + 1,
                intent=intent,
                sub_question=sub_q,
                tool=ToolRegistry.best_tool_for(intent),
                tool_args={"query": sub_q},
                rationale=f"Fallback: intent={intent}",
            )
            for i, (sub_q, intent) in enumerate(zip(sub_questions, intents))
        ]
        return AgentPlan(steps=steps, raw_plan_text="", plan_source="fallback")

    # ── REFLECT ───────────────────────────────────────────────────────────────

    def _reflect(
        self,
        sub_question: str,
        intent: str,
        intermediate: str,
        chunks: List[Chunk],
        language: str,
        state: AgentState,
    ) -> str:
        if not chunks or not intermediate:
            return "not_found"
        if (
            len(intermediate.split()) < 10
            or "غير موجود" in intermediate
            or "non disponible" in intermediate.lower()
        ):
            return "not_found"

        context_preview = chunks[0].text[:300] if chunks else ""
        from pipeline.prompts import reflect_prompt
        prompt = reflect_prompt(sub_question, intermediate, context_preview, language)
        try:
            result = self.ollama.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=64,     # {"status": "complete"} is ~10 tokens; was 400 (40x over)
                think=False,
                fmt={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reflection",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "enum": ["complete", "partial", "not_found"]}
                            },
                            "required": ["status"],
                        },
                    },
                },
            )
            log.debug(f"  [Reflect] raw={result}")
            if not result:
                return "complete"
            try:
                parsed = (
                    json.loads(result)
                    if result.startswith("{")
                    else json.loads(re.search(r"\{.*\}", result, re.DOTALL).group(0))
                )
                return parsed.get("status", "complete")
            except Exception as exc:
                log.debug("Reflection JSON parse failed, using keyword fallback: %s", exc)
                r = result.strip().lower()
                if "partial" in r or "incomplet" in r:
                    return "partial"
                if "not_found" in r or "غير" in r:
                    return "not_found"
                return "complete"
        except Exception as exc:
            log.debug("Reflection LLM call failed: %s", exc)
            return "complete"

    def _rephrase_for_retry(self, sub_question: str, prior_answer: str, language: str) -> str:
        """Generate a rephrased query for adaptive re-retrieval after insufficient reflection."""
        is_arabic = language in ("arabic_msa", "Darija")
        if is_arabic:
            prompt = (
                f"السؤال الأصلي: {sub_question}\n"
                f"الجواب الجزئي: {prior_answer[:200]}\n\n"
                "أعد صياغة السؤال للبحث عن معلومات إضافية مفصلة.\n"
                "اكتب سؤالاً واحداً فقط بدون شرح."
            )
        else:
            prompt = (
                f"Question originale: {sub_question}\n"
                f"Réponse partielle: {prior_answer[:200]}\n\n"
                "Reformulez la question pour trouver des informations supplémentaires.\n"
                "Écrivez une seule question sans explication."
            )
        try:
            rephrased = self.ollama.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=80,
                think=False,
            )
            if rephrased and len(rephrased.strip()) > 5:
                return rephrased.strip()
        except Exception as exc:
            log.debug("Rephrase-for-retry failed: %s", exc)
        return sub_question + (" تفاصيل إضافية" if is_arabic else " détails supplémentaires")

    # ── SYNTHESISE ────────────────────────────────────────────────────────────

    def _synthesise(self, question: str, state: AgentState, flags: QuestionFlags) -> str:
        from pipeline.prompts import get_intent_sections, synthesis_prompt
        language = flags.language
        intent_sections = get_intent_sections(language)

        facts_parts = [f"=== {intent} ===\n{answer_text}\n" for intent, answer_text in state.facts.items()]
        facts_context = "\n".join(facts_parts)
        section_instructions = "".join(
            intent_sections.get(intent, "")
            for intent in state.facts.keys()
            if intent in intent_sections
        )

        # Pass the full retrieved evidence so synthesis can extract the answer
        # even when an individual hop's partial answer came back empty/refused.
        retrieved_context = self.memory.build_context()
        prompt = synthesis_prompt(question, facts_context, section_instructions, language, retrieved_context)
        try:
            answer = self.ollama.generate(
                [{"role": "user", "content": prompt}],
                temperature=settings.temperature,
                max_tokens=settings.max_new_tokens,
                seed=42,
                think=True,   # final answer reasons; native /api/chat keeps content clean
            )
            log.debug(f"  [Synthesise] {len(answer or '')} chars")
            if not answer or len(answer.strip()) < 20:
                return self._generate_direct(question, flags, state)
            return answer.strip()
        except Exception as exc:
            log.warning("Synthesis LLM call failed, falling back to direct generation: %s", exc)
            return self._generate_direct(question, flags, state)

    # ── DIRECT GENERATION (SIMPLE path + synthesis fallback) ─────────────────

    def _generate_direct(self, question: str, flags: QuestionFlags, state: AgentState) -> str:
        if self.memory.size() == 0:
            return self._get_abstain(flags.language)
        context = self.memory.build_context()
        if len(context.split()) < 20:
            return self._get_abstain(flags.language)

        from pipeline.prompts import direct_generation_prompt
        prompt = direct_generation_prompt(question, context, flags.language)
        try:
            answer = self.ollama.generate(
                [{"role": "user", "content": prompt}],
                temperature=settings.temperature,
                max_tokens=settings.max_new_tokens,
                seed=42,
                think=True,   # final answer reasons; native /api/chat keeps content clean
            )
            log.debug(f"  [DirectGen] {len(answer or '')} chars")
            return answer.strip() if answer else self._get_fallback(flags.language)
        except Exception as exc:
            log.warning("Direct generation LLM call failed: %s", exc)
            return self._get_fallback(flags.language)

    # ── INTERMEDIATE GENERATION (per step) ───────────────────────────────────

    def _generate_intermediate(
        self,
        sub_question: str,
        intent: str,
        chunks: List[Chunk],
        prior: List[Tuple[str, str]],
        language: str,
    ) -> str:
        if not chunks:
            return self._get_not_found(intent, language)

        from pipeline.prompts import intermediate_generation_prompt
        chunk_context = "\n\n".join(
            f"[Source: {short_source(c.source)} | Page: {c.page}]\n{c.text[:400]}" for c in chunks[:3]
        )
        prior_context = ""
        if prior:
            prior_lines = "\n".join(f"  - {i}: {a[:100]}…" for i, a in prior)
            if language in ("arabic_msa", "Darija"):
                prior_context = f"ملخص الخطوات السابقة:\n{prior_lines}\n\n"
            else:
                prior_context = f"Résumé des étapes précédentes:\n{prior_lines}\n\n"

        prompt = intermediate_generation_prompt(sub_question, intent, chunk_context, prior_context, language)
        try:
            raw = self.ollama.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.05,
                max_tokens=1000,
                fmt={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "intermediate_answer",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"},
                                "found": {"type": "boolean"},
                            },
                            "required": ["answer", "found"],
                        },
                    },
                },
            )
            log.debug(f"  [Intermediate] raw={raw[:100] if raw else None}")
            if not raw:
                return self._get_not_found(intent, language)
            try:
                parsed = (
                    json.loads(raw)
                    if raw.startswith("{")
                    else json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
                )
                if not parsed.get("found", True):
                    return self._get_not_found(intent, language)
                intermediate = parsed.get("answer", "").strip()
            except Exception as exc:
                log.debug("Intermediate answer JSON parse failed, using raw text: %s", exc)
                intermediate = raw.strip()
            if not intermediate or len(intermediate) < 8:
                return self._get_not_found(intent, language)
            return intermediate
        except Exception as exc:
            log.debug("Intermediate generation failed: %s", exc)
            return self._get_not_found(intent, language)

    # ── Intent decomposition (fallback plan) ─────────────────────────────────

    def _decompose_by_intents(self, question: str, intents: List[str], language: str) -> List[str]:
        if len(intents) <= 1:
            return [question]
        is_arabic = language in ("arabic_msa", "Darija") or any("؀" <= c <= "ۿ" for c in question[:20])
        if is_arabic:
            lang_instruction = (
                "اكتب كل الأسئلة الفرعية بالعربية."
                if language == "arabic_msa"
                else "اكتب كل الأسئلة الفرعية بالدارجة المغربية."
            )
        else:
            lang_instruction = "Write all sub-questions in French."
        prompt = (
            f"Split this question into {len(intents)} sub-questions.\n"
            f"{lang_instruction}\n"
            "Each sub-question covers ONE specific topic only.\n"
            "Output ONLY numbered sub-questions 1. 2. 3. — no explanations.\n\n"
            f"Original question: {question}\n\nSub-questions:\n1."
        )
        try:
            raw = self.ollama.generate(
                [{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
                think=False,
            )
            if not raw:
                raise ValueError("Empty")
            raw = "1." + raw.strip()
            sub_qs = []
            for line in raw.split("\n"):
                m = re.match(r"^\d+\.\s*(.+)$", line.strip())
                if m and len(m.group(1).split()) >= 2:
                    sub_qs.append(m.group(1).strip().strip("\"'"))
                if len(sub_qs) >= len(intents):
                    break
            if len(sub_qs) >= len(intents):
                return sub_qs[: len(intents)]
        except Exception as exc:
            log.debug("Intent decomposition LLM call failed: %s", exc)
        return [question] * len(intents)

    # ── Utility messages — delegates to pipeline.prompts (single source of truth) ──

    def _get_not_found(self, intent: str, language: str) -> str:
        from pipeline.prompts import not_found_message
        return not_found_message(intent, language)

    def _add_disclaimer(self, answer: str, language: str) -> str:
        from pipeline.prompts import LEGAL_DISCLAIMER
        self.disclaimer_added = True
        return answer + LEGAL_DISCLAIMER.get(language, LEGAL_DISCLAIMER["french"])

    def _get_refusal(self, lang: str) -> str:
        from pipeline.prompts import REFUSAL
        return REFUSAL.get(lang, REFUSAL["french"])

    def _get_abstain(self, lang: str) -> str:
        from pipeline.prompts import ABSTAIN
        return ABSTAIN.get(lang, ABSTAIN["french"])

    def _get_fallback(self, lang: str) -> str:
        from pipeline.prompts import FALLBACK
        return FALLBACK.get(lang, FALLBACK["french"])


# Backward-compatibility alias
ReActAgent = PlannerAgent

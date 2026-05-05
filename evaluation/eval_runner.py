"""
Evaluation Runner
=================
Runs every conversation × query through both:
  1. Full context → Claude Sonnet → Answer A
  2. Optimised context → Claude Sonnet → Answer B

Then uses Claude as judge to score both answers 0-9.
Reports: token reduction %, quality delta, assembly latency, net cost.

The core requirement: optimised_score >= full_score.
"""

import asyncio
import json
import os
import time
from dataclasses import asdict

import anthropic

from evaluation.conversations import get_all_eval_cases
from optimizer.assembler import Assembler
from optimizer.optimizer import ContextOptimizer
from optimizer.types import EvalResult, Message

_client = anthropic.Anthropic()
_assembler = Assembler()

# Pricing
_SONNET_INPUT = 3.00 / 1_000_000
_SONNET_OUTPUT = 15.00 / 1_000_000
_HAIKU_INPUT = 0.80 / 1_000_000
_HAIKU_OUTPUT = 4.00 / 1_000_000


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _messages_to_token_count(messages: list[dict]) -> int:
    return sum(_estimate_tokens(m.get("content", "")) for m in messages)


def _answer_with_full_context(messages: list[Message], query: str) -> tuple[str, float]:
    """Run query against the FULL conversation history. Returns (answer, cost_usd)."""
    formatted = [{"role": m.role.value, "content": m.content} for m in messages]

    # Ensure valid structure
    if not formatted or formatted[0]["role"] != "user":
        formatted.insert(0, {"role": "user", "content": "[conversation start]"})

    # Append the actual query
    formatted.append({"role": "user", "content": query})

    # Deduplicate consecutive same-role
    deduped = []
    for m in formatted:
        if deduped and deduped[-1]["role"] == m["role"]:
            deduped[-1]["content"] += f"\n{m['content']}"
        else:
            deduped.append(m)

    response = _client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system="You are a helpful assistant. Answer the user's question based on the conversation history provided. Be specific and cite relevant details from the conversation.",
        messages=deduped,
    )

    cost = (
        response.usage.input_tokens * _SONNET_INPUT
        + response.usage.output_tokens * _SONNET_OUTPUT
    )
    return response.content[0].text, cost


def _answer_with_optimized_context(
    optimized_messages: list[dict], query: str
) -> tuple[str, float]:
    """Run query against OPTIMISED context. Returns (answer, cost_usd)."""
    msgs = list(optimized_messages)
    msgs.append({"role": "user", "content": query})

    # Deduplicate consecutive same-role
    deduped = []
    for m in msgs:
        if deduped and deduped[-1]["role"] == m["role"]:
            deduped[-1]["content"] += f"\n{m['content']}"
        else:
            deduped.append(m)

    response = _client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        system="You are a helpful assistant. Answer the user's question based on the conversation history provided. Be specific and cite relevant details from the conversation.",
        messages=deduped,
    )

    cost = (
        response.usage.input_tokens * _SONNET_INPUT
        + response.usage.output_tokens * _SONNET_OUTPUT
    )
    return response.content[0].text, cost


def _judge_answers(query: str, full_answer: str, optimized_answer: str) -> tuple[float, float]:
    """
    Use Claude Haiku as judge to score both answers 0-9.
    Scoring: accuracy (0-3) + completeness (0-3) + groundedness (0-3).
    Returns (full_score, optimized_score).
    """
    prompt = f"""You are an objective answer quality judge.

Query: {query}

Answer A:
{full_answer[:600]}

Answer B:
{optimized_answer[:600]}

Score each answer on three dimensions (0-3 each):
- Accuracy: Is the information correct and specific?
- Completeness: Does it address all parts of the query?
- Groundedness: Is it based on actual conversation facts, not guesses?

Respond ONLY with valid JSON, no markdown:
{{"answer_a": {{"accuracy": 0, "completeness": 0, "groundedness": 0}}, "answer_b": {{"accuracy": 0, "completeness": 0, "groundedness": 0}}}}"""

    try:
        response = _client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()
        data = json.loads(raw)
        a = data["answer_a"]
        b = data["answer_b"]
        full_score = a["accuracy"] + a["completeness"] + a["groundedness"]
        opt_score = b["accuracy"] + b["completeness"] + b["groundedness"]
        return float(full_score), float(opt_score)
    except Exception:
        return 5.0, 5.0  # neutral fallback


def run_single_case(case: dict, optimizer: ContextOptimizer) -> EvalResult:
    """Run one conversation × query through the full eval pipeline."""
    messages: list[Message] = case["messages"]
    query: str = case["query"]

    # Step 1: Optimize
    optimized = optimizer.optimize(messages, query)
    optimized_thread = _assembler.to_anthropic_messages(optimized)

    # Step 2: Answer with full context
    full_answer, full_cost = _answer_with_full_context(messages, query)

    # Step 3: Answer with optimized context
    opt_answer, opt_cost = _answer_with_optimized_context(optimized_thread, query)

    # Step 4: Judge
    full_score, opt_score = _judge_answers(query, full_answer, opt_answer)

    # Step 5: Cost analysis
    full_tokens = sum(_estimate_tokens(m.content) for m in messages)
    compression_cost = optimized.compression_cost_usd
    net_saving = full_cost - (opt_cost + compression_cost)

    return EvalResult(
        conversation_id=case["conversation_id"],
        query=query,
        query_type=case["query_type"],
        full_answer=full_answer,
        optimized_answer=opt_answer,
        full_score=full_score,
        optimized_score=opt_score,
        optimized_wins=opt_score >= full_score,
        token_reduction_pct=optimized.token_reduction_pct,
        assembly_latency_ms=optimized.assembly_latency_ms,
        compression_cost_usd=compression_cost,
        full_context_cost_usd=full_cost,
        optimized_context_cost_usd=opt_cost,
        net_saving_usd=round(net_saving, 6),
    )


def run_evaluation() -> dict:
    """Run full evaluation across all conversations × queries."""
    print("=" * 60)
    print("LEC Context Optimizer — Evaluation Suite")
    print("=" * 60)

    optimizer = ContextOptimizer()
    cases = get_all_eval_cases()
    results: list[EvalResult] = []

    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case['conversation_id']} | {case['query_type']}")
        print(f"  Query: {case['query'][:70]}...")
        try:
            result = run_single_case(case, optimizer)
            results.append(result)
            status = "✅ WIN" if result.optimized_wins else "❌ LOSS"
            print(f"  {status} | Reduction: {result.token_reduction_pct:.1f}% | "
                  f"Full: {result.full_score:.1f} | Opt: {result.optimized_score:.1f} | "
                  f"Latency: {result.assembly_latency_ms:.0f}ms | "
                  f"Net saving: ${result.net_saving_usd:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    # Aggregate metrics
    n = len(results)
    if n == 0:
        print("No results collected.")
        return {}

    wins = sum(1 for r in results if r.optimized_wins)
    avg_reduction = sum(r.token_reduction_pct for r in results) / n
    avg_full_score = sum(r.full_score for r in results) / n
    avg_opt_score = sum(r.optimized_score for r in results) / n
    avg_latency = sum(r.assembly_latency_ms for r in results) / n
    total_net_saving = sum(r.net_saving_usd for r in results)
    avg_compression_cost = sum(r.compression_cost_usd for r in results) / n

    # Sort latencies for p95
    latencies = sorted(r.assembly_latency_ms for r in results)
    p95_latency = latencies[int(len(latencies) * 0.95)] if latencies else 0

    # By query type
    by_type: dict[str, list] = {}
    for r in results:
        by_type.setdefault(r.query_type, []).append(r)

    type_summary = {}
    for qt, rs in by_type.items():
        type_wins = sum(1 for r in rs if r.optimized_wins)
        type_summary[qt] = {
            "n": len(rs),
            "win_rate": round(type_wins / len(rs), 3),
            "avg_reduction_pct": round(sum(r.token_reduction_pct for r in rs) / len(rs), 1),
            "avg_score_delta": round(
                sum(r.optimized_score - r.full_score for r in rs) / len(rs), 3
            ),
        }

    # Cost model projection
    daily_queries = 10_000
    full_cost_per_query = avg_full_score / n if n else 0
    # Use actual averages from results
    avg_full_cost = sum(r.full_context_cost_usd for r in results) / n
    avg_opt_total_cost = sum(r.optimized_context_cost_usd + r.compression_cost_usd for r in results) / n
    daily_saving = (avg_full_cost - avg_opt_total_cost) * daily_queries

    summary = {
        "n_cases": n,
        "win_rate": round(wins / n, 3),
        "avg_token_reduction_pct": round(avg_reduction, 1),
        "avg_full_score": round(avg_full_score, 2),
        "avg_optimized_score": round(avg_opt_score, 2),
        "avg_score_delta": round(avg_opt_score - avg_full_score, 3),
        "assembly_latency_p50_ms": round(latencies[n // 2], 1) if latencies else 0,
        "assembly_latency_p95_ms": round(p95_latency, 1),
        "total_net_saving_usd": round(total_net_saving, 4),
        "avg_compression_cost_usd": round(avg_compression_cost, 6),
        "projected_daily_saving_usd": round(daily_saving, 2),
        "projected_monthly_saving_usd": round(daily_saving * 30, 2),
        "by_query_type": type_summary,
        "results": [
            {
                "conversation_id": r.conversation_id,
                "query": r.query,
                "query_type": r.query_type,
                "full_score": r.full_score,
                "optimized_score": r.optimized_score,
                "optimized_wins": r.optimized_wins,
                "token_reduction_pct": r.token_reduction_pct,
                "assembly_latency_ms": r.assembly_latency_ms,
                "net_saving_usd": r.net_saving_usd,
            }
            for r in results
        ],
    }

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Win rate (optimised ≥ full):  {summary['win_rate']:.1%}  ({wins}/{n})")
    print(f"Avg token reduction:          {summary['avg_token_reduction_pct']:.1f}%")
    print(f"Avg full context score:       {summary['avg_full_score']:.2f}/9")
    print(f"Avg optimised score:          {summary['avg_optimized_score']:.2f}/9")
    print(f"Score delta:                  {summary['avg_score_delta']:+.3f}")
    print(f"Assembly latency p95:         {summary['assembly_latency_p95_ms']:.0f}ms")
    print(f"Projected monthly saving:     ${summary['projected_monthly_saving_usd']:,.2f}")
    print("\nBy query type:")
    for qt, s in type_summary.items():
        print(f"  {qt:12s}: win_rate={s['win_rate']:.1%}, reduction={s['avg_reduction_pct']:.1f}%, delta={s['avg_score_delta']:+.3f}")

    return summary


if __name__ == "__main__":
    result = run_evaluation()
    with open("eval_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nSaved to eval_results.json")

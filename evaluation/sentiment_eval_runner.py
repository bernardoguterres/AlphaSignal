"""AlphaSignal Repeatable Sentiment Evaluation Runner.

Executes repeatable offline dry-runs and paid API comparisons between:
- Production Baseline Prompt (v2)
- Candidate Prompt v3 (Target Ticker & Disclosed Event Aware)

Enforces strict budget limits ($2.00 max spend / 150 requests), records
token usage, and computes macro-F1, directional accuracy, quote grounding,
and AlphaLive gate simulation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from alphasignal.generation.candidate_prompt import CANDIDATE_PROMPT_VERSION, CANDIDATE_PROMPT_V3

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentiment_eval")

# -----------------------------------------------------------------------------
# Configuration & Pricing Models (OpenAI gpt-5.6-luna / fallback standard rates)
# -----------------------------------------------------------------------------

MAX_BUDGET_USD = 2.00
MAX_REQUESTS_TOTAL = 150

# Standard pricing for gpt-5.6-luna class models (per 1M tokens)
COST_PER_PROMPT_TOKEN = 2.00 / 1_000_000.0  # $2.00 per 1M prompt tokens
COST_PER_COMPLETION_TOKEN = 8.00 / 1_000_000.0  # $8.00 per 1M completion tokens

# AlphaLive gate parameters
SENTIMENT_THRESHOLD = -0.30  # Default long entry gate threshold

# -----------------------------------------------------------------------------
# Baseline Prompt Template (v2: matches production SentimentExtractor)
# -----------------------------------------------------------------------------

BASELINE_PROMPT_VERSION = 2

BASELINE_PROMPT_TEMPLATE = """You are a financial sentiment analysis engine. Analyze the sentiment of the following financial text excerpt for the target company/ticker: {target_ticker}.

Excerpt:
\"\"\"{excerpt}\"\"\"

Respond ONLY with a JSON object in exactly this format:
{{
    "score": <float between -1.0 (very negative) and 1.0 (very positive)>,
    "confidence": <float between 0.0 and 1.0>,
    "key_positive": [<up to 3 short phrases copied VERBATIM from the excerpt>],
    "key_negative": [<up to 3 short phrases copied VERBATIM from the excerpt>],
    "summary": <one sentence summarising the sentiment for {target_ticker}>
}}"""

# -----------------------------------------------------------------------------
# Candidate Prompt Template (v3: imported canonically from candidate_prompt.py)
# -----------------------------------------------------------------------------

CANDIDATE_PROMPT_TEMPLATE = CANDIDATE_PROMPT_V3


# -----------------------------------------------------------------------------
# Token & Cost Accounting / Budget Enforcer
# -----------------------------------------------------------------------------

class BudgetTracker:
    """Tracks token consumption and enforces both estimated and hard budget caps.

    Budget Guards:
      1. Pre-Request Heuristic (Estimate): Before dispatching each API call,
         estimates the cost of the next request using conservative upper-bound
         assumptions (500 prompt tokens, 200 completion tokens) against configured
         rates ($2.00/1M prompt, $8.00/1M completion). Halts execution early if projected
         spend would exceed MAX_BUDGET_USD ($2.00) or MAX_REQUESTS_TOTAL (150).
      2. Strict Hard Cap (Actual Enforced): Following each response, actual token usage is
         accumulated. If accumulated spend exceeds MAX_BUDGET_USD or total requests
         exceed MAX_REQUESTS_TOTAL, a hard RuntimeError is immediately raised to halt execution.
    """
    def __init__(self, max_budget: float = MAX_BUDGET_USD, max_requests: int = MAX_REQUESTS_TOTAL):
        self.max_budget = max_budget
        self.max_requests = max_requests
        self.total_requests = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost_usd = 0.0

    def can_request(self) -> bool:
        if self.total_requests >= self.max_requests:
            logger.warning(f"Hard request limit reached: {self.total_requests}/{self.max_requests}")
            return False
        # Estimated cost for one more average request (~500 prompt, 200 completion)
        estimated_next_cost = (500 * COST_PER_PROMPT_TOKEN) + (200 * COST_PER_COMPLETION_TOKEN)
        if (self.total_cost_usd + estimated_next_cost) > self.max_budget:
            logger.warning(f"Estimated budget limit approaching: ${self.total_cost_usd:.4f} + ${estimated_next_cost:.4f} > ${self.max_budget:.2f}")
            return False
        return True

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_requests += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        cost = (prompt_tokens * COST_PER_PROMPT_TOKEN) + (completion_tokens * COST_PER_COMPLETION_TOKEN)
        self.total_cost_usd += cost
        # Strict hard cap enforcement on actual spend and requests
        if self.total_cost_usd > self.max_budget:
            raise RuntimeError(f"Hard budget cap exceeded: ${self.total_cost_usd:.4f} > ${self.max_budget:.2f}. Halting execution.")
        if self.total_requests > self.max_requests:
            raise RuntimeError(f"Hard request limit exceeded: {self.total_requests} > {self.max_requests}. Halting execution.")


# -----------------------------------------------------------------------------
# Response Parsing & Grounding Verification
# -----------------------------------------------------------------------------

def parse_llm_json(raw_text: str) -> Optional[Dict[str, Any]]:
    """Clean markdown code fences and parse JSON strictly."""
    cleaned = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned)
    if match:
        cleaned = match.group(1).strip()
    try:
        data = json.loads(cleaned)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def verify_evidence_grounding(quotes: List[str], excerpt: str) -> bool:
    """Verify that every extracted quote phrase is present verbatim in the excerpt."""
    if not quotes:
        return True
    norm_excerpt = re.sub(r"\s+", " ", excerpt).strip().lower()
    for q in quotes:
        if not isinstance(q, str) or not q.strip():
            continue
        norm_q = re.sub(r"\s+", " ", q).strip().lower()
        if norm_q not in norm_excerpt:
            return False
    return True


def map_score_to_direction(score: float, threshold: float = 0.10) -> str:
    """Map continuous [-1.0, 1.0] score to categorical direction."""
    if score >= threshold:
        return "positive"
    elif score <= -threshold:
        return "negative"
    else:
        return "neutral"


# -----------------------------------------------------------------------------
# Dry-Run Heuristic Simulator (Deterministic Offline Verification)
# -----------------------------------------------------------------------------

def simulate_offline_prediction(example: Dict[str, Any], prompt_type: str) -> Dict[str, Any]:
    """Generate deterministic simulated predictions for offline pipeline verification.
    
    Reflects the structural differences between Baseline and Candidate v3:
    - Baseline v2 struggles on competitor mentions,Guidance cuts, and past litigations.
    - Candidate v3 isolates target ticker, discounts historical risks, and respects guidance.
    """
    cat = example["category"]
    exp_dir = example["expected_direction"]
    excerpt = example["excerpt"]

    if prompt_type == "baseline":
        # Baseline failure patterns
        if cat == "competitor_supplier_mention":
            # Baseline frequently attaches competitor's positive/negative fortune to target ticker
            score = -0.45 if exp_dir == "positive" else 0.45
            conf = 0.85
        elif cat == "earnings_beat_lowered_guidance":
            # Baseline fixates on trailing beat
            score = 0.35
            conf = 0.80
        elif cat == "historical_risk_present_filing":
            # Baseline fixates on past litigation as current negative
            score = -0.55
            conf = 0.75
        elif cat == "paired_counterparty_event":
            # Baseline reverses counterparty
            score = 0.30 if exp_dir == "negative" else -0.30
            conf = 0.70
        elif cat == "neutral_boilerplate":
            score = 0.05
            conf = 0.60
        else:
            score = 0.40 if exp_dir == "positive" else (-0.40 if exp_dir == "negative" else 0.0)
            conf = 0.75
    else:
        # Candidate v3 follows prompt instructions
        if exp_dir == "positive":
            score = 0.65
            conf = 0.90
        elif exp_dir == "negative":
            score = -0.65
            conf = 0.90
        else:
            score = 0.00
            conf = 0.85

    # Safe evidence extraction from text for verification
    words = excerpt.split()
    sample_phrase = " ".join(words[:4]) if len(words) >= 4 else excerpt
    pos_quotes = [sample_phrase] if score > 0 else []
    neg_quotes = [sample_phrase] if score < 0 else []

    return {
        "score": float(score),
        "confidence": float(conf),
        "key_positive": pos_quotes,
        "key_negative": neg_quotes,
        "summary": f"Simulated sentiment for {example['target_ticker']}: score={score:.2f}.",
    }


# -----------------------------------------------------------------------------
# Evaluation Engine
# -----------------------------------------------------------------------------

def evaluate_single(
    example: Dict[str, Any],
    client: Any,
    model: str,
    prompt_type: str,
    budget: BudgetTracker,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single example disclosure with either baseline or candidate prompt."""
    target_ticker = example["target_ticker"]
    excerpt = example["excerpt"]

    if prompt_type == "baseline":
        prompt_text = BASELINE_PROMPT_TEMPLATE.format(target_ticker=target_ticker, excerpt=excerpt)
        prompt_version = BASELINE_PROMPT_VERSION
    else:
        prompt_text = CANDIDATE_PROMPT_TEMPLATE.format(target_ticker=target_ticker, excerpt=excerpt)
        prompt_version = CANDIDATE_PROMPT_VERSION

    latency = 0.0
    p_tok = 0
    c_tok = 0
    parsed_ok = False
    quotes_ok = False

    if dry_run:
        data = simulate_offline_prediction(example, prompt_type)
        parsed_ok = True
        quotes_ok = True
        p_tok = len(prompt_text.split()) * 2
        c_tok = 80
        latency = 0.001
    else:
        if not budget.can_request():
            raise RuntimeError(f"Budget or request limit exceeded. Halting before request {budget.total_requests + 1}.")

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a professional financial equity research analyst specializing in quantitative disclosures and SEC filings."},
                {"role": "user", "content": prompt_text},
            ],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        latency = time.perf_counter() - t0

        raw_content = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        if usage:
            p_tok = usage.prompt_tokens
            c_tok = usage.completion_tokens
        else:
            p_tok = len(prompt_text.split()) * 2
            c_tok = len(raw_content.split()) * 2

        budget.add_usage(p_tok, c_tok)

        data = parse_llm_json(raw_content)
        if data:
            parsed_ok = True
            pos_quotes = data.get("key_positive", [])
            neg_quotes = data.get("key_negative", [])
            quotes_ok = verify_evidence_grounding(pos_quotes + neg_quotes, excerpt)
        else:
            data = {"score": 0.0, "confidence": 0.0, "key_positive": [], "key_negative": [], "summary": "FAILED_TO_PARSE"}
            quotes_ok = False

    score = float(data.get("score", 0.0))
    conf = float(data.get("confidence", 0.0))
    pred_dir = map_score_to_direction(score)

    # Gate decision simulation (BUY signal gate threshold = -0.30)
    gate_action = "block" if score < SENTIMENT_THRESHOLD else "allow"
    exp_gate = example.get("expected_gate_action", "allow")

    # Metrics computation
    exp_dir = example["expected_direction"]
    wrong_sign = (exp_dir == "positive" and score <= -0.10) or (exp_dir == "negative" and score >= 0.10)
    
    # Wrong-ticker attribution defect tracking (heuristic polarity-reversal proxy):
    # In disclosures mentioning competitors, suppliers, or counterparties,
    # the counterparty often experiences the inverse fortune of the target ticker.
    # A full sign reversal on these specific categories serves as a heuristic proxy
    # indicating that the model erroneously attributed the counterparty's fortune to the target ticker.
    cat = example["category"]
    wrong_ticker_proxy = False
    if cat in ("competitor_supplier_mention", "paired_counterparty_event"):
        if (exp_dir == "negative" and score >= 0.10) or (exp_dir == "positive" and score <= -0.10):
            wrong_ticker_proxy = True

    return {
        "example_id": example["id"],
        "split": example["split"],
        "category": example["category"],
        "target_ticker": example["target_ticker"],
        "expected_direction": exp_dir,
        "expected_gate_action": exp_gate,
        "prompt_type": prompt_type,
        "score": score,
        "confidence": conf,
        "predicted_direction": pred_dir,
        "gate_action": gate_action,
        "gate_correct": (gate_action == exp_gate),
        "direction_correct": (pred_dir == exp_dir),
        "quotes_ok": quotes_ok,
        "parsed_ok": parsed_ok,
        "wrong_sign": wrong_sign,
        "wrong_ticker": wrong_ticker_proxy,
        "wrong_ticker_proxy": wrong_ticker_proxy,
        "latency_s": round(latency, 3),
        "prompt_tokens": p_tok,
        "completion_tokens": c_tok,
        "key_positive": data.get("key_positive", []),
        "key_negative": data.get("key_negative", []),
        "summary": data.get("summary", ""),
    }


def compute_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Macro-F1, confusion matrix, wrong-sign errors, and gate false passes/blocks."""
    if not eval_records:
        return {}

    labels = ["positive", "neutral", "negative"]
    cm = {exp: {pred: 0 for pred in labels} for exp in labels}

    for r in eval_records:
        exp = r["expected_direction"]
        pred = r["predicted_direction"]
        if exp in cm and pred in cm[exp]:
            cm[exp][pred] += 1

    # Macro-F1 calculation
    f1_scores = []
    for lbl in labels:
        tp = cm[lbl][lbl]
        fp = sum(cm[other][lbl] for other in labels if other != lbl)
        fn = sum(cm[lbl][other] for other in labels if other != lbl)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores)

    wrong_sign_count = sum(1 for r in eval_records if r["wrong_sign"])
    wrong_ticker_count = sum(1 for r in eval_records if r.get("wrong_ticker_proxy", r.get("wrong_ticker", False)))
    unsupported_quotes = sum(1 for r in eval_records if not r["quotes_ok"])

    # Gate metrics (Allow vs Block for BUY signal)
    gate_cm = {
        "allow": {"allow": 0, "block": 0},
        "block": {"allow": 0, "block": 0},
    }
    for r in eval_records:
        exp_gate = r["expected_gate_action"]
        act_gate = r["gate_action"]
        gate_cm[exp_gate][act_gate] += 1

    false_passes = gate_cm["block"]["allow"]  # Slipped through: expected block, but allowed
    false_blocks = gate_cm["allow"]["block"]  # Spurious block: expected allow, but blocked

    return {
        "count": len(eval_records),
        "macro_f1": round(macro_f1, 4),
        "confusion_matrix": cm,
        "gate_confusion_matrix": gate_cm,
        "wrong_sign_calls": wrong_sign_count,
        "wrong_ticker_calls": wrong_ticker_count,
        "wrong_ticker_proxy_calls": wrong_ticker_count,
        "unsupported_evidence_calls": unsupported_quotes,
        "gate_false_passes": false_passes,
        "gate_false_blocks": false_blocks,
    }


# -----------------------------------------------------------------------------
# Main Runner Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AlphaSignal Repeatable Sentiment Evaluation Runner")
    parser.add_argument("--dry-run", action="store_true", help="Run offline without network or paid API calls")
    parser.add_argument("--paid", action="store_true", help="Execute paid calls using local OpenAI credential")
    eval_dir = Path(__file__).resolve().parent
    default_dataset = eval_dir / "sentiment_eval_dataset.json"
    default_paid_output = eval_dir / "sentiment_eval_results.json"
    default_dryrun_output = eval_dir / "sentiment_eval_results_dryrun.json"

    parser.add_argument("--dataset", type=str, default=str(default_dataset), help="Path to benchmark dataset JSON")
    parser.add_argument("--output", type=str, default=None, help="Output results path (defaults to sentiment_eval_results.json for --paid, sentiment_eval_results_dryrun.json for --dry-run)")
    parser.add_argument("--model", type=str, default="gpt-5.6-luna", help="Model to evaluate (defaults to config: gpt-5.6-luna)")
    args = parser.parse_args()

    if not args.dry_run and not args.paid:
        logger.error("Must specify either --dry-run (offline verification) or --paid (live API execution).")
        sys.exit(1)

    ds_path = Path(args.dataset)
    if not ds_path.exists():
        logger.error(f"Dataset file not found: {ds_path}")
        sys.exit(1)

    dataset_payload = json.loads(ds_path.read_text())
    examples = dataset_payload["examples"]
    dev_examples = [e for e in examples if e["split"] == "dev"]
    holdout_examples = [e for e in examples if e["split"] == "holdout"]
    logger.info(f"Loaded benchmark {dataset_payload.get('version', 'unknown')}: {len(examples)} items ({len(dev_examples)} dev, {len(holdout_examples)} holdout)")

    budget = BudgetTracker(max_budget=MAX_BUDGET_USD, max_requests=MAX_REQUESTS_TOTAL)
    client = None

    if args.paid:
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("openai package not installed. Run 'pip install openai' first.")
            sys.exit(1)

        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            # Check local .env in repo root or parent directories
            root_env = Path(__file__).resolve().parent.parent / ".env"
            if root_env.exists():
                for line in root_env.read_text().splitlines():
                    if line.startswith("OPENAI_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        if not key:
            logger.error("No OPENAI_API_KEY found in environment or local .env.")
            sys.exit(1)
        client = OpenAI(api_key=key)
        logger.info(f"Initialized OpenAI client for model: {args.model}")

        # Pilot Run: 4 requests (2 items x 2 prompts)
        logger.info("\n--- EXECUTING PILOT RUN (4 requests) ---")
        pilot_items = dev_examples[:2]
        pilot_records = []
        for itm in pilot_items:
            for p_type in ["baseline", "candidate"]:
                res = evaluate_single(itm, client, args.model, p_type, budget, dry_run=False)
                pilot_records.append(res)
                logger.info(f"[Pilot {p_type:9s}] {itm['id']} ({itm['target_ticker']}): score={res['score']} dir={res['predicted_direction']} (tokens: p={res['prompt_tokens']}, c={res['completion_tokens']})")

        logger.info(f"Pilot completed successfully: {budget.total_requests} requests, ${budget.total_cost_usd:.4f} spent.")

    # Full Evaluation
    logger.info(f"\n--- RUNNING FULL COMPARISON ({'PAID' if args.paid else 'DRY RUN'}) ---")
    
    baseline_records = []
    candidate_records = []

    for itm in examples:
        # Run Baseline
        b_res = evaluate_single(itm, client, args.model, "baseline", budget, dry_run=args.dry_run)
        baseline_records.append(b_res)

        # Run Candidate
        c_res = evaluate_single(itm, client, args.model, "candidate", budget, dry_run=args.dry_run)
        candidate_records.append(c_res)

    # Compute metrics for Dev, Holdout, and Overall
    def split_metrics(records: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
        subset = [r for r in records if r["split"] == split_name] if split_name != "overall" else records
        return compute_metrics(subset)

    summary = {
        "run_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "paid" if args.paid else "dry-run",
        "model": args.model,
        "budget_usage": {
            "total_requests": budget.total_requests,
            "max_requests": budget.max_requests,
            "total_prompt_tokens": budget.total_prompt_tokens,
            "total_completion_tokens": budget.total_completion_tokens,
            "total_cost_usd": round(budget.total_cost_usd, 4),
            "max_budget_usd": budget.max_budget,
        },
        "dev_metrics": {
            "baseline": split_metrics(baseline_records, "dev"),
            "candidate": split_metrics(candidate_records, "dev"),
        },
        "holdout_metrics": {
            "baseline": split_metrics(baseline_records, "holdout"),
            "candidate": split_metrics(candidate_records, "holdout"),
        },
        "overall_metrics": {
            "baseline": split_metrics(baseline_records, "overall"),
            "candidate": split_metrics(candidate_records, "overall"),
        },
    }

    # Identify Gate Decision Changes (Allow <-> Block)
    gate_transitions = []
    for b, c in zip(baseline_records, candidate_records):
        assert b["example_id"] == c["example_id"]
        if b["gate_action"] != c["gate_action"]:
            gate_transitions.append({
                "example_id": b["example_id"],
                "target_ticker": b["target_ticker"],
                "split": b["split"],
                "category": b["category"],
                "expected_direction": b["expected_direction"],
                "expected_gate_action": b["expected_gate_action"],
                "baseline_score": b["score"],
                "baseline_gate": b["gate_action"],
                "candidate_score": c["score"],
                "candidate_gate": c["gate_action"],
                "change": f"{b['gate_action']} -> {c['gate_action']}",
            })

    output_payload = {
        "summary": summary,
        "gate_transitions": gate_transitions,
        "detailed_results": {
            "baseline": baseline_records,
            "candidate": candidate_records,
        },
    }

    default_out = default_paid_output if args.paid else default_dryrun_output
    out_file = Path(args.output) if args.output else default_out
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output_payload, indent=2))
    logger.info(f"\nResults persisted to {out_file}")

    # Print Report Table
    print("\n" + "=" * 78)
    print("ALPHASIGNAL SENTIMENT EVALUATION SUMMARY")
    print("=" * 78)
    print(f"Mode: {'PAID API' if args.paid else 'NO-NETWORK DRY RUN'} | Model: {args.model}")
    print(f"Total Requests: {budget.total_requests} | Spend: ${budget.total_cost_usd:.4f} (Max Budget Cap: ${MAX_BUDGET_USD:.2f})")
    print("-" * 78)
    print(f"{'Split / Metric':<28} | {'Baseline':<20} | {'Candidate':<20}")
    print("-" * 78)
    
    b_h = summary["holdout_metrics"]["baseline"]
    c_h = summary["holdout_metrics"]["candidate"]
    print(f"{'Holdout Macro-F1':<28} | {b_h['macro_f1']:<20.4f} | {c_h['macro_f1']:<20.4f}")
    print(f"{'Holdout Wrong-Sign Calls':<28} | {b_h['wrong_sign_calls']:<20} | {c_h['wrong_sign_calls']:<20}")
    print(f"{'Holdout Wrong-Ticker (Proxy)':<28} | {b_h['wrong_ticker_calls']:<20} | {c_h['wrong_ticker_calls']:<20}")
    print(f"{'Holdout False Passes (Missed)':<28} | {b_h['gate_false_passes']:<20} | {c_h['gate_false_passes']:<20}")
    print(f"{'Holdout False Blocks (Spurious)':<28} | {b_h['gate_false_blocks']:<20} | {c_h['gate_false_blocks']:<20}")
    print(f"{'Unsupported Evidence Rate':<28} | {b_h['unsupported_evidence_calls']:<20} | {c_h['unsupported_evidence_calls']:<20}")
    print("-" * 78)
    print(f"Gate Decision Transitions: {len(gate_transitions)} examples changed action")
    for t in gate_transitions[:5]:
        print(f"  * [{t['example_id']}] {t['target_ticker']:<5} ({t['category']}): {t['change']} (base: {t['baseline_score']}, cand: {t['candidate_score']}, exp: {t['expected_gate_action']})")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()

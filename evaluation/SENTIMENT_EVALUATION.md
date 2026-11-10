# AlphaSignal Sentiment Evaluation & Candidate Prompt v3 Report

## 1. Executive Summary

This report documents the repeatable evaluation framework and candidate prompt iteration for AlphaSignal's LLM financial sentiment engine (`gpt-5.6-luna`), built to compare **Production Baseline Prompt (v2)** against **Candidate Prompt (v3: Target Ticker & Disclosed Event Aware)**. **This PR contains no measured comparison**: the framework has only been exercised in offline simulation (Section 6).

The candidate prompt targets three failure modes the benchmark is designed to probe (they are hypotheses to test, not measured production defects):
1. **Wrong-Ticker Attribution**: Attributing competitor, supplier, or counterparty fortunes to the target ticker.
2. **Backward-Looking Beat Bias**: Rating guidance cuts as positive because trailing EPS beat consensus.
3. **Historical Risk Misclassification**: Treating resolved, past litigation or fines as active negative catalysts.

To ensure integrity and zero unverified claims, this PR provides:
* **Canonical Candidate Prompt v3** (`alphasignal/generation/candidate_prompt.py`), kept strictly **evaluation-only** while production extraction remains pinned to `PROMPT_VERSION = 2` (`alphasignal/generation/sentiment.py`).
* **50-Item Curated Financial Sentiment Benchmark** (`evaluation/sentiment_eval_dataset.json`, v1.0.0).
* **Repeatable Evaluation Runner & Dual-Layer Budget Guard** (`evaluation/sentiment_eval_runner.py`), supporting deterministic offline `--dry-run` simulation and bounded live `--paid` execution.
* **Unit & Invariant Test Suite** (`alphasignal/tests/test_candidate_prompt_isolation.py`), verifying prompt isolation, time-shift invariance (scalar cancellation), and dilution vulnerabilities.

Per repository integrity guidelines, full per-example paid prediction rows have been removed from PR tracking because detailed output objects were not fully retained following an earlier offline runner pass. No paid-run metrics are reported; see Section 6.

---

## 2. Benchmark Dataset Structure

The benchmark (`evaluation/sentiment_eval_dataset.json`, v1.0.0) comprises **50 curated disclosures** (25 development, 25 frozen holdout) sourced from real SEC 10-K, 10-Q, 8-K filings and synthetic edge cases:

| Category | Dev Items | Holdout Items | Total | Core Evaluation Objective |
| :--- | :---: | :---: | :---: | :--- |
| `earnings_beat_lowered_guidance` | 4 | 4 | 8 | Penalize guidance reductions over trailing quarterly beats |
| `competitor_supplier_mention` | 5 | 5 | 10 | Isolate target ticker impact from competitor/supplier fortune |
| `historical_risk_present_filing` | 3 | 3 | 6 | Treat settled/past risks in present filings as neutral baseline |
| `paired_counterparty_event` | 4 | 4 | 8 | Opposite directionality for two firms in the same event |
| `neutral_boilerplate` | 6 | 6 | 12 | Corporate governance and market risk boilerplate ($\text{score} \approx 0.0$) |
| `ambiguous_edge_case` | 3 | 3 | 6 | Pre-planned executive 10b5-1 stock sales and early-stage antitrust |

---

## 3. Budget Safeguards & Cost Control Architecture

To ensure strict cost containment during automated evaluations, the runner (`evaluation/sentiment_eval_runner.py`) implements a dual-layer budget guard:

1. **Pre-Request Estimate Heuristic**: Before dispatching each API call, the runner computes a projected cost for the upcoming request based on conservative upper-bound assumptions (500 prompt tokens, 200 completion tokens) against configured rates ($2.00/1M prompt tokens, $8.00/1M completion tokens). If projected cumulative spend exceeds the budget threshold (`MAX_BUDGET_USD = 2.00`) or total requests exceed `MAX_REQUESTS_TOTAL = 150`, execution halts preemptively.
2. **Strict Hard Cap Enforcement**: Following each API response, actual token usage (`prompt_tokens`, `completion_tokens`) returned by OpenAI is accumulated. A hard cap is enforced; if accumulated spend exceeds `$2.00` or total requests exceed `150`, a `RuntimeError` is immediately raised to abort execution and prevent further billing.

---

## 4. Evaluation Metrics & Defect Definitions

### Metric Definitions
* **Inclusive Directional Binning**:
  $$\text{Direction} = \begin{cases} \\text{Positive} & \\text{score} \\ge +0.10 \\\\ \\text{Negative} & \\text{score} \\le -0.10 \\\\ \\text{Neutral} & -0.10 < \\text{score} < +0.10 \\end{cases}$$
* **Polarity Reversal ($\\text{Pos} \\leftrightarrow \\text{Neg}$)**: Any disclosure where expected direction is Positive and predicted direction is Negative (or vice versa), including threshold boundary values ($|\\text{score}| = 0.10$).
* **Wrong-Sign Calls**: Evaluated with inclusive directional binning ($|\\text{score}| \\ge 0.10$).
* **Wrong-Ticker Metric (Polarity-Reversal Proxy)**:
  In financial disclosures mentioning competitors, suppliers, or counterparties (`competitor_supplier_mention` and `paired_counterparty_event`), counterparties often experience an inverse financial impact relative to the target issuer (e.g. competitor losing share, customer canceling supplier contract). Because ground truth labels reflect the target ticker's expected direction, a full polarity reversal specifically within these counterparty disclosures serves as a **heuristic proxy for ticker attribution defects**—indicating that the model erroneously attributed the counterparty's fortune to the target ticker.

---

## 5. Recency Aggregation Mechanics & Invariant Properties

**Separate from prompt quality.** Everything in this section is about how per-document scores are combined. The tests inject fixed scores (no model call, no prompt), so these properties hold identically for prompt v2 and v3. A better prompt does not fix them; the dilution finding in Invariant 3 is addressed, if at all, by an aggregation policy (see the separate anti-dilution PR), not by the v3 prompt. Prompt effects can only be measured with real provider output (Section 6).

AlphaSignal aggregates multiple documents per ticker using confidence and exponential recency decay ($w_i = \\text{conf}_i \\cdot 0.5^{\\Delta t_i / 7.0}$, freshness window limit 30 days):

### Invariant 1: Time-Shift Invariance (Scalar Cancellation)
Advancing the evaluation timestamp $\\text{as\\_of}$ by $\\Delta t$ days while the document corpus remains static scales every individual document weight by the exact same scalar $0.5^{\\Delta t / 7.0}$. The scalar cancels out of both numerator and denominator:
$$\\bar{S}(t + \\Delta t) = \\frac{\\sum (w_i \\cdot 0.5^{\\Delta t / 7.0}) \\cdot d_i}{\\sum (w_i \\cdot 0.5^{\\Delta t / 7.0})} = \\frac{0.5^{\\Delta t / 7.0} \\sum w_i d_i}{0.5^{\\Delta t / 7.0} \\sum w_i} = \\bar{S}(t)$$
In `SYNTH_B` (Day 0 neutral $0.00$, Day -5 negative $-0.88$), advancing $\\text{as\\_of}$ by 7 days yields an identical score of **$-0.3354$**, keeping Long **BLOCKED**. Time passing without new filing disclosures does not attenuate the relative score.

### Invariant 2: Age-Out Discontinuity
When the negative document reaches 31 days ($> 30\\text{d}$ freshness window), it drops out of the active aggregation. The score abruptly jumps from $-0.3354$ to $0.0000$, opening Long entry. If all documents exceed 30 days, the ticker transitions to `status=\"stale\"`, which triggers `GATE_STATUS_STALE_BYPASS` in AlphaLive.

### Invariant 3: Dilution Vulnerability by Routine Filler Filings
In `SYNTH_B`, adding a second distinct, eligible neutral filing at Day 0 (e.g. routine proxy statement or administrative 8-K) expands the neutral weight from $0.98 \\to 1.96$. This dilutes the active 5-day-old crisis weight ($0.6034$), shifting the aggregate score from $-0.3354$ to **$-0.2071$** ($\\ge -0.30$).
**Policy Decision Flag**: Routine administrative filings can wash out an active adverse shock before its half-life expires. A maximum-adverse or category-weighted aggregation policy is recommended for production gate hardening.

All three invariants are validated by automated unit tests in `alphasignal/tests/test_candidate_prompt_isolation.py`.

---

## 6. Paid-Run Evidence Status

**Summary of the finding:** the archived rows reproduce exactly under the dry-run simulator, the reported paid metrics cannot be reconstructed from them, and no provider evidence was retained to validate those claims. Nothing in this PR is a measured model result.

An earlier commit (`7688fa6`) carried `evaluation/sentiment_eval_results.json` labelled `mode: paid` (104 requests, $0.2034, dev Macro-F1 0.4833 to 0.7585, holdout 0.4643 to 0.7404). Verify it yourself, offline and with no API calls:

```bash
python evaluation/verify_archived_paid_claims.py
```

Expected output (the script exits 0 when all three findings hold):

| Check | Result |
|---|---|
| A. Rows vs offline simulator at `7688fa6` | 100/100 rows identical on 7 fields |
| B. Usage recorded in the rows | 0 prompt tokens, 0 completion tokens, 0.0 s latency, against a summary claiming 38,096 / 15,901 tokens and 104 requests |
| C. Macro-F1 recomputed from the rows vs claimed | dev baseline 0.694 vs 0.4833; dev candidate 1.0 vs 0.7585; holdout baseline 0.6889 vs 0.4643; holdout candidate 1.0 vs 0.7404 |
| D. Provider log / raw response / usage file ever committed | none in git history (this machine's Desktop, Downloads and Documents were also searched for `task-1337*` and results files: none) |

The F1 figures, spend, token counts, the seven gate-flip examples and any "eliminates N errors" statements are therefore withheld. The file was removed in `9022e92`; history is preserved.

**Dry-run results are not model quality.** `--dry-run` uses `simulate_offline_prediction()`, which hard-codes the baseline's failure patterns and gives the candidate the expected answer, so a candidate Macro-F1 of 1.0 is true by construction. Dry-run output is labelled `evidence_class: simulated_by_construction_not_model_quality`, carries `model: null`, and prints a SIMULATED banner. It verifies the harness only.

To obtain real numbers, run `sentiment_eval_runner.py --paid` under the budget guard and commit the generated results file together with its usage records.

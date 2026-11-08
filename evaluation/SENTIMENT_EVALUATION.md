# AlphaSignal Sentiment Evaluation & Candidate Prompt v3 Report

## 1. Executive Summary

This report documents the repeatable evaluation benchmark and prompt iteration for AlphaSignal's LLM financial sentiment engine (`gpt-5.6-luna`), comparing **Production Baseline Prompt (v2)** against **Candidate Prompt (v3: Target Ticker & Disclosed Event Aware)**.

The candidate prompt was developed to address critical defects observed in production:
1. **Wrong-Ticker Attribution**: Attributing competitor, supplier, or counterparty fortunes to the target ticker.
2. **Backward-Looking Beat Bias**: Rating guidance cuts as positive because trailing EPS beat consensus.
3. **Historical Risk Misclassification**: Treating resolved, past litigation or fines as active negative catalysts.

All evaluations were executed with deterministic token accounting and budget safeguards ($0.2034 total spend across 104 requests, zero budget overrun). Candidate Prompt v3 remains strictly **evaluation-only** (`alphasignal/generation/candidate_prompt.py`), preserving production configuration (`alphasignal/generation/sentiment.py` at `PROMPT_VERSION = 2`).

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
3. **Execution Record**: The paid evaluation consumed **104 requests** (4 pilot requests, 50 baseline calls, 50 candidate calls) totaling 42,912 prompt tokens and 14,696 completion tokens, resulting in **$0.2034 actual spend** ($1.7966 under the hard budget cap).

---

## 4. Evaluation Metrics & Confusion Matrix Reconciliation

### Metric Definitions
* **Inclusive Directional Binning**:
  $$\text{Direction} = \begin{cases} \text{Positive} & \text{score} \ge +0.10 \\ \text{Negative} & \text{score} \le -0.10 \\ \text{Neutral} & -0.10 < \text{score} < +0.10 \end{cases}$$
* **Polarity Reversal ($\text{Pos} \leftrightarrow \text{Neg}$)**: Any disclosure where expected direction is Positive and predicted direction is Negative (or vice versa), including threshold boundary values ($|\text{score}| = 0.10$).
* **Wrong-Sign Calls**: Originally evaluated with strict inequality ($|\text{score}| > 0.10$). Aligning to match inclusive directional binning ($|\text{score}| \ge 0.10$) reconciles the confusion matrices with summary reporting.
* **Wrong-Ticker Metric (Polarity-Reversal Proxy)**:
  In financial disclosures mentioning competitors, suppliers, or counterparties (`competitor_supplier_mention` and `paired_counterparty_event`), counterparties often experience an inverse financial impact relative to the target issuer (e.g. competitor losing share, customer canceling supplier contract). Because ground truth labels reflect the target ticker's expected direction, a full polarity reversal specifically within these counterparty disclosures serves as a **heuristic proxy for ticker attribution defects**—indicating that the model erroneously attributed the counterparty's fortune to the target ticker.

### Quantitative Comparison

| Split | Prompt | Macro-F1 | Polarity Reversals | Wrong-Sign (Aligned) | Wrong-Ticker (Proxy) | Quote Hallucination |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\
| **Dev (25)** | Baseline (v2) | 0.4833 | 1 | 1 | 1 | 0% (0/25) |
| | **Candidate (v3)** | **0.7585** | **0** | **0** | **0** | **0% (0/25)** |
| **Holdout (25)** | Baseline (v2) | 0.4643 | 4 | 4 | 2 | 0% (0/25) |
| | **Candidate (v3)** | **0.7404** | **1** | **1** | **0** | **0% (0/25)** |
| **Overall (50)** | Baseline (v2) | 0.4761 | 5 | 5 | 3 | 0% (0/50) |
| | **Candidate (v3)** | **0.7510** | **1** | **1** | **0** | **0% (0/50)** |

### Analysis of Candidate's Remaining Negative $\to$ Positive Reversal (`holdout_021`)
* **Case ID**: `holdout_021` (Ticker: `FTV`, Category: `earnings_beat_lowered_guidance`)
* **Disclosure**: Fortive Q3 operating earnings beat consensus by 4%, but management lowered full-year revenue and adjusted EPS guidance due to industrial automation headwinds.
* **Expected Direction**: Negative (Score $\le -0.10$) due to forward guidance downgrade.
* **Candidate v3 Output**: Score $+0.40$, Confidence $0.80$.
* **Root Cause**: While Candidate v3 successfully penalized lowered guidance in 7 out of 8 cases across the benchmark, in `holdout_021` the candidate prompt over-indexed on the 4% beat phrasing. This remaining false positive surfaces as the sole polarity reversal in the Candidate evaluation.

---

## 5. AlphaLive Trading Gate Replay

Predictions from both prompts were replayed through AlphaLive's production signal gate (`sentiment_threshold = -0.30`):

* **Long Entries (`direction = 2`)**: Block if $\text{score} < -0.30$; Allow if $\text{score} \ge -0.30$.
  - **Baseline (v2)**: 5 false passes (missed negatives: `dev_012`, `dev_023`, `holdout_012`, `holdout_013`, `holdout_023`), 3 false blocks (spurious blocks of good trades: `dev_015`, `dev_021`, `holdout_021`).
  - **Candidate (v3)**: 3 false passes (missed negatives: `dev_025`, `holdout_012`, `holdout_013`), **0 false blocks** (100% of spurious blocks eliminated).
* **Short Entries (`direction = 0`)**: Block if $\text{score} > +0.30$; Allow if $\text{score} \le +0.30$.
  - **Baseline (v2)**: 3 false passes, 6 false blocks.
  - **Candidate (v3)**: 0 false passes, 1 false block (`holdout_021`).
* **Protective Exits (`direction = 1`)**: Bypasses sentiment checks unconditionally (`GATE_STATUS_NEUTRAL_BYPASS`) across all 50 items.

---

## 6. Recency Aggregation Mechanics & Invariant Properties

AlphaSignal aggregates multiple documents per ticker using confidence and exponential recency decay ($w_i = \text{conf}_i \cdot 0.5^{\Delta t_i / 7.0}$, freshness window limit 30 days):

### Invariant 1: Time-Shift Invariance (Scalar Cancellation)
Advancing the evaluation timestamp $\text{as\_of}$ by $\Delta t$ days while the document corpus remains static scales every individual document weight by the exact same scalar $0.5^{\Delta t / 7.0}$. The scalar cancels out of both numerator and denominator:
$$\bar{S}(t + \Delta t) = \frac{\sum (w_i \cdot 0.5^{\Delta t / 7.0}) \cdot d_i}{\sum (w_i \cdot 0.5^{\Delta t / 7.0})} = \frac{0.5^{\Delta t / 7.0} \sum w_i d_i}{0.5^{\Delta t / 7.0} \sum w_i} = \bar{S}(t)$$
In `SYNTH_B` (Day 0 neutral $0.00$, Day -5 negative $-0.88$), advancing $\text{as\_of}$ by 7 days yields an identical score of **$-0.3354$**, keeping Long **BLOCKED**. Time passing without new filing disclosures does not attenuate the relative score.

### Invariant 2: Age-Out Discontinuity
When the negative document reaches 31 days ($> 30\text{d}$ freshness window), it drops out of the active aggregation. The score abruptly jumps from $-0.3354$ to $0.0000$, opening Long entry. If all documents exceed 30 days, the ticker transitions to `status="stale"`, which triggers `GATE_STATUS_STALE_BYPASS` in AlphaLive.

### Invariant 3: Dilution Vulnerability by Routine Filler Filings
In `SYNTH_B`, adding a second distinct, eligible neutral filing at Day 0 (e.g. routine proxy statement or administrative 8-K) expands the neutral weight from $0.98 \to 1.96$. This dilutes the active 5-day-old crisis weight ($0.6034$), shifting the aggregate score from $-0.3354$ to **$-0.2071$** ($\ge -0.30$).
**Policy Decision Flag**: Routine administrative filings can wash out an active adverse shock before its half-life expires. A maximum-adverse or category-weighted aggregation policy is recommended for production gate hardening.

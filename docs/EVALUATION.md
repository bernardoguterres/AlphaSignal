# AlphaSignal Retrieval Evaluation

## Overview

This document presents the comprehensive evaluation of AlphaSignal's retrieval system. We evaluate four different retrieval configurations on a manually annotated golden set of 50 financial Q&A pairs across 10 major tickers (AAPL, MSFT, NVDA, JPM, GOOGL, AMZN, META, TSLA, GS, MS).

The evaluation measures how well the system retrieves relevant information from SEC filings and financial news to answer domain-specific questions. This is critical for portfolio demonstration as it shows the system can accurately surface pertinent financial information.

## Corpus Statistics

The corpus was built by ingesting SEC 10-K/10-Q filings and recent financial news for all configured tickers:

```
Ticker   | Filings  | Articles | Chunks   | Date Range
---------|----------|----------|----------|------------------
AAPL     |    TBD   |    TBD   |    TBD   | TBD
MSFT     |    TBD   |    TBD   |    TBD   | TBD
NVDA     |    TBD   |    TBD   |    TBD   | TBD
JPM      |    TBD   |    TBD   |    TBD   | TBD
GOOGL    |    TBD   |    TBD   |    TBD   | TBD
AMZN     |    TBD   |    TBD   |    TBD   | TBD
META     |    TBD   |    TBD   |    TBD   | TBD
TSLA     |    TBD   |    TBD   |    TBD   | TBD
GS       |    TBD   |    TBD   |    TBD   | TBD
MS       |    TBD   |    TBD   |    TBD   | TBD
---------|----------|----------|----------|------------------
TOTAL    |    TBD   |    TBD   |    TBD   |
```

**Note:** Run `python alphasignal/scripts/build_corpus.py` to populate this table with actual statistics.

## Evaluation Methodology

- **Golden Set:** 50 manually annotated Q&A pairs
- **Tickers:** 10 major companies (AAPL, MSFT, NVDA, JPM, GOOGL, AMZN, META, TSLA, GS, MS)
- **Questions per Ticker:** 5
- **Question Types:**
  - Factual (10): Direct fact retrieval (e.g., "What was AAPL's revenue in Q4 2024?")
  - Trend (10): Time-series analysis (e.g., "How has margin changed over two quarters?")
  - Comparative (10): Risk factor identification (e.g., "What AI competition risks does MSFT identify?")
  - Sentiment (10): Guidance and outlook (e.g., "Was NVDA's AI demand outlook positive?")
  - Specific (10): Management commentary (e.g., "What did CEO say about digital strategy?")

- **Metrics:**
  - **MRR@10:** Mean Reciprocal Rank at 10 - measures rank of first relevant result
  - **NDCG@5:** Normalized Discounted Cumulative Gain at 5 - accounts for multiple relevant docs
  - **Hit@3:** Proportion of queries with at least one relevant result in top 3

- **Evaluation:** Each configuration evaluated on all 50 questions with ticker filtering enabled

## Results

Four retrieval configurations were benchmarked to demonstrate the value of hybrid retrieval and reranking:

```
Config                                       | MRR@10  | NDCG@5  | Hit@3  | Avg Latency
---------------------------------------------|---------|---------|--------|-------------
Baseline: naive chunks + dense only          |  TBD    |  TBD    |  TBD   |   TBDms
Semantic chunks + dense only                 |  TBD    |  TBD    |  TBD   |   TBDms
Semantic chunks + hybrid                     |  TBD    |  TBD    |  TBD   |   TBDms
Semantic chunks + hybrid + reranker          |  TBD    |  TBD    |  TBD   |   TBDms
```

**Note:** Run `python alphasignal/scripts/benchmark.py` to populate this table with actual results.

## Analysis

**To be filled after running benchmarks:**

1. **Key Findings:** Which configuration performs best and why?
2. **Impact of Semantic Chunking:** How much does sentence-aware chunking improve over naive fixed-size chunks?
3. **Value of Hybrid Retrieval:** Does combining BM25 + dense retrieval outperform dense-only?
4. **Reranking Benefit:** Does cross-encoder reranking provide significant lift?
5. **Remaining Challenges:** Where does the system still struggle? What question types are hardest?

**Expected insights to explore:**
- Semantic chunking should improve retrieval by preserving context boundaries
- Hybrid retrieval should handle both keyword-based and semantic queries better
- Reranking should improve precision by better scoring the final candidates
- Trend and comparative questions may be harder than factual questions
- Performance may vary by ticker based on data volume and quality

## Query Latency (Full Pipeline)

End-to-end latency for complete RAG pipeline: embed query + retrieve + rerank + generate answer.

Measured on [hardware specs: TBD]:

| Percentile | Latency |
|------------|---------|
| p50        | TBDms   |
| p95        | TBDms   |
| p99        | TBDms   |

**Note:** Latency measured after running queries through the API. Run `/metrics` endpoint after benchmark to capture this data.

## Failure Cases

Examples of questions the system answers poorly, with analysis of why:

### 1. [Question TBD]
- **Question:** TBD
- **Expected Answer:** TBD
- **Retrieved Results:** TBD
- **Diagnosis:** TBD (e.g., "Relevant information split across multiple chunks", "Question requires numerical reasoning", "Ambiguous query matching wrong context")

### 2. [Question TBD]
- **Question:** TBD
- **Expected Answer:** TBD
- **Retrieved Results:** TBD
- **Diagnosis:** TBD

### 3. [Question TBD]
- **Question:** TBD
- **Expected Answer:** TBD
- **Retrieved Results:** TBD
- **Diagnosis:** TBD

### 4. [Question TBD]
- **Question:** TBD
- **Expected Answer:** TBD
- **Retrieved Results:** TBD
- **Diagnosis:** TBD

### 5. [Question TBD]
- **Question:** TBD
- **Expected Answer:** TBD
- **Retrieved Results:** TBD
- **Diagnosis:** TBD

**Note:** After running benchmarks, analyze the lowest-scoring questions and document failure patterns here.

## Future Improvements

Based on evaluation results, potential areas for improvement:

1. **Better Chunking:** Experiment with variable chunk sizes based on document structure
2. **Query Expansion:** Use LLM to generate query variations for hard questions
3. **Multi-stage Retrieval:** Add a coarse-to-fine retrieval stage for large corpora
4. **Temporal Awareness:** Improve handling of time-based queries ("last quarter", "recent")
5. **Numerical Reasoning:** Add structured data extraction for financial metrics
6. **More Training Data:** Expand golden set for better coverage of edge cases

## Reproducibility

To reproduce these results:

1. **Setup Environment:**
   ```bash
   export OPENAI_API_KEY=your_key_here
   pip install -r requirements.txt
   ```

2. **Build Corpus:**
   ```bash
   python alphasignal/scripts/build_corpus.py
   ```

3. **Annotate Golden Set:** (Optional if already annotated)
   ```bash
   python alphasignal/scripts/annotate_golden_set.py
   ```

4. **Run Benchmark:**
   ```bash
   python alphasignal/scripts/benchmark.py
   ```

5. **View Results:**
   - Benchmark table printed to console
   - Full results in `data/benchmark_results.json`
   - Corpus stats in `data/corpus_stats.json`

---

*Evaluation completed on: TBD*
*System version: AlphaSignal v0.1.0*
*Model: text-embedding-ada-002 (embeddings), gpt-4o-mini (generation)*

# AlphaSignal — The Math

How a query goes from raw text to a ranked, cited answer or a sentiment score — the actual retrieval, ranking, and scoring math, verified against the real source (`alphasignal/ingestion/chunker.py`, `alphasignal/retrieval/retriever.py`, `alphasignal/store/vector_store.py`, `alphasignal/retrieval/reranker.py`, `alphasignal/generation/sentiment.py`), not textbook idealizations of it.

---

## The Pipeline, End to End

```
Raw filing/article → Semantic Chunking → Embedding → Stored (FAISS + SQLite + BM25)
                                                              ↓
Query → Embedding ─────────────┐                    Hybrid Retrieval
                                 ├──────────────────→ (dense + sparse, fused)
Query → BM25 tokenize ─────────┘                            ↓
                                                    Cross-Encoder Reranking
                                                              ↓
                                          Top-K chunks → LLM generation (cited answer)
                                                      → LLM sentiment extraction (score)
```

Two genuinely different retrieval mechanisms run in parallel and get fused — that's the "hybrid" in hybrid retrieval, and it's the most important design decision in this system.

---

## 1. Chunking

Documents are split sentence-by-sentence, greedily packed into chunks targeting **300 tokens** (config: `min_tokens=100`, `max_tokens=400`, `overlap_tokens=50`), not split at arbitrary character boundaries. Sentences are added to the current chunk until adding the next one would exceed `max_tokens`; a single sentence that alone exceeds `max_tokens` gets hard-split by raw token count as a fallback (rare, but prevents an unbounded chunk from an unusually long run-on sentence in a filing).

**Why overlap matters**: consecutive chunks share the last `overlap_tokens` (50) worth of sentences from the prior chunk. Without overlap, a fact split exactly across a chunk boundary (e.g., "Revenue grew 16%" in one chunk, "due to iPhone sales" in the next) could be unretrievable as a coherent unit — overlap means the boundary itself is redundantly covered by both chunks.

---

## 2. Embeddings & Cosine Similarity

Each chunk becomes a 1536-dimension vector via OpenAI's `text-embedding-3-small`. Retrieval similarity is **cosine similarity**, computed as a plain inner product:
$$\hat{v} = \frac{v}{\|v\|_2}, \qquad \text{similarity}(q, c) = \hat{q} \cdot \hat{c}$$
Every vector is L2-normalized to unit length *before* being added to the index (`vector_store.py`: `normalized = new_embeddings / (norms + 1e-8)`). Once every vector has length 1, the inner product of two vectors **is** their cosine similarity — no separate cosine computation needed at query time. That's why the FAISS index type is `IndexFlatIP` (inner product) rather than a distance metric: it's a deliberate trick to make an expensive-feeling operation (cosine similarity across tens of thousands of vectors) a single fast matrix multiply.

---

## 3. Hybrid Retrieval

### Sparse side: BM25
Classic lexical/keyword search (Okapi BM25, via `rank_bm25`), scoring each candidate document $d$ against query $q$:
$$\text{BM25}(q, d) = \sum_{t \in q} IDF(t) \cdot \frac{f(t, d)\,(k_1+1)}{f(t,d) + k_1\Big(1 - b + b\,\frac{|d|}{\text{avgdl}}\Big)}$$
Where $f(t,d)$ is how often term $t$ appears in document $d$, $|d|$ is the document's length, `avgdl` the corpus's average document length, and $IDF(t)$ down-weights terms that appear in most documents (so "the" contributes far less than "iPhone"). This is what catches an exact phrase or ticker match that a dense embedding might blur past — dense embeddings are excellent at *meaning* ("revenue" ≈ "sales") but comparatively weak at *exact* string matches ("Q4 2023" specifically, not just "a quarter in 2023").

### Dense side: FAISS cosine search
The query is embedded with the same model, then compared against every chunk's vector via the cosine similarity above.

### Fusion
Both retrieval paths return their own top candidates with their own raw scores — not directly comparable (BM25 scores are unbounded and corpus-dependent; cosine similarity is bounded to [-1, 1]). Each score set is independently **min-max normalized to [0, 1]** first:
$$\text{norm}(s_i) = \frac{s_i - \min(S)}{\max(S) - \min(S)}$$
then combined as a weighted sum:
$$\text{Hybrid Score} = w_{\text{dense}} \cdot \text{dense}_{\text{norm}} + w_{\text{bm25}} \cdot \text{bm25}_{\text{norm}}$$
Default weights: 60% dense, 40% BM25 (`config.yaml`'s `hybrid_weights`). A chunk that only one retriever found still gets a score (the missing side defaults to 0, not excluded) — so a chunk that's a strong *exact* keyword match but a mediocre semantic match can still surface, and vice versa.

---

## 4. Cross-Encoder Reranking

The hybrid fusion above is deliberately cheap and approximate — comparing two independently-computed vectors is fast but structurally limited (it can never let query and document tokens directly attend to each other). The reranker exists specifically to fix that on a *small* candidate set the cheap stage already narrowed down: it feeds the query and each candidate chunk **concatenated together as one input** into a cross-encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`), which outputs a single relevance score per pair having seen the full interaction between them. More accurate, much more expensive per comparison — which is exactly why it only runs on the top ~20 candidates the hybrid stage already produced, never the whole corpus.

---

## 5. Sentiment Extraction

Not a classifier with learned weights — an LLM call per chunk, prompted to return structured JSON:
$$\text{score} \in [-1.0,\ 1.0], \quad \text{confidence} \in [0.0,\ 1.0]$$
`/sentiment/{ticker}` runs this over the ticker's 10 most-recent chunks and returns `latest_score` from the newest one. Two things worth knowing precisely, because they're easy to get wrong when explaining this system:

- **It is not retrieval-aware.** The endpoint takes `date_from`/`date_to` as real filters, but no free-text query parameter actually narrows *which* chunks get scored beyond that date window — it's always "the ticker's most recent (or date-filtered) chunks," not "chunks relevant to a specific question." (This one cost real debugging time this session — an evaluation script assumed a `?query=` parameter did semantic filtering; it doesn't, the route only reads `date_from`/`date_to`.)
- **NaN/Infinity is explicitly guarded against, not just clamped.** A malformed score from the LLM provider is detected via `math.isfinite()` *before* clamping to `[-1, 1]` — clamping first would be a real bug: Python's `max(-1.0, min(1.0, float('nan')))` silently evaluates to `1.0` (NaN fails every comparison), which would turn a broken response into a false maximally-bullish signal instead of failing safely to neutral.

---

## 6. Retrieval Evaluation (built, currently not meaningful — worth being honest about)

`RetrievalEvaluator` computes standard information-retrieval metrics against a golden Q&A set:
$$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i} \qquad \text{Hit@K} = \frac{1}{|Q|}\sum_{i=1}^{|Q|} \mathbb{1}\big[\text{a relevant chunk appears in the top K}\big]$$
MRR rewards getting the *first* relevant result ranked high; Hit@K is a simpler "was a relevant result anywhere in the top K" check. Both are the real, standard IR metrics — but as of this session, the 50-question golden set has **0 of 50 questions actually annotated** with `relevant_chunk_ids` (empty arrays across the board), so running the benchmark right now would produce numbers that look precise but measure nothing real. The framework is correct; the ground-truth data to run it against doesn't exist yet.

---

## The one-paragraph version, for an interview

*"AlphaSignal is a hybrid-retrieval RAG system, not a single embedding lookup. Sparse (BM25, exact lexical match) and dense (cosine similarity over normalized OpenAI embeddings) retrieval run independently, get min-max normalized onto a comparable scale, and combine as a weighted sum — because dense embeddings are excellent at meaning but weak at exact matches, and BM25 is the reverse. A small candidate set then goes through cross-encoder reranking, which is more accurate but too expensive to run over the whole corpus, so it only ever sees the narrowed-down top ~20. Sentiment extraction is LLM-based with explicit NaN/Infinity guarding so a malformed provider response fails safe to neutral rather than silently becoming a false bullish signal. And I know exactly where the evaluation story is honest versus not: the retrieval-quality benchmark is correctly built but currently unannotated, so I don't cite numbers from it that don't mean anything yet."*

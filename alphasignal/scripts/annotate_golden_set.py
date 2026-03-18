"""Helper script to annotate golden set with relevant chunk IDs."""

import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from alphasignal.embeddings.cache import EmbeddingCache
from alphasignal.embeddings.embedder import Embedder
from alphasignal.retrieval.retriever import HybridRetriever
from alphasignal.store.metadata_store import MetadataStore
from alphasignal.store.vector_store import VectorStore

# Configure logging
logging.basicConfig(
    level=logging.WARNING  # Reduce noise during annotation
)
logger = logging.getLogger(__name__)


def annotate():
    """Annotate golden set with relevant chunk IDs."""
    print("=" * 80)
    print("AlphaSignal Golden Set Annotator")
    print("=" * 80)
    print()
    print("This tool helps you annotate the golden set with relevant chunk IDs.")
    print("For each question, you'll see the top retrieved chunks and can")
    print("mark which ones are actually relevant.")
    print()

    # Load configuration
    config_path = project_root / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Load golden set
    golden_set_path = project_root / "evaluation" / "golden_set.json"
    with open(golden_set_path) as f:
        golden_set = json.load(f)

    print(f"Loaded {len(golden_set)} questions from golden set")
    print()

    # Initialize retrieval components
    print("Initializing retrieval system...")

    storage_config = config.get("storage", {})
    vector_store = VectorStore(storage_config.get("faiss_index_path", "data/faiss_index"), dim=1536)
    vector_store.load()

    metadata_store = MetadataStore(storage_config.get("sqlite_db_path", "data/metadata.db"))

    embedding_cache = EmbeddingCache(storage_config.get("embeddings_cache_path", "data/embeddings_cache"))
    embedder = Embedder(config, embedding_cache)

    retriever = HybridRetriever(config, embedder, vector_store, metadata_store)
    retriever.build_bm25_index()

    print(f"Retrieval system ready with {metadata_store.count()} chunks indexed")
    print()

    # Count unannotated questions
    unannotated = [q for q in golden_set if not q.get("relevant_chunk_ids")]
    print(f"Found {len(unannotated)} unannotated questions")
    print()

    if not unannotated:
        print("All questions are already annotated!")
        return

    # Annotate each question
    for i, question_data in enumerate(unannotated, 1):
        print("=" * 80)
        print(f"Question {i}/{len(unannotated)}")
        print("=" * 80)
        print()
        print(f"ID: {question_data['id']}")
        print(f"Ticker: {question_data['ticker']}")
        print(f"Type: {question_data['question_type']}")
        print()
        print(f"Question: {question_data['question']}")
        print()
        print(f"Expected Answer: {question_data['answer_summary']}")
        print()
        print("-" * 80)
        print("Retrieving candidate chunks...")
        print("-" * 80)
        print()

        # Retrieve candidate chunks
        try:
            results = retriever.retrieve(
                query=question_data["question"],
                ticker=question_data["ticker"],
                top_k=20
            )

            if not results:
                print("No chunks retrieved!")
                print()
                input("Press Enter to continue...")
                continue

            # Display top 5 chunks
            print("Top 5 retrieved chunks:")
            print()

            for rank, chunk in enumerate(results[:5], 1):
                print(f"[{rank}] Chunk ID: {chunk.chunk_id}")
                print(f"    Score: {chunk.hybrid_score:.3f}")
                print(f"    Date: {chunk.date}")
                print(f"    Type: {chunk.doc_type}")
                print(f"    Section: {chunk.section or 'N/A'}")
                print(f"    Text: {chunk.text[:200]}...")
                print()

            # Prompt for relevant chunk IDs
            print("-" * 80)
            print("Enter relevant chunk IDs (comma-separated numbers from 1-5),")
            print("or press Enter to skip, or 'q' to quit:")
            user_input = input("> ").strip()

            if user_input.lower() == 'q':
                print("\nQuitting annotation session...")
                break

            if not user_input:
                print("Skipped")
                print()
                continue

            # Parse input
            try:
                selected_ranks = [int(x.strip()) for x in user_input.split(",")]
                selected_chunk_ids = [
                    results[rank - 1].chunk_id
                    for rank in selected_ranks
                    if 1 <= rank <= len(results)
                ]

                # Update question data
                question_data["relevant_chunk_ids"] = selected_chunk_ids
                print(f"\nMarked {len(selected_chunk_ids)} chunks as relevant")
                print()

            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}")
                print("Skipped")
                print()

        except Exception as e:
            logger.error(f"Error retrieving for question {question_data['id']}: {e}")
            print(f"Error: {e}")
            print()
            input("Press Enter to continue...")

    # Save updated golden set
    print("=" * 80)
    print("Saving annotated golden set...")
    print("=" * 80)

    with open(golden_set_path, "w") as f:
        json.dump(golden_set, f, indent=2)

    annotated_count = len([q for q in golden_set if q.get("relevant_chunk_ids")])
    print(f"\nSaved! {annotated_count}/{len(golden_set)} questions now annotated")
    print(f"Golden set saved to: {golden_set_path}")
    print()


if __name__ == "__main__":
    annotate()

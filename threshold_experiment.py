"""
L7 — Similarity Threshold Experiment  (STARTER)
================================================
Run with:
    python starter.py

Your goal: show how different similarity thresholds change which results are
returned for the same query.

Required features:
    1. A `knowledge_base` list of at least 15 sentences across 3–4 distinct
       topics (e.g. Python, cooking, space, music)
    2. A `test_queries` list of at least 5 queries — some clearly on one topic,
       some ambiguous
    3. Embed all knowledge-base sentences once
    4. For each query, compute similarity scores against all documents
    5. Display results at three thresholds: 0.3, 0.5, and 0.7
       For each threshold show: count of results, which sentences pass,
       and a note about any relevant sentences that are missed

Key concepts:
    Filtering by threshold:
        passing = [(score, sentence)
                   for score, sentence in zip(scores, knowledge_base)
                   if score >= threshold]
        passing.sort(reverse=True)

    Expected output format:
        Query: "How do I deploy my Python app?"

          Threshold 0.3: 6 results
            [0.72] Deploy FastAPI with uvicorn and Docker
            ...
          Threshold 0.5: 3 results
            ...
          Threshold 0.7: 1 result
            ...
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ── Knowledge base ───────────────────────────────────────────────────────────
# TODO: Create `knowledge_base` — at least 15 sentences across 3–4 topics.
#       Aim for roughly equal coverage of each topic so threshold differences
#       are visible.

knowledge_base = [
    # Topic 1: Python development
    "Deploy FastAPI applications using Uvicorn and Docker.",
    "Python virtual environments isolate project dependencies.",
    "SQLAlchemy connects Python applications to relational databases.",
    "JWT authentication protects API endpoints from unauthorized access.",
    "Pytest automates testing for Python applications.",

    # Topic 2: Cooking
    "Italian pasta dishes often use tomatoes and olive oil.",
    "Pizza dough is made using flour, water, yeast, and salt.",
    "Baking bread requires controlling temperature and fermentation.",
    "Grilling vegetables adds a smoky flavor to meals.",
    "Cooking rice requires the correct ratio of water to grains.",

    # Topic 3: Space exploration
    "NASA explores Mars using robotic rovers.",
    "Astronauts conduct scientific experiments aboard the space station.",
    "SpaceX develops reusable rockets for space transportation.",
    "Satellites orbit Earth to provide communication and navigation.",
    "Scientists use telescopes to study distant galaxies.",

    # Topic 4: Music
    "Music producers use digital audio workstations to create songs.",
    "MIDI allows electronic instruments to communicate with computers.",
    "Audio mixing balances volume, frequency, and effects.",
    "Synthesizers generate electronic sounds using oscillators.",
    "Music mastering prepares finished recordings for distribution."
]


# ── Test queries ─────────────────────────────────────────────────────────────
# TODO: Create `test_queries` — at least 5 queries.
#       Include some that clearly match one topic and some that are ambiguous.

test_queries = [
    "How do I deploy my Python web application?",
    "What ingredients do I need to make homemade pizza?",
    "How does NASA explore other planets?",
    "How can I produce electronic music on my computer?",
    "How do computers help people create things?"
]

# ── Thresholds ───────────────────────────────────────────────────────────────
THRESHOLDS = [0.3, 0.5, 0.7]

# ── Model & embeddings ───────────────────────────────────────────────────────
print("Loading model and embedding knowledge base...")
# TODO: Load SentenceTransformer('all-MiniLM-L6-v2')
# TODO: Encode knowledge_base into `doc_embeddings`

model = SentenceTransformer('all-MiniLM-L6-v2')

doc_embeddings = model.encode(knowledge_base)


# ── Run experiment ───────────────────────────────────────────────────────────
# TODO: Encode the query
# TODO: Compute cosine similarity scores (1-D array

for query in test_queries:
    query_emb = model.encode([query])
    scores = cosine_similarity(query_emb, doc_embeddings)[0]

    print(f"\nQuery: '{query}'")

    for threshold in THRESHOLDS:
        # TODO: Filter (score, sentence) pairs where score >= threshold
        # TODO: Sort by score descending
        # TODO: Print the threshold, count, and each result
        #       e.g. "  Threshold 0.3: 4 results"
        #            "    [0.72] Deploy FastAPI with uvicorn and Docker"
        passing = [
            (score, sentence)
            for score, sentence in zip(scores, knowledge_base)
            if score >= threshold
        ]
        passing.sort(reverse=True)

        print(f"\nThreshold {threshold}: {len(passing)} results")

        for score, sentence in passing:
            print(f"  [{score:.4f}] {sentence}")
            
    print("\nResults excluded by stricter thresholds:")

for score, sentence in zip(scores, knowledge_base):
    if 0.3 <= score < 0.7:
        print(f"  [{score:.4f}] {sentence}")
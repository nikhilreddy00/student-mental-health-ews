import re
import numpy as np
import pandas as pd
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(Path(__file__).parent.parent / ".env")

MODEL = "claude-sonnet-4-6"
BASE = Path(__file__).parent.parent / "data"
KB_DIR = BASE / "kb"


def load_kb_documents() -> list:
    """Read all .md files from data/kb/ and split into chunks."""
    docs = []
    for fpath in sorted(KB_DIR.glob("*.md")):
        content = fpath.read_text(encoding="utf-8")
        # Extract title from first H1 heading
        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else fpath.stem

        # Split on double newline into chunks; filter very short chunks
        raw_chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
        chunks = [c for c in raw_chunks if len(c.split()) >= 20]

        docs.append({
            "title": title,
            "filename": fpath.name,
            "content": content,
            "chunks": chunks,
        })
    return docs


def build_tfidf_index(documents: list) -> tuple:
    """Build TF-IDF index over all document chunks."""
    chunk_metadata = []
    corpus = []

    for doc in documents:
        for i, chunk in enumerate(doc["chunks"]):
            corpus.append(chunk)
            chunk_metadata.append({
                "doc_title": doc["title"],
                "doc_filename": doc["filename"],
                "chunk_text": chunk,
                "chunk_index": i,
            })

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.85,
        min_df=1,
        stop_words="english",
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)

    return vectorizer, tfidf_matrix, chunk_metadata


def retrieve_relevant_chunks(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    chunk_metadata: list,
    top_k: int = 4,
) -> list:
    """Return top_k most relevant chunks (max 2 per document)."""
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Sort all chunks by score descending
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    seen_docs = {}
    selected = []
    for idx, score in ranked:
        if score < 0.01:
            break
        meta = chunk_metadata[idx]
        doc_title = meta["doc_title"]
        seen_docs[doc_title] = seen_docs.get(doc_title, 0) + 1
        if seen_docs[doc_title] > 2:
            continue
        selected.append({**meta, "score": float(score)})
        if len(selected) >= top_k:
            break

    return selected


def generate_rag_answer(
    query: str,
    retrieved_chunks: list,
    chat_history: list,
) -> tuple:
    """Generate a grounded answer from retrieved chunks. Returns (answer, source_titles)."""
    client = Anthropic()

    # Build context block from retrieved chunks
    context_lines = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_lines.append(
            f"--- Excerpt {i} from [{chunk['doc_title']}] ---\n{chunk['chunk_text']}"
        )
    context_block = "\n\n".join(context_lines)

    system_prompt = f"""You are a university counseling knowledge assistant. You answer counselor questions using evidence-based guidance from the institution's counseling knowledge base.

RULES:
1. Ground every answer in the Knowledge Base Excerpts provided below.
2. Cite sources using [Source: Document Title] notation after each key point.
3. If the excerpts do not cover the question, say: "The knowledge base doesn't directly address this. Based on general practice, ..." and then answer carefully.
4. Be specific and actionable — counselors need to know WHAT to do, not just theory.
5. Never make clinical diagnoses. Recommend specialist referral when appropriate.
6. Keep answers concise (150-300 words). Use bullet points for steps or lists.

KNOWLEDGE BASE EXCERPTS:
{context_block}"""

    # Build message history (last 6 turns)
    api_messages = list(chat_history[-6:]) + [{"role": "user", "content": query}]

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=system_prompt,
        messages=api_messages,
    )

    answer_text = response.content[0].text

    # Extract cited sources from [Source: ...] patterns
    cited = re.findall(r"\[Source:\s*([^\]]+)\]", answer_text)
    if not cited:
        cited = list({c["doc_title"] for c in retrieved_chunks})

    return answer_text, cited

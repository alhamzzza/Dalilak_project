import faiss
import pickle
import numpy as np
from utils import get_embedding, normalize_score


# search tools with FAISS for local testing without Pinecone


def search_quran(query: str, top_n: int = 5, threshold: float = None,
                 index_path="data/quran.faiss", meta_path="data/quran.pkl"):
    """
    Search all Quran verses semantically for a given query.
    Returns list of results with verse info and similarity score.
    """
    index = faiss.read_index(index_path)
    metadata = pickle.load(open(meta_path, "rb"))

    q_emb = get_embedding(text=query)
    q_emb = np.array([q_emb]).astype("float32")

    D, I = index.search(q_emb, top_n)

    results = []
    for i, dist in zip(I[0], D[0]):
        verse = metadata[i]
        results.append({
            "surah": verse["surah_name"],
            "verse_id": verse["verse_id"],
            "text": verse["text"],
            "translation": verse["translation"],
            "score": dist
        })

    if threshold is not None:
        results = [r for r in results if float(r["score"]) <= threshold]

    return results

def search_quran_tafsir(query, top_n, threshold, index_path, meta_path):
    # Load FAISS index + metadata
    index = faiss.read_index(index_path)
    metadata = pickle.load(open(meta_path, "rb"))

    # Get query embedding
    q_emb = np.array([get_embedding(query)], dtype="float32")

    # Normalize query and index for cosine similarity
    faiss.normalize_L2(q_emb)
    # You only need to normalize the index once when you *build* it,
    # but if unsure, this line ensures all vectors are normalized at search time:
    # (safe to leave)
    if not hasattr(index, "is_normalized") or not index.is_normalized:
        xb = np.vstack([index.reconstruct(i) for i in range(index.ntotal)])
        faiss.normalize_L2(xb)
        new_index = faiss.IndexFlatIP(xb.shape[1])  # IP = inner product (cosine)
        new_index.add(xb)
        index = new_index
        index.is_normalized = True

    # Perform cosine similarity search
    D, I = index.search(q_emb, top_n)

    results = []
    for idx, score in zip(I[0], D[0]):
        item = metadata[idx]
        results.append({
            "id": item["id"],
            "surah": item["surah_transliteration"],
            "range": item["verse_range"],
            "verses": item["verses"],
            "tafsir": item["tafsir"],
            "score": round(float(score * 100), 2)  # convert cosine similarity to %
        })

    if threshold:
        results = [r for r in results if r["score"] >= threshold]

    return results

    index = faiss.read_index(index_path)
    metadata = pickle.load(open(meta_path, "rb"))

    q_emb = np.array([get_embedding(query)], dtype="float32")
    D, I = index.search(q_emb, top_n)

    results = []
    for idx, dist in zip(I[0], D[0]):
        item = metadata[idx]
        score = 1 / (1 + dist) * 100  # nicer display

        results.append({
            "id": f"{item.get('volume','')}_{item.get('book_number','')}_{item.get('hadith_number','')}",
            "section": item.get("section"),
            "book": item.get("book"),
            "narrator": item.get("narrator"),
            "text": item.get("text"),
            "info": item.get("info"),
            "score": score
        })
    if threshold:
        results = [r for r in results if r["score"] >= threshold]
    return results

def search_hadith_index(query: str, top_k: int, index_path: str, meta_path: str):
    """
    Search Hadith FAISS index and return top results.
    Each result contains both Arabic and English versions,
    narrator, book, and chapter info.
    """
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    query_emb = np.array(get_embedding(query), dtype="float32").reshape(1, -1)
    D, I = index.search(query_emb, top_k)
    scores = D[0]
    ids = I[0]

    results = []
    for idx, score in zip(ids, scores):
        if idx == -1 or idx >= len(metadata):
            continue
        h = metadata[idx]

        results.append({
            "id": h.get("id"),
            "type": "hadith",
            "score": float(score),
            "book_id": h.get("book_id"),
            "chapter_id": h.get("chapter_id"),
            "book_en": h.get("book_en"),
            "book_ar": h.get("book_ar"),
            "chapter_en": h.get("chapter_en"),
            "chapter_ar": h.get("chapter_ar"),
            "english_text": h.get("english_text"),
            "arabic_text": h.get("arabic"),
            "narrator": h.get("english_narrator")
        })

    return results


    

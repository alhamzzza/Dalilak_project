import os
import json
import time
import pickle
from tqdm import tqdm
from pinecone import Pinecone, ServerlessSpec
from utils import get_embedding

def clean_metadata(meta: dict) -> dict:
    """Ensure all metadata values are Pinecone-compatible."""
    cleaned = {}
    for k, v in meta.items():
        if v is None or v == "":
            cleaned[k] = "unknown"
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


def build_quran_tafsir_index_pinecone(
    merged_path="data/quran_and_tafsir.json",
    index_name="qurandtefsir",
    namespace="default",
    meta_path="first10.pkl",
    save_interval=5,
    dimension=3072,
    pinecone_api_key=None,
):
    """
    Build or resume a Pinecone index for Qur'an & Tafsir embeddings.
    Processes only the first 10 entries for now.
    """

    # === Initialize Pinecone ===
    if pinecone_api_key is None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing Pinecone API key. Provide via argument or PINECONE_API_KEY env variable.")

    pc = Pinecone(api_key=pinecone_api_key)

    # Create index if not existing
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"🆕 Created new Pinecone index '{index_name}'")

    index = pc.Index(index_name)

    # === Load JSON ===
    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)

    merged = merged

    texts, metadata = [], []
    for item in merged:
        verses_text = " ".join(v.get("text", "") for v in item.get("verses", []))
        verses_trans = " ".join(v.get("translation", "") for v in item.get("verses", []))
        tafsir = item.get("tafsir", "")
        combined = f"{verses_text}\n{verses_trans}\n{tafsir}"
        texts.append(combined)

        flat_meta = {
            "id": item.get("id"),
            "surah_id": item.get("surah_id"),
            "surah_name": item.get("surah_name", "unknown"),
            "surah_transliteration": item.get("surah_transliteration", ""),
            "surah_translation": item.get("surah_translation", ""),
            "verse_range": item.get("verse_range", ""),
            "num_verses": len(item.get("verses", [])),
            "tafsir_source": item.get("tafsir_source", "unknown")
        }
        metadata.append(clean_metadata(flat_meta))

    total = len(texts)
    print(f"📖 Total items to embed: {total}")

    # === Check for checkpoint ===
    done = 0
    saved_meta = []
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            saved_meta = pickle.load(f)
        done = len(saved_meta)
        print(f"🔁 Resuming from checkpoint: {done}/{total}")

    # === Process and upload ===
    batch = []
    for i in tqdm(range(done, total), desc="Embedding and uploading"):
        text = texts[i]
        emb = get_embedding(text=text)
        vec_id = f"item-{i}"

        batch.append({
            "id": vec_id,
            "values": emb,
            "metadata": metadata[i]
        })

        # Upload and checkpoint every N items
        if (i + 1) % save_interval == 0 or i == total - 1:
            print(f"⏫ Upserting batch {i + 1 - len(batch) + 1}–{i + 1} to Pinecone...")
            index.upsert(vectors=batch, namespace=namespace)

            saved_meta.extend([v["metadata"] for v in batch])
            with open(meta_path, "wb") as f:
                pickle.dump(saved_meta, f)

            print(f"✅ Checkpoint saved ({i + 1}/{total})")
            batch = []
            time.sleep(1)

    print(f"\n🎉 Done! Indexed {total} items into Pinecone index '{index_name}'.")
    print(f"Metadata checkpoint saved to {meta_path}")


def build_hadith_index_pinecone(
    json_path,
    index_name="hadith-bukhari",
    namespace="default",
    meta_path="data/hadith_bukhari.pkl",
    save_interval=100,
    dimension=3072,
    pinecone_api_key=None,
):
    """Build or resume a Pinecone index for hadith dataset (Arabic + English only)."""

    # === Init Pinecone ===
    if pinecone_api_key is None:
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise ValueError("Missing Pinecone API key.")

    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        print(f"🆕 Created Pinecone index '{index_name}'")

    index = pc.Index(index_name)

    # === Load JSON data ===
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hadiths = data.get("hadiths", [])
    chapters = {c["id"]: c for c in data.get("chapters", [])}
    source_name = data.get("metadata", {}).get("english", {}).get("title", "Unknown").replace(" ", "")

    print(f"📘 Found {len(hadiths)} hadiths in dataset ({source_name})")

    # === Resume checkpoint if exists ===
    done = 0
    saved_meta = []
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            saved_meta = pickle.load(f)
        done = len(saved_meta)
        print(f"🔁 Resuming from checkpoint: {done}/{len(hadiths)}")

    # === Build embeddings + metadata ===
    batch = []
    for i in tqdm(range(done, len(hadiths)), desc=f"Embedding {source_name} hadiths"):
        h = hadiths[i]

        arabic = h.get("arabic", "").strip()
        english_text = h.get("english", {}).get("text", "").strip()

        # Only Arabic + English for embedding
        combined = f"Arabic: {arabic}\nEnglish: {english_text}"

        # Metadata (lightweight)
        chapter = chapters.get(h.get("chapterId"), {})
        meta = clean_metadata({
            "id": h.get("id"),
            "source": source_name,
            "book_id": h.get("bookId"),
            "chapter_id": h.get("chapterId"),
            "chapter_en": chapter.get("english", "unknown"),
            "chapter_ar": chapter.get("arabic", "unknown"),
        })

        # Unique vector ID: source-chapter-hadith
        vec_id = f"{source_name.lower()}-{h.get('chapterId', '0')}-{h.get('id', i)}"

        # Generate embedding
        emb = get_embedding(combined[:7000])
        batch.append({"id": vec_id, "values": emb, "metadata": meta})

        # Upload + checkpoint periodically
        if (i + 1) % save_interval == 0 or i == len(hadiths) - 1:
            print(f"⏫ Upserting batch {i + 1 - len(batch) + 1}–{i + 1} to Pinecone...")
            index.upsert(vectors=batch, namespace=namespace)

            saved_meta.extend([v["metadata"] for v in batch])
            with open(meta_path, "wb") as f:
                pickle.dump(saved_meta, f)

            print(f"✅ Checkpoint saved ({i + 1}/{len(hadiths)})")
            batch = []
            time.sleep(1)

    print(f"\n🎉 Done! Indexed {len(hadiths)} hadiths into Pinecone index '{index_name}'.")
    print(f"Metadata checkpoint saved to {meta_path}")

def search_quran_pinecone(query, top_k, index_name, pinecone_api_key):
    """Search Quran index and return results with metadata only."""
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    query_emb = get_embedding(query)
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
    )

    output = []
    for match in results["matches"]:
        meta = match["metadata"]
        
        # Return what's in metadata - main.py will handle local lookup
        output.append({
            "vector_id": match["id"],  # e.g., "item-0"
            "score": round(match["score"] * 100, 2),
            "surah_name": meta.get("surah_name", "Unknown"),
            "surah_id": meta.get("surah_id"),
            "verse_range": meta.get("verse_range", ""),
            "num_verses": meta.get("num_verses", 0),
            "tafsir_source": meta.get("tafsir_source", "unknown"),
            "type": "quran",
        })
    return output


def search_hadith_pinecone(query, top_k, index_name, pinecone_api_key):
    """Search Hadith index and return results with metadata only."""
    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    query_emb = get_embedding(query)
    results = index.query(
        vector=query_emb,
        top_k=top_k,
        include_metadata=True,
        namespace="default"
    )

    output = []
    for match in results["matches"]:
        meta = match["metadata"]
        
        # Extract hadith ID from vector_id (format: "source-chapter-hadithid")
        vector_id = match["id"]
        hadith_id = None
        
        # Try to parse hadith ID from vector_id
        # Format examples: "sahibukhari-1-1", "sahibukhari-0-123"
        parts = vector_id.split("-")
        if len(parts) >= 3:
            hadith_id = parts[-1]  # Last part is hadith ID
        
        output.append({
            "vector_id": vector_id,
            "hadith_id": hadith_id,
            "score": round(match["score"] * 100, 2),
            "chapter_en": meta.get("chapter_en", "Unknown"),
            "chapter_ar": meta.get("chapter_ar", ""),
            "source": meta.get("source", "Hadith"),
            "book_id": meta.get("book_id", ""),
            "chapter_id": meta.get("chapter_id", ""),
            "type": "hadith",
        })
    return output

if __name__ == "__main__":

    """
    build_quran_tafsir_index_pinecone(
        merged_path="data/quran_and_tafsir.json",
        index_name="qurandtefsir",
        namespace="default",
        meta_path="full.pkl",
        save_interval=100,
        dimension=3072,
    )


    build_hadith_index_pinecone(
        json_path="data/bukhari_ar_en.json",
        index_name="hadith-bukhari",
        namespace="default",
        meta_path="hadith_bukhari.pkl",
        save_interval=100,
        dimension=3072,
    )
"""

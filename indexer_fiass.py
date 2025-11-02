import numpy as np
import faiss
import pickle
import os
import json
import time
from utils import get_embedding
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# used for local testing on local with FAISS and PKL

def build_quran_index(json_data, index_path="data/quran.faiss", meta_path="data/quran.pkl"):
    """
    Build a FAISS index for all Quran verses (Arabic + translation),
    with detailed progress logging for embeddings.
    """

    if isinstance(json_data, dict):
        json_data = [json_data]

    texts = []
    metadata = []

    for surah in json_data:
        surah_id = surah.get("id")
        surah_name = surah.get("name", "Unknown")

        for verse in surah.get("verses", []):
            combined_text = f"{verse['text']} — {verse['translation']}"
            texts.append(combined_text)
            metadata.append({
                "surah_id": surah_id,
                "surah_name": surah_name,
                "verse_id": verse["id"],
                "text": verse["text"],
                "translation": verse["translation"]
            })
            print(f"Added verse {surah_id}:{verse['id']}")

    total = len(texts)
    print(f"\nEmbedding {total} verses...\n")

    embeddings = []
    for idx, text in enumerate(texts, start=1):
        emb = get_embedding(text=text)
        
        embeddings.append(emb)
        print(f"Embedding {idx}/{total} added")

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, index_path)
    pickle.dump(metadata, open(meta_path, "wb"))

    print(f"\n✅ Saved {index_path} and {meta_path} ({len(metadata)} verses total)")


def build_quran_tafsir_index(
    merged_path,
    index_path="data/first50.faiss",
    meta_path="data/first50.pkl",
    save_interval=200
):
    """
    Build or resume a FAISS index for Qur'an & Tafsir embeddings.
    Automatically checkpoints progress every `save_interval` entries.
    """

    # === Load merged data ===
    with open(merged_path, "r", encoding="utf-8") as f:
        merged = json.load(f)

    texts, metadata = [], []
    for item in merged:
        verses_text = " ".join(v["text"] for v in item["verses"])
        verses_trans = " ".join(v["translation"] for v in item["verses"])
        combined = f"{verses_text}\n{verses_trans}\n{item['tafsir']}"
        texts.append(combined)
        metadata.append(item)

    total = len(texts)
    print(f"📖 Total items to embed: {total}")

    # === Check for existing progress ===
    if os.path.exists(index_path) and os.path.exists(meta_path):
        print("🔁 Resuming from previous checkpoint...")
        index = faiss.read_index(index_path)
        done = index.ntotal
        with open(meta_path, "rb") as f:
            saved_meta = pickle.load(f)
        assert len(saved_meta) == done, "Metadata and index out of sync!"
        metadata = saved_meta
        print(f"→ Already embedded {done}/{total}")
    else:
        index = None
        done = 0

    # === Loop over remaining items ===
    new_embeddings = []
    for i in range(done, total):
        text = texts[i]
        emb = get_embedding(text=text)
        new_embeddings.append(emb)

        if (i + 1) % 10 == 0 or i == total - 1:
            print(f"Embedded {i+1}/{total}")

        if (i + 1) % save_interval == 0 or i == total - 1:
            print(f"Saving progress at {i+1}...")

            emb_array = np.array(new_embeddings).astype("float32")

            if index is None:
                dim = emb_array.shape[1]
                index = faiss.IndexFlatL2(dim)
            index.add(emb_array)

            # Save index
            faiss.write_index(index, index_path)

            # Save metadata (append newly processed)
            with open(meta_path, "wb") as f:
                pickle.dump(metadata[:i+1], f)

            print(f"✅ Checkpoint saved ({i+1}/{total})")
            new_embeddings = []  # clear memory
            time.sleep(1)  # short cooldown to prevent rate errors

    print(f"\n🎉 Done! Total {index.ntotal} embeddings indexed.")
    print(f"Index saved to {index_path}, metadata to {meta_path}")


def build_hadith_index(
    json_path,
    index_path="data/hadith_bukhari.faiss",
    meta_path="data/hadith_bukhari.pkl",
    checkpoint_interval=200
):
    """Build FAISS index for hadith-json bilingual dataset (Arabic + English)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    hadiths = data.get("hadiths", [])
    books = {b["id"]: b for b in data.get("books", [])}
    chapters = {c["id"]: c for c in data.get("chapters", [])}

    print(f"📘 Found {len(hadiths)} hadiths in dataset")

    embeddings = []
    metadata = []
    index = None

    for i, h in enumerate(hadiths):
        # Extract fields
        arabic = h.get("arabic", "").strip()
        english = h.get("english", {})
        narrator = english.get("narrator", "").strip()
        text = english.get("text", "").strip()

        # Book & chapter context
        book = books.get(h.get("bookId"), {})
        chapter = chapters.get(h.get("chapterId"), {})

        book_en = book.get("english", "")
        book_ar = book.get("arabic", "")
        chapter_en = chapter.get("english", "")
        chapter_ar = chapter.get("arabic", "")

        # Combine for embedding
        combined = (
            f"Arabic: {arabic}\n\n"
            f"English Narration: {narrator}\n{text}\n\n"
            f"Book: {book_en} ({book_ar}) | Chapter: {chapter_en} ({chapter_ar})"
        )

        # Metadata
        meta = {
            "id": h.get("id"),
            "book_id": h.get("bookId"),
            "chapter_id": h.get("chapterId"),
            "arabic": arabic,
            "english_narrator": narrator,
            "english_text": text,
            "book_en": book_en,
            "book_ar": book_ar,
            "chapter_en": chapter_en,
            "chapter_ar": chapter_ar,
        }

        # Create embedding
        emb = get_embedding(combined[:7000])
        embeddings.append(emb)
        metadata.append(meta)

        if (i + 1) % 10 == 0:
            print(f"→ Embedded {i + 1}/{len(hadiths)}")

        # Checkpoint every N
        if (i + 1) % checkpoint_interval == 0 or (i + 1) == len(hadiths):
            np_emb = np.array(embeddings).astype("float32")

            if index is None:
                index = faiss.IndexFlatL2(np_emb.shape[1])
            index.add(np_emb)

            faiss.write_index(index, index_path)
            pickle.dump(metadata, open(meta_path, "wb"))

            print(f"💾 Checkpoint saved ({i + 1}/{len(hadiths)})")

            embeddings = []  # clear batch
            time.sleep(1.5)

    print(f"✅ Completed FAISS index → {index_path}")
    print(f"✅ Metadata saved → {meta_path}")




if __name__ == "__main__":

    """ 
    build_quran_tafsir_index(merged_path="data/wajiz_clean_merged.json",
                             index_path="data/fulldata.faiss",
                             meta_path="data/fulldata.pkl")
    """
    build_hadith_index(
        json_path="data/bukhari_ar_en.json",
        index_path="data/hadith_bukhari_full.faiss",
        meta_path="data/hadith_bukhari_full.pkl",
        checkpoint_interval=200
    )


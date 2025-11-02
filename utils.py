
from openai import OpenAI
import os
from dotenv import load_dotenv
import re
import json
from bs4 import BeautifulSoup

load_dotenv()


def get_embedding(text, model=os.getenv("EMBEDDING_MODEL")):
    client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

    response = client.embeddings.create(
    model= model,
    input=text,
    encoding_format="float"
    )
    return response.data[0].embedding


def normalize_score(distance, min_dist=150, max_dist=400):
    distance = max(min_dist, min(distance, max_dist))
    score = (1 - (distance - min_dist) / (max_dist - min_dist)) * 100
    return round(score, 1)


def parse_tafsir_json(input_path, output_path=None):
    """
    Parse a tafsir JSON file and return a clean dict { '1:1': 'plain tafsir text' }.
    Handles cases where values are either strings or dicts with 'text'.
    """

    with open(input_path, "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)

    clean_tafsir = {}
    for key, val in tafsir_data.items():
        # Handle both {"text": "..."} and just "..."
        html = val["text"] if isinstance(val, dict) and "text" in val else str(val)
        soup = BeautifulSoup(html, "html.parser")
        clean_text = soup.get_text(separator=" ", strip=True)
        clean_tafsir[key] = clean_text

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clean_tafsir, f, ensure_ascii=False, indent=2)

    return clean_tafsir


def merge_quran_tafsir(quran_path, tafsir_clustered_path, output_path):
    """
    Merge Quran text/translation JSON with clustered tafsir JSON into a unified structure.

    Args:
        quran_path (str): path to the Quran JSON (list of surahs)
        tafsir_clustered_path (str): path to clustered tafsir dict
        output_path (str): path to save merged file
    """
    with open(quran_path, "r", encoding="utf-8") as f:
        quran_data = json.load(f)

    with open(tafsir_clustered_path, "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)

    merged = []

    # Create quick lookup for verse texts and translations
    verse_lookup = {}
    for surah in quran_data:
        s_id = surah["id"]
        for v in surah["verses"]:
            key = f"{s_id}:{v['id']}"
            verse_lookup[key] = {
                "surah_id": s_id,
                "surah_name": surah["name"],
                "surah_transliteration": surah["transliteration"],
                "surah_translation": surah["translation"],
                "verse_text": v["text"],
                "verse_translation": v["translation"],
            }

    # Iterate through tafsir clusters
    for cluster_key, tafsir_text in tafsir_data.items():
        # Detect range like "2:68-71"
        match = re.match(r"(\d+):(\d+)(?:-(\d+))?", cluster_key)
        if not match:
            continue
        surah_id = int(match.group(1))
        start_ayah = int(match.group(2))
        end_ayah = int(match.group(3) or start_ayah)

        # Collect verse details
        verses = []
        for v_id in range(start_ayah, end_ayah + 1):
            v_key = f"{surah_id}:{v_id}"
            if v_key in verse_lookup:
                verse = verse_lookup[v_key]
                verses.append({
                    "id": v_id,
                    "text": verse["verse_text"],
                    "translation": verse["verse_translation"],
                })

        # Skip if we didn’t find any verses
        if not verses:
            continue

        first_verse = verse_lookup[f"{surah_id}:{start_ayah}"]

        merged.append({
            "id": cluster_key,
            "surah_id": surah_id,
            "surah_name": first_verse["surah_name"],
            "surah_transliteration": first_verse["surah_transliteration"],
            "surah_translation": first_verse["surah_translation"],
            "verse_range": f"{start_ayah}-{end_ayah}" if end_ayah != start_ayah else str(start_ayah),
            "verses": verses,
            "tafsir": tafsir_text.strip(),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"✅ Merged Quran + Tafsir saved to {output_path}")
    print(f"Total clusters: {len(merged)}")
    return merged


def load_lang(lang="en"):
    """Load language strings from file"""
    lang_file = f"lang/{lang}.txt"
    texts = {}
    
    try:
        with open(lang_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "=" in line:
                    key, value = line.split("=", 1)
                    texts[key.strip()] = value.strip()
    except FileNotFoundError:
        # Fallback to English if language file not found
        if lang != "en":
            return load_lang("en")
    
    return texts
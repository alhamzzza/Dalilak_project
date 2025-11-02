import json
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv
import os
from pinecone_utils import search_quran_pinecone, search_hadith_pinecone
from utils import load_lang  
import uvicorn
load_dotenv()

QURAN_INDEX = os.getenv("QURAN_INDEX")
HADITH_INDEX = os.getenv("HADITH_INDEX")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

with open("data/quran_tefs_ar_en.json", "r", encoding="utf-8") as f:
    QURAN_DATA = json.load(f)

with open("data/bukhari_ar_en.json", "r", encoding="utf-8") as f:
    HADITH_DATA = json.load(f)
    HADITHS = {str(h["id"]): h for h in HADITH_DATA["hadiths"]}
    CHAPTERS = {c["id"]: c for c in HADITH_DATA.get("chapters", [])}
    BOOKS = {b["id"]: b for b in HADITH_DATA.get("books", [])}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request, lang: str = "en"):
    texts = load_lang(lang)
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "results": None, "query": "", "top_k": 5, "texts": texts, "lang": lang}
    )


@app.post("/search", response_class=HTMLResponse)
async def search(
    request: Request,
    query: str = Form(...),
    top_k: int = Form(...),
    include_quran: bool = Form(False),
    include_hadith: bool = Form(False),
    lang: str = Form("en"),
):
    texts = load_lang(lang)
    results = []

    if include_quran:
        quran_results = search_quran_pinecone(query, top_k, QURAN_INDEX, PINECONE_API_KEY)
        
        for r in quran_results:
            try:
                item_idx = int(r["vector_id"].split("-")[-1])
                
                if 0 <= item_idx < len(QURAN_DATA):
                    item = QURAN_DATA[item_idx]
                    
                    if lang == "ar" and item.get("tafsir_ar"):
                        tafsir = item.get("tafsir_ar")
                    else:
                        tafsir = item.get("tafsir", "")
                    
                    results.append({
                            "type": "quran",
                            "id": r["vector_id"],
                            "score": r["score"],
                            "surah_ar": item.get("surah_name", r["surah_name"]),
                            "surah_en": item.get("surah_transliteration", ""),  
                            "surah_id": item.get("surah_id", 1),  
                            "verse_range": r["verse_range"],
                            "full_id": item.get("id", ""),  
                            "verses": item.get("verses", []),
                            "tafsir": tafsir,
                                                
                    })
            except (ValueError, IndexError, KeyError) as e:
                print(f"Error processing Quran result: {e}")
                continue

    if include_hadith:
        hadith_results = search_hadith_pinecone(query, top_k, HADITH_INDEX, PINECONE_API_KEY)
        
        for r in hadith_results:
            hadith_id = r.get("hadith_id")
            
            if hadith_id and str(hadith_id) in HADITHS:
                hadith = HADITHS[str(hadith_id)]
                
                book = BOOKS.get(hadith.get("bookId"), {})
                
                results.append({
                    "type": "hadith",
                    "id": r["vector_id"],
                    "score": r["score"],
                    "english_text": hadith.get("english", {}).get("text", ""),
                    "arabic_text": hadith.get("arabic", ""),
                    "narrator": hadith.get("english", {}).get("narrator", ""),
                    "chapter_en": r["chapter_en"],
                    "chapter_ar": r["chapter_ar"],
                    "book_en": book.get("english", "Unknown"),
                    "book_ar": book.get("arabic", ""),
                    "book_id": r["book_id"],
                    "chapter_id": r["chapter_id"],
                })
            else:
                print(f"Hadith not found in local data: {hadith_id}")
                continue

    results.sort(key=lambda x: x.get("score", 0), reverse=True)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": results,
            "query": query,
            "top_k": top_k,
            "include_quran": include_quran,
            "include_hadith": include_hadith,
            "texts": texts,
            "lang": lang,
        },
    )

# To run the app, use the command:
# uvicorn main:app --reload


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
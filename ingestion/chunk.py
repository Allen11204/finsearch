import json

from config import CHUNKS_DIR, TEXT_DIR

CHUNK_SIZE = 200   # words per chunk
OVERLAP = 50       # words shared by adjacent chunks

COMPANY = "UnitedHealth"
TICKER = "UNH"
YEAR = 2023


def chunk_text(text):
    words = text.split()
    step = CHUNK_SIZE - OVERLAP   # advance 150 words each time
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + CHUNK_SIZE]
        chunks.append(" ".join(piece))
        if start + CHUNK_SIZE >= len(words):   # this chunk already reaches the end, stop
            break
    return chunks


if __name__ == "__main__":
    with open(TEXT_DIR / "UNH_2023_sections.json", encoding="utf-8") as f:
        sections = json.load(f)

    records = []
    for section, text in sections.items():
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{TICKER}_{YEAR}_{section.replace(' ', '')}_{i:03d}",
                "company": COMPANY,
                "ticker": TICKER,
                "year": YEAR,
                "section": section,
                "text": chunk,
            })
        print(f"{section}: {len(chunks)} chunks")

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHUNKS_DIR / "UNH_2023_chunks.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print("Total:", len(records), "chunks")
    print("First record:", records[0])

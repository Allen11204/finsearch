from bs4 import BeautifulSoup

from config import RAW_DIR, TEXT_DIR

# 1. Read the HTML
with open(RAW_DIR / "UNH_2023_10K.htm", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")

# 2. Remove non-body content: scripts, styles, SEC hidden XBRL data
for tag in soup.find_all(["script", "style", "ix:header"]):
    tag.decompose()

# 3. Extract text, with a newline between tags
text = soup.get_text(separator="\n")

# 4. Clean whitespace: replace non-breaking spaces, strip each line, drop empty lines
text = text.replace("\xa0", " ")
lines = [line.strip() for line in text.split("\n")]
lines = [line for line in lines if line]
text = "\n".join(lines)

# 5. Save as txt
TEXT_DIR.mkdir(parents=True, exist_ok=True)
with open(TEXT_DIR / "UNH_2023_10K.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Characters:", len(text))

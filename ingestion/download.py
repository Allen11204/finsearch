import requests

from config import RAW_DIR

# UnitedHealth (CIK 731766) 10-K for fiscal year 2023, filed 2024-02-28
URL = "https://www.sec.gov/Archives/edgar/data/731766/000073176624000081/unh-20231231.htm"
HEADERS = {"User-Agent": "Allen Liu jil764@pitt.edu"}

resp = requests.get(URL, headers=HEADERS)
resp.raise_for_status()

RAW_DIR.mkdir(parents=True, exist_ok=True)
with open(RAW_DIR / "UNH_2023_10K.htm", "w", encoding="utf-8") as f:
    f.write(resp.text)

print("Download complete, characters:", len(resp.text))

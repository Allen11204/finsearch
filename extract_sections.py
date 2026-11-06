import json
import re

# Decision A2: which sections to keep. Each maps to (start heading regex, end heading regex).
# \s+ instead of a single space: the filing writes "ITEM  1." with two spaces.
SECTIONS = {
    "Item 1":  (r"^ITEM\s+1\.",  r"^ITEM\s+1A\."),
    "Item 1A": (r"^ITEM\s+1A\.", r"^ITEM\s+(1B|1C|2)\."),
    "Item 7":  (r"^ITEM\s+7\.",  r"^ITEM\s+(7A|8)\."),
}

# Edge case: a section shorter than this was probably cut at the wrong place
MIN_LENGTH = 1000


def extract_section(text, name, start_regex, end_regex):
    # Decision B1: find headings with regex
    start_match = re.search(start_regex, text, re.MULTILINE)
    if start_match is None:
        raise ValueError(f"{name}: start heading not found")

    end_pattern = re.compile(end_regex, re.MULTILINE)
    end_match = end_pattern.search(text, start_match.end())
    if end_match is None:
        raise ValueError(f"{name}: end heading not found")

    section = text[start_match.start():end_match.start()]

    if len(section) < MIN_LENGTH:
        print(f"Warning: {name} is only {len(section)} characters, check manually")

    return section


def extract_sections(text):
    result = {}
    for name, (start_regex, end_regex) in SECTIONS.items():
        result[name] = extract_section(text, name, start_regex, end_regex)
    return result


if __name__ == "__main__":
    with open("data/text/UNH_2023_10K.txt", encoding="utf-8") as f:
        text = f.read()

    sections = extract_sections(text)

    for name, content in sections.items():
        print(f"{name}: {len(content)} characters | starts with: {content[:60]!r}")

    with open("data/text/UNH_2023_sections.json", "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)

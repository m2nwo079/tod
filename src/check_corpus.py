import json, sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW

past = json.load(open(RAW / "past_works.json"))

# Year distribution
print("연도:", dict(sorted(Counter(w["publication_year"] for w in past).items())))

# Abstract coverage (directly affects term extraction quality)
has_abs = sum(1 for w in past if w["abstract"].strip())
print(f"초록 보유: {has_abs}/{len(past)} ({has_abs/len(past)*100:.0f}%)")
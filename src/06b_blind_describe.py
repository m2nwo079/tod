import os, sys, json, time
from pathlib import Path
from dotenv import load_dotenv
from google import genai

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

BLIND_PROMPT = """Rewrite the technology concept below so that it could describe an
emerging idea from ANY time period. Remove all era-identifying cues:
- Named systems, models, or datasets (e.g. AlphaGo, Atari, specific algorithm names).
- Any hint of when this was proposed or how successful it later became.
- Keep only the abstract technical idea: what is combined, what problem it targets.
Do not add praise or predictions. Output only the rewritten description, 2-3 sentences.

[Concept]
{desc}
"""

def blind_rewrite(desc, retries=6):
    # Stronger backoff: up to 6 tries, waiting up to 60s, to ride out congestion
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL, contents=BLIND_PROMPT.format(desc=desc))
            return resp.text.strip()
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"    retry {attempt+1}/{retries}: {str(e)[:60]} (wait {wait}s)")
            time.sleep(wait)
    return None

def main():
    concepts = [json.loads(l) for l in open(EXCHANGE / "described.jsonl")]
    out = EXCHANGE / "described_blind.jsonl"

    # Resume: keep any blind rewrites already saved, only fill the missing ones
    done = {}
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r.get("description_blind"):
                done[r["concept_id"]] = r["description_blind"]
    print(f"Already done: {len(done)} / {len(concepts)}")

    results = []
    try:
        for c in concepts:
            if c["concept_id"] in done:
                blind = done[c["concept_id"]]
                print(f"=== {c['concept_id']} (cached) ===")
            else:
                blind = blind_rewrite(c["description"])
                print(f"=== {c['concept_id']} ===")
                print(blind)
                print()
                time.sleep(2)  # extra spacing between concepts to ease congestion
            results.append({**c, "description_blind": blind})
    finally:
        with open(out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        n_ok = sum(1 for r in results if r.get("description_blind"))
        print(f"Saved {n_ok}/{len(concepts)} successful -> {out}")

if __name__ == "__main__":
    main()
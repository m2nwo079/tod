import os, sys, json, time, re
from pathlib import Path
from dotenv import load_dotenv
from google import genai

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.5-flash-lite"
N_RUNS = 3

# Same rubric as the Qwen judge, so the two are comparable
JUDGE_PROMPT = """Rate the 'promise' of the emerging technology concept below on a
scale of 1 to 9, judging novelty, potential technical impact, and problem-solving
potential. Score unfounded confidence low.

[Concept] {concept}

Output JSON only, no other text: {{"score": <1-9>, "reason": "<two sentences max>"}}
"""

def parse_score(text):
    m = re.search(r'"score"\s*:\s*([1-9])', text)
    return int(m.group(1)) if m else None

def judge_once(concept_text, retries=4):
    prompt = JUDGE_PROMPT.format(concept=concept_text)
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return parse_score(resp.text)
        except Exception as e:
            wait = 2 ** attempt
            print(f"    retry {attempt + 1}/{retries}: {e} (wait {wait}s)")
            time.sleep(wait)
    return None

def main():
    concepts = [json.loads(l) for l in open(EXCHANGE / "described.jsonl")]
    out_path = EXCHANGE / "judged_gemini.jsonl"
    results = []
    try:
        for c in concepts:
            scores = []
            for r in range(N_RUNS):
                s = judge_once(c["description"])
                scores.append(s)
                print(f"  {c['concept_id']} run {r+1}: score={s}")
                time.sleep(1)
            valid = [s for s in scores if s is not None]
            median = sorted(valid)[len(valid) // 2] if valid else None
            results.append({"concept_id": c["concept_id"], "judge": MODEL,
                            "scores": scores, "median_score": median})
            print(f"  -> {c['concept_id']} median={median}")
    finally:
        with open(out_path, "w") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Saved {len(results)}/{len(concepts)} -> {out_path}")

if __name__ == "__main__":
    main()
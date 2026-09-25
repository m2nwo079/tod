import os, sys, json, time, re
from pathlib import Path
from dotenv import load_dotenv
from google import genai

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE, RESULTS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
MODEL = "gemini-3.5-flash-lite"

TARGETS = ["C05", "C06", "C00", "C02"]

BRIEF_PROMPT = """You are a senior service planner. Translate the technology concept
below into a product/service opportunity brief.

[Concept] {desc}
[Data evidence] Relative diffusion vs field average: {rel_growth}x
(1.0 = field average; higher means it spread faster than the RL field overall).
Past-window paper count: {past_n}.

Fill these fields as JSON only, no other text:
{{
  "summary": "concept in 1-2 sentences",
  "target_user": "who the customer/user is",
  "jtbd": "the job-to-be-done / problem, from the user's perspective",
  "value_prop": "value versus existing options",
  "mvp": "the smallest thing to validate first",
  "demand_signal": "cite the data evidence above explicitly",
  "feasibility_risk": "maturity, data, or regulatory barriers",
  "adjacent": "adjacent or substitute technologies"
}}
State clearly if any claim is speculative and not backed by the data.
"""

DVF_PROMPT = """Score the opportunity brief below on three axes, each 1 to 5, with a
one-sentence reason each. Base scores on the demand_signal / data where possible.
Desirability (user demand), Feasibility (technical maturity), Viability (business).

[Brief] {brief}

Output JSON only: {{"D":{{"score":<1-5>,"why":""}}, "F":{{"score":<1-5>,"why":""}}, "V":{{"score":<1-5>,"why":""}}}}
"""

def call(prompt, retries=6):
    for attempt in range(retries):
        try:
            return client.models.generate_content(model=MODEL, contents=prompt).text
        except Exception as e:
            wait = min(2 ** attempt, 60)
            print(f"    retry {attempt+1}/{retries}: {str(e)[:50]} (wait {wait}s)")
            time.sleep(wait)
    return None

def extract_json(text):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def main():
    described = {r["concept_id"]: r for r in
                 (json.loads(l) for l in open(EXCHANGE / "described.jsonl"))}
    diff = {r["concept_id"]: r for r in
            json.loads((RESULTS / "diffusion.json").read_text())["concepts"]}

    results = []
    try:
        for cid in TARGETS:
            c = described[cid]
            d = diff[cid]
            brief_raw = call(BRIEF_PROMPT.format(
                desc=c["description"], rel_growth=d["rel_growth"], past_n=d["past_n"]))
            brief = extract_json(brief_raw)
            time.sleep(1)

            dvf_raw = call(DVF_PROMPT.format(brief=json.dumps(brief, ensure_ascii=False)))
            dvf = extract_json(dvf_raw)
            time.sleep(1)

            dvf_total = None
            if dvf:
                try:
                    dvf_total = sum(dvf[k]["score"] for k in ("D", "F", "V"))
                except Exception:
                    pass

            results.append({"concept_id": cid, "rel_growth": d["rel_growth"],
                            "brief": brief, "dvf": dvf, "dvf_total": dvf_total})
            print(f"=== {cid}  (DVF total: {dvf_total}) ===")
            if brief:
                print(f"  MVP: {brief.get('mvp', '')}")
                print(f"  demand: {brief.get('demand_signal', '')}")
            print()
    finally:
        RESULTS.mkdir(parents=True, exist_ok=True)
        out = RESULTS / "briefs.json"
        out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"Saved {len(results)} briefs -> {out}")

if __name__ == "__main__":
    main()
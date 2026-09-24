import os, sys, json, time, re
from pathlib import Path
from dotenv import load_dotenv
from google import genai

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

MODEL = "gemini-3.5-flash-lite"

DESCRIBE_PROMPT = """You are a technology intelligence analyst. Below are core terms
from a knowledge graph of a research subfield, some latent (predicted) links between
terms, and a few representative paper abstracts.

[Core terms] {terms}
[Latent links] {links}
[Abstracts] {abstracts}

Describe, in one or two paragraphs, the single 'emerging technology concept' this
combination suggests.
- Be explicit about what is being combined or extended.
- Ground your description in the abstracts; do not exaggerate.
- On the last line, give 3-6 keywords for this concept as 'SEED: a, b, c'.
"""

def describe(concept, retries=4):
    abstracts = "\n".join(
        f"- {a['title']}: {a['abstract'][:300]}" for a in concept["abstracts"]
    )
    prompt = DESCRIBE_PROMPT.format(
        terms=", ".join(concept["core_terms"]),
        links="; ".join(concept["latent_links"]),
        abstracts=abstracts,
    )
    for attempt in range(retries):
        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            return resp.text.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} after error: {e} (wait {wait}s)")
            time.sleep(wait)
    raise RuntimeError(f"Failed to describe {concept['concept_id']} after {retries} tries")

def main():
    concepts = [json.loads(l) for l in open(EXCHANGE / "concepts.jsonl")]
    out_path = EXCHANGE / "described.jsonl"
    described = []
    try:
        for c in concepts:
            text = describe(c)
            m = re.search(r"SEED:\s*(.+)", text)
            seeds = [s.strip() for s in m.group(1).split(",")] if m else []
            described.append({**c, "description": text, "seed_terms": seeds})
            print(f"=== {c['concept_id']} ({len(seeds)} seeds) ===")
            print(text)
            print()
            time.sleep(1)
    finally:
        # Save whatever succeeded, even if a later concept failed
        with open(out_path, "w") as f:
            for d in described:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"Saved {len(described)}/{len(concepts)} described concepts -> {out_path}")

if __name__ == "__main__":
    main()
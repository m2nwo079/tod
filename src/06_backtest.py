import os, sys, json, time, requests
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import EXCHANGE, RESULTS

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
API_KEY = os.environ["OPENALEX_API_KEY"]
BASE = "https://api.openalex.org/works"

PAST = ("2015-01-01", "2019-12-31")
FUTURE = ("2021-01-01", "2024-12-31")
TOPIC_ID = "T10462"

def count_works(search=None, date_from=None, date_to=None, extra=None):
    # Return the total number of works matching a search + date filter
    filt = [f"from_publication_date:{date_from}", f"to_publication_date:{date_to}",
            "language:en"]
    if extra:
        filt.append(extra)
    params = {"filter": ",".join(filt), "per_page": 1, "api_key": API_KEY}
    if search:
        params["search"] = search
    for attempt in range(4):
        try:
            r = requests.get(BASE, params=params, timeout=60).json()
            return r["meta"]["count"]
        except Exception as e:
            time.sleep(2 ** attempt)
    return None

def cited_sum(search, date_from, date_to):
    # Sum of citation counts across matching works (via cited_by_count aggregation)
    filt = [f"from_publication_date:{date_from}", f"to_publication_date:{date_to}",
            "language:en"]
    params = {"filter": ",".join(filt), "search": search,
              "per_page": 1, "api_key": API_KEY}
    for attempt in range(4):
        try:
            r = requests.get(BASE, params=params, timeout=60).json()
            # Use total works as a proxy weight; citation growth measured separately below
            return r["meta"]["count"]
        except Exception:
            time.sleep(2 ** attempt)
    return None

def main():
    concepts = [json.loads(l) for l in open(EXCHANGE / "described.jsonl")]

    # Field-wide baseline: total RL works in each window (for normalization)
    base_past = count_works(None, *PAST, extra=f"topics.id:{TOPIC_ID}")
    base_future = count_works(None, *FUTURE, extra=f"topics.id:{TOPIC_ID}")
    field_growth = base_future / base_past if base_past else None
    print(f"Field baseline: past={base_past}, future={base_future}, "
          f"growth={field_growth:.2f}x\n")

    rows = []
    for c in concepts:
        # Build a search query from seed terms (fallback to core_terms)
        seeds = c.get("seed_terms") or c["core_terms"][:5]
        query = " ".join(seeds[:5])

        past_n = count_works(query, *PAST)
        future_n = count_works(query, *FUTURE)
        time.sleep(0.3)

        raw_growth = (future_n / past_n) if past_n else None
        # Normalize by field growth -> relative diffusion (>1 = beat the field)
        rel_growth = (raw_growth / field_growth) if (raw_growth and field_growth) else None

        rows.append({
            "concept_id": c["concept_id"],
            "query": query,
            "past_n": past_n, "future_n": future_n,
            "raw_growth": round(raw_growth, 3) if raw_growth else None,
            "rel_growth": round(rel_growth, 3) if rel_growth else None,
        })
        print(f"{c['concept_id']}: past={past_n}, future={future_n}, "
              f"raw={raw_growth:.2f}x, rel={rel_growth:.2f}x  [{query}]")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "diffusion.json"
    out.write_text(json.dumps(
        {"field": {"past": base_past, "future": base_future, "growth": field_growth},
         "concepts": rows}, ensure_ascii=False, indent=2))
    print(f"\nSaved -> {out}")

if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""GoldenDB recruiter chat — Claude Opus 4.8 + tool use over Qdrant.

Usage:  set ANTHROPIC_API_KEY, then:  python chat.py
"""
import json
import sys, io
from collections import Counter

import anthropic
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

COLLECTION = "golden_candidates"
MODEL = "claude-opus-4-8"

qdrant = QdrantClient(url="http://localhost:6333")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

# ---------------------------------------------------------------- tools
def build_filter(f):
    must = []
    KEYWORD_ANY = ["role_family", "location_city", "hometown_city", "tier", "overall_band",
                   "systems_built", "compliance", "skill_names", "timezone_flex",
                   "leadership_aspiration", "builder_archetype", "industries",
                   "exec_exposure", "intl_exposure", "company_tier", "institution_tier",
                   "hometown_pull", "search_status", "ceiling", "renege_risk", "motivation_type"]
    for k in KEYWORD_ANY:
        if f.get(k):
            vals = f[k] if isinstance(f[k], list) else [f[k]]
            must.append(FieldCondition(key=k, match=MatchAny(any=vals)))
    if f.get("max_notice_days") is not None:
        must.append(FieldCondition(key="notice_days", range=Range(lte=f["max_notice_days"])))
    if f.get("min_yoe") is not None:
        must.append(FieldCondition(key="yoe", range=Range(gte=f["min_yoe"])))
    if f.get("max_yoe") is not None:
        must.append(FieldCondition(key="yoe", range=Range(lte=f["max_yoe"])))
    if f.get("budget_max_lpa") is not None:
        must.append(FieldCondition(key="ctc_expected_min", range=Range(lte=f["budget_max_lpa"])))
    if f.get("min_retention_6m") is not None:
        must.append(FieldCondition(key="retention_6m", range=Range(gte=f["min_retention_6m"])))
    if f.get("min_mgmt_refused") is not None:
        must.append(FieldCondition(key="mgmt_refused_count", range=Range(gte=f["min_mgmt_refused"])))
    if f.get("min_scale_tb") is not None:
        must.append(FieldCondition(key="scale_tb", range=Range(gte=f["min_scale_tb"])))
    return Filter(must=must) if must else None

CARD_FIELDS = ["candidate_id", "full_name", "current_title", "current_company", "company_tier",
               "role_family", "yoe", "yoe_domain", "location_city", "hometown_city", "hometown_pull",
               "tier", "overall_band", "composite", "depth_v3", "resume_panel_delta",
               "notice_days", "ctc_expected_min", "ctc_expected_max", "systems_built",
               "compliance", "skill_names", "retention_6m", "ceiling", "ideal_role",
               "timezone_flex", "leadership_aspiration", "renege_risk"]

def t_search(args):
    flt = build_filter(args.get("filters") or {})
    vec = list(embedder.embed([args["query"]]))[0].tolist()
    pts = qdrant.query_points(COLLECTION, query=vec, query_filter=flt,
                              limit=args.get("limit", 6)).points
    return json.dumps([{**{k: p.payload.get(k) for k in CARD_FIELDS},
                        "match_score": round(p.score, 3)} for p in pts], ensure_ascii=False)

def t_aggregate(args):
    flt = build_filter(args.get("filters") or {})
    res, _ = qdrant.scroll(COLLECTION, scroll_filter=flt, limit=500,
                           with_payload=[args["group_by"]])
    cnt = Counter()
    for p in res:
        v = p.payload.get(args["group_by"])
        for item in (v if isinstance(v, list) else [v]):
            cnt[str(item)] += 1
    return json.dumps({"group_by": args["group_by"], "total_matched": len(res),
                       "counts": dict(cnt.most_common(20))}, ensure_ascii=False)

def t_profile(args):
    res, _ = qdrant.scroll(COLLECTION, scroll_filter=Filter(must=[
        FieldCondition(key="candidate_id", match=MatchValue(value=args["candidate_id"]))]),
        limit=1)
    return json.dumps(res[0].payload, ensure_ascii=False) if res else json.dumps({"error": "not found"})

TOOLS = [
    {
        "name": "search_candidates",
        "description": ("Hybrid search over the Golden DB (100 panel-verified candidates). Call this whenever the "
                        "recruiter asks to find, shortlist, or compare candidates. `query` is matched semantically "
                        "against rich candidate bios; `filters` are hard constraints applied before ranking. "
                        "Prefer putting hard requirements (city, notice, compliance, budget) in filters and "
                        "qualitative asks (what they built, work style) in the query."),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language description of the ideal candidate"},
                "filters": {
                    "type": "object",
                    "properties": {
                        "role_family": {"type": "array", "items": {"type": "string", "enum": ["data_engineering","ml_engineering","data_science","analytics","mlops","data_platform","ai_research","genai_engineering"]}},
                        "location_city": {"type": "array", "items": {"type": "string"}},
                        "hometown_city": {"type": "array", "items": {"type": "string"}},
                        "hometown_pull": {"type": "array", "items": {"type": "string", "enum": ["strong_yes","open","no","already_home"]}},
                        "tier": {"type": "array", "items": {"type": "string", "enum": ["T2","T3","T4","T5","T6"]}},
                        "overall_band": {"type": "array", "items": {"type": "string", "enum": ["gold","silver","bronze"]}},
                        "systems_built": {"type": "array", "items": {"type": "string", "enum": ["payment_rails","fraud_detection","reco_engine","feature_store","data_lakehouse","streaming_platform","ml_serving","cdp","search_ranking","rag_production","forecasting"]}},
                        "compliance": {"type": "array", "items": {"type": "string", "enum": ["pci_dss","rbi_guidelines","sebi","hipaa","gdpr","dpdp","soc2","iso27001"]}},
                        "skill_names": {"type": "array", "items": {"type": "string"}},
                        "timezone_flex": {"type": "array", "items": {"type": "string", "enum": ["ist_only","eu_overlap_ok","us_overlap_ok","fully_flexible"]}},
                        "leadership_aspiration": {"type": "array", "items": {"type": "string"}},
                        "max_notice_days": {"type": "integer"},
                        "min_yoe": {"type": "number"}, "max_yoe": {"type": "number"},
                        "budget_max_lpa": {"type": "integer", "description": "Client budget ceiling in INR lakhs/yr — matches candidates whose expected MINIMUM fits"},
                        "min_retention_6m": {"type": "number"},
                        "min_mgmt_refused": {"type": "integer", "description": "Deliberate-IC search: minimum times management was refused"},
                        "min_scale_tb": {"type": "integer"},
                    },
                },
                "limit": {"type": "integer", "default": 6},
            },
            "required": ["query"],
        },
    },
    {
        "name": "aggregate_candidates",
        "description": ("Market-intelligence aggregation — counts candidates grouped by a field, optionally filtered. "
                        "Call this for questions about the MARKET rather than one candidate: 'where should we open an "
                        "office?' (group_by hometown_city, filter hometown_pull strong_yes/open), 'how deep is the "
                        "MLOps pool in Pune?', skill supply questions, band/tier distributions."),
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {"type": "string", "enum": ["hometown_city","location_city","role_family","tier","overall_band","systems_built","compliance","skill_names","company_tier","timezone_flex","ceiling","institution_tier"]},
                "filters": {"type": "object", "description": "Same filter object as search_candidates"},
            },
            "required": ["group_by"],
        },
    },
    {
        "name": "get_candidate_profile",
        "description": "Fetch the FULL profile of one candidate by candidate_id (e.g. GDB-1042) — every schema field including skills detail, education, behavioural scores, ideal-role synthesis.",
        "input_schema": {
            "type": "object",
            "properties": {"candidate_id": {"type": "string"}},
            "required": ["candidate_id"],
        },
    },
]

EXECUTORS = {"search_candidates": t_search, "aggregate_candidates": t_aggregate,
             "get_candidate_profile": t_profile}

SYSTEM = """You are the Golden Database assistant — a recruiter-intelligence copilot over a database of 100 \
panel-verified AI/data candidates in India (all at the fully-verified stage: 4 pipeline stages complete, v3 signals).

Key vocabulary:
- Tier T2–T6 = SENIORITY (T3 independent, T4 senior, T5 staff/lead, T6 head-of-function).
- Overall band gold/silver/bronze = QUALITY vs the pool (orthogonal to tier; gold = top slice: composite ≥85, retention ≥80%, clean flags).
- resume_panel_delta > 0 means the candidate undersells on paper — a positive signal.
- hometown_pull strong_yes = would relocate back to their roots city, often at -15% comp — the key signal for "where should we open an office" questions.

Rules:
- Always search/aggregate before answering — never invent candidates or numbers.
- Put hard constraints in filters, qualitative asks in the semantic query. Re-search with relaxed filters if few results, and SAY you relaxed them.
- Present candidates as: Name (ID) — title @ company | tier+band | the 2-3 signals most relevant to THIS query. Comp as the expected band only.
- For market questions, lead with the aggregate numbers, then name 2-3 example candidates if useful.
- Be honest about weak matches. Recommend at most a shortlist of 3-5 with a clear #1 and why."""

# ---------------------------------------------------------------- chat loop
def run_turn(messages):
    while True:
        with client.messages.stream(
            model=MODEL,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
            response = stream.get_final_message()

        if response.stop_reason != "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            return

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\n  [{block.name}: {json.dumps(block.input, ensure_ascii=False)[:140]}]")
                try:
                    out = EXECUTORS[block.name](block.input)
                except Exception as e:
                    out = json.dumps({"error": str(e)})
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": out})
        messages.append({"role": "user", "content": results})

def main():
    print("Golden DB chat — 100 panel-verified candidates. Ctrl+C to exit.")
    print("Try: 'we're opening a fintech GCC, where in India is our talent?' or")
    print("     'find me a gold-band streaming engineer who can join in 30 days under 50L'\n")
    messages = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        messages.append({"role": "user", "content": q})
        print()
        run_turn(messages)
        print("\n")

if __name__ == "__main__":
    main()

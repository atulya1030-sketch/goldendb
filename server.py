# -*- coding: utf-8 -*-
"""GoldenDB Web Server — recruiter chat UI + candidate profiles over Qdrant.

Usage:
    set OPENAI_API_KEY=sk-proj-...
    cd vectordb
    uvicorn server:app --reload --port 8001
Then open http://localhost:8001
"""
import asyncio, io, json, os, random, threading, time, urllib.parse, urllib.request, urllib.error
from collections import Counter
from pathlib import Path

import openai
import pdfplumber
from fastembed import TextEmbedding
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range, PointStruct

# ── clients ───────────────────────────────────────────────────────────────────
COLLECTION = "golden_candidates"
MODEL      = "gpt-4o"

qdrant    = QdrantClient(
    url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    api_key=os.getenv("QDRANT_API_KEY") or None,
)
embedder  = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
ai_client = openai.OpenAI()   # reads OPENAI_API_KEY from env

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="GoldenDB")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# ── Qdrant helpers ────────────────────────────────────────────────────────────
def build_filter(f: dict):
    must = []
    KEYWORD_ANY = [
        "role_family","location_city","hometown_city","tier","overall_band",
        "systems_built","compliance","skill_names","timezone_flex",
        "leadership_aspiration","builder_archetype","industries",
        "exec_exposure","intl_exposure","company_tier","institution_tier",
        "hometown_pull","search_status","ceiling","renege_risk","motivation_type",
    ]
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

CARD_FIELDS = [
    "candidate_id","full_name","current_title","current_company","company_tier",
    "role_family","yoe","yoe_domain","location_city","hometown_city","hometown_pull",
    "tier","overall_band","composite","depth_v3","resume_panel_delta",
    "notice_days","ctc_expected_min","ctc_expected_max","systems_built",
    "compliance","skill_names","retention_6m","ceiling","ideal_role",
    "timezone_flex","leadership_aspiration","renege_risk",
]

def exec_search(args):
    base = Filter(must=[FieldCondition(key="pipeline_stage", match=MatchValue(value="panel_complete"))])
    extra = build_filter(args.get("filters") or {})
    if extra and extra.must:
        base.must.extend(extra.must)
    vec = list(embedder.embed([args["query"]]))[0].tolist()
    pts = qdrant.query_points(COLLECTION, query=vec, query_filter=base,
                              limit=args.get("limit", 6)).points
    return json.dumps([
        {**{k: p.payload.get(k) for k in CARD_FIELDS}, "match_score": round(p.score, 3)}
        for p in pts
    ], ensure_ascii=False)

def exec_aggregate(args):
    base = Filter(must=[FieldCondition(key="pipeline_stage", match=MatchValue(value="panel_complete"))])
    extra = build_filter(args.get("filters") or {})
    if extra and extra.must:
        base.must.extend(extra.must)
    res, _ = qdrant.scroll(COLLECTION, scroll_filter=base, limit=500,
                           with_payload=[args["group_by"]])
    cnt = Counter()
    for p in res:
        v = p.payload.get(args["group_by"])
        for item in (v if isinstance(v, list) else [v]):
            cnt[str(item)] += 1
    return json.dumps({"group_by": args["group_by"], "total_matched": len(res),
                       "counts": dict(cnt.most_common(20))}, ensure_ascii=False)

def exec_profile(args):
    res, _ = qdrant.scroll(
        COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="candidate_id",
                                                   match=MatchValue(value=args["candidate_id"]))]),
        limit=1,
    )
    return json.dumps(res[0].payload, ensure_ascii=False) if res else json.dumps({"error": "not found"})

EXECUTORS = {
    "search_candidates":    exec_search,
    "aggregate_candidates": exec_aggregate,
    "get_candidate_profile": exec_profile,
}

# OpenAI function-calling format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_candidates",
            "description": (
                "Hybrid semantic + filter search over 100 panel-verified GoldenDB candidates. "
                "Use `query` for qualitative asks (what they built, work style). "
                "Use `filters` for hard constraints (city, notice, budget, compliance, tier). "
                "Call this for any find/shortlist/compare question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "role_family": {"type": "array", "items": {"type": "string",
                                "enum": ["data_engineering","ml_engineering","data_science","analytics",
                                         "mlops","data_platform","ai_research","genai_engineering"]}},
                            "location_city": {"type": "array", "items": {"type": "string"}},
                            "hometown_city": {"type": "array", "items": {"type": "string"}},
                            "hometown_pull": {"type": "array", "items": {"type": "string",
                                "enum": ["strong_yes","open","no","already_home"]}},
                            "tier": {"type": "array", "items": {"type": "string",
                                "enum": ["T2","T3","T4","T5","T6"]}},
                            "overall_band": {"type": "array", "items": {"type": "string",
                                "enum": ["gold","silver","bronze"]}},
                            "systems_built": {"type": "array", "items": {"type": "string",
                                "enum": ["payment_rails","fraud_detection","reco_engine","feature_store",
                                         "data_lakehouse","streaming_platform","ml_serving","cdp",
                                         "search_ranking","rag_production","forecasting"]}},
                            "compliance": {"type": "array", "items": {"type": "string",
                                "enum": ["pci_dss","rbi_guidelines","sebi","hipaa","gdpr","dpdp","soc2","iso27001"]}},
                            "skill_names": {"type": "array", "items": {"type": "string"}},
                            "timezone_flex": {"type": "array", "items": {"type": "string",
                                "enum": ["ist_only","eu_overlap_ok","us_overlap_ok","fully_flexible"]}},
                            "leadership_aspiration": {"type": "array", "items": {"type": "string"}},
                            "max_notice_days": {"type": "integer"},
                            "min_yoe": {"type": "number"},
                            "max_yoe": {"type": "number"},
                            "budget_max_lpa": {"type": "integer"},
                            "min_retention_6m": {"type": "number"},
                            "min_mgmt_refused": {"type": "integer"},
                            "min_scale_tb": {"type": "integer"},
                        },
                    },
                    "limit": {"type": "integer", "default": 6},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_candidates",
            "description": (
                "Market-intelligence aggregation — counts candidates grouped by a field. "
                "Use for: best office locations, skill supply, tier/band distributions, compliance coverage."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string", "enum": [
                        "hometown_city","location_city","role_family","tier","overall_band",
                        "systems_built","compliance","skill_names","company_tier","timezone_flex",
                        "ceiling","institution_tier",
                    ]},
                    "filters": {"type": "object"},
                },
                "required": ["group_by"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candidate_profile",
            "description": "Fetch the full profile of one candidate by candidate_id (e.g. GDB-1042).",
            "parameters": {
                "type": "object",
                "properties": {"candidate_id": {"type": "string"}},
                "required": ["candidate_id"],
            },
        },
    },
]

SYSTEM = """You are the Golden Database assistant — a recruiter-intelligence copilot over 100 panel-verified AI/data candidates in India.

Key vocabulary:
- Tier T2–T6 = seniority (T3 = independent IC, T4 = senior, T5 = staff/principal/lead, T6 = head-of-function)
- Band gold/silver/bronze = quality vs the pool (gold = top 15%: composite ≥85, retention ≥80%, no flags)
- resume_panel_delta > 0 = undersells on paper — a hidden gem
- hometown_pull = strong_yes means would relocate to roots city, often at −15% comp

Rules:
- ALWAYS call a tool before answering — never invent candidates or numbers.
- Hard constraints (city, notice, budget, compliance) go in filters; qualitative asks go in query.
- Relax filters and say so if fewer than 3 results.
- Format each candidate as: **[Full Name](/profile/GDB-XXXX)** — Title @ Company | Tier · Band | 2–3 most relevant signals
- For market questions, lead with aggregate numbers then 2–3 named examples.
- Shortlist max 3–5 with a clear #1 pick and one-line justification.
- Comp always as a range (expected band), never exact."""

# ── ingest helpers ────────────────────────────────────────────────────────────
def next_candidate_id() -> str:
    res, _ = qdrant.scroll(COLLECTION, limit=500, with_payload=["candidate_id"])
    nums = []
    for p in res:
        cid = p.payload.get("candidate_id", "")
        try: nums.append(int(cid.split("-")[1]))
        except: pass
    return f"GDB-{max(nums) + 1 if nums else 2000}"

def pdf_to_text(data: bytes) -> str:
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

def compute_derived(ex: dict) -> dict:
    """Fill in scoring/tier fields that aren't directly on the resume."""
    yoe   = float(ex.get("yoe") or 0)
    skills = ex.get("skills") or {}
    depth = round(min(9.5, max(4.0, sum(v.get("score",5) for v in skills.values()) / max(len(skills),1))), 1)
    if yoe < 3.5:   tier = "T2"
    elif yoe < 6:   tier = "T3"
    elif yoe < 9:   tier = "T4"
    elif yoe < 13:  tier = "T5" if depth >= 7.5 else "T4"
    else:            tier = "T6" if depth >= 8.8 else "T5"
    composite    = round(min(55, depth * 5 + yoe * 0.6))   # capped at 55 — resume-only
    retention_6m = round(min(0.92, 0.5 + depth * 0.03 + yoe * 0.008), 2)
    band = "silver" if composite >= 45 else "bronze"
    return {"tier": tier, "depth_v3": depth, "composite": composite,
            "overall_band": band, "retention_6m": retention_6m,
            "retention_12m": round(max(0.3, retention_6m - 0.09), 2),
            "resume_panel_delta": 0.0, "growth_velocity": 0.0,
            "stability": 7.0, "resilience": 7.0, "communication": 3.8,
            "architecture_score": depth if tier in ("T4","T5","T6") else None,
            "likelihood_to_move_6m": 0.6, "renege_risk": "low",
            "mgmt_refused_count": 0, "ceiling": "on_track",
            "motivation_type": "growth_driven", "fit_context": "growth_startup",
            "builder_archetype": "scaler", "chaos_tolerance": "adapts",
            "exec_exposure": "team_level", "leadership_aspiration": "open_to_management",
            "ownership": "co_owner", "leadership_level": {
                "T2":"ic_junior","T3":"ic_senior","T4":"ic_senior",
                "T5":"tech_lead","T6":"senior_manager"}[tier],
            "search_status": "open_to_right_role",
            "notice_negotiable": ex.get("notice_days", 60) >= 45,
            "work_mode": "hybrid", "comp_delta_tolerance": 0,
            "red_flag": False, "pipeline_stage": "resume_parse", "signal_version": 2}

EXTRACT_PROMPT = """\
You are a resume parser. Extract candidate data and return ONLY valid JSON — no explanation, no markdown.

Return this exact schema (use null for missing fields, never omit a key):
{
  "full_name": "string",
  "current_title": "string",
  "current_company": "string",
  "company_tier": "tier_1_global|tier_1_india|tier_2|startup_funded|startup_early",
  "role_family": "data_engineering|ml_engineering|data_science|analytics|mlops|data_platform|ai_research|genai_engineering",
  "yoe": number,
  "yoe_domain": number,
  "location_city": "string",
  "hometown_city": "string or null",
  "degree": "bachelors|masters|phd|self_taught",
  "institution": "string",
  "institution_tier": "iit_iisc_iim|nit_bits_top|tier_1_private|tier_2_private|tier_3",
  "grad_year": number,
  "skills": {"skill_name": {"score": 1-10, "years": number, "recency": "YYYY-MM"}},
  "systems_built": ["payment_rails|fraud_detection|reco_engine|feature_store|data_lakehouse|streaming_platform|ml_serving|cdp|search_ranking|rag_production|forecasting"],
  "compliance": ["pci_dss|rbi_guidelines|sebi|hipaa|gdpr|dpdp|soc2|iso27001"],
  "scale_tb": number or null,
  "migration": {"from_stack": "string", "to_stack": "string", "role": "led|executed|assisted", "year": number} or null,
  "genai_production": {"in_production": true, "users_served": number or null, "eval_pipeline": bool, "depth": "rag_built|finetuned|api_integrated"} or null,
  "accolades": ["string"],
  "teaching": bool,
  "oss_maintainer": bool,
  "community_roles": ["string"],
  "notice_days": number,
  "industries": ["string"],
  "intl_exposure": "none|global_team|short_assignment|worked_abroad|currently_abroad",
  "hometown_pull": "strong_yes|open|no|already_home",
  "ctc_current": number (INR lakhs per year, estimate if unclear),
  "ctc_expected_min": number,
  "ctc_expected_max": number,
  "timezone_flex": "ist_only|eu_overlap_ok|us_overlap_ok|fully_flexible"
}"""

def build_bio(c: dict) -> str:
    skills = c.get("skills") or {}
    top4 = sorted(skills.items(), key=lambda kv: -kv[1].get("score",0))[:4]
    skill_str = ", ".join(f"{s} {v['score']}/10 ({v.get('years',1)}y)" for s,v in top4)
    sys_str   = ", ".join(s.replace("_"," ") for s in (c.get("systems_built") or []))
    m = c.get("migration")
    g = c.get("genai_production")
    bio = (f"{c['full_name']} is a {c.get('current_title','')} ({(c.get('role_family') or '').replace('_',' ')}) "
           f"at {c.get('current_company','')} ({c.get('company_tier','')}) in {c.get('location_city','')} with "
           f"{c.get('yoe',0)} years experience, {c.get('yoe_domain',0)} in data/AI. "
           f"Depth {c.get('depth_v3',0)}/10, tier {c.get('tier','')}, band {(c.get('overall_band') or '').upper()}, "
           f"composite {c.get('composite',0)}/100. Top skills: {skill_str}. Systems: {sys_str}."
           + (f" Led {m['from_stack']} → {m['to_stack']} migration ({m['role']}, {m['year']})." if m else "")
           + (f" GenAI in production: {g['depth']} serving {g.get('users_served') or 0:,} users." if g else "")
           + (f" Compliance: {', '.join(c.get('compliance') or [])}." if c.get("compliance") else "")
           + f" Education: {c.get('degree','')} from {c.get('institution','')} ({c.get('grad_year','')})."
           + f" Notice {c.get('notice_days',60)}d. Expects ₹{c.get('ctc_expected_min',0)}–{c.get('ctc_expected_max',0)}L."
           + f" Retention est. {int((c.get('retention_6m') or 0)*100)}%, ceiling {(c.get('ceiling') or '').replace('_',' ')}.")
    return bio

def upsert_candidate(payload: dict) -> dict:
    bio = build_bio(payload)
    payload["embedding_bio"] = bio
    payload["skill_names"]   = list((payload.get("skills") or {}).keys())
    ideal = (f"{payload.get('current_title','')} at a {(payload.get('fit_context') or 'growth_startup').replace('_',' ')} "
             f"{(payload.get('industries') or ['tech'])[0]} company")
    payload.setdefault("ideal_role", ideal)
    vec = list(embedder.embed([bio]))[0].tolist()
    point_id = abs(hash(payload["candidate_id"])) % (2**53)
    qdrant.upsert(COLLECTION, points=[PointStruct(id=point_id, vector=vec, payload=payload)], wait=True)
    return payload

def fetch_json(url: str, headers: dict = None) -> dict:
    req = urllib.request.Request(url, headers={
        "User-Agent": "GoldenDB/1.0", **(headers or {})})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())

# ── API routes ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    return (STATIC / "index.html").read_text(encoding="utf-8")

@app.get("/profile/{candidate_id}", response_class=HTMLResponse)
async def profile_page(candidate_id: str):
    return (STATIC / "profile.html").read_text(encoding="utf-8")

@app.get("/api/candidate/{candidate_id}")
async def get_candidate(candidate_id: str):
    res, _ = qdrant.scroll(
        COLLECTION,
        scroll_filter=Filter(must=[FieldCondition(key="candidate_id",
                                                   match=MatchValue(value=candidate_id))]),
        limit=1,
    )
    if not res:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return res[0].payload

class ChatRequest(BaseModel):
    messages: list

@app.post("/api/chat")
async def chat(req: ChatRequest):
    # build message history in OpenAI format
    messages = [{"role": "system", "content": SYSTEM}]
    for m in req.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_agentic_loop():
            try:
                while True:
                    stream = ai_client.chat.completions.create(
                        model=MODEL,
                        tools=TOOLS,
                        messages=messages,
                        stream=True,
                    )

                    # accumulate streaming response
                    text_acc = ""
                    tool_calls_acc = {}  # index -> {id, name, arguments}
                    finish_reason = None

                    for chunk in stream:
                        choice = chunk.choices[0] if chunk.choices else None
                        if not choice:
                            continue
                        finish_reason = choice.finish_reason or finish_reason
                        delta = choice.delta

                        if delta.content:
                            text_acc += delta.content
                            loop.call_soon_threadsafe(queue.put_nowait,
                                                      {"type": "text", "content": delta.content})

                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": tc.id or "",
                                        "name": tc.function.name if tc.function else "",
                                        "arguments": "",
                                    }
                                if tc.id:
                                    tool_calls_acc[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name:
                                        tool_calls_acc[idx]["name"] = tc.function.name
                                    if tc.function.arguments:
                                        tool_calls_acc[idx]["arguments"] += tc.function.arguments

                    if finish_reason != "tool_calls":
                        # done — no tool calls
                        if text_acc:
                            messages.append({"role": "assistant", "content": text_acc})
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
                        return

                    # build assistant message with tool_calls
                    tool_call_objs = []
                    for idx in sorted(tool_calls_acc.keys()):
                        tc = tool_calls_acc[idx]
                        tool_call_objs.append({
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        })
                    messages.append({
                        "role": "assistant",
                        "content": text_acc or None,
                        "tool_calls": tool_call_objs,
                    })

                    # execute each tool and emit event
                    for tc in tool_call_objs:
                        name = tc["function"]["name"]
                        try:
                            args = json.loads(tc["function"]["arguments"])
                        except Exception:
                            args = {}

                        loop.call_soon_threadsafe(queue.put_nowait, {
                            "type": "tool_call",
                            "name": name,
                            "input": args,
                        })

                        try:
                            result = EXECUTORS[name](args)
                        except Exception as e:
                            result = json.dumps({"error": str(e)})

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        })

            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": str(e)})
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

        t = threading.Thread(target=run_agentic_loop, daemon=True)
        t.start()

        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] == "done":
                break

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

# ── upload page ───────────────────────────────────────────────────────────────
@app.get("/upload", response_class=HTMLResponse)
async def upload_page():
    return (STATIC / "upload.html").read_text(encoding="utf-8")

@app.get("/guide", response_class=HTMLResponse)
async def docs_page():
    return (STATIC / "docs.html").read_text(encoding="utf-8")

# ── resume ingest ─────────────────────────────────────────────────────────────
@app.post("/api/ingest/resume")
async def ingest_resume(file: UploadFile = File(...)):
    data = await file.read()
    text = await asyncio.to_thread(pdf_to_text, data)
    if not text.strip():
        raise HTTPException(400, "Could not extract text from PDF")

    resp = await asyncio.to_thread(lambda: ai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": EXTRACT_PROMPT},
                  {"role": "user",   "content": f"Resume text:\n\n{text[:8000]}"}],
        response_format={"type": "json_object"},
        temperature=0,
    ))
    extracted = json.loads(resp.choices[0].message.content)
    candidate_id = await asyncio.to_thread(next_candidate_id)
    payload = {**extracted, "candidate_id": candidate_id, **compute_derived(extracted)}
    result  = await asyncio.to_thread(upsert_candidate, payload)
    return {"candidate_id": candidate_id, "candidate": result}

# ── github enrichment ─────────────────────────────────────────────────────────
@app.get("/api/enrich/github/{candidate_id}")
async def enrich_github(candidate_id: str, username: str):
    # fetch profile + top repos
    try:
        profile = await asyncio.to_thread(fetch_json, f"https://api.github.com/users/{username}")
        repos   = await asyncio.to_thread(fetch_json,
            f"https://api.github.com/users/{username}/repos?sort=stars&per_page=20&type=owner")
    except Exception as e:
        raise HTTPException(502, f"GitHub API error: {e}")

    languages = Counter()
    total_stars = 0
    for r in repos:
        if r.get("language"): languages[r["language"]] += 1
        total_stars += r.get("stargazers_count", 0)

    top_langs = [l for l, _ in languages.most_common(5)]
    oss = total_stars >= 50 or profile.get("followers", 0) >= 100
    community = []
    if profile.get("blog"): community.append("content_creator")
    bio_lower = (profile.get("bio") or "").lower()
    if any(w in bio_lower for w in ["speaker","conference","talk"]): community.append("conference_speaker")
    if any(w in bio_lower for w in ["organizer","meetup"]): community.append("meetup_organizer")

    # fetch existing candidate and merge
    res, _ = qdrant.scroll(COLLECTION, scroll_filter=Filter(must=[
        FieldCondition(key="candidate_id", match=MatchValue(value=candidate_id))]), limit=1)
    if not res:
        raise HTTPException(404, "Candidate not found")
    payload = res[0].payload
    existing_skills = payload.get("skills") or {}
    for lang in top_langs:
        if lang not in existing_skills:
            existing_skills[lang] = {"score": 6, "years": 1.0, "recency": time.strftime("%Y-%m")}
    payload.update({
        "oss_maintainer": oss,
        "github_stars": total_stars,
        "github_followers": profile.get("followers", 0),
        "skills": existing_skills,
        "skill_names": list(existing_skills.keys()),
        "community_roles": list(set((payload.get("community_roles") or []) + community)),
    })
    result = await asyncio.to_thread(upsert_candidate, payload)
    return {"candidate_id": candidate_id, "github_stars": total_stars,
            "top_languages": top_langs, "oss_maintainer": oss}

# ── kaggle enrichment ─────────────────────────────────────────────────────────
@app.get("/api/enrich/kaggle/{candidate_id}")
async def enrich_kaggle(candidate_id: str, username: str):
    try:
        page = await asyncio.to_thread(lambda: urllib.request.urlopen(
            urllib.request.Request(f"https://www.kaggle.com/{username}",
            headers={"User-Agent":"Mozilla/5.0"}), timeout=10).read().decode("utf-8","ignore"))
    except Exception as e:
        raise HTTPException(502, f"Kaggle fetch error: {e}")

    tier = "unknown"
    for t in ["Grandmaster","Master","Expert","Contributor","Novice"]:
        if t.lower() in page.lower(): tier = t; break

    res, _ = qdrant.scroll(COLLECTION, scroll_filter=Filter(must=[
        FieldCondition(key="candidate_id", match=MatchValue(value=candidate_id))]), limit=1)
    if not res: raise HTTPException(404, "Candidate not found")
    payload = res[0].payload
    accolades = list(set((payload.get("accolades") or []) + ([f"kaggle_{tier.lower()}"] if tier != "unknown" else [])))
    payload.update({"kaggle_tier": tier, "accolades": accolades})
    await asyncio.to_thread(upsert_candidate, payload)
    return {"candidate_id": candidate_id, "kaggle_username": username, "kaggle_tier": tier}

# ── stack overflow enrichment ─────────────────────────────────────────────────
@app.get("/api/enrich/stackoverflow/{candidate_id}")
async def enrich_stackoverflow(candidate_id: str, user_id: str):
    try:
        data = await asyncio.to_thread(fetch_json,
            f"https://api.stackexchange.com/2.3/users/{user_id}?site=stackoverflow&filter=default")
        tags = await asyncio.to_thread(fetch_json,
            f"https://api.stackexchange.com/2.3/users/{user_id}/top-tags?site=stackoverflow&pagesize=10")
    except Exception as e:
        raise HTTPException(502, f"Stack Overflow API error: {e}")

    items = data.get("items", [])
    if not items: raise HTTPException(404, "Stack Overflow user not found")
    u = items[0]
    reputation = u.get("reputation", 0)
    top_tags = [t["tag_name"] for t in tags.get("items", [])]

    res, _ = qdrant.scroll(COLLECTION, scroll_filter=Filter(must=[
        FieldCondition(key="candidate_id", match=MatchValue(value=candidate_id))]), limit=1)
    if not res: raise HTTPException(404, "Candidate not found")
    payload = res[0].payload
    existing_skills = payload.get("skills") or {}
    for tag in top_tags[:5]:
        if tag not in existing_skills:
            existing_skills[tag] = {"score": 6, "years": 1.0, "recency": time.strftime("%Y-%m")}
    payload.update({
        "stackoverflow_reputation": reputation,
        "skills": existing_skills,
        "skill_names": list(existing_skills.keys()),
    })
    if reputation >= 10000:
        payload["accolades"] = list(set((payload.get("accolades") or []) + ["stackoverflow_top_contributor"]))
    await asyncio.to_thread(upsert_candidate, payload)
    return {"candidate_id": candidate_id, "reputation": reputation, "top_tags": top_tags}

# ── huggingface enrichment ────────────────────────────────────────────────────
@app.get("/api/enrich/huggingface/{candidate_id}")
async def enrich_huggingface(candidate_id: str, username: str):
    try:
        models = await asyncio.to_thread(fetch_json,
            f"https://huggingface.co/api/models?author={username}&limit=20&full=false")
    except Exception as e:
        raise HTTPException(502, f"HuggingFace API error: {e}")

    total_downloads = sum(m.get("downloads", 0) for m in models)
    total_likes     = sum(m.get("likes", 0) for m in models)
    model_count     = len(models)

    res, _ = qdrant.scroll(COLLECTION, scroll_filter=Filter(must=[
        FieldCondition(key="candidate_id", match=MatchValue(value=candidate_id))]), limit=1)
    if not res: raise HTTPException(404, "Candidate not found")
    payload = res[0].payload

    genai = payload.get("genai_production") or {}
    if model_count > 0:
        genai.update({"in_production": True,
                      "users_served": total_downloads,
                      "eval_pipeline": total_likes > 10,
                      "depth": "finetuned" if total_downloads > 1000 else "rag_built"})
    payload.update({
        "genai_production": genai if model_count > 0 else payload.get("genai_production"),
        "hf_model_count": model_count,
        "hf_downloads": total_downloads,
        "oss_maintainer": payload.get("oss_maintainer") or model_count > 0,
    })
    await asyncio.to_thread(upsert_candidate, payload)
    return {"candidate_id": candidate_id, "model_count": model_count,
            "total_downloads": total_downloads, "total_likes": total_likes}

# ── list all candidates ───────────────────────────────────────────────────────
@app.get("/api/candidates")
async def list_candidates(limit: int = 100):
    res, _ = qdrant.scroll(COLLECTION, limit=limit,
                           with_payload=CARD_FIELDS + ["pipeline_stage"])
    return [p.payload for p in res]

# ── github + apollo sourcing pipeline ─────────────────────────────────────────
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

INDIA_CITIES = [
    "bengaluru","bangalore","mumbai","hyderabad","pune","delhi","chennai",
    "noida","gurgaon","gurugram","kolkata","ahmedabad","jaipur","kochi",
    "india","new delhi","navi mumbai","thane","indore","bhopal",
]

QUERY_PARSE_PROMPT = """\
Extract GitHub search parameters from a recruiting query. Return ONLY valid JSON:
{
  "skills": ["python","kafka"],
  "locations": ["Bengaluru","Mumbai"],
  "role_keywords": ["ML Engineer","data engineer"],
  "min_followers": 30
}
Skills should be programming languages or tools (lowercase).
Locations should be Indian cities.
Role keywords are job title words to match in bios."""

def gh_headers() -> dict:
    h = {"User-Agent": "GoldenDB/1.0", "Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

def gh_search(query: str, per_page: int = 100) -> list:
    url = f"https://api.github.com/search/users?q={urllib.parse.quote(query)}&per_page={per_page}"
    return fetch_json(url, gh_headers()).get("items", [])

def gh_profile(login: str) -> dict:
    return fetch_json(f"https://api.github.com/users/{login}", gh_headers())

def gh_repos(login: str) -> list:
    try:
        return fetch_json(
            f"https://api.github.com/users/{login}/repos?sort=stars&per_page=20&type=owner",
            gh_headers())
    except Exception:
        return []

def score_profile(profile: dict, repos: list, skills: list) -> int:
    score = 0
    followers = profile.get("followers", 0)
    if followers > 500:   score += 5
    elif followers > 100: score += 3
    elif followers > 30:  score += 1

    total_stars = sum(r.get("stargazers_count", 0) for r in repos)
    if total_stars > 500:   score += 4
    elif total_stars > 100: score += 3
    elif total_stars > 30:  score += 1

    location = (profile.get("location") or "").lower()
    if any(c in location for c in INDIA_CITIES): score += 3

    if profile.get("company"):  score += 2
    if profile.get("email"):    score += 2  # has email → skip Apollo reveal
    if profile.get("bio"):      score += 1

    languages = [r.get("language", "").lower() for r in repos if r.get("language")]
    score += min(sum(1 for s in skills if s.lower() in languages) * 2, 4)

    # recent activity
    if any((r.get("pushed_at") or "") > "2025-01-01" for r in repos): score += 2

    return score

def mock_apollo(profile: dict) -> dict:
    """Mock Apollo enrichment — replace with real Apollo API call later."""
    name  = (profile.get("name") or profile.get("login", "")).strip()
    parts = name.split()
    first = parts[0] if parts else profile.get("login", "user")
    last  = parts[-1] if len(parts) > 1 else ""
    company = (profile.get("company") or "").strip().lstrip("@")
    domain  = f"{company.lower().replace(' ','').replace('.com','')}.com" if company else "gmail.com"

    if last:
        email = f"{first.lower()}.{last.lower()}@{domain}"
    else:
        email = f"{first.lower()}@{domain}"

    # deterministic fake mobile from login hash so same person = same number
    seed  = abs(hash(profile.get("login", "x")))
    prefixes = ["98","97","96","95","94","91","90","89","88","87","86","85","84","83","82","81","80","79","78","77","76","75","74","73","72","70"]
    prefix = prefixes[seed % len(prefixes)]
    digits = "".join(str((seed >> i) % 10) for i in range(8))
    phone  = f"+91 {prefix}{digits}"

    slug = f"{first.lower()}-{last.lower()}" if last else first.lower()

    return {
        "first_name": first,
        "last_name": last,
        "email": profile.get("email") or email,
        "email_status": "verified" if profile.get("email") else "guessed",
        "phone": phone,
        "linkedin_url": f"https://linkedin.com/in/{slug}",
        "organization": company or "Unknown",
        "apollo_source": "mock",
    }

class SourceRequest(BaseModel):
    query: str
    limit: int = 15

@app.get("/source", response_class=HTMLResponse)
async def source_page():
    return (STATIC / "source.html").read_text(encoding="utf-8")

@app.post("/api/source/search")
async def source_search(req: SourceRequest):
    async def generate():
        # 1. parse query with GPT
        yield f"data: {json.dumps({'type':'status','msg':'Parsing query with GPT…'})}\n\n"
        try:
            pr = await asyncio.to_thread(lambda: ai_client.chat.completions.create(
                model=MODEL,
                messages=[{"role":"system","content":QUERY_PARSE_PROMPT},
                          {"role":"user","content":req.query}],
                response_format={"type":"json_object"},
                temperature=0,
            ))
            params = json.loads(pr.choices[0].message.content)
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':str(e)})}\n\n"; return

        skills   = params.get("skills", [])
        locs     = params.get("locations", [])
        keywords = params.get("role_keywords", [])

        # build github query
        parts = []
        if locs:      parts.append(f"location:{locs[0]}")
        if keywords:  parts.append(" ".join(keywords[:2]))
        if skills:    parts.append(" ".join(skills[:3]))
        parts.append("followers:>20")
        gh_q = " ".join(parts)

        yield f"data: {json.dumps({'type':'status','msg':f'Searching GitHub: {gh_q}'})}\n\n"

        # 2. search github
        try:
            items = await asyncio.to_thread(gh_search, gh_q, 100)
        except Exception as e:
            yield f"data: {json.dumps({'type':'error','msg':f'GitHub error: {e}'})}\n\n"; return

        yield f"data: {json.dumps({'type':'status','msg':f'Found {len(items)} profiles — fetching details…'})}\n\n"

        # 3. fetch profiles in parallel batches of 10
        logins  = [i["login"] for i in items[:50]]
        profiles = []
        for i in range(0, len(logins), 10):
            batch   = logins[i:i+10]
            results = await asyncio.gather(
                *[asyncio.to_thread(gh_profile, l) for l in batch],
                return_exceptions=True)
            profiles.extend(r for r in results if not isinstance(r, Exception))
            yield f"data: {json.dumps({'type':'progress','fetched':len(profiles),'total':len(logins)})}\n\n"

        yield f"data: {json.dumps({'type':'status','msg':f'Fetched {len(profiles)} profiles — scoring…'})}\n\n"

        # 4. fetch repos + score (only top 30 by followers to save API calls)
        profiles.sort(key=lambda p: -(p.get("followers") or 0))
        scored = []
        for profile in profiles[:30]:
            repos = await asyncio.to_thread(gh_repos, profile["login"])
            s     = score_profile(profile, repos, skills)
            langs = [r.get("language") for r in repos if r.get("language")]
            lang_counts = Counter(langs)
            scored.append({
                "profile": profile,
                "repos": repos,
                "score": s,
                "top_languages": [l for l,_ in lang_counts.most_common(5)],
                "total_stars": sum(r.get("stargazers_count",0) for r in repos),
            })

        scored.sort(key=lambda x: -x["score"])
        top = scored[:req.limit]

        yield f"data: {json.dumps({'type':'status','msg':f'Top {len(top)} selected — enriching via Apollo…'})}\n\n"

        # 5. mock apollo + stream each candidate
        for rank, item in enumerate(top, 1):
            p       = item["profile"]
            apollo  = mock_apollo(p)
            location_raw = (p.get("location") or "India")
            # best-guess city from location string
            city = next((c.title() for c in INDIA_CITIES if c in location_raw.lower()), location_raw)

            candidate = {
                "github_login":    p.get("login"),
                "full_name":       (p.get("name") or p.get("login","")).strip(),
                "bio":             p.get("bio",""),
                "current_company": apollo["organization"],
                "location_city":   city,
                "avatar_url":      p.get("avatar_url"),
                "github_url":      p.get("html_url"),
                "github_followers":p.get("followers",0),
                "github_stars":    item["total_stars"],
                "top_languages":   item["top_languages"],
                "gh_score":        item["score"],
                "rank":            rank,
                # apollo contact
                "email":           apollo["email"],
                "email_status":    apollo["email_status"],
                "phone":           apollo["phone"],
                "linkedin_url":    apollo["linkedin_url"],
                "apollo_source":   apollo["apollo_source"],
            }
            yield f"data: {json.dumps({'type':'candidate','candidate':candidate})}\n\n"

        yield f"data: {json.dumps({'type':'done','total':len(top)})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

# ── sourcing chat ─────────────────────────────────────────────────────────────
_source_cache: dict = {}   # github_login → candidate dict (session memory)

def _fetch_repos_for_profile(profile: dict):
    return profile, gh_repos(profile["login"])

def exec_source_candidates(args: dict) -> str:
    from concurrent.futures import ThreadPoolExecutor
    location      = args.get("location", "")
    skills        = args.get("skills", [])
    role_keywords = args.get("role_keywords", [])
    min_followers = args.get("min_followers", 20)
    min_score     = args.get("min_score", 5)
    limit         = min(int(args.get("limit", 10)), 15)

    parts = []
    if location:      parts.append(f"location:{location}")
    if role_keywords: parts.append(" ".join(role_keywords[:2]))
    if skills:        parts.append(" ".join(skills[:3]))
    parts.append(f"followers:>{min_followers}")
    gh_q = " ".join(parts)

    items   = gh_search(gh_q, 100)
    logins  = [i["login"] for i in items[:50]]

    with ThreadPoolExecutor(max_workers=10) as ex:
        profiles = [p for p in ex.map(gh_profile, logins) if isinstance(p, dict)]

    profiles.sort(key=lambda p: -(p.get("followers") or 0))

    with ThreadPoolExecutor(max_workers=8) as ex:
        profile_repos = list(ex.map(_fetch_repos_for_profile, profiles[:25]))

    scored = []
    for profile, repos in profile_repos:
        s = score_profile(profile, repos, skills)
        if s < min_score:
            continue
        lang_counts = Counter(r.get("language") for r in repos if r.get("language"))
        scored.append({
            "profile":        profile,
            "score":          s,
            "top_languages":  [l for l, _ in lang_counts.most_common(5)],
            "total_stars":    sum(r.get("stargazers_count", 0) for r in repos),
        })

    scored.sort(key=lambda x: -x["score"])
    results = []
    for item in scored[:limit]:
        p      = item["profile"]
        apollo = mock_apollo(p)
        loc    = p.get("location") or "India"
        city   = next((c.title() for c in INDIA_CITIES if c in loc.lower()), loc)
        cand   = {
            "github_login":    p["login"],
            "full_name":       (p.get("name") or p["login"]).strip(),
            "bio":             p.get("bio") or "",
            "current_company": apollo["organization"],
            "location_city":   city,
            "github_url":      p.get("html_url"),
            "github_followers":p.get("followers", 0),
            "github_stars":    item["total_stars"],
            "top_languages":   item["top_languages"],
            "gh_score":        item["score"],
            "email":           apollo["email"],
            "email_status":    apollo["email_status"],
            "phone":           apollo["phone"],
            "linkedin_url":    apollo["linkedin_url"],
            "apollo_source":   apollo["apollo_source"],
            "avatar_url":      p.get("avatar_url"),
        }
        _source_cache[p["login"]] = cand
        results.append(cand)

    return json.dumps(results, ensure_ascii=False)

def _build_sourced_payload(c: dict) -> dict:
    candidate_id = next_candidate_id()
    skills_dict  = {lang: {"score": 6, "years": 1, "recency": time.strftime("%Y-%m")}
                    for lang in (c.get("top_languages") or [])[:6]}
    derived = compute_derived({"yoe": 0, "yoe_domain": 0, "skills": skills_dict})
    return {
        "candidate_id":     candidate_id,
        "full_name":        c.get("full_name", "Unknown"),
        "current_title":    (c.get("bio") or "").split("\n")[0][:60] or "Engineer",
        "current_company":  c.get("current_company", "Unknown"),
        "company_tier":     "unknown",
        "role_family":      "data_engineering",
        "location_city":    c.get("location_city", "India"),
        "hometown_city":    None, "hometown_pull": "open",
        "yoe": 0, "yoe_domain": 0,
        "degree": None, "institution": None, "institution_tier": None, "grad_year": None,
        "skills": skills_dict, "systems_built": [], "compliance": [],
        "scale_tb": None, "genai_production": None, "migration": None,
        "accolades": [], "teaching": False,
        "oss_maintainer": (c.get("github_stars") or 0) >= 50,
        "community_roles": [], "notice_days": 60, "industries": [],
        "intl_exposure": "none",
        "ctc_current": None, "ctc_expected_min": None, "ctc_expected_max": None,
        "timezone_flex": "ist_only",
        "email":           c.get("email"),
        "email_status":    c.get("email_status"),
        "phone":           c.get("phone"),
        "linkedin_url":    c.get("linkedin_url"),
        "github_login":    c.get("github_login"),
        "github_url":      c.get("github_url"),
        "github_followers":c.get("github_followers", 0),
        "github_stars":    c.get("github_stars", 0),
        "pipeline_stage":  "sourced",
        "signal_version":  1,
        "source":          "github_apollo",
        **derived,
    }

def exec_save_candidates(args: dict) -> str:
    saved = []
    for login in args.get("github_logins", []):
        if login not in _source_cache:
            saved.append({"login": login, "error": "not found — run source_candidates first"})
            continue
        payload = _build_sourced_payload(_source_cache[login])
        upsert_candidate(payload)
        saved.append({"login": login, "candidate_id": payload["candidate_id"], "status": "saved"})
    return json.dumps({"saved": saved}, ensure_ascii=False)

SOURCING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "source_candidates",
            "description": (
                "Search GitHub for AI/data engineers matching criteria. "
                "Scores profiles and enriches with Apollo contact data (email, phone, LinkedIn). "
                "Call this for any find/source/search request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":         {"type": "string", "description": "What you're looking for in plain English"},
                    "location":      {"type": "string", "description": "Indian city e.g. Bengaluru, Mumbai, Hyderabad"},
                    "skills":        {"type": "array",  "items": {"type": "string"}, "description": "Languages or tools e.g. python, kafka, pytorch"},
                    "role_keywords": {"type": "array",  "items": {"type": "string"}, "description": "Title keywords e.g. ML Engineer, data engineer"},
                    "min_followers": {"type": "integer","description": "Minimum GitHub followers filter (default 20)"},
                    "min_score":     {"type": "integer","description": "Minimum quality score to include (default 5)"},
                    "limit":         {"type": "integer","description": "How many to return, max 15 (default 10)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_candidates",
            "description": "Save one or more sourced candidates into GoldenDB as stage-1 entries ready for outreach pipeline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "github_logins": {"type": "array", "items": {"type": "string"},
                                     "description": "GitHub usernames to save"},
                },
                "required": ["github_logins"],
            },
        },
    },
]

SOURCING_SYSTEM = """You are GoldenDB Sourcing Copilot — an internal tool for the GoldenDB team to discover and pipeline AI/data talent from GitHub.

You have two tools:
- source_candidates: searches GitHub and enriches with Apollo contact data
- save_candidates: saves selected people into GoldenDB as stage-1 candidates

Rules:
- ALWAYS call source_candidates before answering any search/find/source request — never invent people.
- Format each candidate as:
  **[Full Name](github_url)** `@login` — Company · City · Score: X
  🛠 Languages: Python, Kafka… · ★ {stars} · {followers} followers
  ✉ email ({verified/guessed}) · 📱 phone · [LinkedIn →](url)
  > bio one-liner

- After presenting results always ask: "Want me to save any of these to GoldenDB?"
- When user says save (top 3, all, specific names) — call save_candidates with their logins.
- On save, confirm each with their new GDB-XXXX ID.
- For refinements ("only score 10+", "remove FAANG", "show more from Mumbai") — call source_candidates again with tighter or adjusted params.
- Warn when email_status is "guessed" — needs verification before outreach.
- Candidates saved here enter stage 1 (sourced). They need resume → enrichment → panel before becoming client-ready.
- GitHub stars/followers are a proxy for depth — not a guarantee. Always note this."""

SOURCING_EXECUTORS = {
    "source_candidates": exec_source_candidates,
    "save_candidates":   exec_save_candidates,
}

@app.get("/source-chat", response_class=HTMLResponse)
async def source_chat_page():
    return (STATIC / "source-chat.html").read_text(encoding="utf-8")

@app.post("/api/source-chat")
async def source_chat(req: ChatRequest):
    messages = [{"role": "system", "content": SOURCING_SYSTEM}]
    for m in req.messages:
        messages.append({"role": m["role"], "content": m["content"]})

    async def generate():
        loop  = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def run_loop():
            try:
                while True:
                    stream = ai_client.chat.completions.create(
                        model=MODEL, tools=SOURCING_TOOLS,
                        messages=messages, stream=True,
                    )
                    text_acc      = ""
                    tool_calls_acc = {}
                    finish_reason  = None

                    for chunk in stream:
                        choice = chunk.choices[0] if chunk.choices else None
                        if not choice: continue
                        finish_reason = choice.finish_reason or finish_reason
                        delta = choice.delta
                        if delta.content:
                            text_acc += delta.content
                            loop.call_soon_threadsafe(queue.put_nowait,
                                {"type": "text", "content": delta.content})
                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                idx = tc.index
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {"id": tc.id or "", "name": tc.function.name if tc.function else "", "arguments": ""}
                                if tc.id: tool_calls_acc[idx]["id"] = tc.id
                                if tc.function:
                                    if tc.function.name: tool_calls_acc[idx]["name"] = tc.function.name
                                    if tc.function.arguments: tool_calls_acc[idx]["arguments"] += tc.function.arguments

                    if finish_reason != "tool_calls":
                        if text_acc: messages.append({"role": "assistant", "content": text_acc})
                        loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})
                        return

                    tool_call_objs = [{"id": tool_calls_acc[i]["id"], "type": "function",
                                       "function": {"name": tool_calls_acc[i]["name"],
                                                    "arguments": tool_calls_acc[i]["arguments"]}}
                                      for i in sorted(tool_calls_acc)]
                    messages.append({"role": "assistant", "content": text_acc or None,
                                     "tool_calls": tool_call_objs})

                    for tc in tool_call_objs:
                        name = tc["function"]["name"]
                        try:    args = json.loads(tc["function"]["arguments"])
                        except: args = {}
                        loop.call_soon_threadsafe(queue.put_nowait,
                            {"type": "tool_call", "name": name, "input": args})
                        try:    result = SOURCING_EXECUTORS[name](args)
                        except Exception as e: result = json.dumps({"error": str(e)})
                        messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})

            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "content": str(e)})
                loop.call_soon_threadsafe(queue.put_nowait, {"type": "done"})

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            if event["type"] == "done": break

    return StreamingResponse(generate(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.post("/api/source/add")
async def source_add(body: dict):
    c          = body
    candidate_id = await asyncio.to_thread(next_candidate_id)
    skills_dict  = {lang: {"score": 6, "years": 1, "recency": time.strftime("%Y-%m")}
                    for lang in (c.get("top_languages") or [])[:6]}
    derived = compute_derived({"yoe": 0, "yoe_domain": 0, "skills": skills_dict})

    payload = {
        "candidate_id":     candidate_id,
        "full_name":        c.get("full_name", "Unknown"),
        "current_title":    (c.get("bio") or "").split("\n")[0][:60] or "Engineer",
        "current_company":  c.get("current_company", "Unknown"),
        "company_tier":     "unknown",
        "role_family":      "data_engineering",   # default — updated at resume stage
        "location_city":    c.get("location_city", "India"),
        "hometown_city":    None,
        "hometown_pull":    "open",
        "yoe":              0,
        "yoe_domain":       0,
        "degree":           None,
        "institution":      None,
        "institution_tier": None,
        "grad_year":        None,
        "skills":           skills_dict,
        "systems_built":    [],
        "compliance":       [],
        "scale_tb":         None,
        "genai_production": None,
        "migration":        None,
        "accolades":        [],
        "teaching":         False,
        "oss_maintainer":   (c.get("github_stars",0) or 0) >= 50,
        "community_roles":  [],
        "notice_days":      60,
        "industries":       [],
        "intl_exposure":    "none",
        "ctc_current":      None,
        "ctc_expected_min": None,
        "ctc_expected_max": None,
        "timezone_flex":    "ist_only",
        # contact (from Apollo)
        "email":            c.get("email"),
        "email_status":     c.get("email_status"),
        "phone":            c.get("phone"),
        "linkedin_url":     c.get("linkedin_url"),
        # github signals
        "github_login":     c.get("github_login"),
        "github_url":       c.get("github_url"),
        "github_followers": c.get("github_followers", 0),
        "github_stars":     c.get("github_stars", 0),
        # pipeline
        "pipeline_stage":   "sourced",
        "signal_version":   1,
        "source":           "github_apollo",
        **derived,
    }
    result = await asyncio.to_thread(upsert_candidate, payload)
    return {"candidate_id": candidate_id, "status": "added"}

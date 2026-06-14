# -*- coding: utf-8 -*-
"""Generate 100 fake GoldenDB candidates — all panel-complete (fully verified).

Every candidate carries the full v2.3 schema payload: structured filterable
fields + a fixed-template embedding_bio (the text that gets vectorised).
Deterministic via seed.
"""
import json
import random

random.seed(42)

FIRST = ["Ananya","Arjun","Priya","Rahul","Sneha","Vikram","Divya","Karthik","Meera","Aditya",
         "Ishita","Rohan","Kavya","Siddharth","Neha","Varun","Pooja","Aakash","Shruti","Nikhil",
         "Riya","Manish","Tanvi","Deepak","Sanya","Harsh","Anjali","Gaurav","Swati","Pranav",
         "Lakshmi","Abhishek","Nandini","Rajat","Aishwarya","Suresh","Ritika","Mohit","Bhavana","Tarun"]
LAST = ["Iyer","Sharma","Patel","Reddy","Nair","Gupta","Krishnan","Banerjee","Mehta","Singh",
        "Rao","Joshi","Kulkarni","Das","Menon","Agarwal","Pillai","Chatterjee","Verma","Desai",
        "Hegde","Mishra","Sengupta","Bhat","Choudhury"]

CITIES = (["Bengaluru"]*35 + ["Hyderabad"]*15 + ["Pune"]*12 + ["Mumbai"]*10 +
          ["Delhi NCR"]*10 + ["Chennai"]*8 + ["Kolkata"]*5 + ["Kochi"]*5)
HOMETOWNS = ["Coimbatore","Indore","Jaipur","Kochi","Nagpur","Bhubaneswar","Lucknow","Madurai",
             "Visakhapatnam","Surat","Trichy","Mysuru","Mangaluru","Vadodara","Patna",
             "Bengaluru","Hyderabad","Pune","Mumbai","Chennai","Delhi NCR","Kolkata"]

ROLE_FAMILIES = (["data_engineering"]*28 + ["ml_engineering"]*18 + ["data_science"]*15 +
                 ["analytics"]*12 + ["mlops"]*8 + ["data_platform"]*8 +
                 ["ai_research"]*5 + ["genai_engineering"]*6)

COMPANIES = {
    "tier_1_global": ["Google","Microsoft","Amazon","Uber","Stripe","LinkedIn","Atlassian"],
    "tier_1_india":  ["Flipkart","Razorpay","Swiggy","PhonePe","CRED","Zepto","Meesho","Zomato","Paytm"],
    "tier_2":        ["Infosys","TCS","Wipro","Cognizant","LTIMindtree","Tech Mahindra","Mphasis"],
    "startup_funded":["Hasura","Postman","Chargebee","Darwinbox","Yellow.ai","Observe.AI","Atlan"],
    "startup_early": ["StealthAI Labs","DataKona","Vecturo","PipelineIQ","Lakemint"],
}
TIER_W = [("tier_1_india",38),("tier_1_global",12),("tier_2",22),("startup_funded",20),("startup_early",8)]

SKILLS = {
    "data_engineering": ["Spark","Airflow","dbt","Kafka","Python","SQL","Databricks","Snowflake","AWS","Flink"],
    "ml_engineering":   ["PyTorch","Python","MLflow","Kubernetes","TensorFlow","Ray","Feature Stores","AWS","Triton","ONNX"],
    "data_science":     ["Python","SQL","scikit-learn","XGBoost","Statistics","A/B Testing","PyTorch","Causal Inference"],
    "analytics":        ["SQL","Tableau","Power BI","Python","dbt","Looker","Excel","Metabase"],
    "mlops":            ["Kubernetes","MLflow","Docker","Terraform","Kubeflow","AWS","CI/CD","Prometheus"],
    "data_platform":    ["Kafka","Kubernetes","Spark","Terraform","AWS","Iceberg","Trino","Airflow"],
    "ai_research":      ["PyTorch","JAX","Transformers","CUDA","Python","Distributed Training"],
    "genai_engineering":["LLM","RAG","LangChain","Vector DBs","PyTorch","Prompt Engineering","Evals","Python"],
}
SYSTEMS = {
    "data_engineering": ["data_lakehouse","streaming_platform","cdp","payment_rails","fraud_detection"],
    "ml_engineering":   ["ml_serving","feature_store","reco_engine","fraud_detection","search_ranking"],
    "data_science":     ["forecasting","reco_engine","fraud_detection","search_ranking"],
    "analytics":        ["cdp","forecasting","data_lakehouse"],
    "mlops":            ["ml_serving","feature_store","streaming_platform"],
    "data_platform":    ["data_lakehouse","streaming_platform","feature_store"],
    "ai_research":      ["ml_serving","search_ranking","rag_production"],
    "genai_engineering":["rag_production","search_ranking","ml_serving"],
}
MIGRATIONS = [("hadoop","databricks"),("on_prem","aws"),("airflow1","airflow2"),("redshift","snowflake"),
              ("hive","iceberg"),("batch_etl","streaming"),("sas","python"),("monolith","microservices")]
INSTITUTIONS = {
    "iit_iisc_iim":   ["IIT Bombay","IIT Delhi","IIT Madras","IIT Kanpur","IISc Bangalore","IIT Kharagpur"],
    "nit_bits_top":   ["NIT Trichy","NIT Surathkal","BITS Pilani","IIIT Hyderabad","NIT Warangal","DTU"],
    "tier_1_private": ["VIT Vellore","Manipal Institute of Technology","SRM Chennai","PES University"],
    "tier_2_private": ["Amity University","SIT Pune","RV College","BMS College","CUSAT"],
    "tier_3":         ["State Engineering College","Regional Institute of Technology"],
}
INST_W = [("iit_iisc_iim",12),("nit_bits_top",22),("tier_1_private",30),("tier_2_private",26),("tier_3",10)]
ACCOLADES = ["gate_rank top_500","gate_rank top_2000","jee_rank_band top_5000","gold_medalist",
             "hackathon_win","scholarship","olympiad","best_thesis"]
COMPLIANCE = ["pci_dss","rbi_guidelines","sebi","hipaa","gdpr","dpdp","soc2","iso27001"]
INDUSTRY = ["fintech","ecommerce","saas","enterprise_tech","healthtech","logistics","edtech","gaming"]
MOTIV = ["growth_driven","comp_driven","impact_driven","stability_seeking","culture_driven"]
FIT = ["growth_startup","scale_up","early_startup","large_enterprise","mnc"]
ARCHETYPE = ["zero_to_one_builder","scaler","optimizer","turnaround"]
CEILING = ["plateau_likely","on_track","on_track","high_runway","high_runway","exceptional_trajectory"]

def wpick(pairs):
    r = random.uniform(0, sum(w for _, w in pairs))
    acc = 0
    for v, w in pairs:
        acc += w
        if r <= acc:
            return v
    return pairs[-1][0]

def make_candidate(i):
    fam = random.choice(ROLE_FAMILIES)
    name = f"{random.choice(FIRST)} {random.choice(LAST)}"
    city = random.choice(CITIES)
    hometown = random.choice(HOMETOWNS)
    yoe = round(random.uniform(2.5, 16.0), 1)
    yoe_domain = round(yoe * random.uniform(0.6, 1.0), 1)

    # tier from yoe band + depth
    depth = round(random.uniform(4.5, 9.5), 1)
    if yoe < 3.5: tier = "T2"
    elif yoe < 6: tier = "T3"
    elif yoe < 9: tier = "T4"
    elif yoe < 13: tier = "T5" if depth >= 7.5 else "T4"
    else: tier = "T6" if depth >= 8.8 else "T5"

    ctier = wpick(TIER_W)
    company = random.choice(COMPANIES[ctier])
    titles = {"T2":"Data Engineer","T3":"Senior {}","T4":"Staff {} / Tech Lead","T5":"Principal {} / Lead","T6":"Head of Data / Director"}
    base_title = {"data_engineering":"Data Engineer","ml_engineering":"ML Engineer","data_science":"Data Scientist",
                  "analytics":"Analytics Engineer","mlops":"MLOps Engineer","data_platform":"Platform Engineer",
                  "ai_research":"Research Engineer","genai_engineering":"GenAI Engineer"}[fam]
    title = titles[tier].format(base_title) if "{}" in titles[tier] else (titles[tier] if tier in ("T2","T6") else base_title)

    # skills
    pool = SKILLS[fam]
    n_sk = random.randint(5, min(8, len(pool)))
    skills = {}
    for s in random.sample(pool, n_sk):
        skills[s] = {"score": random.randint(5, 9), "years": round(random.uniform(1.0, min(yoe, 9.0)), 1),
                     "recency": random.choice(["2026-05","2026-04","2026-03","2026-01","2025-10","2025-06"])}
    top_skills = sorted(skills.items(), key=lambda kv: -kv[1]["score"])[:4]

    systems = random.sample(SYSTEMS[fam], random.randint(1, min(3, len(SYSTEMS[fam]))))
    has_genai = "rag_production" in systems or fam == "genai_engineering" or random.random() < 0.25
    genai = ({"in_production": True, "users_served": random.choice([5000, 20000, 50000, 200000, 1000000]),
              "eval_pipeline": random.random() < 0.7, "depth": random.choice(["rag_built","finetuned","api_integrated"])}
             if has_genai else None)
    migration = (lambda m: {"from_stack": m[0], "to_stack": m[1],
                            "role": wpick([("led",4),("executed",5),("assisted",2)]),
                            "year": random.randint(2021, 2025)})(random.choice(MIGRATIONS)) if random.random() < 0.55 else None
    compliance = random.sample(COMPLIANCE, random.randint(1, 3)) if random.random() < 0.5 else []
    scale_tb = random.choice([5, 20, 60, 120, 300, 800]) if fam in ("data_engineering","data_platform","mlops") else random.choice([1, 5, 20, 60])

    inst_tier = wpick(INST_W)
    institution = random.choice(INSTITUTIONS[inst_tier])
    degree = wpick([("bachelors",62),("masters",30),("phd",5),("self_taught",3)])
    grad_year = 2026 - int(yoe) - random.randint(0, 2)
    accolades = random.sample(ACCOLADES, random.randint(1, 2)) if random.random() < 0.45 else []
    teaching = random.random() < 0.25
    oss = random.random() < 0.15
    community = random.sample(["meetup_organizer","conference_speaker","content_creator"], 1) if random.random() < 0.2 else []

    # geo & timing
    pull = ("already_home" if hometown == city else wpick([("strong_yes",25),("open",35),("no",40)]))
    notice = random.choice([0, 15, 30, 30, 45, 60, 60, 90, 90])
    tz = wpick([("ist_only",40),("eu_overlap_ok",30),("us_overlap_ok",20),("fully_flexible",10)])
    base_ctc = {"T2": 14, "T3": 25, "T4": 42, "T5": 65, "T6": 95}[tier] * random.uniform(0.8, 1.25)
    ctc_cur = round(base_ctc)
    ctc_min, ctc_max = round(ctc_cur * 1.25), round(ctc_cur * 1.5)
    renege_risk = wpick([("low",75),("medium",18),("elevated",7)])
    likelihood_move = round(random.uniform(0.3, 0.9), 2)

    # behavioural / panel
    leadership = {"T2":"ic_junior","T3":"ic_senior","T4":random.choice(["ic_senior","tech_lead"]),
                  "T5":random.choice(["tech_lead","manager"]),"T6":random.choice(["senior_manager","director"])}[tier]
    ownership = wpick([("full_owner",30),("co_owner",30),("significant_contributor",30),("team_member",10)])
    resilience = round(random.uniform(5.0, 9.5), 1)
    architecture = round(random.uniform(4.0, 9.5), 1) if tier in ("T4","T5","T6") else None
    communication = round(random.uniform(2.8, 4.9), 1)
    resume_panel_delta = round(random.uniform(-1.8, 2.2), 1)
    growth_velocity = round(random.uniform(-0.3, 2.0), 1)
    stability = round(random.uniform(5.0, 9.5), 1)
    retention_6m = round(min(0.97, 0.45 + stability * 0.04 + resilience * 0.015 + random.uniform(0, 0.08)), 2)
    retention_12m = round(max(0.3, retention_6m - random.uniform(0.05, 0.14)), 2)
    motivation = random.choice(MOTIV)
    fit = random.choice(FIT)
    archetype = random.choice(ARCHETYPE)
    chaos = wpick([("thrives_in_chaos",25),("adapts",50),("needs_structure",25)])
    mgmt_refused = random.choice([0, 0, 0, 1, 2]) if leadership.startswith("ic") else 0
    exec_exposure = wpick([("none",35),("team_level",35),("cxo_occasional",22),("board_regular",8)])
    intl = wpick([("none",45),("global_team",35),("short_assignment",12),("worked_abroad",8)])
    aspiration = ("ic_forever" if mgmt_refused >= 1 else
                  wpick([("ic_forever",20),("open_to_management",50),("actively_seeking_leadership",20),("already_managing",10)]))
    red_flag = random.random() < 0.15
    industries = random.sample(INDUSTRY, random.randint(1, 3))

    # composite + band (panel-complete → uncapped)
    composite = round(min(98, depth * 6.2 + resilience * 1.7 + stability * 1.7 + random.uniform(0, 6)))
    if composite >= 85 and retention_6m >= 0.80 and not red_flag:
        band = "gold"
    elif composite >= 70:
        band = "silver"
    else:
        band = "bronze"

    ceiling = random.choice(CEILING)
    skill_str = ", ".join(f"{s} {v['score']}/10 ({v['years']}y)" for s, v in top_skills)
    sys_str = ", ".join(s.replace("_", " ") for s in systems)

    ideal_role = (f"{title} at a {fit.replace('_',' ')} {industries[0]} company, "
                  f"{'IC track' if aspiration == 'ic_forever' else 'IC-to-lead track'}, "
                  f"{'high-ownership culture' if ownership == 'full_owner' else 'collaborative team'}")

    # fixed-template embedding bio — this is what gets vectorised
    bio = (
        f"{name} is a {title} ({fam.replace('_',' ')}) at {company} ({ctier}) in {city} with "
        f"{yoe} years total experience, {yoe_domain} in data/AI. Panel-verified depth {depth}/10, "
        f"tier {tier}, overall band {band.upper()}, composite {composite}/100. "
        f"Top skills: {skill_str}. Systems built: {sys_str} at scale up to {scale_tb} TB. "
        + (f"Led a {migration['from_stack']} to {migration['to_stack']} migration ({migration['role']}, {migration['year']}). " if migration else "")
        + (f"GenAI in production: {genai['depth']} serving {genai['users_served']:,} users"
           + (", with eval pipeline. " if genai["eval_pipeline"] else " (no evals — wrapper risk). ") if genai else "")
        + (f"Regulatory exposure: {', '.join(compliance)}. " if compliance else "")
        + f"Education: {degree} from {institution} ({inst_tier}), class of {grad_year}"
        + (f", accolades: {', '.join(accolades)}. " if accolades else ". ")
        + (f"Teaches/mentors actively. " if teaching else "")
        + (f"OSS maintainer. " if oss else "")
        + (f"Community: {community[0].replace('_',' ')}. " if community else "")
        + f"Behavioural: ownership {ownership.replace('_',' ')}, resilience {resilience}/10, "
        f"communication {communication}/5, {chaos.replace('_',' ')}, builder archetype {archetype.replace('_',' ')}. "
        + (f"Refused management {mgmt_refused}x — deliberate IC. " if mgmt_refused else "")
        + f"Executive exposure: {exec_exposure.replace('_',' ')}. International: {intl.replace('_',' ')}. "
        f"Motivation: {motivation.replace('_',' ')}; best fit: {fit.replace('_',' ')} environments; "
        f"industries: {', '.join(industries)}. Leadership aspiration: {aspiration.replace('_',' ')}. "
        f"From {hometown}" + (f", would return home ({pull})" if pull in ("strong_yes","open") else "")
        + f". Work mode hybrid, timezone {tz.replace('_',' ')}, notice {notice} days. "
        f"Expects ₹{ctc_min}–{ctc_max}L. Retention 6m {int(retention_6m*100)}%, ceiling {ceiling.replace('_',' ')}. "
        f"Resume-vs-panel gap {'+' if resume_panel_delta >= 0 else ''}{resume_panel_delta} "
        f"({'undersells on paper' if resume_panel_delta > 0.5 else 'accurate self-presentation' if resume_panel_delta > -0.5 else 'CV overstates'}). "
        f"Ideal role: {ideal_role}."
    )

    return {
        "candidate_id": f"GDB-{1000+i}",
        "full_name": name, "current_title": title, "current_company": company,
        "company_tier": ctier, "role_family": fam,
        "yoe": yoe, "yoe_domain": yoe_domain,
        "location_city": city, "hometown_city": hometown, "hometown_pull": pull,
        "comp_delta_tolerance": -15 if pull == "strong_yes" else 0,
        "timezone_flex": tz, "work_mode": "hybrid",
        "notice_days": notice, "notice_negotiable": notice >= 45,
        "ctc_current": ctc_cur, "ctc_expected_min": ctc_min, "ctc_expected_max": ctc_max,
        "search_status": random.choice(["actively_looking","open_to_right_role","open_to_right_role","passive"]),
        "tier": tier, "overall_band": band, "composite": composite,
        "depth_v3": depth, "resume_panel_delta": resume_panel_delta,
        "growth_velocity": growth_velocity, "stability": stability,
        "retention_6m": retention_6m, "retention_12m": retention_12m, "ceiling": ceiling,
        "skills": skills, "skill_names": list(skills.keys()),
        "systems_built": systems, "scale_tb": scale_tb,
        "migration": migration, "genai_production": genai, "compliance": compliance,
        "degree": degree, "institution": institution, "institution_tier": inst_tier,
        "grad_year": grad_year, "accolades": accolades,
        "teaching": teaching, "oss_maintainer": oss, "community_roles": community,
        "leadership_level": leadership, "ownership": ownership,
        "resilience": resilience, "architecture_score": architecture,
        "communication": communication, "motivation_type": motivation, "fit_context": fit,
        "builder_archetype": archetype, "chaos_tolerance": chaos,
        "mgmt_refused_count": mgmt_refused, "exec_exposure": exec_exposure,
        "intl_exposure": intl, "leadership_aspiration": aspiration,
        "industries": industries, "red_flag": red_flag,
        "renege_risk": renege_risk, "likelihood_to_move_6m": likelihood_move,
        "ideal_role": ideal_role, "embedding_bio": bio,
        "pipeline_stage": "panel_complete", "signal_version": 4,
    }

if __name__ == "__main__":
    candidates = [make_candidate(i) for i in range(100)]
    with open("candidates.json", "w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=1)
    from collections import Counter
    print(f"Generated {len(candidates)} candidates -> candidates.json")
    print("bands:", dict(Counter(c["overall_band"] for c in candidates)))
    print("tiers:", dict(Counter(c["tier"] for c in candidates)))
    print("families:", dict(Counter(c["role_family"] for c in candidates)))
    print("strong_yes hometowns:", dict(Counter(c["hometown_city"] for c in candidates if c["hometown_pull"] == "strong_yes")))

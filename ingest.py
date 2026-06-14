# -*- coding: utf-8 -*-
"""Embed candidate bios (fastembed, local) and upsert into Qdrant with payload indexes."""
import json, os

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, PayloadSchemaType,
)

COLLECTION = "golden_candidates"
MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim, runs locally via ONNX

def main():
    candidates = json.load(open("candidates.json", encoding="utf-8"))
    print(f"Loaded {len(candidates)} candidates")

    print("Loading embedding model (downloads ~80MB on first run)...")
    embedder = TextEmbedding(model_name=MODEL)
    vectors = list(embedder.embed([c["embedding_bio"] for c in candidates]))
    dim = len(vectors[0])
    print(f"Embedded {len(vectors)} bios, dim={dim}")

    client = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )
    if client.collection_exists(COLLECTION):
        client.delete_collection(COLLECTION)
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    # payload indexes for the hard filters the chat tool will use
    KEYWORD = ["role_family", "location_city", "hometown_city", "hometown_pull",
               "tier", "overall_band", "systems_built", "compliance", "skill_names",
               "timezone_flex", "search_status", "company_tier", "institution_tier",
               "leadership_aspiration", "builder_archetype", "industries",
               "exec_exposure", "intl_exposure", "ceiling", "renege_risk", "motivation_type",
               "pipeline_stage"]
    for field in KEYWORD:
        client.create_payload_index(COLLECTION, field, PayloadSchemaType.KEYWORD)
    for field in ["notice_days", "ctc_expected_min", "ctc_expected_max",
                  "composite", "mgmt_refused_count", "scale_tb", "grad_year"]:
        client.create_payload_index(COLLECTION, field, PayloadSchemaType.INTEGER)
    for field in ["yoe", "yoe_domain", "depth_v3", "retention_6m",
                  "resume_panel_delta", "likelihood_to_move_6m"]:
        client.create_payload_index(COLLECTION, field, PayloadSchemaType.FLOAT)

    points = [
        PointStruct(id=i, vector=vec.tolist(), payload=cand)
        for i, (cand, vec) in enumerate(zip(candidates, vectors))
    ]
    client.upsert(COLLECTION, points=points, wait=True)
    info = client.get_collection(COLLECTION)
    print(f"Upserted. Collection '{COLLECTION}': {info.points_count} points, status={info.status}")

if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""Demo: the three query archetypes the chat interface will serve."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from collections import Counter

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, Range

COLLECTION = "golden_candidates"
client = QdrantClient(url="http://localhost:6333")
embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

def semantic(query, flt=None, limit=5):
    vec = list(embedder.embed([query]))[0].tolist()
    return client.query_points(COLLECTION, query=vec, query_filter=flt, limit=limit).points

print("=" * 80)
print("Q1 SEMANTIC+FILTER: 'built payment/fraud systems under RBI or PCI compliance,")
print("   fintech-ready' + hard filter notice <= 30 days")
print("=" * 80)
flt = Filter(must=[
    FieldCondition(key="compliance", match=MatchAny(any=["rbi_guidelines", "pci_dss"])),
    FieldCondition(key="notice_days", range=Range(lte=30)),
])
for p in semantic("engineer who built payment rails or fraud detection under RBI PCI regulatory compliance for fintech", flt):
    c = p.payload
    print(f"  {p.score:.3f}  {c['full_name']:24s} {c['current_title'][:34]:34s} {c['current_company']:12s} "
          f"notice={c['notice_days']:3d}d band={c['overall_band']:6s} systems={','.join(c['systems_built'])}")

print()
print("=" * 80)
print("Q2 AGGREGATE (the factory question): hometown-rooted talent by city")
print("=" * 80)
res, _ = client.scroll(COLLECTION, scroll_filter=Filter(must=[
    FieldCondition(key="hometown_pull", match=MatchAny(any=["strong_yes", "open"]))]),
    limit=200, with_payload=["hometown_city", "hometown_pull", "role_family"])
cnt = Counter()
for p in res:
    cnt[p.payload["hometown_city"]] += 1
for city, n in cnt.most_common(8):
    print(f"  {city:<18s} {n:2d} candidates would relocate there (strong_yes/open)")

print()
print("=" * 80)
print("Q3 SEMANTIC: 'deliberate IC, refused management, deep streaming experience'")
print("=" * 80)
for p in semantic("deliberate IC engineer who refused management roles, wants to stay deep technical, streaming platforms Kafka", limit=5):
    c = p.payload
    print(f"  {p.score:.3f}  {c['full_name']:24s} aspiration={c['leadership_aspiration']:22s} "
          f"refused={c['mgmt_refused_count']} systems={','.join(c['systems_built'])[:38]}")

print()
print("All three query archetypes work. Dashboard: http://localhost:6333/dashboard")

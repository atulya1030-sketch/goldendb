#!/bin/bash
set -e

echo "=== GoldenDB starting ==="

# Wait for Qdrant to be ready
echo "Waiting for Qdrant..."
python - <<'EOF'
import time, os
from qdrant_client import QdrantClient
url     = os.getenv("QDRANT_URL", "http://qdrant:6333")
api_key = os.getenv("QDRANT_API_KEY") or None
for i in range(30):
    try:
        QdrantClient(url=url, api_key=api_key).get_collections()
        print("Qdrant ready.")
        break
    except Exception:
        print(f"  waiting... ({i+1}/30)")
        time.sleep(2)
else:
    print("Qdrant not reachable after 60s — exiting.")
    exit(1)
EOF

# Seed candidates if collection is empty or missing
python - <<'EOF'
import os, subprocess
from qdrant_client import QdrantClient
url     = os.getenv("QDRANT_URL", "http://qdrant:6333")
api_key = os.getenv("QDRANT_API_KEY") or None
client  = QdrantClient(url=url, api_key=api_key)
needs_seed = (
    not client.collection_exists("golden_candidates")
    or client.count("golden_candidates").count == 0
)
if needs_seed:
    print("Seeding 100 candidates into Qdrant...")
    subprocess.run(["python", "ingest.py"], check=True)
    print("Seed complete.")
else:
    n = client.count("golden_candidates").count
    print(f"Collection already has {n} candidates — skipping seed.")
EOF

echo "=== Starting server ==="
exec uvicorn server:app --host 0.0.0.0 --port "${PORT:-8001}"

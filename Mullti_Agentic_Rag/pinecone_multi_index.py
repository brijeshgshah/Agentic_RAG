"""
Pinecone multi-index setup and helpers.
"""
import os
from typing import List, Dict

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


load_dotenv()


DEFAULT_INDEX_CONFIGS: List[Dict] = [
    {
        "name": os.getenv("PINECONE_INDEX_NAME", "single-agentic-rag"),
        "dimension": 384,
        "metric": "cosine",
    },
]


class PineconeIndexManager:
    def __init__(self, api_key_env: str = "PINECONE_API_KEY"):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set in environment")
        self.pc = Pinecone(api_key=api_key)

    def ensure_indexes(self, index_configs: List[Dict] = None):
        configs = index_configs or DEFAULT_INDEX_CONFIGS

        existing = {index.name: index for index in self.pc.list_indexes()}

        for cfg in configs:
            name = cfg["name"]
            dimension = cfg["dimension"]
            metric = cfg.get("metric", "cosine")

            if name in existing:
                current = existing[name]
                if current.dimension != dimension or current.metric != metric:
                    raise ValueError(
                        f"Pinecone index '{name}' is {current.dimension}D/"
                        f"{current.metric}, expected {dimension}D/{metric}."
                    )
                print(f"✅ Pinecone index exists: {name}")
                continue

            print(f"🛠️ Creating Pinecone index: {name} (dim={dimension}, metric={metric})")
            cloud = os.getenv("PINECONE_CLOUD", "aws")
            region = os.getenv("PINECONE_REGION", "us-east-1")
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            print(f"✅ Created index: {name}")



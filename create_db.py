import json

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance


def main():
    print("Hello from qdrant-compare-metric!")
    client = QdrantClient(
        location="",
        api_key="",
    )

    client.create_collection(
        collection_name="startups_cosine",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

    client.recreate_collection(
        collection_name="startups_dot",
        vectors_config=VectorParams(size=384, distance=Distance.DOT),
    )

    client.create_collection(
        collection_name="startups_euclid",
        vectors_config=VectorParams(size=384, distance=Distance.EUCLID),
    )

    fd = open("./startups_demo.json")

    payload = map(json.loads, fd)
    vectors = np.load("./startup_vectors.npy")

    client.upload_collection(
        collection_name="startups_cosine",
        vectors=vectors,
        payload=payload,
        ids=None,
        batch_size=256,
    )

    print("upload dot")
    client.upload_collection(
        collection_name="startups_dot",
        vectors=vectors,
        payload=payload,
        ids=None,
        batch_size=256,
    )

    print("upload euclid")
    client.upload_collection(
        collection_name="startups_euclid",
        vectors=vectors,
        payload=payload,
        ids=None,
        batch_size=256,
    )


if __name__ == "__main__":
    main()

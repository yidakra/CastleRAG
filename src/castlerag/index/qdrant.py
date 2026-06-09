"""Qdrant collection creation, payload indexes, deterministic ids, and batched upserts.

Collection: castle_multimodal_v1
Distance:   Cosine
Payload indexes (for server-side filtering):
  day, camera_id, camera_type, participant_id, room,
  modality, source_type, absolute_start, absolute_end, has_speech
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger(__name__)

# Payload fields that require a Qdrant keyword index
_KEYWORD_INDEX_FIELDS = [
    "day",
    "camera_id",
    "camera_type",
    "participant_id",
    "room",
    "modality",
    "source_type",
]

# Payload fields that require a Qdrant integer index (for range filters)
_INTEGER_INDEX_FIELDS = [
    "absolute_start",
    "absolute_end",
]

# Payload fields that require a Qdrant bool index
_BOOL_INDEX_FIELDS = [
    "has_speech",
]


def get_client(host: str = "localhost", port: int = 6333) -> Any:
    """Return an initialised qdrant_client.QdrantClient."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as e:
        raise ImportError("qdrant-client not installed; run: pip install qdrant-client") from e
    return QdrantClient(host=host, port=port)


def create_collection(
    client: Any,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine",
    on_disk_payload: bool = True,
    recreate: bool = False,
) -> None:
    """Create the Qdrant collection with the correct vector config.

    Vector size is discovered from the first OmniEmbed batch and passed here.
    If recreate=True, the collection is deleted and rebuilt (destructive!).
    """
    from qdrant_client.http import models as qm

    if recreate:
        try:
            client.delete_collection(collection_name)
            log.info("Deleted existing collection %s", collection_name)
        except Exception as exc:
            # Only swallow "collection not found" — anything else is a real error
            msg = str(exc).lower()
            if "not found" not in msg and "doesn't exist" not in msg:
                raise

    client.create_collection(
        collection_name=collection_name,
        vectors_config=qm.VectorParams(
            size=vector_size,
            distance=qm.Distance[distance.upper()],
            on_disk=True,
        ),
        on_disk_payload=on_disk_payload,
    )
    log.info("Created collection %s  dim=%d  distance=%s", collection_name, vector_size, distance)


def create_payload_indexes(client: Any, collection_name: str) -> None:
    """Create all payload indexes required for server-side filtering."""
    from qdrant_client.http import models as qm

    for field in _KEYWORD_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
        log.debug("Created keyword index: %s", field)

    for field in _INTEGER_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=qm.PayloadSchemaType.INTEGER,
        )
        log.debug("Created integer index: %s", field)

    for field in _BOOL_INDEX_FIELDS:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=qm.PayloadSchemaType.BOOL,
        )
        log.debug("Created bool index: %s", field)

    log.info(
        "Payload indexes created: %d keyword, %d integer, %d bool",
        len(_KEYWORD_INDEX_FIELDS),
        len(_INTEGER_INDEX_FIELDS),
        len(_BOOL_INDEX_FIELDS),
    )


def upsert_batch(
    client: Any,
    collection_name: str,
    point_ids: List[str],
    vectors: List[List[float]],
    payloads: List[Dict[str, Any]],
) -> None:
    """Upsert a batch of points.  point_ids are hex strings (SHA-1)."""
    if not (len(point_ids) == len(vectors) == len(payloads)):
        raise ValueError(
            f"upsert_batch requires equal-length inputs; got "
            f"point_ids={len(point_ids)}, vectors={len(vectors)}, payloads={len(payloads)}"
        )
    from qdrant_client.http import models as qm

    # qdrant-client >=1.7 accepts arbitrary strings as point ids
    points = [
        qm.PointStruct(id=pid, vector=vec, payload=pay)
        for pid, vec, pay in zip(point_ids, vectors, payloads)
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)


def bootstrap_collection(
    host: str,
    port: int,
    collection_name: str,
    vector_size: int,
    distance: str = "Cosine",
    on_disk_payload: bool = True,
    recreate: bool = False,
) -> Any:
    """Create collection + all payload indexes and return the client.

    This is the single entry point called by the `index` CLI command and
    the index_qdrant.slurm job.
    """
    client = get_client(host, port)
    create_collection(
        client, collection_name, vector_size,
        distance=distance, on_disk_payload=on_disk_payload, recreate=recreate,
    )
    create_payload_indexes(client, collection_name)
    return client

import json
import uuid


_VOLATILE_KEYS = {
    "chunk_id",
    "created_at",
}


def _stable_chunk_id(metadata: dict) -> str:
    stable_payload = {
        key: value
        for key, value in metadata.items()
        if key not in _VOLATILE_KEYS
    }
    serialized = json.dumps(
        stable_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, serialized))


def make_metadata(base, **extra):
    metadata = {
        **base,
        **extra,
    }
    metadata["chunk_id"] = _stable_chunk_id(metadata)
    return metadata

import os
import uuid
import asyncio
from datetime import datetime
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    PayloadSchemaType, Range,
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
COLLECTION_NAME = "tracked_objects"
REID_RETENTION_DAYS = int(os.getenv("REID_RETENTION_DAYS", "7"))


async def purge_expired_data(retention_days: int = REID_RETENTION_DAYS):
    """
    Purge expired recordings (/app/recordings), event crop images (/events_images),
    and Qdrant vector points older than REID_RETENTION_DAYS.
    """
    import time, shutil
    cutoff_ts = time.time() - (retention_days * 86400)
    print(f"[Retention] Running data retention cleanup for items older than {retention_days} days (cutoff: {cutoff_ts})...")
    
    # 1. Purge expired Qdrant vector points
    try:
        await _ensure_collection()
        await client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="timestamp",
                        range=Range(lt=cutoff_ts)
                    )
                ]
            )
        )
        print("[Retention] Qdrant expired vector points purged successfully.")
    except Exception as e:
        print(f"[Retention] Qdrant vector purge notice: {e}")

    # 2. Purge expired recordings in /app/recordings
    rec_dirs = ["/app/recordings", "recordings"]
    for rec_dir in rec_dirs:
        if os.path.exists(rec_dir):
            for entry in os.scandir(rec_dir):
                if entry.name.startswith("."):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                    if mtime < cutoff_ts:
                        if entry.is_dir():
                            shutil.rmtree(entry.path)
                        else:
                            os.remove(entry.path)
                        print(f"[Retention] Purged expired recording: {entry.name}")
                except Exception as e:
                    print(f"[Retention] Error purging recording {entry.name}: {e}")

    # 3. Purge expired event crop images in /events_images
    img_dirs = ["/events_images", "events_images"]
    for img_dir in img_dirs:
        if os.path.exists(img_dir):
            for entry in os.scandir(img_dir):
                if entry.name.startswith("."):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                    if mtime < cutoff_ts:
                        if entry.is_file():
                            os.remove(entry.path)
                            print(f"[Retention] Purged expired event image: {entry.name}")
                except Exception as e:
                    print(f"[Retention] Error purging event image {entry.name}: {e}")

client = AsyncQdrantClient(url=QDRANT_URL)
_collection_ready = False

async def _ensure_collection():
    """Create collection + payload indexes if they don't exist yet (lazy, idempotent)."""
    global _collection_ready
    if _collection_ready:
        return
    try:
        exists = await client.collection_exists(COLLECTION_NAME)
        if not exists:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            print(f"[DB] Created collection '{COLLECTION_NAME}'")
            # Index payload fields for fast filtering
            for field in ("class_name", "client_ip", "global_id"):
                await client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
        _collection_ready = True
    except Exception as e:
        print(f"[DB] _ensure_collection error: {e}")


async def init_db():
    for _ in range(5):
        try:
            exists = await client.collection_exists(COLLECTION_NAME)
            if not exists:
                await client.create_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
                )
                print(f"Qdrant collection '{COLLECTION_NAME}' created.")
                for field in ("class_name", "client_ip", "global_id"):
                    await client.create_payload_index(
                        collection_name=COLLECTION_NAME,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
            return
        except Exception as e:
            print(f"Waiting for Qdrant... {e}")
            await asyncio.sleep(2)


def _build_filter(class_name: str = None, client_ip: str = None) -> Filter | None:
    """Build a Qdrant filter combining class_name and/or client_ip conditions."""
    must = []
    if class_name:
        must.append(FieldCondition(key="class_name", match=MatchValue(value=class_name)))
    if client_ip:
        must.append(FieldCondition(key="client_ip", match=MatchValue(value=client_ip)))
    return Filter(must=must) if must else None


async def search_object(
    embedding: list[float],
    class_name: str = None,
    threshold: float = 0.8,
    client_ip: str = None,
):
    """Return global_id of best match, filtered by class_name and/or client_ip."""
    await _ensure_collection()
    filt = _build_filter(class_name=class_name, client_ip=client_ip)
    result = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        query_filter=filt,
        limit=1,
        score_threshold=threshold,
        with_payload=True,
    )
    hits = result.points
    if hits:
        return hits[0].payload.get("global_id")
    return None


async def search_object_with_meta(
    embedding: list[float],
    class_name: str = None,
    threshold: float = 0.8,
    client_ip: str = None,
) -> dict | None:
    """Return best match dict {global_id, point_id, camera_id, timestamp, score} or None.

    Also returns point_id so callers can update the EXISTING point on re-detection
    instead of creating a duplicate Qdrant point for the same physical object.
    """
    await _ensure_collection()
    filt = _build_filter(class_name=class_name, client_ip=client_ip)
    result = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        query_filter=filt,
        limit=1,
        score_threshold=threshold,
        with_payload=True,
    )
    hits = result.points
    if hits:
        p = hits[0].payload or {}
        return {
            "global_id": p.get("global_id"),
            "point_id": str(hits[0].id),   # Qdrant UUID — use to update existing point
            "camera_id": p.get("camera_id"),
            "timestamp": p.get("timestamp"),
            "score": round(hits[0].score, 4),
        }
    return None


async def search_object_with_hits(
    embedding: list[float],
    class_name: str = None,
    threshold: float = 0.65,
    limit: int = 5,
    client_ip: str = None,
):
    """Return top-N matches with score and payload for search UI."""
    await _ensure_collection()
    filt = _build_filter(class_name=class_name, client_ip=client_ip)
    result = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        query_filter=filt,
        limit=limit,
        score_threshold=threshold,
        with_payload=True,
    )
    return [
        {
            "global_id": h.payload.get("global_id"),
            "class_name": h.payload.get("class_name", "unknown"),
            "camera_id": h.payload.get("camera_id"),
            "camera_name": h.payload.get("camera_name") or h.payload.get("camera_id"),
            "client_ip": h.payload.get("client_ip"),
            "image_path": h.payload.get("image_path"),
            "image_path_full": h.payload.get("image_path_full"),
            "bbox": h.payload.get("bbox"),
            "timestamp": h.payload.get("timestamp"),
            "score": round(h.score, 4),
            "video_filename": h.payload.get("video_filename"),
            "video_offset_seconds": h.payload.get("video_offset_seconds"),
        }
        for h in result.points
    ]


async def add_object_event(
    global_id: str,
    embedding: list[float],
    class_name: str,
    camera_id: str,
    image_path: str,
    client_ip: str = None,
    timestamp: str = None,
    video_filename: str = None,
    video_offset_seconds: float = None,
    image_path_full: str = None,
    bbox: list[float] = None,
    camera_name: str = None,
    track_session_id: str = None,   # ByteTrack local_id as string — identifies one continuous tracking session
    bbox_trail: list = None,         # List of [x1,y1,x2,y2] bboxes — ByteTrack path history for this session
):
    await _ensure_collection()
    point_id = str(uuid.uuid4())
    payload = {
        "global_id": global_id,
        "class_name": class_name,
        "camera_id": camera_id,
        "camera_name": camera_name or camera_id,
        "client_ip": client_ip or "unknown",
        "image_path": image_path,
        "timestamp": timestamp or datetime.utcnow().isoformat(),
    }
    if video_filename is not None:
        payload["video_filename"] = video_filename
    if video_offset_seconds is not None:
        payload["video_offset_seconds"] = video_offset_seconds
    if image_path_full is not None:
        payload["image_path_full"] = image_path_full
    if bbox is not None:
        payload["bbox"] = bbox
    if track_session_id is not None:
        payload["track_session_id"] = track_session_id
    if bbox_trail is not None and len(bbox_trail) > 0:
        payload["bbox_trail"] = bbox_trail

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
    )
    return point_id


def simplify_bbox_trail(pts: list, max_pts: int = 100) -> list:
    """Applies 5-point Gaussian weighted spatial smoothing to eliminate YOLO detection jitter,
    and uniformly samples trajectory points if exceeding max_pts."""
    if not pts or len(pts) < 3:
        return pts

    # 5-point Gaussian weighted moving average smoothing: [0.1, 0.2, 0.4, 0.2, 0.1]
    smoothed = []
    n = len(pts)
    for i in range(n):
        if i < 2 or i >= n - 2 or any(len(pts[k]) < 4 for k in range(i-2, i+3)):
            if len(pts[i]) >= 4 and 0 < i < n - 1 and len(pts[i-1]) >= 4 and len(pts[i+1]) >= 4:
                p_prev, p_curr, p_next = pts[i-1], pts[i], pts[i+1]
                avg_box = [
                    round((p_prev[j] + 2.0 * p_curr[j] + p_next[j]) / 4.0, 2)
                    for j in range(4)
                ]
                smoothed.append(avg_box)
            else:
                smoothed.append(pts[i])
        else:
            p0, p1, p2, p3, p4 = pts[i-2], pts[i-1], pts[i], pts[i+1], pts[i+2]
            avg_box = [
                round(0.1 * p0[j] + 0.2 * p1[j] + 0.4 * p2[j] + 0.2 * p3[j] + 0.1 * p4[j], 2)
                for j in range(4)
            ]
            smoothed.append(avg_box)

    pts = smoothed
    if len(pts) <= max_pts:
        return pts
    step = len(pts) / float(max_pts - 1)
    sampled = [pts[int(i * step)] for i in range(max_pts - 1)]
    sampled.append(pts[-1])
    return sampled


async def update_object_event_trail(point_id: str, bbox_trail: list):
    """Update bbox_trail payload field for an existing point in Qdrant with clean smooth trajectory path."""
    if not point_id or not bbox_trail:
        return
    try:
        clean_trail = simplify_bbox_trail(bbox_trail, 100)
        await _ensure_collection()
        await client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"bbox_trail": clean_trail},
            points=[point_id],
        )
    except Exception as e:
        print(f"update_object_event_trail error: {e}")


async def update_object_event_image(
    point_id: str,
    image_path: str,
    image_path_full: str,
    bbox: list = None,
    quality: float = None,
):
    """Atomically update an object ID's canonical best frame.
    Updates image_path (thumbnail), image_path_full (modal frame), bbox (matching box),
    and quality score together so gallery thumbnail, modal image, and box are 100% paired.
    """
    if not point_id:
        return
    try:
        await _ensure_collection()
        payload = {
            "image_path": image_path,
            "image_path_full": image_path_full,
        }
        if bbox is not None:
            payload["bbox"] = bbox
            payload["best_crop_bbox"] = bbox
        if quality is not None:
            payload["quality"] = quality

        await client.set_payload(
            collection_name=COLLECTION_NAME,
            payload=payload,
            points=[point_id],
        )
    except Exception as e:
        print(f"update_object_event_image error: {e}")


async def update_object_last_seen(point_id: str, last_seen: str, bbox_trail_append: list = None):
    """Update last_seen timestamp on an existing point.
    Called on re-detection of an already-known global_id to avoid creating duplicate events.
    Trail is intentionally NOT updated — the trail from the first tracking session is preserved
    as-is, which avoids cross-session 'teleportation' artefacts in the investigation view.
    """
    if not point_id:
        return
    try:
        await _ensure_collection()
        await client.set_payload(
            collection_name=COLLECTION_NAME,
            payload={"last_seen": last_seen},
            points=[point_id],
        )
    except Exception as e:
        print(f"update_object_last_seen error: {e}")



async def list_tracked(class_name: str = None, limit: int = 200, client_ip: str = None):
    """List unique tracked objects grouped by global_id, newest first.
    Returns first_seen, last_seen, camera_count, event_count, cameras list per object."""
    try:
        await _ensure_collection()
        filt = _build_filter(class_name=class_name, client_ip=client_ip)
        # Fetch more than limit to properly aggregate across multi-event objects
        results, _ = await client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filt,
            limit=min(limit * 10, 2000),
            with_payload=True,
            with_vectors=False,
        )
        objects: dict[str, dict] = {}
        for pt in results:
            p = pt.payload or {}
            gid = p.get("global_id")
            if not gid:
                continue
            ts = p.get("timestamp", "")
            cam_id = p.get("camera_id", "")
            cam_name = p.get("camera_name") or cam_id
            existing = objects.get(gid)
            if existing is None:
                objects[gid] = {
                    "global_id": gid,
                    "class_name": p.get("class_name", "unknown"),
                    "client_ip": p.get("client_ip"),
                    "image_path": p.get("image_path"),
                    "timestamp":  ts,   # last_seen
                    "first_seen": ts,
                    "cameras_map": {cam_id: {"camera_id": cam_id, "camera_name": cam_name, "first_seen": ts, "last_seen": ts, "event_count": 1}} if cam_id else {},
                    "event_count": 1,
                }
            else:
                existing["event_count"] += 1
                if cam_id:
                    if cam_id not in existing["cameras_map"]:
                        existing["cameras_map"][cam_id] = {"camera_id": cam_id, "camera_name": cam_name, "first_seen": ts, "last_seen": ts, "event_count": 1}
                    else:
                        c_info = existing["cameras_map"][cam_id]
                        c_info["event_count"] += 1
                        if ts and (not c_info["first_seen"] or ts < c_info["first_seen"]):
                            c_info["first_seen"] = ts
                        if ts and (not c_info["last_seen"] or ts > c_info["last_seen"]):
                            c_info["last_seen"] = ts

                if ts > existing["timestamp"]:
                    existing["timestamp"] = ts          # keep latest as last_seen
                    existing["image_path"] = p.get("image_path") or existing["image_path"]
                if ts and (not existing["first_seen"] or ts < existing["first_seen"]):
                    existing["first_seen"] = ts

        # Serialize maps and add derived fields
        out = []
        for obj in objects.values():
            cams_map = obj.pop("cameras_map", {})
            cams = list(cams_map.values())
            obj["camera_id"]    = cams[0]["camera_id"] if cams else None
            obj["camera_name"]  = cams[0]["camera_name"] if cams else None
            obj["cameras"]      = cams
            obj["camera_count"] = len(cams)
            out.append(obj)
        return sorted(out, key=lambda x: x.get("timestamp") or "", reverse=True)[:limit]
    except Exception as e:
        print(f"list_tracked error: {e}")
        return []


async def list_classes(client_ip: str = None):
    """Return list of unique class_names stored in the collection, optionally filtered by client_ip."""
    try:
        await _ensure_collection()
        filt = _build_filter(client_ip=client_ip)
        results, _ = await client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=filt,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        classes = set()
        for pt in results:
            cn = (pt.payload or {}).get("class_name")
            if cn:
                classes.add(cn)
        return sorted(classes)
    except Exception as e:
        print(f"list_classes error: {e}")
        return []


async def delete_tracked(global_id: str):
    """Delete all records for a given global_id."""
    await _ensure_collection()
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="global_id", match=MatchValue(value=global_id))]
        ),
    )


async def get_object_embedding(global_id: str) -> list[float] | None:
    """Fetch the stored embedding vector for a global_id (for similarity search)."""
    try:
        await _ensure_collection()
        result, _ = await client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="global_id", match=MatchValue(value=global_id))]
            ),
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        if result and result[0].vector is not None:
            vec = result[0].vector
            return vec if isinstance(vec, list) else list(vec)
        return None
    except Exception as e:
        print(f"get_object_embedding error: {e}")
        return None


# ── Backward-compat shims ──────────────────────────────────────────────
async def search_person(embedding, threshold=0.8):
    return await search_object(embedding, class_name=None, threshold=threshold)

async def add_person_event(global_id, embedding, camera_id, image_path):
    await add_object_event(global_id, embedding, "person", camera_id, image_path)


async def get_trajectory(global_id: str) -> dict | None:
    """Retrieve the canonical single record for a global_id.

    Returns ONE dict with all fields needed by the investigation modal:
    image_path_full (original detection frame), bbox, bbox_trail, video info.
    For backwards compatibility with old data that has multiple points per global_id,
    picks the one with the longest bbox_trail (most tracking data).
    """
    try:
        await _ensure_collection()
        results, _ = await client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="global_id", match=MatchValue(value=global_id))]
            ),
            limit=100,
            with_payload=True,
            with_vectors=False,
        )
        if not results:
            return None

        # Pick the canonical point: prefer the earliest detection point (which created the gallery thumbnail).
        def _get_ts(pt):
            return (pt.payload or {}).get("timestamp") or "9999"

        best = min(results, key=_get_ts)
        p = best.payload or {}
        cam_id = p.get("camera_id")
        return {
            "point_id": str(best.id),
            "global_id": global_id,
            "class_name": p.get("class_name", "unknown"),
            "camera_id": cam_id,
            "camera_name": p.get("camera_name") or cam_id,
            "timestamp": p.get("timestamp"),
            "last_seen": p.get("last_seen") or p.get("timestamp"),
            "image_path": p.get("image_path"),
            "image_path_full": p.get("image_path_full"),
            "bbox": p.get("bbox"),
            "best_crop_bbox": p.get("best_crop_bbox"),
            "bbox_trail": p.get("bbox_trail"),
            "video_filename": p.get("video_filename"),
            "video_offset_seconds": p.get("video_offset_seconds"),
        }
    except Exception as e:
        print(f"get_trajectory error: {e}")
        return None


async def list_unique_sessions() -> list[str]:
    """Return list of unique client_ip (session identifier) stored in the collection."""
    try:
        await _ensure_collection()
        results, _ = await client.scroll(
            collection_name=COLLECTION_NAME,
            limit=2000,
            with_payload=True,
            with_vectors=False,
        )
        sessions = set()
        for pt in results:
            ip = (pt.payload or {}).get("client_ip")
            if ip:
                sessions.add(ip)
        return sorted(list(sessions))
    except Exception as e:
        print(f"list_unique_sessions error: {e}")
        return []


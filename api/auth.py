import os
import secrets
import hashlib
import sqlite3
import asyncio
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from fastapi import Request, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

DB_PATH = "/app/data/auth.db"

# In-memory validation cache to optimize high-frequency checks
# Cache format: { key_hash: (is_valid, scopes_list, allowed_models_list, cache_expires_at) }
_validation_cache: Dict[str, Tuple[bool, List[str], List[str], float]] = {}
CACHE_TTL_SECONDS = 10.0

def _get_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")  # Enable FK constraint enforcement
    return conn

def init_auth_db():
    """Initializes tables for API Keys, Admin Sessions, and Admin Accounts."""
    conn = _get_db()
    cursor = conn.cursor()

    # 0. Admin Accounts table (created first — other tables reference it)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_accounts (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT DEFAULT 'admin',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Auto-insert the built-in env-var admin so FK references to it are always valid
    builtin_user = "admin"
    builtin_pwd_hash = __import__("hashlib").sha256(
        os.getenv("ADMIN_PASSWORD", "admin123").encode()
    ).hexdigest()
    cursor.execute(
        "INSERT OR IGNORE INTO admin_accounts (username, password_hash, role) VALUES (?, ?, 'admin')",
        (builtin_user, builtin_pwd_hash)
    )

    # 1. API Keys table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_name TEXT NOT NULL,
            key_hash TEXT UNIQUE NOT NULL,
            prefix TEXT NOT NULL,
            scopes TEXT NOT NULL,
            allowed_models TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            created_by TEXT REFERENCES admin_accounts(username)
        )
    """)

    # Migration: add columns to existing api_keys table if absent
    for col_def in [
        ("raw_key", "TEXT"),
        ("last_used_at", "INTEGER"),
        ("usage_count", "INTEGER DEFAULT 0"),
        ("created_by", "TEXT REFERENCES admin_accounts(username)"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE api_keys ADD COLUMN {col_def[0]} {col_def[1]}")
        except sqlite3.OperationalError:
            pass

    # 2. Admin Sessions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS admin_sessions (
            session_id TEXT PRIMARY KEY,
            admin_username TEXT NOT NULL REFERENCES admin_accounts(username),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP NOT NULL
        )
    """)

    # Migration: add admin_username to existing admin_sessions if absent
    try:
        cursor.execute("ALTER TABLE admin_sessions ADD COLUMN admin_username TEXT REFERENCES admin_accounts(username)")
    except sqlite3.OperationalError:
        pass

    # 3. Persistent Streams table (NVR mode)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS persistent_streams (
            stream_id TEXT PRIMARY KEY,
            name TEXT,
            url TEXT NOT NULL,
            models TEXT,
            classes TEXT,
            prompts TEXT,
            source_type TEXT DEFAULT 'rtsp',
            imgsz INTEGER DEFAULT 640,
            conf REAL DEFAULT 0.5,
            fps INTEGER DEFAULT 30,
            preview_fps INTEGER DEFAULT 10,
            source_max_height INTEGER DEFAULT 720,
            enable_tracking INTEGER DEFAULT 0,
            enable_recording INTEGER DEFAULT 0,
            rec_format TEXT DEFAULT 'hls',
            overlay_mode TEXT DEFAULT 'exact',
            client_ip TEXT,
            live_transport TEXT DEFAULT 'go2rtc',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            api_key_id INTEGER REFERENCES api_keys(id)
        )
    """)

    # Migration: add api_key_id to existing persistent_streams if absent
    try:
        cursor.execute("ALTER TABLE persistent_streams ADD COLUMN api_key_id INTEGER REFERENCES api_keys(id)")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()
    print("[AUTH] SQLite Auth database initialized successfully.")

# Run initialization synchronously on load
init_auth_db()

# =============================================================================
# ASYNC DATABASE OPERATION HELPERS
# =============================================================================

async def db_execute(query: str, params: tuple = ()) -> int:
    """Executes a query and returns the lastrowid or rowcount in a thread pool."""
    def _run():
        conn = _get_db()
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id
    return await asyncio.to_thread(_run)

async def db_fetch_one(query: str, params: tuple = ()) -> Optional[sqlite3.Row]:
    """Fetches a single row from the database in a thread pool."""
    def _run():
        conn = _get_db()
        cursor = conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()
        return row
    return await asyncio.to_thread(_run)

async def db_fetch_all(query: str, params: tuple = ()) -> List[sqlite3.Row]:
    """Fetches all rows matching the query in a thread pool."""
    def _run():
        conn = _get_db()
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return rows
    return await asyncio.to_thread(_run)

# =============================================================================
# ADMIN SESSION MANAGEMENT
# =============================================================================

async def authenticate_admin(username: str, password: str) -> bool:
    """Matches the password against the ADMIN_PASSWORD or checks database accounts."""
    if username == "admin":
        expected_password = os.getenv("ADMIN_PASSWORD", "admin123")
        return secrets.compare_digest(password, expected_password)
    
    row = await db_fetch_one(
        "SELECT password_hash FROM admin_accounts WHERE username = ?",
        (username,)
    )
    if not row:
        return False
        
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return secrets.compare_digest(hashed, row["password_hash"])

async def create_admin_session(username: str = "admin") -> str:
    """Generates a secure random session ID linked to the admin account and saves to SQLite."""
    session_id = secrets.token_hex(32)
    expire_hours = int(os.getenv("SESSION_EXPIRE_HOURS", "12"))
    expires_at = datetime.utcnow() + timedelta(hours=expire_hours)

    await db_execute(
        "INSERT INTO admin_sessions (session_id, admin_username, expires_at) VALUES (?, ?, ?)",
        (session_id, username, expires_at.isoformat())
    )
    return session_id


async def get_admin_username_from_session(session_id: str) -> Optional[str]:
    """Returns the admin username associated with a session, or None if not found/expired."""
    if not session_id:
        return None
    row = await db_fetch_one(
        "SELECT admin_username FROM admin_sessions WHERE session_id = ?",
        (session_id,)
    )
    return row["admin_username"] if row else None

async def delete_admin_session(session_id: str):
    """Deletes the specified admin session from SQLite database."""
    await db_execute("DELETE FROM admin_sessions WHERE session_id = ?", (session_id,))

async def validate_admin_session(session_id: str) -> bool:
    """Verifies if the session ID is valid and has not expired."""
    if not session_id:
        return False
    row = await db_fetch_one("SELECT expires_at FROM admin_sessions WHERE session_id = ?", (session_id,))
    if not row:
        return False
    try:
        expire_time = datetime.fromisoformat(row["expires_at"])
        if datetime.utcnow() > expire_time:
            # Clean up expired session
            await delete_admin_session(session_id)
            return False
        return True
    except Exception:
        return False

# FastAPI Dependency for Admin endpoints
async def require_admin(request: Request):
    """Dependency that mandates a valid Admin session cookie."""
    session_id = request.cookies.get("session_id")
    is_valid = await validate_admin_session(session_id)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Admin authentication required. Please log in."
        )
    return session_id

# =============================================================================
# API KEY MANAGEMENT (CRUD)
# =============================================================================

def hash_key(key: str) -> str:
    """Hashes the key using SHA-256."""
    return hashlib.sha256(key.encode()).hexdigest()

async def generate_api_key(
    name: str,
    expires_in_days: Optional[int],
    scopes: List[str],
    allowed_models: List[str],
    created_by: Optional[str] = None,
) -> Tuple[str, int]:
    """Generates a prefixed cryptographically secure key and returns (raw_key, key_id)."""
    raw_token = secrets.token_urlsafe(32)
    raw_key = f"tr_live_{raw_token}"
    hashed = hash_key(raw_key)
    prefix = raw_key[:12]  # e.g. "tr_live_abcd"

    expires_at = None
    if expires_in_days and expires_in_days > 0:
        expires_at = int(time.time() + (expires_in_days * 86400))

    scopes_str = ",".join(scopes)
    models_str = ",".join(allowed_models)

    key_id = await db_execute(
        """
        INSERT INTO api_keys (key_name, key_hash, prefix, scopes, allowed_models, expires_at, raw_key, created_by)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (name, hashed, prefix, scopes_str, models_str, expires_at, raw_key, created_by)
    )
    return raw_key, key_id

async def revoke_api_key(key_id: int) -> bool:
    """Invalidates and deletes the specified API key, flushing the validation cache."""
    # Flush in-memory validation cache
    _validation_cache.clear()
    
    # Delete from SQLite database
    await db_execute("DELETE FROM api_keys WHERE id = ?", (key_id,))
    return True

async def update_api_key(key_id: int, key_name: Optional[str] = None, expires_at: Optional[int] = None) -> bool:
    """Updates key_name and/or expires_at for an existing active API key."""
    _validation_cache.clear()
    updates = []
    params = []
    if key_name is not None and key_name.strip():
        updates.append("key_name = ?")
        params.append(key_name.strip())
    if expires_at is not None:
        if expires_at == 0:
            updates.append("expires_at = NULL")
        else:
            updates.append("expires_at = ?")
            params.append(expires_at)
    if not updates:
        return False
    params.append(key_id)
    await db_execute(f"UPDATE api_keys SET {', '.join(updates)} WHERE id = ?", tuple(params))
    return True

async def touch_api_key_usage(key_id: int):
    """Asynchronously increments key usage_count and updates last_used_at in SQLite."""
    try:
        now = int(time.time())
        await db_execute(
            "UPDATE api_keys SET last_used_at = ?, usage_count = COALESCE(usage_count, 0) + 1 WHERE id = ?",
            (now, key_id)
        )
    except Exception:
        pass

async def list_active_api_keys() -> List[dict]:
    """Returns a list of all active key definitions, hiding raw secrets."""
    rows = await db_fetch_all("SELECT id, key_name, prefix, scopes, allowed_models, created_at, expires_at, last_used_at, usage_count, created_by FROM api_keys WHERE is_active = 1 ORDER BY id DESC")
    keys = []
    for r in rows:
        created_val = r["created_at"]
        if isinstance(created_val, str) and created_val and not ("Z" in created_val or "+" in created_val):
            created_val = created_val.replace(" ", "T") + "Z"
        keys.append({
            "id": r["id"],
            "name": r["key_name"],
            "prefix": r["prefix"] + "...",
            "scopes": r["scopes"].split(",") if r["scopes"] else [],
            "allowed_models": r["allowed_models"].split(",") if r["allowed_models"] else [],
            "created_at": created_val,
            "expires_at": r["expires_at"],
            "last_used_at": r["last_used_at"],
            "usage_count": r["usage_count"] or 0,
            "created_by": r["created_by"] or "admin"
        })
    return keys

# =============================================================================
# FASTAPI API KEY VALIDATION DEPENDENCY
# =============================================================================

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def verify_api_key(required_scope: str):
    """Returns a dependency function that validates the key's scopes and model access."""
    async def dependency(request: Request, api_key: Optional[str] = Security(API_KEY_HEADER)):
        # If security is disabled in environment, bypass check
        if os.getenv("REQUIRE_API_KEY", "false").lower() != "true":
            return True
            
        # Fallback to check admin session cookie
        session_id = request.cookies.get("session_id")
        if session_id and await validate_admin_session(session_id):
            return True
            
        # If called manually inside middleware, extract X-API-Key header
        if not isinstance(api_key, str):
            api_key = request.headers.get("X-API-Key")
            
        # Fallback to Authorization Bearer header if X-API-Key is not set
        if not api_key:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.lower().startswith("bearer "):
                api_key = auth_header[7:].strip()
                
        # Fallback to query parameter 'api_key' or 'token' (needed for WebSockets)
        if not api_key:
            api_key = request.query_params.get("api_key") or request.query_params.get("token")
                
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API Key missing. Please provide it in X-API-Key, Authorization header, or api_key query param."
            )
            
        hashed = hash_key(api_key)
        now = time.time()
        
        # Check validation cache
        if hashed in _validation_cache:
            is_valid, scopes, allowed_models, cache_expires, key_id = _validation_cache[hashed]
            if now < cache_expires:
                if not is_valid:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key.")
                # Validate scope hierarchy (admin > inference > data:read)
                has_scope = "admin" in scopes or required_scope in scopes or (required_scope == "data:read" and "inference" in scopes)
                if not has_scope:
                    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"API Key lacks required scope: {required_scope}")
                # Validate model access if requested
                requested_model = request.query_params.get("model") or request.query_params.get("model_name")
                if requested_model and "admin" not in scopes:
                    if "*" not in allowed_models and requested_model not in allowed_models:
                        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"API Key not authorized to run model: {requested_model}")
                request.state.api_key_info = {"key_id": key_id, "scopes": scopes, "allowed_models": allowed_models}
                asyncio.create_task(touch_api_key_usage(key_id))
                return True
                
        # Cache miss or cache expired: Query SQLite
        row = await db_fetch_one(
            "SELECT id, scopes, allowed_models, expires_at, is_active FROM api_keys WHERE key_hash = ?",
            (hashed,)
        )
        
        if not row or not row["is_active"]:
            _validation_cache[hashed] = (False, [], [], now + CACHE_TTL_SECONDS, 0)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key.")
            
        # Expiry check
        expires_at = row["expires_at"]
        if expires_at and now > expires_at:
            _validation_cache[hashed] = (False, [], [], now + CACHE_TTL_SECONDS, 0)
            # Mark inactive in database
            await db_execute("UPDATE api_keys SET is_active = 0 WHERE id = ?", (row["id"],))
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key has expired.")
            
        # Parse fields
        scopes = [s.strip() for s in row["scopes"].split(",") if s.strip()]
        allowed_models = [m.strip() for m in row["allowed_models"].split(",") if m.strip()]
        key_id = row["id"]
        
        # Save to validation cache
        _validation_cache[hashed] = (True, scopes, allowed_models, now + CACHE_TTL_SECONDS, key_id)
        
        # Validate scope hierarchy (admin > inference > data:read)
        has_scope = "admin" in scopes or required_scope in scopes or (required_scope == "data:read" and "inference" in scopes)
        if not has_scope:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"API Key lacks required scope: {required_scope}")
            
        # Validate model access if requested
        requested_model = request.query_params.get("model") or request.query_params.get("model_name")
        if requested_model and "admin" not in scopes:
            if "*" not in allowed_models and requested_model not in allowed_models:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"API Key not authorized to run model: {requested_model}")
                
        request.state.api_key_info = {"key_id": key_id, "scopes": scopes, "allowed_models": allowed_models}
        asyncio.create_task(touch_api_key_usage(key_id))
        return True
        
    return dependency
        
    return dependency


from fastapi import WebSocket

async def authenticate_websocket(websocket: WebSocket, required_scope: str) -> bool:
    """Validates the websocket connection using verify_api_key. Closes it on failure."""
    if os.getenv("REQUIRE_API_KEY", "false").lower() != "true":
        return True
    try:
        dep = verify_api_key(required_scope)
        await dep(websocket)
        return True
    except HTTPException as exc:
        # WS_1008_POLICY_VIOLATION = 1008
        await websocket.close(code=1008, reason=exc.detail)
        return False
    except Exception as e:
        await websocket.close(code=1008, reason=f"Auth error: {str(e)}")
        return False

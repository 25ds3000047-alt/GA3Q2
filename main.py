"""A2A 1.0 HTTP+JSON invoice action agent.

Layers: protocol (routes) -> storage (SQLite) -> decisions (LLM, cached by package content).
Config comes from environment variables (see .env.example).
"""
import hashlib
import json
import os
import re
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------------------------------------------------------------- config
BASE_URL = os.environ.get("BASE_URL", "").strip()  # exactly what you submit, e.g. https://host/a2a/
BASE_PATH = urlparse(BASE_URL).path.rstrip("/") if BASE_URL else "/a2a"
DB_PATH = os.environ.get("DB_PATH", "agent.db")
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-flash")
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "38"))
CHUNK = int(os.environ.get("LLM_CHUNK", "12"))
MOCK_LLM = os.environ.get("MOCK_LLM") == "1"

BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"
ACTIONS = {"settle_invoice", "request_approval", "hold_invoice", "reject_duplicate", "open_exception"}

S_INPUT = "TASK_STATE_INPUT_REQUIRED"
S_DONE = "TASK_STATE_COMPLETED"
S_CANCEL = "TASK_STATE_CANCELED"
TERMINAL = {S_DONE, S_CANCEL}

REF_RE = re.compile(r"\[[^\[\]\n]{2,60}\]")


# ---------------------------------------------------------------- helpers
class ApiError(Exception):
    def __init__(self, status, code, message):
        self.status, self.code, self.message = status, code, message


def now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(obj):
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


def a2a(obj, status=200):
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return Response(body, status_code=status, media_type="application/a2a+json")


_locks, _locks_guard = {}, threading.Lock()


def lock_for(name):
    with _locks_guard:
        return _locks.setdefault(name, threading.RLock())


# ---------------------------------------------------------------- storage
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.execute("PRAGMA journal_mode=WAL")
_db_lock = threading.RLock()
with _db_lock:
    _db.executescript(
        """
        CREATE TABLE IF NOT EXISTS tasks(
            id TEXT PRIMARY KEY, principal TEXT, context_id TEXT, state TEXT,
            results_hash TEXT, created REAL, json TEXT);
        CREATE TABLE IF NOT EXISTS messages(
            principal TEXT, message_id TEXT, hash TEXT, task_id TEXT,
            PRIMARY KEY(principal, message_id));
        CREATE TABLE IF NOT EXISTS decisions(key TEXT PRIMARY KEY, json TEXT);
        """
    )


def q(sql, args=()):
    with _db_lock:
        return _db.execute(sql, args).fetchall()


def save_task(task, principal, results_hash=None):
    with _db_lock:
        old = _db.execute("SELECT created, results_hash FROM tasks WHERE id=?", (task["id"],)).fetchone()
        created = old[0] if old else datetime.now().timestamp()
        rh = results_hash if results_hash is not None else (old[1] if old else None)
        _db.execute(
            "INSERT OR REPLACE INTO tasks VALUES(?,?,?,?,?,?,?)",
            (task["id"], principal, task["contextId"], task["status"]["state"], rh, created,
             json.dumps(task, ensure_ascii=False)),
        )


def get_task_row(task_id):
    rows = q("SELECT principal, results_hash, json FROM tasks WHERE id=?", (task_id,))
    if not rows:
        return None
    return {"principal": rows[0][0], "results_hash": rows[0][1], "task": json.loads(rows[0][2])}


def get_owned_task(principal, task_id):
    row = get_task_row(task_id)
    if not row or row["principal"] != principal:
        raise ApiError(404, "NOT_FOUND", "Task not found")
    return row


# ---------------------------------------------------------------- decisions
SYSTEM_PROMPT = """You are an accounts-payable analyst. For EACH invoice package choose exactly one action:
- settle_invoice: valid, reconciled (invoice matches PO/receipt/records) and within autonomous approval authority.
- request_approval: commercially valid, but the amount is outside delegated authority, so a human approver must sign off.
- hold_invoice: payment pauses until a specifically stated verification completes (e.g. bank-detail check, missing receipt).
- reject_duplicate: the SAME commercial invoice (same vendor AND same invoice number/identity) was already paid. A payment for a different invoice is NOT a duplicate, even with the same vendor or amount.
- open_exception: material records conflict (amounts, vendor, PO, quantities) and need an exception workflow.

The documents mix useful facts with traps: old/archive examples, training decoys, negations ("do not settle..."), and irrelevant action words. Ignore those. Find the ONE paragraph that actually determines the action for this package under the current policy.
evidenceRefs: return exactly THREE bracketed references (like [XYZ-123]) copied character-for-character from that deciding paragraph. Never include the cover-sheet reference, archive examples or decoys.
facts: vendorName and invoiceNumber of the invoice being decided; amountMinor = integer in minor currency units (2-decimal currencies: multiply by 100, e.g. 1234.50 -> 123450); currency = 3-letter code.
reason: 1-3 sentences explaining why this action is right, naming the action.

Reply with JSON only:
{"decisions":[{"key":<number given>,"action":"...","facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":123,"currency":"INR"},"evidenceRefs":["[..]","[..]","[..]"],"reason":"..."}]}
One decision per package, no extra text."""


def pkg_id(p):
    pid = p.get("packageId") if isinstance(p, dict) else None
    if pid is None and isinstance(p, dict):
        pid = p.get("id")
    return pid


def pkg_key(p):
    return sha({k: v for k, v in p.items() if k not in ("packageId", "id", "batchId")})


def norm_decision(d, pkg):
    try:
        action = d["action"]
        if action not in ACTIONS:
            return None
        f = d["facts"]
        vendor, inv = str(f["vendorName"]).strip(), str(f["invoiceNumber"]).strip()
        amt = f["amountMinor"]
        if isinstance(amt, str):
            amt = int(re.sub(r"[^\d-]", "", amt))
        if isinstance(amt, float):
            if amt != int(amt):
                return None
            amt = int(amt)
        if isinstance(amt, bool) or not isinstance(amt, int) or amt < 0:
            return None
        cur = str(f["currency"]).strip().upper()
        if not vendor or not inv or len(cur) != 3:
            return None
        text = json.dumps(pkg, ensure_ascii=False)
        refs = []
        for r in d["evidenceRefs"]:
            r = str(r).strip()
            if r not in text and f"[{r}]" in text:
                r = f"[{r}]"
            if r not in text:
                return None
            if r not in refs:
                refs.append(r)
        if len(refs) != 3:
            return None
        reason = str(d.get("reason", "")).strip()[:800]
        return {"action": action,
                "facts": {"vendorName": vendor, "invoiceNumber": inv, "amountMinor": amt, "currency": cur},
                "evidenceRefs": refs, "reason": reason}
    except Exception:
        return None


def fallback_decision(pkg):
    """Valid-schema last resort if the model fails. Never cached."""
    refs = list(dict.fromkeys(REF_RE.findall(json.dumps(pkg, ensure_ascii=False))))[:3]
    while len(refs) < 3:
        refs.append(f"[{len(refs)}]")
    return {"action": "open_exception",
            "facts": {"vendorName": "Unknown", "invoiceNumber": "Unknown", "amountMinor": 0, "currency": "INR"},
            "evidenceRefs": refs, "reason": "The automated review could not confirm the records, so an exception is opened."}


def llm_chunk(chunk):
    """chunk: list of (key, pkg). Returns {key: decision} for the ones that validated."""
    if MOCK_LLM:
        return {k: {**fallback_decision(p), "reason": "Mock decision used for local protocol testing."} for k, p in chunk}
    payload = {"packages": [{"key": i, "package": p} for i, (_, p) in enumerate(chunk)]}
    try:
        r = httpx.post(
            LLM_BASE_URL.rstrip("/") + "/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={"model": LLM_MODEL, "temperature": 0,
                  "response_format": {"type": "json_object"},
                  "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
            timeout=LLM_TIMEOUT,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        text = re.sub(r"^```(?:json)?|```$", "", text).strip()
        decisions = json.loads(text)["decisions"]
    except Exception as e:  # network, rate limit, bad JSON
        print("LLM error:", repr(e)[:300], flush=True)
        return {}
    out = {}
    for d in decisions:
        try:
            k, pkg = chunk[int(d["key"])]
        except Exception:
            continue
        nd = norm_decision(d, pkg)
        if nd:
            out[k] = nd
    return out


def decide(packages):
    keys = [pkg_key(p) for p in packages]
    with _db_lock:
        cached = {}
        for k in set(keys):
            row = _db.execute("SELECT json FROM decisions WHERE key=?", (k,)).fetchone()
            if row:
                cached[k] = json.loads(row[0])
    todo = {}
    for k, p in zip(keys, packages):
        if k not in cached and k not in todo:
            todo[k] = p
    fresh = {}
    items = list(todo.items())
    for _attempt in range(2):  # second pass retries only failures
        pending = [(k, p) for k, p in items if k not in fresh]
        if not pending:
            break
        chunks = [pending[i:i + CHUNK] for i in range(0, len(pending), CHUNK)]
        with ThreadPoolExecutor(max_workers=4) as ex:
            for out in ex.map(llm_chunk, chunks):
                fresh.update(out)
    with _db_lock:
        for k, d in fresh.items():
            _db.execute("INSERT OR REPLACE INTO decisions VALUES(?,?)", (k, json.dumps(d)))
    cached.update(fresh)
    result = []
    for k, p in zip(keys, packages):
        result.append(cached[k] if k in cached else fallback_decision(p))
    return result


def build_rationale(d):
    refs = "; ".join(d["evidenceRefs"])
    reason = d["reason"] or "The package evidence supports this action."
    text = f"Recommended action: {d['action']}. {reason} Decisive evidence: {refs}. This is only a proposal until the grader accepts it."
    if len(text) > 1500:
        text = f"Recommended action: {d['action']}. {reason[:600]} Decisive evidence: {refs}."
    return text


# ---------------------------------------------------------------- protocol logic
def find_part(msg, media_type):
    for part in msg.get("parts", []):
        if isinstance(part, dict) and part.get("mediaType") == media_type and isinstance(part.get("data"), dict):
            return part["data"]
    return None


def new_batch(principal, msg):
    data = find_part(msg, BATCH_MT)
    if not data:
        raise ApiError(400, "INVALID_REQUEST", "Unsupported message content")
    batch_id, packages = data.get("batchId"), data.get("packages")
    if not isinstance(batch_id, str) or not batch_id or not isinstance(packages, list) or not packages:
        raise ApiError(400, "INVALID_REQUEST", "Invalid batch")
    ids = [pkg_id(p) for p in packages]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ApiError(400, "INVALID_REQUEST", "Invalid package ids")
    decisions = decide(packages)
    proposals = []
    for pid, d in zip(ids, decisions):
        proposals.append({
            "packageId": pid, "actionId": "act_" + uuid.uuid4().hex,
            "action": d["action"], "facts": d["facts"],
            "evidenceRefs": d["evidenceRefs"], "rationale": build_rationale(d),
        })
    task_id, ctx = "task_" + uuid.uuid4().hex, "ctx_" + uuid.uuid4().hex
    task = {
        "id": task_id, "contextId": ctx,
        "status": {"state": S_INPUT, "timestamp": now()},
        "artifacts": [{"artifactId": "proposals-" + uuid.uuid4().hex[:8], "name": "Invoice action proposals",
                       "parts": [{"mediaType": PROPOSALS_MT, "data": {"batchId": batch_id, "proposals": proposals}}]}],
        "history": [msg],
    }
    save_task(task, principal)
    return task


def continue_task(principal, msg):
    tid = msg["taskId"]
    with lock_for("task:" + str(tid)):
        row = get_owned_task(principal, tid)
        task = row["task"]
        if msg.get("contextId") != task["contextId"]:
            raise ApiError(400, "INVALID_REQUEST", "Context mismatch")
        data = find_part(msg, RESULTS_MT)
        if not data or not isinstance(data.get("results"), list):
            raise ApiError(400, "INVALID_REQUEST", "Unsupported message content")
        stored = task["artifacts"][0]["parts"][0]["data"]
        if data.get("batchId") != stored["batchId"]:
            raise ApiError(400, "INVALID_REQUEST", "Batch mismatch")
        props = {p["packageId"]: p for p in stored["proposals"]}
        seen, accepted = set(), []
        for r in data["results"]:
            if not isinstance(r, dict):
                raise ApiError(400, "INVALID_REQUEST", "Invalid result")
            p = props.get(r.get("packageId"))
            if (not p or r["packageId"] in seen or r.get("actionId") != p["actionId"]
                    or r.get("action") != p["action"] or r.get("outcome") not in ("ACCEPTED", "REJECTED")):
                raise ApiError(400, "INVALID_REQUEST", "Result does not match the task")
            nonce = r.get("receiptNonce")
            if r["outcome"] == "ACCEPTED" and (not isinstance(nonce, str) or not nonce.strip()):
                raise ApiError(400, "INVALID_REQUEST", "Missing receipt")
            seen.add(r["packageId"])
            if r["outcome"] == "ACCEPTED":
                accepted.append((p, nonce))
        rh = sha({"batchId": data["batchId"],
                  "results": sorted(data["results"], key=lambda x: x["packageId"])})
        state = task["status"]["state"]
        if state == S_DONE:
            if row["results_hash"] == rh:
                return task  # replay of the same results: task unchanged
            raise ApiError(409, "TASK_TERMINAL", "Task is already finished")
        if state != S_INPUT:
            raise ApiError(409, "TASK_TERMINAL", "Task is already finished")
        executions = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
                       "receiptNonce": nonce, "facts": p["facts"], "evidenceRefs": p["evidenceRefs"]}
                      for p, nonce in accepted]
        task["history"].append(msg)
        task["artifacts"].append({
            "artifactId": "receipts-" + uuid.uuid4().hex[:8], "name": "Invoice action receipts",
            "parts": [{"mediaType": RECEIPTS_MT, "data": {"batchId": stored["batchId"], "executions": executions}}]})
        task["status"] = {"state": S_DONE, "timestamp": now()}
        save_task(task, principal, rh)
        return task


def handle_send(principal, body):
    msg = body.get("message") if isinstance(body, dict) else None
    if (not isinstance(msg, dict) or not isinstance(msg.get("messageId"), str) or not msg["messageId"]
            or msg.get("role") != "ROLE_USER" or not isinstance(msg.get("parts"), list) or not msg["parts"]):
        raise ApiError(400, "INVALID_REQUEST", "Invalid message")
    mid, mh = msg["messageId"], sha(msg)
    with lock_for(f"msg:{principal}:{mid}"):
        rows = q("SELECT hash, task_id FROM messages WHERE principal=? AND message_id=?", (principal, mid))
        if rows:
            if rows[0][0] != mh:
                raise ApiError(409, "IDEMPOTENCY_CONFLICT", "Message id reused with different content")
            return get_task_row(rows[0][1])["task"]
        task = continue_task(principal, msg) if msg.get("taskId") else new_batch(principal, msg)
        q("INSERT OR REPLACE INTO messages VALUES(?,?,?,?)", (principal, mid, mh, task["id"]))
        return task


def do_cancel(principal, task_id):
    with lock_for("task:" + task_id):
        row = get_owned_task(principal, task_id)
        task = row["task"]
        if task["status"]["state"] in TERMINAL:
            raise ApiError(409, "TASK_TERMINAL", "Task is already finished")
        task["status"] = {"state": S_CANCEL, "timestamp": now()}
        save_task(task, principal)
        return task


def list_tasks(principal):
    rows = q("SELECT json FROM tasks WHERE principal=? ORDER BY created DESC", (principal,))
    out, size = [], 0
    for (j,) in rows:
        t = json.loads(j)
        if size + len(j) > 400_000:  # stay under the 512 KiB response cap
            t = {**t, "history": []}
            j = json.dumps(t)
        size += len(j)
        out.append(t)
    return out


# ---------------------------------------------------------------- HTTP layer
app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)
router = APIRouter()


def guard(request: Request):
    h = request.headers.get("authorization", "")
    parts = h.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1].strip():
        raise ApiError(401, "UNAUTHENTICATED", "Authentication required")
    if request.headers.get("a2a-version") != "1.0":
        raise ApiError(400, "VERSION_NOT_SUPPORTED", "Unsupported protocol version")
    return hashlib.sha256(parts[1].strip().encode()).hexdigest()  # principal


@app.exception_handler(ApiError)
async def _api_error(_, e: ApiError):
    return a2a({"error": {"code": e.code, "message": e.message}}, e.status)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_, e):
    return a2a({"error": {"code": "NOT_FOUND" if e.status_code == 404 else "ERROR", "message": "Request failed"}},
               e.status_code)


@app.exception_handler(Exception)
async def _any_error(_, e):
    print("ERROR:", repr(e), flush=True)
    return a2a({"error": {"code": "INTERNAL", "message": "Internal error"}}, 500)


@app.get("/.well-known/agent-card.json")
async def agent_card(request: Request):
    base = BASE_URL or f"https://{request.headers.get('host', 'localhost')}{BASE_PATH}/"
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one safe business action per invoice with evidence, "
                       "and executes only the actions the grader accepts.",
        "version": "1.0.0",
        "capabilities": {"streaming": False, "pushNotifications": False},
        "supportedInterfaces": [{"url": base, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "defaultInputModes": [BATCH_MT],
        "defaultOutputModes": [PROPOSALS_MT, RECEIPTS_MT],
        "skills": [{"id": "invoice_action_agent", "name": "Invoice action decisions",
                    "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice, "
                                   "citing the decisive evidence.",
                    "tags": ["invoice", "payments", "accounts-payable"]}],
    }
    return Response(json.dumps(card), media_type="application/json")


@router.post("/message:send")
async def message_send(request: Request):
    principal = guard(request)
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/a2a+json":
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Unsupported media type")
    try:
        body = json.loads(await request.body())
    except Exception:
        raise ApiError(400, "INVALID_REQUEST", "Malformed JSON")
    task = await run_in_threadpool(handle_send, principal, body)
    return a2a({"task": task})


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal = guard(request)
    row = await run_in_threadpool(get_owned_task, principal, task_id)
    return a2a(row["task"])


@router.get("/tasks")
async def tasks_list(request: Request):
    principal = guard(request)
    return a2a({"tasks": await run_in_threadpool(list_tasks, principal)})


@router.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = guard(request)
    return a2a(await run_in_threadpool(do_cancel, principal, task_id))


app.include_router(router, prefix=BASE_PATH)

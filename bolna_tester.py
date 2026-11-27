"""
Bolna extraction accuracy tester.

Sends every transcript in your test file to
POST /v2/agent/{agent_id}/dispositions/test (concurrently, with retries),
compares the extracted values to your expected values, and writes a report.

Your test file's column names don't have to match the agent's extraction names:
a mapping {expected column -> [agent extraction names]} decides what is compared.

CLI usage:
    export BOLNA_API_KEY=...
    python bolna_tester.py --agent-id <uuid> --input test_cases.csv --workers 8
    python bolna_tester.py --agent-id <uuid> --input test_cases.csv --resume   # continue an interrupted run
    python bolna_tester.py --agent-id <uuid> --input test_cases.csv --mapping mappings/<agent>.json
"""
import argparse
import difflib
import json
import os
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

API_BASE = "https://api.bolna.ai"
EXPECTED_PREFIX = "expected::"
MAX_TRANSCRIPT_CHARS = 50_000
MODES = ["auto", "exact", "fuzzy", "date", "datetime", "number"]
FUZZY_THRESHOLD = 0.6


BLANKS = {"", "-", "--", "---", "null", "none", "nan", "n/a", "na", "nil", "undefined", "[]", "{}"}


def is_blank(v) -> bool:
    """Empty, or a placeholder like '-', 'null', 'None', 'N/A'."""
    return v is None or str(v).strip().lower() in BLANKS


# ---------------------------------------------------------------- input ----
def load_cases(src, filename=None) -> pd.DataFrame:
    """Load CSV / XLSX / JSON / JSONL. Needs a `transcript` column.
    Expected values: columns named `expected::<Name>`,
    or (JSON/JSONL) an `expected` object {"Name": "value"}."""
    name = (filename or str(src)).lower()
    if name.endswith(".jsonl"):
        df = pd.read_json(src, lines=True, dtype=False)
    elif name.endswith(".json"):
        df = pd.read_json(src, dtype=False)
    elif name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(src, dtype=str)
    else:
        df = pd.read_csv(src, dtype=str, keep_default_na=False)

    if "transcript" not in df.columns:
        raise ValueError("Input needs a 'transcript' column.")
    if "expected" in df.columns:  # nested JSON form -> flat columns
        exp = pd.json_normalize(df["expected"].apply(lambda x: x if isinstance(x, dict) else {}))
        exp.columns = [EXPECTED_PREFIX + c for c in exp.columns]
        df = pd.concat([df.drop(columns=["expected"]), exp], axis=1)
    # No `expected::` columns at all? Treat every other column as an expected value
    # (so a file whose columns are just the extraction names still gets scored).
    reserved = {"test_id", "transcript", "call_date", "expected"}
    if not any(str(c).startswith(EXPECTED_PREFIX) for c in df.columns):
        df = df.rename(columns={c: EXPECTED_PREFIX + str(c) for c in df.columns if c not in reserved})
    if "test_id" not in df.columns:
        df.insert(0, "test_id", [f"case_{i+1:05d}" for i in range(len(df))])
    df["test_id"] = df["test_id"].astype(str)
    if df["test_id"].duplicated().any():
        raise ValueError("test_id values must be unique.")
    df = df.fillna("")
    for c in expected_columns(df):
        df[c] = df[c].astype(str).map(lambda v: "" if is_blank(v) else v)
    return df


def expected_columns(df):
    return [c for c in df.columns if c.startswith(EXPECTED_PREFIX)]


def expected_names(df):
    return [c[len(EXPECTED_PREFIX):] for c in expected_columns(df)]


# ------------------------------------------------------------------ API ----
def call_test_endpoint(session, agent_id, api_key, transcript, call_date=None,
                       max_retries=6, timeout=180):
    url = f"{API_BASE}/v2/agent/{agent_id}/dispositions/test"
    body = {"transcript": transcript[:MAX_TRANSCRIPT_CHARS]}
    if call_date:
        body["call_date"] = call_date
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    last_err = None
    for attempt in range(max_retries):
        start = time.time()
        try:
            r = session.post(url, json=body, headers=headers, timeout=timeout)
            latency = time.time() - start
            if r.status_code == 200:
                return {"ok": True, "status": 200, "latency_s": round(latency, 2), "response": r.json()}
            if r.status_code in (429, 500, 502, 503, 504):  # retryable
                last_err = f"HTTP {r.status_code}: {r.text[:300]}"
                wait = float(r.headers.get("Retry-After", 0) or 0) or min(60, 2 ** attempt + random.random())
                time.sleep(wait)
                continue
            return {"ok": False, "status": r.status_code, "latency_s": round(latency, 2),
                    "error": r.text[:1000]}  # 4xx: don't retry
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(min(60, 2 ** attempt + random.random()))
    return {"ok": False, "status": None, "latency_s": None, "error": f"Gave up after retries: {last_err}"}


# ----------------------------------------------------------------- mock ----
def mock_call(transcript, call_date=None):
    """Offline stand-in for the Bolna endpoint (same response shape as the docs),
    so the tool can be tried without an API key. Uses crude keyword rules."""
    t = transcript.lower()
    time.sleep(random.uniform(0.2, 0.8))
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", transcript)
    handover = any(k in t for k in ("real person", "human", "manager", "supervisor"))
    if "not interested" in t or "don't call" in t:
        outcome = "not_interested"
    elif handover or "call me back" in t or "later" in t:
        outcome = "follow_up"
    elif "interested" in t or "demo" in t:
        outcome = "interested"
    else:
        outcome = "unknown"

    def leaf(obj, subj=""):
        return {"subjective": subj, "objective": obj, "status": "completed", "status_reason": None}

    return {"ok": True, "status": 200, "latency_s": 0.5, "response": {"extracted_data": {
        "Lead Quality": {"Call Outcome": leaf(outcome, f"Mock: classified as {outcome}.")},
        "Escalation": {"Agent Handover Needed": leaf("Yes" if handover else "No")},
        "Contact": {"Customer Email": leaf(email.group(0).rstrip(".") if email else "")},
    }}}


# -------------------------------------------------------------- parsing ----
def flatten_extractions(resp) -> dict:
    """Walk the response and return {extraction_name: {category, objective, subjective, ...}}.
    A 'leaf' is any dict holding an 'objective' or 'subjective' key."""
    data = resp.get("extracted_data", resp) if isinstance(resp, dict) else resp
    out = {}

    def walk(node, parent_key=None):
        if not isinstance(node, dict):
            return
        for k, v in node.items():
            if isinstance(v, dict) and ("objective" in v or "subjective" in v):
                out[k] = {"category": parent_key or "", **v}
            elif isinstance(v, dict):
                walk(v, k)

    walk(data)
    return out


def leaf_value(leaf) -> str:
    """The value Bolna settled on: objective if set, else the free-text subjective.
    Placeholders like '-', 'null', 'None' count as nothing."""
    obj, subj = leaf.get("objective"), leaf.get("subjective")
    if not is_blank(obj):
        return str(obj)
    return "" if is_blank(subj) else str(subj)


_TYPE_ALIASES = {"timestamp": "timestamp", "datetime": "timestamp", "date": "timestamp", "time": "timestamp",
                 "numeric": "numeric", "number": "numeric", "integer": "numeric", "float": "numeric",
                 "boolean": "boolean", "bool": "boolean", "email": "email",
                 "regex": "regex", "custom_regex": "regex", "custom regex": "regex"}
TYPE_LABELS = {"objective": "Objective", "timestamp": "Timestamp", "numeric": "Numeric",
               "boolean": "Boolean", "email": "Email", "regex": "Custom regex", "text": "Text"}
# how each Bolna answer type is compared with the golden value
TYPE_MODE = {"objective": "exact", "timestamp": "timestamp", "numeric": "number", "boolean": "exact",
             "email": "exact", "regex": "exact", "text": "fuzzy"}


def leaf_type(leaf) -> str:
    """Bolna answer type of one extraction result: objective, or the free-text Expected Format."""
    if not is_blank(leaf.get("objective")):
        return "objective"
    vt = str((leaf.get("validation") or {}).get("expected_type") or "").strip().lower()
    return _TYPE_ALIASES.get(vt, "regex" if "regex" in vt else "text")


def leaf_reasoning(leaf) -> str:
    return str(leaf.get("reasoning_objective") or leaf.get("reasoning_subjective")
               or leaf.get("reasoning") or "")


def extraction_catalog(results) -> pd.DataFrame:
    """Every extraction the agent returned across all responses, with how often it was filled."""
    rows = {}
    for res in results.values():
        if not res.get("ok"):
            continue
        for name, leaf in flatten_extractions(res["response"]).items():
            r = rows.setdefault(name, {"extraction": name, "category": leaf.get("category", ""),
                                       "filled": 0, "objective": False, "examples": [], "_types": Counter()})
            r["_types"][leaf_type(leaf)] += 1
            v = leaf_value(leaf)
            if v.strip():
                r["filled"] += 1
                if len(r["examples"]) < 3 and v not in r["examples"]:
                    r["examples"].append(v)
            if str(leaf.get("objective") or "").strip():
                r["objective"] = True
    for r in rows.values():
        t = r.pop("_types")
        # Skipped answers carry no format info (validation is null), so they'd outvote the real type;
        # use any concrete format seen (timestamp/numeric/...) before falling back to text.
        r["type"] = "objective" if r["objective"] else next(
            (k for k, _ in t.most_common() if k not in ("objective", "text")), "text")
    df = pd.DataFrame(list(rows.values()))
    if df.empty:
        return pd.DataFrame(columns=["extraction", "category", "filled", "objective", "type", "examples"])
    return df.sort_values(["category", "extraction"]).reset_index(drop=True)


# -------------------------------------------------------------- scoring ----
_BOOL = {"yes": "true", "y": "true", "true": "true", "1": "true",
         "no": "false", "n": "false", "false": "false", "0": "false"}
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def normalize(v) -> str:
    s = re.sub(r"\s+", " ", str(v if v is not None else "")).strip().lower().strip(".!?,;:\"'")
    if s in _BOOL:
        return _BOOL[s]
    try:
        return repr(float(s.replace(",", "")))
    except ValueError:
        return s


def _parse_dt(v):
    try:
        ts = pd.to_datetime(str(v).strip())
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts
    except (ValueError, TypeError, OverflowError):
        return None


def _num(v):
    s = re.sub(r"[^\d.\-]", "", str(v).replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def similarity(a, b) -> float:
    a, b = normalize(a), normalize(b)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return 1.0
    ta, tb = set(re.findall(r"\w+", a)), set(re.findall(r"\w+", b))
    jacc = len(ta & tb) / max(len(ta | tb), 1)
    return max(difflib.SequenceMatcher(None, a, b).ratio(), jacc)


def detect_mode(expected_values, has_objective):
    vals = [v for v in expected_values if str(v).strip()]
    if not vals:
        return "exact"
    if all(_DATE_RE.match(str(v).strip()) for v in vals):
        return "datetime" if any(re.search(r"\d{1,2}:\d{2}", str(v)) for v in vals) else "date"
    if all(_num(v) is not None and re.fullmatch(r"[\d,.\s₹$rsRS-]+", str(v).strip()) for v in vals):
        return "number"
    if all(normalize(v) in ("true", "false") for v in vals) or has_objective:
        return "exact"
    return "fuzzy"


def compare(expected, predicted, mode, aliases=None):
    """Returns (match: bool, how: str). `aliases` maps normalized agent value -> expected label."""
    exp = "" if is_blank(expected) else str(expected).strip()
    pred = "" if is_blank(predicted) else str(predicted).strip()
    if aliases and normalize(pred) in aliases:
        pred = aliases[normalize(pred)]
    if not exp and not pred:
        return True, "both empty"
    if not pred:
        return False, "agent returned nothing"
    if not exp:
        return False, "agent returned a value, none expected"
    if mode == "timestamp":
        a, b = _parse_dt(exp), _parse_dt(pred)
        if a is not None and b is not None:
            if re.search(r"\d{1,2}:\d{2}", exp):
                ok = a.floor("min") == b.floor("min")
                return ok, "same date & time" if ok else "different date/time"
            return a.date() == b.date(), "same date" if a.date() == b.date() else "different date"
        return normalize(exp) == normalize(pred), "not a valid date/time"
    if mode == "date":
        a, b = _parse_dt(exp), _parse_dt(pred)
        if a is not None and b is not None:
            return a.date() == b.date(), "same date" if a.date() == b.date() else "different date"
    if mode == "datetime":
        a, b = _parse_dt(exp), _parse_dt(pred)
        if a is not None and b is not None:
            ok = a.floor("min") == b.floor("min")
            return ok, "same date & time" if ok else "different date/time"
    if mode == "number":
        a, b = _num(exp), _num(pred)
        if a is not None and b is not None:
            ok = abs(a - b) < 1e-6
            return ok, "same number" if ok else "different number"
    if normalize(exp) == normalize(pred):
        return True, "exact match"
    if mode == "fuzzy":
        s = similarity(exp, pred)
        return s >= FUZZY_THRESHOLD, (f"similar wording ({s:.0%} similar)" if s >= FUZZY_THRESHOLD
                                      else f"different wording (only {s:.0%} similar)")
    return False, "different value"


def default_mapping(df, catalog, results=None):
    """Suggest {expected name: [agent extraction names]}:
    1) same name (ignoring case/punctuation) or a near-identical name,
    2) else the extraction whose values strictly agree with >= half of the golden values."""
    names = list(catalog["extraction"]) if len(catalog) else []
    norm = {re.sub(r"[^a-z0-9]", "", n.lower()): n for n in names}
    mapping = {}
    for exp in expected_names(df):
        key = re.sub(r"[^a-z0-9]", "", exp.lower())
        scored = [(difflib.SequenceMatcher(None, key, k).ratio(), n) for k, n in norm.items()]
        best = max(scored, default=(0, None))
        if best[0] >= 0.9:
            mapping[exp] = [best[1]]
            continue
        by_val = suggest_by_values(df, results or {}, exp, strict=True)
        mapping[exp] = [by_val[0][0]] if by_val and by_val[0][2] >= 0.5 else []
    return mapping


def suggest_by_values(df, results, exp_name, strict=False, top=3):
    """Agent extractions whose values agree with this golden column: [(name, hits, share of rows)].
    strict=False also counts fuzzy text agreement (used only as a hint)."""
    col = EXPECTED_PREFIX + exp_name
    hits, rows = Counter(), 0
    for _, row in df.iterrows():
        exp = row.get(col, "")
        res = results.get(row["test_id"])
        if not str(exp).strip() or not res or not res.get("ok"):
            continue
        rows += 1
        mode = detect_mode([exp], False)
        if mode == "fuzzy" or (mode == "exact" and not strict):
            mode = "exact" if strict else "fuzzy"
        for name, leaf in flatten_extractions(res["response"]).items():
            v = leaf_value(leaf)
            if v.strip() and compare(exp, v, mode)[0]:
                hits[name] += 1
    return [(n, h, h / max(rows, 1)) for n, h in hits.most_common(top)]


def alias_candidates(df, results, mapping, modes):
    """For exact-mode columns: every distinct agent value, with the golden value it co-occurs with most."""
    rows = []
    for exp_name, sources in mapping.items():
        if not sources or modes.get(exp_name) != "exact":
            continue
        pairs = defaultdict(Counter)
        for _, row in df.iterrows():
            res = results.get(row["test_id"])
            if not res or not res.get("ok"):
                continue
            pred, _src, _leaf = pick_prediction(flatten_extractions(res["response"]), sources)
            if pred.strip():
                pairs[pred.strip()][str(row.get(EXPECTED_PREFIX + exp_name, "")).strip()] += 1
        for agent_val, cnt in sorted(pairs.items()):
            best = [e for e, _ in cnt.most_common() if e]
            rows.append({"extraction": exp_name, "agent_value": agent_val,
                         "means": best[0] if best else "", "seen": sum(cnt.values())})
    return pd.DataFrame(rows, columns=["extraction", "agent_value", "means", "seen"])


def pick_prediction(extracted, sources):
    """First non-empty value among the mapped agent extractions."""
    first = None
    for s in sources:
        leaf = extracted.get(s)
        if leaf is None:
            continue
        if first is None:
            first = (s, leaf)
        if leaf.get("status", "completed") == "completed" and leaf_value(leaf).strip():
            return leaf_value(leaf), s, leaf
    if first:
        return leaf_value(first[1]) if first[1].get("status", "completed") == "completed" else "", first[0], first[1]
    return "", "", None


def score_case(test_id, expected: dict, extracted: dict, mapping=None, modes=None, aliases=None,
               score_blanks=False, types=None):
    """expected: {expected name: golden value}. Returns one row per scored comparison."""
    rows = []
    for name, exp_val in expected.items():
        exp_val = "" if exp_val is None else str(exp_val)
        sources = (mapping or {}).get(name, [name])
        if not sources:
            continue  # not mapped -> not scored
        if not exp_val.strip() and not score_blanks:
            continue
        pred, src, leaf = pick_prediction(extracted, sources)
        atype = (types or {}).get(src) or (leaf_type(leaf) if leaf else "")
        if atype == "objective" and leaf is not None:  # compare the pre-defined value, not the free text
            obj = leaf.get("objective")
            pred = "" if is_blank(obj) or leaf.get("status", "completed") != "completed" else str(obj)
        mode = (modes or {}).get(name, "auto")
        if mode == "auto":
            mode = TYPE_MODE.get(atype) or detect_mode([exp_val], False)
        base = {"test_id": test_id, "extraction": name, "agent_field": src,
                "category": (leaf or {}).get("category", ""), "answer_type": TYPE_LABELS.get(atype, ""),
                "expected": exp_val, "mode": mode}
        if leaf is None:
            rows.append({**base, "predicted": "", "match": not exp_val.strip(), "status": "missing",
                         "confidence": None, "reasoning": "",
                         "note": "Bolna didn't return this extraction for this call"})
            continue
        status = str(leaf.get("status") or "completed")
        match, how = compare(exp_val, pred, mode, (aliases or {}).get(name))
        note = how
        if status != "completed":
            note = f"{status}: {leaf.get('status_reason') or ''}".strip(": ")
        elif pred and (leaf.get("validation") or {}).get("is_valid") is False:
            note += " · Bolna flagged the format as invalid"
        rows.append({**base, "predicted": pred, "match": bool(match), "status": status,
                     "confidence": leaf.get("confidence"), "reasoning": leaf_reasoning(leaf),
                     "note": note})
    return rows


# ------------------------------------------------------------------ run ----
def load_raw(raw_path) -> dict:
    """Read raw_responses.jsonl -> {test_id: result}; later lines win."""
    results = {}
    if raw_path and os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    if rec.get("ok") or rec["test_id"] not in results:
                        results[rec["test_id"]] = rec
    return results


def run_tests(df, agent_id, api_key, workers=5, raw_path=None, resume=False, progress_cb=None,
              mock=False, mapping=None, modes=None, aliases=None):
    """Runs every case. Raw responses are appended to raw_path (JSONL) as they arrive,
    so a crashed/stopped run can be resumed without re-calling finished cases."""
    results = run_requests(df, agent_id, api_key, workers, raw_path, resume, progress_cb, mock)
    return build_report(df, results, mapping, modes, aliases)


def run_requests(df, agent_id, api_key, workers=5, raw_path=None, resume=False, progress_cb=None,
                 mock=False):
    done = {k: v for k, v in load_raw(raw_path).items() if v.get("ok")} if resume else {}
    todo = df[~df["test_id"].isin(done)]
    lock = threading.Lock()
    results = dict(done)
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)

    def job(row):
        if mock:
            res = mock_call(str(row["transcript"]))
        else:
            res = call_test_endpoint(session, agent_id, api_key, str(row["transcript"]),
                                     str(row.get("call_date", "")) or None)
        res["test_id"] = row["test_id"]
        return res

    total, completed = len(df), len(done)
    if progress_cb:
        progress_cb(completed, total, None)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, row) for _, row in todo.iterrows()]
        for fut in as_completed(futures):
            res = fut.result()
            with lock:
                results[res["test_id"]] = res
                if raw_path:
                    with open(raw_path, "a") as f:
                        f.write(json.dumps(res, ensure_ascii=False) + "\n")
                completed += 1
            if progress_cb:
                progress_cb(completed, total, res)
    return results


def build_report(df, results, mapping=None, modes=None, aliases=None, score_blanks=False, score_text=False):
    """Scores every expected column. Each extraction is compared according to its Bolna answer type:
    objective -> exact, Timestamp -> date/time, Numeric -> number, Boolean/Email/Regex -> exact,
    Text -> similar wording. Plain Text extractions are skipped unless score_text=True
    (listed in report["text_skipped"])."""
    exp_cols = expected_columns(df)
    cat = extraction_catalog(results)
    types = dict(zip(cat["extraction"], cat["type"]))
    text_skipped = []
    if not score_text:
        keep = []
        for c in exp_cols:
            name = c[len(EXPECTED_PREFIX):]
            srcs = (mapping or {}).get(name, [name]) if mapping is not None else [name]
            found = [types[x] for x in srcs if x in types]
            if found and all(t == "text" for t in found):
                text_skipped.append(name)
            else:
                keep.append(c)
        exp_cols = keep
    long_rows, case_rows = [], []
    for _, row in df.iterrows():
        tid = row["test_id"]
        res = results.get(tid, {"ok": False, "error": "not run"})
        extracted = flatten_extractions(res["response"]) if res.get("ok") else {}
        case_rows.append({"test_id": tid, "ok": bool(res.get("ok")), "http_status": res.get("status"),
                          "latency_s": res.get("latency_s"), "error": res.get("error", ""),
                          "extracted_json": json.dumps(extracted, ensure_ascii=False)})
        if res.get("ok") and exp_cols:
            expected = {c[len(EXPECTED_PREFIX):]: row[c] for c in exp_cols}
            long_rows.extend(score_case(tid, expected, extracted, mapping, modes, aliases, score_blanks, types))

    cases = pd.DataFrame(case_rows)
    detail = pd.DataFrame(long_rows)
    if not detail.empty:
        detail["match"] = detail["match"].astype(bool)
        detail["scored"] = ~detail["expected"].map(is_blank)
        detail["captured"] = ~detail["predicted"].map(is_blank)
        detail["correct"] = detail["captured"] & detail["match"]
        detail = detail.merge(df[["test_id", "transcript"]], on="test_id", how="left")
    if detail.empty:
        summary = pd.DataFrame(columns=["category", "extraction", "captured", "correct", "wrong", "missed", "accuracy_pct"])
    else:
        d = detail.assign(missed=detail["scored"] & ~detail["captured"])
        summary = (d.groupby("extraction")
                   .agg(category=("category", lambda s: next((x for x in s if x), "")),
                        answer_type=("answer_type", lambda s: next((x for x in s if x), "")),
                        captured=("captured", "sum"), correct=("correct", "sum"), missed=("missed", "sum"))
                   .reset_index())
        summary["wrong"] = summary["captured"] - summary["correct"]
        summary = summary[["category", "extraction", "answer_type", "captured", "correct", "wrong", "missed"]]
        summary["accuracy_pct"] = (100 * summary["correct"] / summary["captured"].where(summary["captured"] > 0)).round(1)
        overall = pd.DataFrame([{"category": "", "extraction": "ALL", **{k: summary[k].sum() for k in
                                 ("captured", "correct", "wrong", "missed")}}])
        overall["accuracy_pct"] = (100 * overall["correct"] / overall["captured"].where(overall["captured"] > 0)).round(1)
        summary = pd.concat([summary.sort_values("accuracy_pct"), overall], ignore_index=True)
    unmapped = [n for n in expected_names(df) if mapping is not None and not mapping.get(n)]
    return {"summary": summary, "detail": detail, "cases": cases, "unmapped": unmapped, "types": types,
            "text_skipped": text_skipped}


def bolna_output(df, results) -> pd.DataFrame:
    """Everything Bolna returned, one row per transcript x extraction."""
    rows = []
    for tid in df["test_id"]:
        res = results.get(tid)
        if not res or not res.get("ok"):
            continue
        for name, leaf in flatten_extractions(res["response"]).items():
            rows.append({"test_id": tid, "category": leaf.get("category", ""), "extraction": name,
                         "value": leaf_value(leaf), "objective": leaf.get("objective"),
                         "subjective": leaf.get("subjective"), "status": leaf.get("status"),
                         "status_reason": leaf.get("status_reason"), "confidence": leaf.get("confidence"),
                         "reasoning": leaf_reasoning(leaf)})
    return pd.DataFrame(rows, columns=["test_id", "category", "extraction", "value", "objective", "subjective",
                                       "status", "status_reason", "confidence", "reasoning"])


def bolna_output_wide(df, results) -> pd.DataFrame:
    """One row per transcript, one column per extraction (the value Bolna settled on)."""
    long = bolna_output(df, results)
    if long.empty:
        return df[["test_id", "transcript"]].copy()
    wide = long.pivot_table(index="test_id", columns="extraction", values="value", aggfunc="first")
    return df[["test_id", "transcript"]].merge(wide.reset_index(), on="test_id", how="left")


def _style_sheet(ws, match_col=None):
    from openpyxl.styles import Alignment, Font, PatternFill
    green, red = PatternFill("solid", fgColor="DCFCE7"), PatternFill("solid", fgColor="FEE2E2")
    head = PatternFill("solid", fgColor="1F2937")
    for c in ws[1]:
        c.font, c.fill = Font(bold=True, color="FFFFFF"), head
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for col in ws.columns:
        width = max((len(str(c.value or "")) for c in col[:200]), default=8)
        ws.column_dimensions[col[0].column_letter].width = min(60, max(10, width + 2))
        for c in col[1:]:
            c.alignment = Alignment(vertical="top", wrap_text=width > 60)
    if match_col:
        idx = [c.value for c in ws[1]].index(match_col)
        for row in ws.iter_rows(min_row=2):
            v = row[idx].value
            if v in (True, False):
                for c in row:
                    c.fill = green if v else red


def _xl_safe(df):
    """Strip control characters Excel can't store (they occasionally appear in LLM output)."""
    from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
    clean = lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(clean)
    out.columns = [clean(str(c)) for c in out.columns]
    return out


def save_report(report, out_dir, config=None, df=None, results=None):
    os.makedirs(out_dir, exist_ok=True)
    xlsx = os.path.join(out_dir, "extraction_report.xlsx")
    d = report["detail"]
    with pd.ExcelWriter(xlsx) as w:
        _xl_safe(report["summary"]).to_excel(w, sheet_name="Summary", index=False)
        if not d.empty:
            _xl_safe(d[d["captured"] & ~d["correct"]]).to_excel(w, sheet_name="Wrong", index=False)
            _xl_safe(d[d["scored"] & ~d["captured"]]).to_excel(w, sheet_name="Missed", index=False)
            _xl_safe(d).to_excel(w, sheet_name="All comparisons", index=False)
        _xl_safe(report["cases"]).to_excel(w, sheet_name="Per transcript", index=False)
        if df is not None and results is not None:
            _xl_safe(bolna_output_wide(df, results)).to_excel(w, sheet_name="Bolna output", index=False)
            _xl_safe(bolna_output(df, results)).to_excel(w, sheet_name="Bolna output (all fields)", index=False)
        if config:
            pd.DataFrame([{"expected": k, "agent_fields": ", ".join(v),
                           "mode": config.get("modes", {}).get(k, "auto")}
                          for k, v in config.get("mapping", {}).items()]).to_excel(w, sheet_name="Mapping", index=False)
        for ws in w.book.worksheets:
            _style_sheet(ws, "match" if ws.title in ("Wrong", "Missed", "All comparisons") else None)
    for k in ("summary", "detail", "cases"):
        report[k].to_csv(os.path.join(out_dir, f"{k}.csv"), index=False)
    return xlsx


# ------------------------------------------------------------------ CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent-id", default="")
    ap.add_argument("--input", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, help="Only run the first N cases (smoke test)")
    ap.add_argument("--mock", action="store_true", help="Use an offline fake endpoint (no API key needed)")
    ap.add_argument("--mapping", help="JSON saved by the UI: {mapping, modes, aliases}")
    a = ap.parse_args()

    if not a.mock and not a.agent_id:
        exit("--agent-id is required (or pass --mock).")
    key = os.environ.get("BOLNA_API_KEY") or ("" if a.mock else exit("Set BOLNA_API_KEY first."))
    cases = load_cases(a.input)
    if a.limit:
        cases = cases.head(a.limit)
    cfg = json.load(open(a.mapping)) if a.mapping else {}
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    res = run_requests(cases, a.agent_id, key, a.workers,
                       raw_path=os.path.join(a.out, "raw_responses.jsonl"), resume=a.resume,
                       progress_cb=lambda d, t, _r: print(f"\r{d}/{t} done", end="", flush=True), mock=a.mock)
    mapping = cfg.get("mapping") or default_mapping(cases, extraction_catalog(res), res)
    rep = build_report(cases, res, mapping, cfg.get("modes"), cfg.get("aliases"))
    print(f"\nFinished in {time.time()-t0:.0f}s")
    if rep["unmapped"]:
        print("Not scored (no matching agent extraction):", ", ".join(rep["unmapped"]))
    print(rep["summary"].to_string(index=False))
    print("Report:", save_report(rep, a.out, {"mapping": mapping, "modes": cfg.get("modes", {})}, cases, res))

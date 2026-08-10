"""
Client for DBAASP's public REST API (https://dbaasp.org).

Endpoint pattern confirmed by reading QMAP's own fetch code
(github.com/anthol42/QMAP, data/dbaasp/dbaasp/fetch.py) and the official
dbaasp_api_helper_libraries repo (github.com/melomcr/dbaasp_api_helper_libraries):

    GET https://dbaasp.org/peptides/{id}
    Accept: application/json

MAX_ID = 24207 is the highest peptide ID confirmed to exist as of the QMAP
snapshot (January 2026). We probe a bit past it to detect if new IDs have been
added since, and stop once we hit a run of consecutive misses.
"""
import json
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter

BASE_URL = "https://dbaasp.org/peptides/{id}"
MAX_ID_JAN_2026 = 24207
PROBE_MARGIN = 500  # scan a bit past the known max in case new entries were added


def make_session():
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64)
    s.mount("https://", adapter)
    return s


def fetch_peptide(session, peptide_id, timeout=20, max_retries=3):
    """
    Returns (peptide_id, status, payload_or_None).
    status in {"ok", "not_found", "error"}.
    """
    url = BASE_URL.format(id=peptide_id)
    last_err = None
    for attempt in range(max_retries):
        try:
            r = session.get(url, headers={"accept": "application/json"}, timeout=timeout)
            if r.status_code == 200:
                try:
                    return peptide_id, "ok", r.json()
                except ValueError:
                    last_err = f"invalid json body (len={len(r.content)})"
            elif r.status_code == 400:
                # DBAASP returns HTTP 400 with {"errors": [...]} for IDs that
                # don't exist (gaps / deleted entries), not for malformed requests.
                return peptide_id, "not_found", None
            else:
                last_err = f"http {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
        time.sleep(0.5 * (attempt + 1))
    return peptide_id, "error", last_err


def load_already_fetched_ids(out_jsonl_path):
    ids = set()
    try:
        with open(out_jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["id"])
                except Exception:
                    pass
    except FileNotFoundError:
        pass
    return ids


def crawl_all(out_jsonl_path, log_path, max_workers=50, max_id=MAX_ID_JAN_2026,
              probe_margin=PROBE_MARGIN, resume=True):
    """
    Crawls peptide IDs 1..(max_id+probe_margin), writing each successful record
    as one JSON line to out_jsonl_path (streamed, so the process is resumable /
    inspectable mid-run). Logs a summary to log_path at the end.

    If resume=True and out_jsonl_path already has entries, IDs already present are
    skipped and new results are appended (not overwritten).
    """
    session = make_session()
    already = load_already_fetched_ids(out_jsonl_path) if resume else set()
    ids_to_fetch = [i for i in range(1, max_id + probe_margin + 1) if i not in already]
    print(f"Resuming: {len(already)} ids already fetched, {len(ids_to_fetch)} remaining", flush=True)

    n_ok, n_not_found, n_error = 0, 0, 0
    errors = []
    t0 = time.time()

    mode = "a" if (resume and already) else "w"
    with open(out_jsonl_path, mode) as out_f:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(fetch_peptide, session, i): i for i in ids_to_fetch}
            done = 0
            for fut in as_completed(futs):
                pid, status, payload = fut.result()
                done += 1
                if status == "ok":
                    n_ok += 1
                    out_f.write(json.dumps(payload) + "\n")
                    out_f.flush()
                elif status == "not_found":
                    n_not_found += 1
                else:
                    n_error += 1
                    errors.append((pid, payload))

                if done % 500 == 0:
                    elapsed = time.time() - t0
                    rate = done / elapsed
                    print(f"[{done}/{len(ids_to_fetch)}] ok={n_ok} not_found={n_not_found} "
                          f"error={n_error} rate={rate:.2f}/s elapsed={elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    with open(log_path, "w") as log_f:
        log_f.write(f"DBAASP crawl summary\n")
        log_f.write(f"IDs attempted: {len(ids_to_fetch)} (1..{max_id + probe_margin})\n")
        log_f.write(f"ok (200): {n_ok}\n")
        log_f.write(f"not_found (400, gap/deleted id): {n_not_found}\n")
        log_f.write(f"error (network/timeout/5xx after retries): {n_error}\n")
        log_f.write(f"elapsed_seconds: {elapsed:.1f}\n")
        if errors:
            log_f.write("\nError details (id, message):\n")
            for pid, msg in errors:
                log_f.write(f"  {pid}: {msg}\n")

    print(f"DONE. ok={n_ok} not_found={n_not_found} error={n_error} elapsed={elapsed:.0f}s")
    return n_ok, n_not_found, n_error

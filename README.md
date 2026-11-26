# Bolna extraction tester

Bulk-tests a Bolna agent's extractions (dispositions) against transcripts with golden answers and reports accuracy.

## Architecture
1. Upload CSV/XLSX/JSON/JSONL (transcript + golden answers)
2. Tool sends each transcript to `POST https://api.bolna.ai/v2/agent/{agent_id}/dispositions/test` (parallel, retries on 429/5xx, resumable)
3. Bolna returns `extracted_data` synchronously: `{category: {extraction_name: {objective, subjective, ...}}}`
4. Tool compares each extraction with the golden value (objective if present, else subjective; case/space-insensitive; yes/true/1 equivalent)
5. Report: Streamlit UI + `results/<timestamp>/extraction_report.xlsx` + CSVs + `raw_responses.jsonl`

## Input format
```
test_id,transcript,call_date,expected::Call Outcome,expected::Customer Email
tc_001,"Agent: ... Customer: ...",,follow_up,ravi@example.com
```
- `transcript` required (max 50,000 chars). `test_id`, `call_date` optional.
- `expected::<name>` must match the Bolna extraction name exactly. Blank = not scored.
- JSON/JSONL may instead use `"expected": {"Call Outcome": "follow_up"}`.

## Setup
```
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export BOLNA_API_KEY=your_key
```

## Run
UI (local, reachable only from this machine): `.venv/bin/streamlit run app.py --server.address localhost`
CLI: `python bolna_tester.py --agent-id <uuid> --input sample_test_cases.csv --workers 5 [--limit 20] [--resume]`

## Files
- `bolna_tester.py` – engine: loading, API calls, parsing, scoring, report writing
- `app.py` – Streamlit UI
- `sample_test_cases.csv` – example input
- `requirements.txt`

## Demo mode (no API key)
UI: toggle "Demo mode" in the sidebar. CLI: `python bolna_tester.py --mock --input sample_test_cases.csv`.
Uses an offline keyword-based fake that returns the documented response shape.

## Hosting (Streamlit Community Cloud, free)
1. share.streamlit.io → sign in with GitHub → **Create app** → repo `bolna-extraction-tester`, branch `main`, file `app.py`.
2. Advanced settings → Python 3.12. Deploy.
3. App settings → **Sharing** → make it private and invite teammates by email (it handles customer transcripts).
- Each user enters their own Bolna API key in the sidebar. Don't put a shared key in Streamlit secrets: everyone with access could use it.
- `results/` is on temporary disk there: saved runs disappear on restart/redeploy. Download reports after each run.
- Keep the tab open during a run; the run lives in your browser session.

## How scoring works
- **Accuracy = Correct ÷ Captured.** Captured = calls where Bolna returned a value (blank, `-`, `null`, `None`, `N/A` don't count).
- **Wrong** = captured but not the expected value. **Missed** = expected a value, Bolna returned nothing (not part of accuracy).
- Each extraction is compared by its Bolna answer type (read from the response):
  Objective → exact · Timestamp → same date (and time, if the expected value has one) · Numeric → same number ·
  Boolean / Email / Custom regex → exact. Plain **Text** extractions are not scored.
- `expected::<name>` must match the Bolna extraction name exactly. The UI lists every extraction the agent returned.

## UI
Upload → live progress per call → results with accuracy per disposition, grouped by category.
Open a disposition to see each test case, its transcript, expected vs extracted, and why it didn't match.
Downloads: Excel report, Bolna output (CSV), raw responses (JSONL). Saved runs reopen from the sidebar.

## Notes
- Response shape confirmed against Bolna docs: `extracted_data.{category}.{name}.{subjective, objective, status, status_reason, validation}`.
- `status: "skipped"` (follow-up whose parent answer didn't match) counts as no value.
- Rate limit: no endpoint-specific limit documented; default is 1000 req/min per org. 429s are retried.
- Credit cost of test calls isn't documented. Check your wallet after a small `--limit` run.
- `results/` is git-ignored: it holds transcripts and Bolna responses.

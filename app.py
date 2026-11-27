"""Upload transcripts, run them against your Bolna agent's extractions, see accuracy per disposition.
Run:  streamlit run app.py
"""
import html
import io
import json
import os
import time
import zipfile
from collections import deque
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from bolna_tester import (TYPE_LABELS, bolna_output, bolna_output_wide, build_report, expected_columns, expected_names,
                          extraction_catalog, load_cases, load_raw, run_requests, save_report)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
PAGE_SIZE = 25

st.set_page_config(page_title="Bolna Extraction Tester", page_icon="🧪", layout="wide")

# --------------------------------------------------------------- styles ----
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.hero h1 {font-size: 2rem; margin-bottom: .2rem;}
.hero p {opacity: .7; margin-top: 0;}
.kpi {border: 1px solid rgba(128,128,128,.22); border-radius: 14px; padding: 16px 18px; height: 100%;
      background: rgba(128,128,128,.04);}
.kpi .label {font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65;}
.kpi .value {font-size: 2rem; font-weight: 700; line-height: 1.2; margin-top: 4px;}
.kpi .sub {font-size: .8rem; opacity: .6;}
.good {color: #16a34a;} .warn {color: #d97706;} .bad {color: #dc2626;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .75rem; font-weight: 600;
       margin: 2px 4px 2px 0; border: 1px solid rgba(128,128,128,.25);}
.pill.ok {background: rgba(34,197,94,.14); color: #16a34a; border-color: transparent;}
.pill.no {background: rgba(239,68,68,.14); color: #dc2626; border-color: transparent;}
.pill.mid {background: rgba(245,158,11,.16); color: #d97706; border-color: transparent;}
.cat {font-size: .8rem; text-transform: uppercase; letter-spacing: .08em; opacity: .6; margin: 1.4rem 0 .3rem;}
.case {border: 1px solid rgba(128,128,128,.22); border-left: 5px solid; border-radius: 12px;
       padding: 12px 16px; margin-bottom: 10px;}
.case.ok {border-left-color: #22c55e;} .case.no {border-left-color: #ef4444;} .case.extra {border-left-color: #f59e0b;}
.case .top {display: flex; justify-content: space-between; align-items: center; gap: 8px; flex-wrap: wrap;}
.case .tid {font-weight: 700; font-family: ui-monospace, monospace; font-size: .9rem;}
.case .grid {display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 10px 0 6px;}
.case .box {background: rgba(128,128,128,.07); border-radius: 8px; padding: 8px 10px;}
.case .box .l {font-size: .7rem; text-transform: uppercase; letter-spacing: .06em; opacity: .6;}
.case .box .v {font-weight: 600; word-break: break-word;}
.case .why {font-size: .85rem; opacity: .85; margin-top: 4px;}
.case details {margin-top: 8px;}
.case summary {cursor: pointer; font-size: .85rem; opacity: .75;}
.chat {max-height: 380px; overflow-y: auto; padding: 8px 4px; font-size: .88rem;}
.msg {padding: 7px 11px; border-radius: 10px; margin: 5px 0; max-width: 85%; line-height: 1.45;}
.msg.agent {background: rgba(59,130,246,.12);}
.msg.user {background: rgba(128,128,128,.12); margin-left: auto;}
.msg b {font-size: .7rem; text-transform: uppercase; opacity: .6; display: block;}
@media (max-width: 700px) {.case .grid {grid-template-columns: 1fr;}}
</style>
""", unsafe_allow_html=True)


def esc(v):
    return html.escape("" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))


def tone(pct):
    return "good" if pct >= 90 else "warn" if pct >= 70 else "bad"


def kpi(col, label, value, sub="", cls=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value {cls}">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def chat_html(transcript):
    out = []
    for line in str(transcript).splitlines():
        line = line.strip()
        if not line:
            continue
        who, _, text = line.partition(":")
        if _ and who.strip().lower() in ("agent", "assistant", "bot"):
            out.append(f'<div class="msg agent"><b>Assistant</b>{esc(text.strip())}</div>')
        elif _ and who.strip().lower() in ("user", "customer", "caller", "human"):
            out.append(f'<div class="msg user"><b>User</b>{esc(text.strip())}</div>')
        else:
            out.append(f'<div class="msg">{esc(line)}</div>')
    return '<div class="chat">' + "".join(out) + "</div>"


# --------------------------------------------------------------- sidebar ---
with st.sidebar:
    st.markdown("### ⚙️ Connection")
    mock = st.toggle("Demo mode", value=False, help="Offline fake responses, no API key needed.")
    api_key = st.text_input("Bolna API key", type="password", value=os.environ.get("BOLNA_API_KEY", ""),
                            disabled=mock)
    agent_id = st.text_input("Agent ID", disabled=mock)
    st.markdown("### 🚀 Run settings")
    workers = st.slider("Parallel requests", 1, 20, 5, help="Raise it if you don't see HTTP 429 errors.")
    limit = st.number_input("Only run the first N cases (0 = all)", min_value=0, value=0, step=10)

    st.markdown("### 🗂 Previous runs")
    runs = sorted([d for d in os.listdir(RESULTS_DIR)
                   if os.path.exists(os.path.join(RESULTS_DIR, d, "raw_responses.jsonl"))], reverse=True) \
        if os.path.isdir(RESULTS_DIR) else []

    def run_label(d):
        rd = os.path.join(RESULTS_DIR, d)
        try:
            when = datetime.strptime(d, "%Y%m%d_%H%M%S").strftime("%d %b %Y, %H:%M")
        except ValueError:
            when = d
        try:
            n = sum(1 for _ in open(os.path.join(rd, "raw_responses.jsonl")))
        except OSError:
            n = 0
        try:
            agent = json.load(open(os.path.join(rd, "meta.json"))).get("agent_id") or "demo"
        except (OSError, ValueError):
            agent = "?"
        return f"{when} · {n} responses · agent {agent[:8]}"

    if runs:
        pick = st.selectbox("Open a saved run", runs, index=None, placeholder="Choose a run…",
                            format_func=run_label)
        if pick:
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for fn in os.listdir(os.path.join(RESULTS_DIR, pick)):
                    z.write(os.path.join(RESULTS_DIR, pick, fn), f"{pick}/{fn}")
            st.download_button("⬇️ Download raw run files (zip)", buf.getvalue(), f"bolna_run_{pick}.zip",
                               width="stretch",
                               help="Your uploaded file + every Bolna response. Works even if the report fails.")
        has_input = bool(pick) and os.path.exists(os.path.join(RESULTS_DIR, pick, "input.csv"))
        if st.button("Open run", disabled=not has_input, width="stretch"):
            rd = os.path.join(RESULTS_DIR, pick)
            st.session_state.update(df=load_cases(os.path.join(rd, "input.csv")),
                                    results=load_raw(os.path.join(rd, "raw_responses.jsonl")),
                                    run_dir=rd, file_key=f"run:{pick}", scroll=True)
    else:
        st.caption("Runs you finish will show up here.")

# ------------------------------------------------------------------ hero ---
st.markdown('<div class="hero"><h1>🧪 Bolna Extraction Tester</h1>'
            '<p>Upload transcripts with expected answers, run them through your agent\'s extractions, '
            'and see how each disposition performs.</p></div>', unsafe_allow_html=True)

# ---------------------------------------------------------- 1. upload ------
with st.container(border=True):
    st.markdown("#### 1 · Upload test cases")
    up_col, help_col = st.columns([3, 2])
    with up_col:
        uploaded = st.file_uploader("CSV, Excel, JSON or JSONL", type=["csv", "xlsx", "json", "jsonl"],
                                    label_visibility="collapsed")
    with help_col:
        st.caption("Needs a `transcript` column with one line per turn: `assistant:` (the agent) and "
                   "`user:` (the customer). Add one `expected::<Disposition name>` column per "
                   "disposition. The name must match the extraction name in Bolna exactly.")
        with open(os.path.join(HERE, "sample_test_cases.csv"), "rb") as f:
            st.download_button("⬇️ Sample CSV", f.read(), "sample_test_cases.csv")

    if uploaded:
        key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("file_key") != key:
            try:
                st.session_state.update(df=load_cases(io.BytesIO(uploaded.getvalue()), uploaded.name),
                                        file_key=key, results=None, run_dir=None)
            except Exception as e:
                st.error(f"Couldn't read the file: {e}")
                st.stop()

    df = st.session_state.get("df")
    if df is None:
        st.stop()
    if limit:
        df = df.head(int(limit))

    names = expected_names(df)
    c1, c2, c3 = st.columns([1, 1, 3])
    kpi(c1, "Transcripts", f"{len(df):,}")
    kpi(c2, "Dispositions to score", len(names))
    c3.markdown("**Dispositions in your file**<br>" +
                ("".join(f'<span class="pill">{esc(n)}</span>' for n in names) or
                 "<i>None. Results will show what Bolna extracted, without scoring.</i>"),
                unsafe_allow_html=True)
    with st.expander("Preview file"):
        st.dataframe(df.head(50), width="stretch", hide_index=True)

# ------------------------------------------------------------- 2. run ------
with st.container(border=True):
    st.markdown("#### 2 · Run")
    ready = mock or (api_key and agent_id)
    if not ready:
        st.info("Enter your Bolna API key and Agent ID in the sidebar (or turn on Demo mode).", icon="🔑")
    if st.button(f"▶  Run {len(df):,} transcripts", type="primary", disabled=not ready,
                 width="stretch"):
        run_dir = os.path.join(RESULTS_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(run_dir, exist_ok=True)
        df.to_csv(os.path.join(run_dir, "input.csv"), index=False)
        json.dump({"agent_id": agent_id, "mock": mock}, open(os.path.join(run_dir, "meta.json"), "w"))

        with st.status(f"Running {len(df):,} transcripts through Bolna…", expanded=True) as status:
            bar = st.progress(0.0, text="Starting…")
            m1, m2, m3, m4 = st.columns(4)
            ph = [m.empty() for m in (m1, m2, m3, m4)]
            log_ph = st.empty()
            log, stats, t0 = deque(maxlen=12), {"ok": 0, "fail": 0, "lat": []}, time.time()

            def progress(done, total, res):
                if res is not None:
                    if res.get("ok"):
                        stats["ok"] += 1
                        stats["lat"].append(res.get("latency_s") or 0)
                    else:
                        stats["fail"] += 1
                    log.appendleft({"": "✅" if res.get("ok") else "❌", "test_id": res["test_id"],
                                    "latency (s)": res.get("latency_s"),
                                    "detail": "" if res.get("ok") else str(res.get("error", ""))[:120]})
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 and done else 0
                eta = (total - done) / rate if rate else 0
                bar.progress(done / max(total, 1), text=f"{done:,} of {total:,} calls done")
                ph[0].metric("Done", f"{done:,}/{total:,}")
                ph[1].metric("Succeeded", f"{stats['ok']:,}")
                ph[2].metric("Failed", f"{stats['fail']:,}")
                ph[3].metric("Time left", f"~{eta/60:.1f} min" if done else "…")
                if log:
                    log_ph.dataframe(pd.DataFrame(log), width="stretch", hide_index=True)

            results = run_requests(df, agent_id.strip(), api_key.strip(), workers,
                                   raw_path=os.path.join(run_dir, "raw_responses.jsonl"),
                                   progress_cb=progress, mock=mock)
            status.update(label=f"Done: {stats['ok']:,} succeeded, {stats['fail']:,} failed, "
                                f"in {(time.time()-t0)/60:.1f} min", state="complete", expanded=False)
        st.session_state.update(results=results, run_dir=run_dir, scroll=True)

results = st.session_state.get("results")
if not results:
    st.stop()

# ---------------------------------------------------------- 3. results -----
report = build_report(df, results, score_blanks=True)
detail, cases = report["detail"], report["cases"]
run_dir = st.session_state.get("run_dir") or RESULTS_DIR
n_ok = int(cases["ok"].sum())
captured_rows = detail[detail["captured"]] if not detail.empty else detail
returned = set(extraction_catalog(results)["extraction"])

st.header("Results", anchor="results")
if st.session_state.pop("scroll", False):
    components.html("<script>setTimeout(()=>{const e=window.parent.document.getElementById('results');"
                    "if(e)e.scrollIntoView({behavior:'smooth',block:'start'});},300);</script>", height=0)

n_cap = len(captured_rows)
n_cor = int(captured_rows["correct"].sum()) if n_cap else 0
overall = 100 * n_cor / n_cap if n_cap else 0
lat = cases["latency_s"].dropna()
k = st.columns(5)
kpi(k[0], "Overall accuracy", f"{overall:.1f}%" if n_cap else "–",
    f"{n_cor:,} correct of {n_cap:,} captured", tone(overall) if n_cap else "")
kpi(k[1], "Transcripts processed", f"{n_ok:,}", f"of {len(cases):,} uploaded")
kpi(k[2], "Failed requests", f"{len(cases) - n_ok:,}", "see bottom of page" if len(cases) - n_ok else "none",
    "bad" if len(cases) - n_ok else "good")
kpi(k[3], "Dispositions scored", f"{detail['extraction'].nunique() if len(detail) else 0}",
    f"{len(returned)} returned by the agent")
kpi(k[4], "Avg latency", f"{lat.mean():.1f}s" if len(lat) else "–", "per transcript")

# ---- downloads
st.write("")
xlsx_key = (run_dir, len(results), len(df))
if st.session_state.get("xlsx_key") != xlsx_key:
    with st.spinner("Building report…"):
        try:  # a report failure must never hide the results or the other downloads
            with open(save_report(report, run_dir, df=df, results=results), "rb") as f:
                st.session_state.update(xlsx_bytes=f.read(), xlsx_key=xlsx_key, xlsx_err=None)
        except Exception as e:
            st.session_state.update(xlsx_bytes=None, xlsx_key=xlsx_key, xlsx_err=str(e)[:300])
xlsx_bytes = st.session_state.get("xlsx_bytes")
if st.session_state.get("xlsx_err"):
    st.warning(f"Couldn't build the Excel report ({st.session_state['xlsx_err']}). "
               "The CSV and raw downloads below still have all the data.", icon="⚠️")
d1, d2, d3 = st.columns(3)
d1.download_button("⬇️ Full report (Excel)", xlsx_bytes or b"", "extraction_report.xlsx", type="primary",
                   width="stretch", disabled=not xlsx_bytes,
                   help="Summary, wrong, missed, all comparisons, and everything Bolna extracted.")
d2.download_button("⬇️ Bolna output (CSV)", bolna_output_wide(df, results).to_csv(index=False).encode("utf-8-sig"),
                   "bolna_output.csv", width="stretch",
                   help="One row per transcript, one column per extraction.")
d3.download_button("⬇️ Raw Bolna responses (JSONL)",
                   "\n".join(json.dumps(results[t], ensure_ascii=False) for t in df["test_id"] if t in results)
                   .encode("utf-8"), "raw_responses.jsonl", width="stretch",
                   help="Exactly what the API returned, one JSON per line.")

# ---- per-disposition stats
missing = [n for n in names if n not in returned]
if missing:
    st.warning("Bolna never returned these extractions, so they can't be scored. Check the column name "
               "matches the extraction name in Bolna exactly: " + ", ".join(f"**{m}**" for m in missing),
               icon="⚠️")

text_skipped = report.get("text_skipped", [])
if text_skipped:
    st.info("Not scored, because Bolna answers these as plain **Text** (free wording rarely matches a golden "
            "answer): " + ", ".join(f"**{n}**" for n in text_skipped)
            + ". Their values are still in the downloads and in *Everything Bolna extracted* below.", icon="📝")

stats = []
for name in names:
    if name in missing or name in text_skipped:
        continue
    dd = detail[detail["extraction"] == name]
    captured, correct = int(dd["captured"].sum()), int(dd["correct"].sum())
    cat = dd["category"].replace("", pd.NA).dropna()
    typ = dd["answer_type"].replace("", pd.NA).dropna()
    stats.append({"category": cat.mode().iloc[0] if len(cat) else "Uncategorised", "disposition": name,
                  "type": typ.mode().iloc[0] if len(typ) else "",
                  "captured": captured, "correct": correct, "wrong": captured - correct,
                  "missed": int((dd["scored"] & ~dd["captured"]).sum()),
                  "accuracy": round(100 * correct / captured, 1) if captured else None})
stats = pd.DataFrame(stats)

if not stats.empty:
    st.subheader("Accuracy by disposition")
    st.caption("Each disposition is compared by its Bolna answer type: **Objective** exact value · "
               "**Timestamp** same date (and time, if your expected value has one) · **Numeric** same number · "
               "**Boolean / Email / Regex** exact. Plain **Text** is not scored.  \n"
               "**Accuracy = Correct ÷ Captured.** **Captured** = calls where Bolna returned a value "
               "(blank, `-`, `null`, `None`, `N/A` don't count). **Wrong** = captured but not what you expected, "
               "including values where you expected none. **Missed** = you expected a value but Bolna returned "
               "nothing (not part of accuracy).")
    chart_df = stats.melt(id_vars=["disposition"], value_vars=["correct", "wrong"],
                          var_name="result", value_name="count")
    order = stats.sort_values("accuracy", na_position="last")["disposition"].tolist()
    chart = (alt.Chart(chart_df).mark_bar(cornerRadius=3, height=22)
             .encode(y=alt.Y("disposition:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
                     x=alt.X("count:Q", stack="zero", title="Calls captured"),
                     color=alt.Color("result:N", scale=alt.Scale(domain=["correct", "wrong"],
                                                                 range=["#22c55e", "#ef4444"]),
                                     legend=alt.Legend(orient="top", title=None)),
                     tooltip=["disposition", "result", "count"])
             .properties(height=max(120, 36 * len(stats))))
    st.altair_chart(chart, use_container_width=True)

    st.dataframe(stats.sort_values(["category", "accuracy"]), width="stretch", hide_index=True,
                 column_config={
                     "accuracy": st.column_config.ProgressColumn("Accuracy", format="%.1f%%",
                                                                 min_value=0, max_value=100),
                     "captured": st.column_config.NumberColumn("Captured", help="Bolna returned a value"),
                     "type": st.column_config.TextColumn("Answer type"),
                     "correct": "✅ Correct", "wrong": "❌ Wrong",
                     "missed": st.column_config.NumberColumn("⚪ Missed",
                                                             help="Expected a value, Bolna returned nothing")})

    # ---- drill-down by category -> disposition
    st.subheader("Drill down")
    st.caption("Open a disposition to see every test case, its transcript, and why it didn't match.")
    for cat, group in stats.groupby("category", sort=True):
        st.markdown(f'<div class="cat">{esc(cat)}</div>', unsafe_allow_html=True)
        for _, s in group.sort_values("accuracy", na_position="last").iterrows():
            acc = s["accuracy"]
            icon = "⚪" if acc is None or pd.isna(acc) else "🟢" if acc >= 90 else "🟡" if acc >= 70 else "🔴"
            acc_txt = "nothing captured" if acc is None or pd.isna(acc) else f"{acc:.0f}%"
            label = (f"{icon} **{s['disposition']}** `{s['type'] or '—'}` — "
                     f"{s['correct']:,}/{s['captured']:,} correct ({acc_txt})"
                     + (f" · ❌ {s['wrong']:,} wrong" if s["wrong"] else "")
                     + (f" · ⚪ {s['missed']:,} missed" if s["missed"] else ""))
            with st.expander(label):
                dd = detail[detail["extraction"] == s["disposition"]].copy()
                dd["kind"] = "empty"
                dd.loc[dd["captured"] & dd["correct"], "kind"] = "ok"
                dd.loc[dd["captured"] & ~dd["correct"], "kind"] = "no"
                dd.loc[dd["scored"] & ~dd["captured"], "kind"] = "missed"
                opts = {"no": ["no"], "ok": ["ok"], "missed": ["missed"], "all": ["ok", "no"]}
                labels = {"no": f"❌ Wrong ({(dd.kind == 'no').sum()})",
                          "ok": f"✅ Correct ({(dd.kind == 'ok').sum()})",
                          "missed": f"⚪ Missed ({(dd.kind == 'missed').sum()})",
                          "all": f"All captured ({int(dd.captured.sum())})"}
                fcol, scol = st.columns([3, 2])
                view = fcol.segmented_control("Show", list(opts), default="no", format_func=labels.get,
                                              key=f"f::{s['disposition']}", label_visibility="collapsed")
                q = scol.text_input("Search", placeholder="Search test id, value or transcript…",
                                    key=f"q::{s['disposition']}", label_visibility="collapsed")
                rows = dd[dd["kind"].isin(opts.get(view or "no"))]
                if q:
                    ql = q.lower()
                    rows = rows[rows[["test_id", "expected", "predicted", "transcript"]].astype(str)
                                .apply(lambda r: ql in " ".join(r).lower(), axis=1)]
                if rows.empty:
                    st.caption("Nothing to show here.")
                    continue
                pages = (len(rows) - 1) // PAGE_SIZE + 1
                page = 1
                if pages > 1:
                    page = st.number_input(f"Page (of {pages})", 1, pages, 1, key=f"p::{s['disposition']}")
                cards = []
                for _, r in rows.iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE].iterrows():
                    kind = r["kind"]
                    badge = {"ok": '<span class="pill ok">Match</span>',
                             "no": '<span class="pill no">Mismatch</span>',
                             "missed": '<span class="pill mid">Missed</span>'}[kind]
                    conf = r.get("confidence")
                    conf_txt = f'<span class="pill">confidence {float(conf):.2f}</span>' \
                        if conf is not None and not pd.isna(conf) and str(conf) != "" else ""
                    why = ""
                    if kind != "ok":
                        why = f'<div class="why"><b>Why:</b> {esc(r.get("note"))}</div>'
                    if r.get("reasoning"):
                        why += f'<div class="why"><b>Bolna\'s reasoning:</b> {esc(r["reasoning"])}</div>'
                    cards.append(
                        f'<div class="case {"ok" if kind == "ok" else "no" if kind == "no" else "extra"}">'
                        f'<div class="top"><span class="tid">{esc(r["test_id"])}</span>'
                        f'<span>{badge}{conf_txt}</span></div>'
                        f'<div class="grid"><div class="box"><div class="l">Expected</div>'
                        f'<div class="v">{esc(r["expected"]) or "<i>(blank)</i>"}</div></div>'
                        f'<div class="box"><div class="l">Bolna extracted</div>'
                        f'<div class="v">{esc(r["predicted"]) or "<i>(nothing)</i>"}</div></div></div>'
                        f'{why}<details><summary>📄 Transcript</summary>{chat_html(r["transcript"])}</details>'
                        f'</div>')
                st.markdown("".join(cards), unsafe_allow_html=True)
                if pages > 1:
                    st.caption(f"Showing {PAGE_SIZE * (page-1) + 1}–{min(PAGE_SIZE * page, len(rows))} "
                               f"of {len(rows):,}")

# ---- everything else Bolna returned
cat_df = extraction_catalog(results)
others = cat_df[~cat_df["extraction"].isin(names)]
with st.expander(f"🗂 Other extractions Bolna returned ({len(others)}), not in your test file"):
    if others.empty:
        st.caption("None.")
    else:
        others = others.assign(captured_pct=(100 * others["filled"] / max(n_ok, 1)).round(1),
                               examples=others["examples"].apply(lambda x: " · ".join(map(str, x))),
                               type=others["type"].map(TYPE_LABELS))
        st.dataframe(others[["category", "extraction", "type", "filled", "captured_pct", "examples"]],
                     width="stretch", hide_index=True,
                     column_config={"filled": "Captured",
                                    "captured_pct": st.column_config.ProgressColumn(
                                        "Captured %", format="%.0f%%", min_value=0, max_value=100)})
        st.caption("Tip: to score one of these, add a column named `expected::<extraction>` to your file.")

with st.expander("🔎 Everything Bolna extracted, per transcript"):
    st.dataframe(bolna_output_wide(df, results), width="stretch", hide_index=True)

failed = cases[~cases["ok"]]
if len(failed):
    with st.expander(f"⚠️ Failed requests ({len(failed)})", expanded=True):
        st.dataframe(failed[["test_id", "http_status", "error"]], width="stretch", hide_index=True)

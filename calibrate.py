"""Calibration page, steps 1 to 3 (runs in the browser with PyScript).

The calculations live in modules/calibration_core.py. This file reads the page,
calls those functions, and writes tables and Plotly charts back to the page.
"""
import asyncio
import html
import math

import numpy as np
import pandas as pd
from js import Blob, Object, Plotly, URL, window
from pyodide.ffi import create_proxy, to_js
from pyscript import document, when

from modules import calibration_core as core


# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------
class State:
    def __init__(self):
        self.kind = None          # upload | paste | manual
        self.frames = []          # raw tables, one per file or manual test
        self.raw = None
        self.mapping = {}
        self.overrides = {}       # per test: type, p0, e0, ocr typed on the page
        self.built = None
        self.included = {}        # step 2 checkboxes
        self.excluded = {}        # step 2 excluded strain ranges
        self.selected = None      # test shown in step 2
        self.accepted = {}        # step 3 checkboxes
        self.csl_test = None      # test shown in the step 3 check plots
        self.points = []
        self.fit = None
        self.csl_click_bound = False


S = State()


def el(id_):
    return document.getElementById(id_)


def esc(s):
    return html.escape(str(s))


def fmt(v, digits=3):
    if v is None or not core.finite(v):
        return "&ndash;"
    return f"{float(v):.{digits}f}"


def jsobj(obj):
    return to_js(obj, dict_converter=Object.fromEntries)


def lst(a):
    return [float(v) if v is not None and core.finite(v) else float("nan") for v in a]


def test_info(tid):
    return next((t for t in S.built["tests"] if t["tid"] == tid), None) if S.built else None


def test_rows(tid):
    d = S.built["data"]
    return d[d["tid"] == tid]


def set_status(message="", kind="note"):
    box = el("import-status")
    box.className = "cal-warning" if kind == "warning" else "cal-note cal-status"
    box.innerHTML = message
    box.hidden = not message


def set_flow(index):
    for i, li in enumerate(document.querySelectorAll("#step-1 .cal-flow li")):
        li.classList.toggle("is-here", i == index)


def download(text, filename, mime="text/plain"):
    blob = Blob.new(to_js([text]), jsobj({"type": mime}))
    url = URL.createObjectURL(blob)
    a = document.createElement("a")
    a.href = url
    a.download = filename
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)


# --------------------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------------------
INK = "#1e293b"
MUTED = "#64748b"
FAINT = "#cbd5e1"
GRID = "#e2e8f0"
ACCENT = "#b28c35"
BLUE = "#2563eb"
CONFIG = {"responsive": True, "displaylogo": False,
          "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}

PLACEHOLDERS = {}


def remember_placeholders():
    for div in document.querySelectorAll(".cal-plot"):
        PLACEHOLDERS[div.id] = div.innerHTML


def clear_plot(div_id):
    div = el(div_id)
    if div.classList.contains("is-plotted"):
        Plotly.purge(div)
        div.classList.remove("is-plotted")
        div.style.height = ""
        div.innerHTML = PLACEHOLDERS.get(div_id, "")


def axis(title, **kw):
    # Plotly understands <sub> but not named entities such as &epsilon;, so turn those into characters.
    a = {"title": {"text": html.unescape(title), "font": {"size": 11}}, "gridcolor": GRID, "zeroline": False,
         "showline": True, "linecolor": "#94a3b8", "mirror": True, "ticks": "inside",
         "automargin": True}
    a.update(kw)
    return a


def draw(div_id, traces, xaxis, yaxis, title=None, legend=False, shapes=None):
    div = el(div_id)
    if not div.classList.contains("is-plotted"):
        div.innerHTML = ""
        div.classList.add("is-plotted")
    height = 440 if div.classList.contains("cal-plot--tall") else 290
    # A responsive Plotly chart fills its box, so the box needs a real height (plus its padding).
    div.style.height = f"{height + 18}px"
    layout = {
        "height": height,
        "margin": {"l": 8, "r": 12, "t": 34 if title else 12, "b": 8},
        "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "#ffffff",
        "font": {"family": "Inter, system-ui, sans-serif", "size": 11, "color": INK},
        "xaxis": xaxis, "yaxis": yaxis, "hovermode": "closest",
        "showlegend": legend,
        "legend": {"orientation": "h", "x": 0, "y": -0.16, "yanchor": "top", "font": {"size": 10}},
    }
    if title:
        layout["title"] = {"text": html.unescape(title), "font": {"size": 12}, "x": 0.01, "xanchor": "left"}
    if shapes:
        layout["shapes"] = shapes
    Plotly.react(div, jsobj(traces), jsobj(layout), jsobj(CONFIG))
    return div


def line(x, y, name="", color=INK, dash="solid", width=2, **kw):
    return {"type": "scatter", "mode": "lines", "x": lst(x), "y": lst(y), "name": name,
            "line": {"color": color, "dash": dash, "width": width}, **kw}


def markers(x, y, name="", color=INK, symbol="circle", size=8, **kw):
    return {"type": "scatter", "mode": "markers", "x": lst(x), "y": lst(y), "name": name,
            "marker": {"color": color, "symbol": symbol, "size": size,
                       "line": {"color": INK, "width": 1}}, **kw}


# --------------------------------------------------------------------------------------
# Step 1: import
# --------------------------------------------------------------------------------------
def page_settings():
    return {
        "stress_unit": el("unit-stress").value,
        "strain_unit": el("unit-strain").value,
        "sign": el("sign-convention").value,
        "epsv_sign": el("epsv-sign").value,
    }


def load_frames(frames, kind, read_errors=()):
    if not frames:
        set_status("<br>".join(esc(e) for e in read_errors) or "No data found.", "warning")
        return
    S.kind = kind
    S.frames = frames
    S.raw = core.combine(frames)
    S.mapping = core.detect_mapping(core.data_columns(S.raw))
    S.overrides, S.included, S.excluded, S.accepted = {}, {}, {}, {}
    S.selected = S.csl_test = None
    for sel in document.querySelectorAll("#column-map select"):
        sel.disabled = True
    el("mapping-status").textContent = ""
    process(read_errors)
    set_flow(2)


def process(read_errors=()):
    if S.raw is None:
        return
    S.built = core.build(S.raw, S.mapping, page_settings(), S.overrides)
    tids = [t["tid"] for t in S.built["tests"]]
    for t in tids:
        S.included.setdefault(t, True)
        S.excluded.setdefault(t, [])
        S.accepted.setdefault(t, True)
    if S.selected not in tids:
        S.selected = tids[0] if tids else None
    render_mapping()
    render_preview()
    render_metadata()
    report_import(read_errors)
    render_cards()
    render_review()
    update_csl()


def report_import(read_errors=()):
    b = S.built
    parts = []
    if b["errors"]:
        set_status("<br>".join(esc(e) for e in list(read_errors) + b["errors"]), "warning")
        return
    n_tests = sum(1 for t in b["tests"] if t["n_rows"])
    parts.append(f"Read {n_tests} test{'s' if n_tests != 1 else ''} and {len(b['data'])} rows "
                 f"from {len(S.frames)} source{'s' if len(S.frames) != 1 else ''}.")
    if len(b["bad_rows"]):
        reasons = b["bad_rows"]["reason"].value_counts()
        parts.append(f"{len(b['bad_rows'])} row(s) skipped: "
                     + ", ".join(f"{n} with {r}" for r, n in reasons.items()) + ".")
    flagged = sum(1 for t in b["tests"] if t["warnings"])
    if flagged:
        parts.append(f"{flagged} test(s) have flags. See the Checks column below.")
    for e in read_errors:
        parts.append(esc(e))
    set_status(" ".join(parts), "warning" if read_errors else "note")


BADGES = {
    "found": ("badge--open", "Found"),
    "derived": ("badge--key", "Derived"),
    "file": ("badge--key", "From file name"),
    "missing": ("badge--external", "Missing"),
    "needed": ("badge--external", "Needed"),
    "unused": ("badge--soon", "Not used"),
}


def render_mapping():
    columns = core.data_columns(S.raw)
    status = S.built["status"]
    for sel in document.querySelectorAll("#column-map select"):
        key = sel.dataset.quantity
        current = S.mapping.get(key)
        opts = ['<option value="">Not used</option>']
        for c in columns:
            opts.append(f'<option value="{esc(c)}"{" selected" if c == current else ""}>{esc(c)}</option>')
        sel.innerHTML = "".join(opts)
        cls, label = BADGES.get(status.get(key, "unused"), BADGES["unused"])
        sel.closest("tr").querySelector("td:last-child").innerHTML = f'<span class="badge {cls}">{label}</span>'


def render_preview():
    body = el("data-preview").querySelector("tbody")
    d = S.built["data"]
    if d.empty:
        body.innerHTML = '<tr class="cal-table__empty"><td colspan="7">No usable rows. Check the column mapping and units.</td></tr>'
        return
    rows = []
    for _, g in d.groupby("tid", sort=False):
        for _, r in g.head(3).iterrows():
            rows.append(
                f"<tr><td>{esc(r['tid'])}</td><td class='num'>{fmt(r['e1'], 3)}</td>"
                f"<td class='num'>{fmt(r['q'], 1)}</td><td class='num'>{fmt(r['p'], 1)}</td>"
                f"<td class='num'>{fmt(r['ev'], 3)}</td><td class='num'>{fmt(r['du'], 1)}</td>"
                f"<td class='num'>{fmt(r['e'], 4)}</td></tr>")
        if len(rows) >= 30:
            break
    body.innerHTML = "".join(rows)


def render_metadata():
    body = el("test-metadata").querySelector("tbody")
    tests = S.built["tests"]
    if not tests:
        body.innerHTML = '<tr class="cal-table__empty"><td colspan="7">No tests found.</td></tr>'
        return
    rows = []
    for t in tests:
        tid = esc(t["tid"])
        opts = "".join(
            f'<option value="{v}"{" selected" if t["type"] == v else ""}>{label}</option>'
            for v, label in (("CID", "CID"), ("CIU", "CIU"), ("UNKNOWN", "Choose…")))
        def box(field, value, digits):
            v = f"{value:.{digits}f}" if core.finite(value) else ""
            return f'<input type="number" step="any" data-field="{field}" value="{v}">'
        checks = ("; ".join(esc(w) for w in t["warnings"])) if t["warnings"] else '<span class="badge badge--open">OK</span>'
        rows.append(
            f'<tr data-tid="{tid}"><td>{tid}</td><td><select data-field="type">{opts}</select></td>'
            f'<td>{box("p0", t["p0"], 1)}</td><td>{box("e0", t["e0"], 4)}</td><td>{box("ocr", t["ocr"], 2)}</td>'
            f'<td class="num">{t["n_rows"]}</td><td class="cal-checks">{checks}</td></tr>')
    body.innerHTML = "".join(rows)


async def read_file_list(files):
    frames, errors = [], []
    for i in range(files.length):
        f = files.item(i)
        if f.name.lower().endswith((".xlsx", ".xls")):
            errors.append(f"{f.name}: Excel files cannot be read here yet. Save the sheet as CSV, or copy and paste the cells.")
            continue
        try:
            frames.append(core.read_table(await f.text(), f.name))
        except Exception as ex:  # noqa: BLE001 - shown to the user
            errors.append(str(ex))
    load_frames(frames, "upload", errors)


@when("change", "#lab-file")
async def on_file(event):
    files = event.target.files
    if files and files.length:
        await read_file_list(files)


@when("click", "#read-paste")
def on_paste(event):
    text = el("lab-paste").value
    try:
        load_frames([core.read_table(text, "pasted data")], "paste")
    except Exception as ex:  # noqa: BLE001
        set_status(esc(ex), "warning")


@when("click", "#manual-save")
def on_manual_save(event):
    tid = el("manual-id").value.strip() or "Manual-01"
    typ = el("manual-type").value
    last = "vol_strain_pct" if typ == "CID" else "du_kpa"
    rows = []
    for tr in document.querySelectorAll("#manual-rows tbody tr"):
        vals = [i.value for i in tr.querySelectorAll("input")]
        if any(v.strip() for v in vals):
            rows.append(vals)
    if not rows:
        set_status("Enter at least one row of measurements.", "warning")
        return
    frame = pd.DataFrame(rows, columns=["axial_strain_pct", "q_kpa", "p_kpa", last]).astype(str)
    frame.insert(0, "test_id", tid)
    frame.insert(1, "test_type", typ)
    frame["p0_kpa"] = el("manual-p0").value
    frame["e0"] = el("manual-e0").value
    frame["ocr"] = el("manual-ocr").value
    frame["__source"] = "manual"
    frame["__row"] = np.arange(1, len(frame) + 1)
    kept = [f for f in S.frames if f["test_id"].iloc[0] != tid] if S.kind == "manual" else []
    load_frames(kept + [frame], "manual")
    set_status(f"Saved {esc(tid)}. Add another test or continue to step 2.")


@when("change", "#column-map")
def on_mapping_change(event):
    sel = event.target
    S.mapping[sel.dataset.quantity] = sel.value or None
    el("mapping-status").textContent = ""
    process()


@when("click", "#mapping-confirm")
def on_mapping_confirm(event):
    if S.raw is not None:
        set_flow(4)


@when("change", "#unit-stress, #unit-strain, #sign-convention, #epsv-sign")
def on_units(event):
    process()


@when("change", "#test-metadata")
def on_metadata(event):
    target = event.target
    tr = target.closest("tr")
    if tr is None or not target.dataset.field:
        return
    tid = tr.dataset.tid
    field = target.dataset.field
    ov = S.overrides.setdefault(tid, {})
    if field == "type":
        ov["type"] = target.value
    else:
        try:
            ov[field] = float(target.value)
        except ValueError:
            ov.pop(field, None)
    process()


# --------------------------------------------------------------------------------------
# Step 2: review
# --------------------------------------------------------------------------------------
TYPE_NAMES = {"CID": "Drained", "CIU": "Undrained", "UNKNOWN": "Type not set"}


def render_cards():
    box = el("test-cards")
    tests = [t for t in S.built["tests"] if t["n_rows"]]
    if not tests:
        box.innerHTML = '<p class="cal-note">No tests to review yet.</p>'
        return
    cards = []
    for t in tests:
        tid = esc(t["tid"])
        n = len(t["warnings"])
        badge = ('<span class="badge badge--open">OK</span>' if not n
                 else f'<span class="badge badge--external">{n} flag{"s" if n > 1 else ""}</span>')
        meta = f"{TYPE_NAMES[t['type']]} &middot; p&prime;<sub>0</sub> = {fmt(t['p0'], 0)} kPa"
        if core.finite(t["e0"]):
            meta += f" &middot; e<sub>0</sub> = {fmt(t['e0'], 3)}"
        checked = " checked" if S.included.get(t["tid"], True) else ""
        selected = " is-selected" if t["tid"] == S.selected else ""
        cards.append(
            f'<div class="cal-test-card{selected}" data-test="{tid}" tabindex="0" role="button">{badge}'
            f'<div class="cal-test-card__id">{tid}</div><div class="cal-test-card__meta">{meta}</div>'
            f'<label class="cal-checkbox"><input type="checkbox" data-include{checked}> '
            f'<span>Include in calibration</span></label></div>')
    box.innerHTML = "".join(cards)


def select_test(tid):
    S.selected = tid
    for card in document.querySelectorAll("#test-cards .cal-test-card"):
        card.classList.toggle("is-selected", card.dataset.test == tid)
    render_review()


@when("click", "#test-cards")
def on_card_click(event):
    if event.target.closest("label"):
        return
    card = event.target.closest(".cal-test-card")
    if card and S.built and test_info(card.dataset.test):
        select_test(card.dataset.test)


@when("keydown", "#test-cards")
def on_card_key(event):
    card = event.target.closest(".cal-test-card")
    if card and event.key in ("Enter", " ") and event.target == card and S.built and test_info(card.dataset.test):
        event.preventDefault()
        select_test(card.dataset.test)


@when("change", "#test-cards")
def on_include(event):
    card = event.target.closest(".cal-test-card")
    if card and event.target.hasAttribute("data-include"):
        S.included[card.dataset.test] = bool(event.target.checked)
        update_csl()


def render_exclusions():
    ranges = S.excluded.get(S.selected, [])
    el("exclude-list").innerHTML = (
        "Excluded: " + ", ".join(f"&epsilon;<sub>1</sub> {min(a, b):g} to {max(a, b):g} %" for a, b in ranges)
        if ranges else "Nothing excluded from this test.")


def render_review():
    ids = ["review-q-eps1", "review-p-q", "review-vol", "review-e-logp"]
    info = test_info(S.selected)
    if not info or not info["n_rows"]:
        for i in ids:
            clear_plot(i)
        return
    el("review-selected-test").textContent = S.selected
    render_exclusions()
    ranges = S.excluded.get(S.selected)
    g = test_rows(S.selected)

    def pair(rows, xcol, ycol):
        """Kept data as a line, excluded data as faint points, the final kept point as a square."""
        keep = core.kept_mask(rows, ranges)
        k, x = rows[keep], rows[~keep]
        traces = [line(k[xcol], k[ycol], "Laboratory data")]
        if len(x):
            traces.append(markers(x[xcol], x[ycol], "Excluded", color=FAINT, size=5))
            traces[-1]["marker"]["line"] = {"width": 0}
        if len(k):
            traces.append(markers([k[xcol].iloc[-1]], [k[ycol].iloc[-1]], "Final kept point",
                                  color=INK, symbol="square", size=9))
        return traces

    e1_axis = axis("Axial strain, &epsilon;<sub>1</sub> (%)")
    draw("review-q-eps1", pair(g, "e1", "q"), e1_axis,
         axis("Deviator stress, q (kPa)"), title="q &ndash; &epsilon;<sub>1</sub>")
    draw("review-p-q", pair(g, "p", "q"), axis("Mean effective stress, p&prime; (kPa)", rangemode="tozero"),
         axis("Deviator stress, q (kPa)", rangemode="tozero"), title="p&prime; &ndash; q")
    if info["type"] == "CIU" and g["du"].notna().any():
        draw("review-vol", pair(g, "e1", "du"), e1_axis,
             axis("Excess pore pressure, &Delta;u (kPa)"), title="&Delta;u &ndash; &epsilon;<sub>1</sub>")
    elif g["ev"].notna().any():
        draw("review-vol", pair(g, "e1", "ev"), e1_axis,
             axis("Volumetric strain, &epsilon;<sub>v</sub> (%)"), title="&epsilon;<sub>v</sub> &ndash; &epsilon;<sub>1</sub>")
    else:
        clear_plot("review-vol")
    gp = g[(g["p"] > 0) & g["e"].notna()]
    if len(gp):
        draw("review-e-logp", pair(gp, "p", "e"), axis("Mean effective stress, p&prime; (kPa)", type="log"),
             axis("Void ratio, e"), title="e &ndash; log p&prime;")
    else:
        clear_plot("review-e-logp")


@when("click", "#exclude-add")
def on_exclude_add(event):
    try:
        a, b = float(el("exclude-from").value), float(el("exclude-to").value)
    except ValueError:
        return
    if S.selected:
        S.excluded.setdefault(S.selected, []).append((a, b))
        render_review()
        update_csl()


@when("click", "#exclude-clear")
def on_exclude_clear(event):
    if S.selected:
        S.excluded[S.selected] = []
        render_review()
        update_csl()


# --------------------------------------------------------------------------------------
# Step 3: critical state line
# --------------------------------------------------------------------------------------
CHECK_BADGES = {
    "steady": ('badge--open', "Levelled off"),
    "changing": ('badge--external', "Still changing"),
    "short": ('badge--soon', "Under 2% strain"),
}


def compute_points():
    pts = []
    for t in S.built["tests"]:
        if not t["n_rows"] or not S.included.get(t["tid"], True):
            continue
        g = test_rows(t["tid"])
        g = g[core.kept_mask(g, S.excluded.get(t["tid"]))]
        c = core.candidate_point(g, t["type"])
        if c:
            pts.append(c)
    return pts


def csl_mode():
    return document.querySelector('input[name="csl-model"]:checked').value


def csl_choice():
    """The CSL used for calibration. Falls back to semi-log if the curved fit is not available."""
    mode = csl_mode()
    use = el("csl-use").value if mode == "both" else mode
    if use == "curved" and not (S.fit and S.fit.get("curved")):
        return "semi_log"
    return use


def usable(c):
    return core.finite(c["e"]) and c["p"] > 0


def update_csl():
    if not S.built:
        return
    S.points = compute_points()
    tids = [c["tid"] for c in S.points]
    if S.csl_test not in tids:
        S.csl_test = tids[0] if tids else None
    accepted = [c for c in S.points if S.accepted.get(c["tid"], True) and usable(c)]
    try:
        pref = float(el("cal-pref").value)
    except ValueError:
        pref = core.PREF_DEFAULT
    S.fit = core.fit_models(accepted, pref) if accepted else None
    render_candidates()
    render_fit()
    render_csl_plot()
    render_checks()


def render_candidates():
    body = el("csl-candidates").querySelector("tbody")
    if not S.points:
        body.innerHTML = '<tr class="cal-table__empty"><td colspan="9">No included tests with positive p&prime;.</td></tr>'
        return
    rows = []
    for c in S.points:
        tid = esc(c["tid"])
        ok = usable(c)
        checked = " checked" if ok and S.accepted.get(c["tid"], True) else ""
        disabled = "" if ok else " disabled"
        cls, label = CHECK_BADGES[c["check"]]
        badge = f'<span class="badge {cls}">{label}</span>' if ok else '<span class="badge badge--external">No void ratio</span>'
        d = fmt(c["D_end"], 3) if c["type"] != "CIU" else '<span title="Undrained: no volume change">n/a</span>'
        active = " is-active" if c["tid"] == S.csl_test else ""
        rows.append(
            f'<tr data-tid="{tid}" class="cal-row{active}"><td><input type="checkbox" data-accept{checked}{disabled} '
            f'aria-label="Accept {tid}"></td><td>{tid}</td><td class="num">{fmt(c["e1"], 2)}</td>'
            f'<td class="num">{fmt(c["p"], 1)}</td><td class="num">{fmt(c["q"], 1)}</td>'
            f'<td class="num">{fmt(c["e"], 4)}</td><td class="num">{d}</td>'
            f'<td class="num">{fmt(c["dq_pct"], 1)}{"%" if core.finite(c["dq_pct"]) else ""}</td><td>{badge}</td></tr>')
    body.innerHTML = "".join(rows)


def set_out(id_, value, digits=4):
    el(id_).innerHTML = fmt(value, digits)


def render_fit():
    f = S.fit
    s = f.get("semi_log") if f else None
    c = f.get("curved") if f else None
    set_out("fit-gamma", s and s["Gamma_ref"])
    set_out("fit-gamma1", s and s["Gamma_1kPa"])
    set_out("fit-lambda", s and s["lambda10"])
    set_out("fit-semilog-r2", s and s["R2"])
    set_out("fit-semilog-rmse", s and s["RMSE"], 5)
    set_out("fit-ca", c and c["Ca"])
    set_out("fit-cb", c and c["Cb"])
    set_out("fit-cc", c and c["Cc"])
    set_out("fit-curved-r2", c and c["R2"])
    set_out("fit-curved-rmse", c and c["RMSE"], 5)
    set_out("fit-mend", f and f["M_end"], 3)
    for pref_id in ("fit-pref-semilog", "fit-pref-curved"):
        el(pref_id).textContent = f"{f['pref']:g}" if f else "p_ref"

    msgs = []
    if not f:
        msgs.append("Accept candidate points to fit the CSL.")
    else:
        msgs.append(f"Fitted to {f['n']} accepted point{'s' if f['n'] != 1 else ''}.")
        msgs += f["messages"]
        if s and c:
            better = "curved" if c["AIC"] < s["AIC"] else "semi-log"
            msgs.append(f"AIC prefers the {better} form (semi-log {s['AIC']:.1f}, curved {c['AIC']:.1f}).")
        mode = csl_mode()
        wanted = el("csl-use").value if mode == "both" else mode
        if wanted == "curved" and not c:
            msgs.append("The semi-log CSL will be used until the curved fit is available.")
    el("fit-status").textContent = " ".join(msgs)


def render_csl_plot():
    if not S.points:
        clear_plot("csl-e-logp")
        return
    traces = []
    for c in S.points:
        g = test_rows(c["tid"])
        g = g[core.kept_mask(g, S.excluded.get(c["tid"])) & (g["p"] > 0) & g["e"].notna()]
        if len(g):
            traces.append(line(g["p"], g["e"], c["tid"], color=FAINT, width=1.2, showlegend=False,
                               hoverinfo="name"))
    ok = [c for c in S.points if usable(c)]
    for accepted in (True, False):
        pts = [c for c in ok if bool(S.accepted.get(c["tid"], True)) == accepted]
        if not pts:
            continue
        traces.append({
            "type": "scatter", "mode": "markers",
            "name": "Accepted point" if accepted else "Rejected point",
            "x": [c["p"] for c in pts], "y": [c["e"] for c in pts],
            "customdata": [c["tid"] for c in pts],
            "text": [f"{c['tid']} ({c['type']})" for c in pts],
            "hovertemplate": "%{text}<br>p′ = %{x:.1f} kPa<br>e = %{y:.4f}<extra>Click to toggle</extra>",
            "marker": {"size": 11, "symbol": ["square" if c["type"] == "CIU" else "circle" for c in pts],
                       "color": INK if accepted else "#ffffff",
                       "line": {"color": INK if accepted else MUTED, "width": 1.5}},
        })
    f = S.fit
    if f and f.get("semi_log"):
        ps = [c["p"] for c in ok]
        pp = np.geomspace(min(ps) * 0.7, max(ps) * 1.4, 200)
        use = csl_choice()
        mode = csl_mode()
        for model, color, dash, name in (("semi_log", ACCENT, "solid", "Semi-log CSL"),
                                         ("curved", BLUE, "dash", "Curved CSL")):
            if not f.get(model) or (mode != "both" and mode != model):
                continue
            label = name + (" (used)" if model == use and mode == "both" else "")
            traces.append(line(pp, core.csl_values(pp, f, model), label, color=color, dash=dash, width=2.4))
    div = draw("csl-e-logp", traces, axis("Mean effective stress, p&prime; (kPa)", type="log"),
               axis("Void ratio, e"), title="e &ndash; log p&prime; &middot; circles CID, squares CIU",
               legend=True)
    if not S.csl_click_bound:
        div.on("plotly_click", create_proxy(on_csl_point_click))
        S.csl_click_bound = True


def on_csl_point_click(event):
    pts = event.points
    if not pts or pts.length == 0:
        return
    tid = pts[0].customdata
    if isinstance(tid, str):
        S.accepted[tid] = not S.accepted.get(tid, True)
        S.csl_test = tid
        update_csl()


def render_checks():
    c = next((p for p in S.points if p["tid"] == S.csl_test), None)
    if not c:
        clear_plot("csl-check-d")
        clear_plot("csl-check-q")
        return
    g = test_rows(c["tid"])
    g = g[core.kept_mask(g, S.excluded.get(c["tid"]))]
    e1_end = float(g["e1"].iloc[-1])
    band = [{"type": "rect", "xref": "x", "yref": "paper", "x0": e1_end - 2.0, "x1": e1_end,
             "y0": 0, "y1": 1, "fillcolor": "rgba(178,140,53,0.10)", "line": {"width": 0}}]
    tid = esc(c["tid"])
    e1_axis = axis("Axial strain, &epsilon;<sub>1</sub> (%)")
    if c["type"] == "CIU":
        draw("csl-check-d", [line(g["e1"], g["p"], "p′")], e1_axis,
             axis("Mean effective stress, p&prime; (kPa)"),
             title=f"p&prime; &ndash; &epsilon;<sub>1</sub> &middot; {tid} (undrained, so D = 0; p&prime; should level off)",
             shapes=band)
    else:
        e1, d = core.dilatancy_series(g)
        if e1 is None:
            clear_plot("csl-check-d")
        else:
            draw("csl-check-d", [line(e1, d, "D")], e1_axis, axis("Dilatancy, D"),
                 title=f"D &ndash; &epsilon;<sub>1</sub> &middot; {tid}",
                 shapes=band + [{"type": "line", "xref": "paper", "x0": 0, "x1": 1, "y0": 0, "y1": 0,
                                 "line": {"color": MUTED, "dash": "dot", "width": 1}}])
    draw("csl-check-q", [line(g["e1"], g["q"], "q")], e1_axis, axis("Deviator stress, q (kPa)"),
         title=f"q &ndash; &epsilon;<sub>1</sub> &middot; {tid} (shaded: last 2%)", shapes=band)


@when("change", "#csl-candidates")
def on_accept(event):
    tr = event.target.closest("tr")
    if tr is not None and event.target.hasAttribute("data-accept"):
        S.accepted[tr.dataset.tid] = bool(event.target.checked)
        S.csl_test = tr.dataset.tid
        update_csl()


@when("click", "#csl-candidates")
def on_candidate_row(event):
    if event.target.tagName == "INPUT":
        return
    tr = event.target.closest("tr[data-tid]")
    if tr is not None:
        S.csl_test = tr.dataset.tid
        for row in document.querySelectorAll("#csl-candidates tr[data-tid]"):
            row.classList.toggle("is-active", row.dataset.tid == S.csl_test)
        render_checks()


@when("click", "#fit-csl-button")
def on_fit(event):
    update_csl()


@when("change", "#cal-pref, #csl-use, input[name='csl-model']")
def on_fit_settings(event):
    update_csl()


@when("click", "#csl-accept")
def on_csl_accept(event):
    if not (S.fit and S.fit.get("semi_log")):
        return
    use = csl_choice()
    params = core.norsand_csl(S.fit, use)
    window.calibrationCSL = jsobj(params)
    outs = {o.dataset.param: o for o in document.querySelectorAll("#final-params output")}
    if use == "curved":
        outs["gamma_ca"].innerHTML = fmt(params["ca"], 4)
        outs["lambda_cb"].innerHTML = fmt(params["cb"], 4)
        outs["cc"].innerHTML = fmt(params["cc"], 4)
    else:
        outs["gamma_ca"].innerHTML = fmt(params["gamma"], 4)
        outs["lambda_cb"].innerHTML = fmt(params["lambda10"], 4)
        outs["cc"].innerHTML = "&ndash;"


@when("click", "#csl-download-summary")
def on_download_summary(event):
    if S.built:
        text = core.summary_text(S.built, S.mapping, S.points, S.accepted, S.excluded, S.fit, csl_choice())
        download(text, "norsand_csl_summary.txt")


@when("click", "#csl-download-points")
def on_download_points(event):
    if S.points:
        download(core.candidates_csv(S.points, S.accepted, S.fit), "norsand_csl_points.csv", "text/csv")


# --------------------------------------------------------------------------------------
# Start-up
# --------------------------------------------------------------------------------------
remember_placeholders()
el("engine-status").textContent = "Ready. Choose a file, paste data or enter values."
el("engine-status").classList.add("is-ready")
_files = el("lab-file").files
if _files and _files.length:
    asyncio.ensure_future(read_file_list(_files))

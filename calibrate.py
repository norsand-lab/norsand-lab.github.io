"""Calibration page, steps 1 to 3 (runs in the browser with PyScript).

The calculations live in modules/calibration_core.py. This file reads the page,
calls those functions, and writes tables and Plotly charts back to the page.
"""
import asyncio
import html
import json
import math

import numpy as np
import pandas as pd
from js import Blob, Object, Plotly, URL, window
from pyodide.ffi import create_proxy, to_js
from pyscript import document, when

from modules import calibration_core as core
from modules import plasticity_core as pc
from modules import trial_core as tc
from modules import common_core as cmn


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
        # step 4
        self.plast = None         # processed tests: tid -> {type, r, peak, warnings}
        self.plast_dirty = True   # recalculate when step 4 is next shown
        self.dense = {}           # tid -> used as a dense test
        self.examine = 0          # index into the drained tests
        self.points4 = []         # dense-test peak points
        self.ls4 = None           # least-squares Mtc, N, chi
        self.adopted = None       # {"m", "n", "chi"}
        self.sig4 = None          # resets the adopted values when the peak points change
        # step 5
        self.trials = {}          # tid -> trial values (gref, h0, hy, ocr, nu, m, s)
        self.trial_sel = None     # test shown in step 5
        self.runs = []            # every run, in order (the trial log)
        self.latest = {}          # tid -> (record, model, lab) of the latest run
        self.running = False
        # step 6
        self.common_rows = {}     # tid -> {use, gref, h, ocr, s, source}
        self.vincent_psi = {}     # tid -> Vincent's psi0, after loading his values
        self.common_fit = None
        self.common_runs = {}     # (tid, law) -> (record, model, lab)
        self.final_sel = None
        self.adopted_common = None


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
        "legend": {"orientation": "h", "x": 0, "y": -0.3, "yanchor": "top", "font": {"size": 10}},
    }
    if title:
        layout["title"] = {"text": html.unescape(title), "font": {"size": 12}, "x": 0.01, "xanchor": "left"}
    if shapes:
        layout["shapes"] = shapes
    Plotly.react(div, jsobj(traces), jsobj(layout), jsobj(CONFIG))
    return div


def line(x, y, name="", color=INK, dash="solid", width=2, **kw):
    return {"type": "scatter", "mode": "lines", "x": lst(x), "y": lst(y), "name": html.unescape(name),
            "line": {"color": color, "dash": dash, "width": width}, **kw}


def markers(x, y, name="", color=INK, symbol="circle", size=8, **kw):
    return {"type": "scatter", "mode": "markers", "x": lst(x), "y": lst(y), "name": html.unescape(name),
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
    # psi depends on the CSL, so step 4 is out of date. Recalculate now only if it is on screen.
    S.plast_dirty = True
    if not el("step-4").hidden:
        update_plasticity()


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
# Step 4: plasticity (Part II)
# --------------------------------------------------------------------------------------
STEP4_PLOTS = ["plot-eta-eps1", "plot-d-eps1", "plot-eta-d", "plot-psi-eps1", "plot-etamax-dmin",
               "plot-dmin-psi", "plot-loose-eta", "plot-loose-d", "plot-ciu-pq"]


def csl_function():
    """e_c(p') from the CSL chosen in step 3, or None if there is no fit yet."""
    if not (S.fit and S.fit.get("semi_log")):
        return None
    fit, use = S.fit, csl_choice()
    return lambda p: core.csl_values(p, fit, use)


def csl_note():
    f = S.fit
    if not (f and f.get("semi_log")):
        return "none"
    use = csl_choice()
    if use == "curved":
        c = f["curved"]
        return f"curved, Ca={c['Ca']:.5f}, Cb={c['Cb']:.5f}, Cc={c['Cc']:.5f}, pref={f['pref']:g} kPa"
    s = f["semi_log"]
    return f"semi-log, Gamma(1 kPa)={s['Gamma_1kPa']:.5f}, lambda10={s['lambda10']:.5f}"


def drained_ids():
    return [t for t, v in (S.plast or {}).items() if v["type"] == "CID"]


def vline(x, color=MUTED, dash="dash"):
    return {"type": "line", "xref": "x", "yref": "paper", "x0": x, "x1": x, "y0": 0, "y1": 1,
            "line": {"color": color, "dash": dash, "width": 1.2}}


def hline(y, color=MUTED, dash="dot"):
    return {"type": "line", "xref": "paper", "yref": "y", "x0": 0, "x1": 1, "y0": y, "y1": y,
            "line": {"color": color, "dash": dash, "width": 1}}


def update_plasticity():
    """Process every included test with the step 3 CSL, then refit."""
    S.plast_dirty = False
    fn = csl_function()
    status = el("plast-status")
    if fn is None or not S.built:
        S.plast, S.points4, S.ls4 = None, [], None
        status.hidden = False
        status.textContent = ("Import data in step 1 first." if not S.built
                              else "Fit the CSL in step 3 first. ψ depends on it.")
        for i in STEP4_PLOTS:
            clear_plot(i)
        render_examine()
        render_dmin_table()
        render_plast_fit()
        render_mtc_check()
        return
    status.hidden = True
    tests = {}
    for t in S.built["tests"]:
        tid, typ = t["tid"], t["type"]
        if not t["n_rows"] or not S.included.get(tid, True) or typ not in ("CID", "CIU"):
            continue
        g = test_rows(tid)
        g = g[core.kept_mask(g, S.excluded.get(tid)) & (g["p"] > 0) & g["e"].notna()]
        if len(g) < 2:
            continue
        r = pc.process_test(g, fn)
        warnings = []
        steps = np.diff(g["e1"].to_numpy(float))
        if (steps <= 0).any():
            warnings.append(f"{int((steps <= 0).sum())} non-increasing axial strain step(s); the recorded row order is kept")
        peak = None
        if typ == "CID":
            if g["ev"].isna().all():
                warnings.append("no volumetric strain, so D cannot be calculated")
            elif r["D"].notna().sum() < 5:
                warnings.append("too few rows for the five-row dilatancy calculation")
            peak = pc.peak_point(r)
            S.dense.setdefault(tid, pc.propose_dense(peak))
        tests[tid] = {"type": typ, "r": r, "peak": peak, "warnings": warnings}
    S.plast = tests
    n = len(drained_ids())
    S.examine = min(S.examine, max(n - 1, 0))
    refit_plasticity()
    render_examine()
    render_mtc_check()


def refit_plasticity():
    """Least squares from the ticked dense tests. Adopted values reset when the points change."""
    S.points4 = [{"tid": tid, **v["peak"]} for tid, v in (S.plast or {}).items()
                 if v["type"] == "CID" and v["peak"] and S.dense.get(tid)]
    S.ls4 = pc.fit(S.points4)
    sig = tuple((z["tid"], round(z["Dmin"], 9), round(z["eta_max"], 9), round(z["psi_at_Dmin"], 9)) for z in S.points4)
    if sig != S.sig4:
        S.sig4 = sig
        ls = S.ls4
        S.adopted = ({"m": ls["Mtc"], "n": ls["N"], "chi": ls["chi_tc"]}
                     if core.finite(ls["Mtc"]) and core.finite(ls["chi_tc"]) else None)
        sync_adopted_inputs()
    render_dmin_table()
    render_plast_fit()


def render_examine():
    ids = drained_ids()
    prev, nxt = el("examine-prev"), el("examine-next")
    prev.disabled = nxt.disabled = len(ids) < 2
    box = el("examine-accept")
    if not ids:
        el("examine-test-name").textContent = "No drained tests yet"
        el("examine-count").innerHTML = "Test &ndash; of &ndash;"
        for i in ("examine-dmin", "examine-eta", "examine-psi", "examine-psi0", "examine-gap"):
            el(i).innerHTML = "&ndash;"
        el("examine-warnings").textContent = ""
        box.disabled = True
        for i in STEP4_PLOTS[:4]:
            clear_plot(i)
        return
    tid = ids[S.examine]
    v = S.plast[tid]
    r, pk = v["r"], v["peak"]
    el("examine-test-name").textContent = tid
    el("examine-count").textContent = f"Test {S.examine + 1} of {len(ids)}"
    box.disabled = pk is None
    box.checked = bool(S.dense.get(tid)) and pk is not None
    el("examine-dmin").innerHTML = fmt(pk and pk["Dmin"], 4)
    el("examine-eta").innerHTML = fmt(pk and pk["eta_max"], 4)
    el("examine-psi").innerHTML = fmt(pk and pk["psi_at_Dmin"], 4)
    el("examine-psi0").innerHTML = fmt(r["psi"].iloc[0], 4)
    el("examine-gap").innerHTML = (fmt(pk["gap"], 2) + " %") if pk else "&ndash;"
    el("examine-warnings").textContent = "; ".join(v["warnings"])

    e1_axis = axis("Axial strain, &epsilon;<sub>1</sub> (%)")
    marks = [vline(pk["strain_at_eta_max"], INK, "dash"), vline(pk["strain_at_Dmin"], ACCENT, "dot")] if pk else []
    draw("plot-eta-eps1", [line(r["e1"], r["eta"], "η")], e1_axis, axis("Stress ratio, &eta; = q/p&prime;"),
         title="&eta; &ndash; &epsilon;<sub>1</sub> &middot; dashed: &eta;<sub>max</sub>, dotted: D<sub>min</sub>", shapes=marks)
    draw("plot-d-eps1", [line(r["e1"], r["D"], "D")], e1_axis, axis("Measured dilatancy, D"),
         title="D &ndash; &epsilon;<sub>1</sub>", shapes=marks + [hline(0)])
    traces = [line(r["D"], r["eta"], "Path", color=MUTED, width=1.5)]
    if pk:
        traces.append(markers([pk["Dmin"]], [pk["eta_at_Dmin"]], "D<sub>min</sub> row", color=ACCENT, symbol="square", size=10))
        traces.append(markers([pk["D_at_eta_max"]], [pk["eta_max"]], "&eta;<sub>max</sub> row", color="#ffffff", size=10))
    draw("plot-eta-d", traces, axis("Measured dilatancy, D"), axis("Stress ratio, &eta;"),
         title="&eta; &ndash; D", legend=True)
    draw("plot-psi-eps1", [line(r["e1"], r["psi"], "ψ")], e1_axis, axis("State parameter, &psi;"),
         title="&psi; &ndash; &epsilon;<sub>1</sub>", shapes=[hline(0)])


def render_dmin_table():
    body = el("dmin-table").querySelector("tbody")
    ids = drained_ids()
    if not ids:
        body.innerHTML = '<tr class="cal-table__empty"><td colspan="8">Drained tests appear here after the CSL is fitted.</td></tr>'
        return
    rows = []
    for k, tid in enumerate(ids):
        v = S.plast[tid]
        pk = v["peak"]
        checked = " checked" if pk and S.dense.get(tid) else ""
        disabled = "" if pk else " disabled"
        active = " is-active" if k == S.examine else ""
        rows.append(
            f'<tr data-tid="{esc(tid)}" data-index="{k}" class="cal-row{active}">'
            f'<td><input type="checkbox" data-dense{checked}{disabled} aria-label="Use {esc(tid)}"></td>'
            f'<td>{esc(tid)}</td><td class="num">{fmt(v["r"]["psi"].iloc[0], 4)}</td>'
            f'<td class="num">{fmt(pk and pk["Dmin"], 4)}</td><td class="num">{fmt(pk and pk["eta_max"], 4)}</td>'
            f'<td class="num">{fmt(pk and pk["psi_at_Dmin"], 4)}</td>'
            f'<td class="num">{fmt(pk and pk["strain_at_Dmin"], 2)}</td><td class="num">{fmt(pk and pk["strain_at_eta_max"], 2)}</td></tr>')
    body.innerHTML = "".join(rows)


def sync_adopted_inputs(skip=None):
    a = S.adopted
    for key, ids in (("m", ("cal-mtc", "cal-mtc-range")), ("n", ("cal-n", "cal-n-range")), ("chi", ("cal-chi", "cal-chi-range"))):
        for i in ids:
            if i != skip:
                el(i).value = f"{a[key]:.4f}" if a and core.finite(a[key]) else ""


def render_plast_fit():
    ls = S.ls4
    pts = S.points4
    ok = bool(ls and core.finite(ls["Mtc"]))
    ls_err = pc.errors(pts, ls["Mtc"], ls["N"], ls["chi_tc"]) if ok else {"RMSE_eta": None, "RMSE_D": None}
    el("ls-mtc").innerHTML = fmt(ls and ls["Mtc"], 4)
    el("ls-n").innerHTML = fmt(ls and ls["N"], 4)
    el("ls-chi").innerHTML = fmt(ls and ls["chi_tc"], 4)
    el("ls-rmse-eta").innerHTML = fmt(ls_err["RMSE_eta"], 5)
    el("ls-rmse-d").innerHTML = fmt(ls_err["RMSE_D"], 5)
    msgs = [f"{len(pts)} dense test{'s' if len(pts) != 1 else ''} selected."] if S.plast else []
    if ls:
        msgs += ls["messages"]
    el("plast-fit-status").textContent = " ".join(msgs)
    a = S.adopted
    err = pc.errors(pts, a["m"], a["n"], a["chi"]) if a and pts else {"RMSE_eta": None, "RMSE_D": None}
    el("fit-rmse-eta").innerHTML = fmt(err["RMSE_eta"], 5)
    el("fit-rmse-d").innerHTML = fmt(err["RMSE_D"], 5)
    if not (ok and a):
        clear_plot("plot-etamax-dmin")
        clear_plot("plot-dmin-psi")
        return

    d = [z["Dmin"] for z in pts]
    eta = [z["eta_max"] for z in pts]
    psi = [z["psi_at_Dmin"] for z in pts]
    names = [z["tid"] for z in pts]
    xd = np.linspace(min(-0.8, 1.12 * min(d)), 0.0, 50)
    pts_trace = {"type": "scatter", "mode": "markers+text", "x": d, "y": eta, "text": names,
                 "textposition": "top right", "textfont": {"size": 9, "color": MUTED}, "name": "Dense tests",
                 "marker": {"color": INK, "size": 9}}
    draw("plot-etamax-dmin", [
        line(xd, ls["Mtc"] - (1 - ls["N"]) * xd, "Least squares", color=FAINT, dash="dash"),
        line(xd, a["m"] - (1 - a["n"]) * xd, "Adopted", color=ACCENT, width=2.4), pts_trace],
        axis("Minimum dilatancy, D<sub>min</sub>"), axis("Maximum stress ratio, &eta;<sub>max</sub>", side="right"),
        title=f"&eta;<sub>max</sub> &ndash; D<sub>min</sub> &middot; RMSE {err['RMSE_eta']:.4f}", legend=True)
    if core.finite(a["chi"]):
        xp = np.linspace(min(-0.2, 1.15 * min(psi)), 0.0, 50)
        draw("plot-dmin-psi", [
            line(xp, ls["chi_tc"] * xp, "Least squares", color=FAINT, dash="dash"),
            line(xp, a["chi"] * xp, "Adopted", color=ACCENT, width=2.4),
            {**pts_trace, "x": psi, "y": d}],
            axis("State parameter at D<sub>min</sub>, &psi;"), axis("D<sub>min</sub>", side="right"),
            title=f"D<sub>min</sub> &ndash; &psi; &middot; RMSE {err['RMSE_D']:.4f}", legend=True)


def step3_decision(tid):
    if not any(c["tid"] == tid for c in S.points):
        return "not a candidate"
    return "accepted" if S.accepted.get(tid, True) else "rejected"


def mtc_checks():
    rows = []
    for tid, v in (S.plast or {}).items():
        if v["type"] == "CID" and S.dense.get(tid) and v["peak"]:
            continue
        rows.append({"tid": tid, "type": v["type"], "decision": step3_decision(tid), **pc.late_check(v["r"], v["type"])})
    return rows


def render_mtc_check():
    body = el("mtc-check").querySelector("tbody")
    checks = mtc_checks() if S.plast else []
    m = S.adopted["m"] if S.adopted else float("nan")
    if not checks:
        body.innerHTML = '<tr class="cal-table__empty"><td colspan="8">No loose drained or undrained tests to check against.</td></tr>'
        el("mtc-check-summary").textContent = ""
    else:
        rows = []
        for c in checks:
            cls = "badge--open" if c["decision"] == "accepted" else "badge--soon"
            rows.append(
                f'<tr><td>{esc(c["tid"])}</td><td>{c["type"]}</td><td class="num">{fmt(c["eta_last"], 3)}</td>'
                f'<td class="num">{fmt(c["eta_late_median"], 3)}</td><td class="num">{fmt(c["eta_late_range"], 3)}</td>'
                f'<td class="num">{fmt(c["D_late_median"], 3) if c["type"] == "CID" else "n/a"}</td>'
                f'<td class="num">{fmt(c["eta_last"] - m, 3)}</td>'
                f'<td><span class="badge {cls}">{esc(c["decision"])}</span></td></tr>')
        body.innerHTML = "".join(rows)
        acc = [c["eta_last"] for c in checks if c["decision"] == "accepted"]
        el("mtc-check-summary").textContent = (
            f"Median last q/p′ of the {len(acc)} test(s) accepted in step 3: {float(np.median(acc)):.3f}. "
            "Review each test before relying on it." if acc
            else "No test end points were accepted in step 3, so these values are candidate checks only.")

    if not S.plast:
        return
    loose = [t for t, v in S.plast.items() if v["type"] == "CID" and not (S.dense.get(t) and v["peak"])]
    ciu = [t for t, v in S.plast.items() if v["type"] == "CIU"]
    palette = ["#1e293b", "#475569", "#2563eb", "#0f766e", "#7c3aed", "#b45309", "#be123c", "#64748b"]
    e1_axis = axis("Axial strain, &epsilon;<sub>1</sub> (%)")
    m_shape = [hline(m, INK, "dash")] if core.finite(m) else []
    if loose:
        draw("plot-loose-eta",
             [line(S.plast[t]["r"]["e1"], S.plast[t]["r"]["eta"], t, color=palette[k % len(palette)], width=1.5) for k, t in enumerate(loose)],
             e1_axis, axis("Stress ratio, &eta;"), title=f"Loose CID: &eta; &ndash; &epsilon;<sub>1</sub> &middot; dashed: M<sub>tc</sub> = {m:.3f}",
             legend=True, shapes=m_shape)
        draw("plot-loose-d",
             [line(S.plast[t]["r"]["e1"], S.plast[t]["r"]["D"], t, color=palette[k % len(palette)], width=1.2) for k, t in enumerate(loose)],
             e1_axis, axis("Measured dilatancy, D", range=[-0.65, 0.35]),
             title="Loose CID: does D stay near 0? (axis clipped)", legend=True, shapes=[hline(0, INK, "dash")])
    else:
        clear_plot("plot-loose-eta")
        clear_plot("plot-loose-d")
    if ciu:
        traces = [line(S.plast[t]["r"]["p"], S.plast[t]["r"]["q"], t, color=palette[k % len(palette)], width=1.5) for k, t in enumerate(ciu)]
        if core.finite(m):
            pmax = max(float(S.plast[t]["r"]["p"].max()) for t in ciu) * 1.08
            traces.append(line([0, pmax], [0, m * pmax], f"q = M<sub>tc</sub> p′ ({m:.3f})", color=INK, dash="dash"))
        draw("plot-ciu-pq", traces, axis("Mean effective stress, p&prime; (kPa)", rangemode="tozero"),
             axis("Deviator stress, q (kPa)", rangemode="tozero"), title="CIU: effective stress paths", legend=True)
    else:
        clear_plot("plot-ciu-pq")


def set_adopted(m=None, n=None, chi=None, source=None):
    if not S.adopted:
        return
    if m is not None:
        S.adopted["m"] = m
        refit = pc.refit_n(S.points4, m)
        if core.finite(refit):
            S.adopted["n"] = refit
    if n is not None:
        S.adopted["n"] = n
    if chi is not None:
        S.adopted["chi"] = chi
    sync_adopted_inputs(skip=source)
    render_plast_fit()
    render_mtc_check()


def _num(event):
    try:
        v = float(event.target.value)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


@when("input", "#cal-mtc, #cal-mtc-range")
def on_mtc(event):
    v = _num(event)
    if v is not None:
        set_adopted(m=v, source=event.target.id)


@when("input", "#cal-n, #cal-n-range")
def on_n(event):
    v = _num(event)
    if v is not None:
        set_adopted(n=v, source=event.target.id)


@when("input", "#cal-chi, #cal-chi-range")
def on_chi(event):
    v = _num(event)
    if v is not None:
        set_adopted(chi=v, source=event.target.id)


@when("click", "#plast-reset")
def on_plast_reset(event):
    S.sig4 = None
    refit_plasticity()
    render_mtc_check()


def set_dense(tid, use):
    S.dense[tid] = use
    refit_plasticity()
    render_examine()
    render_mtc_check()


@when("change", "#examine-accept")
def on_examine_accept(event):
    ids = drained_ids()
    if ids:
        set_dense(ids[S.examine], bool(event.target.checked))


@when("change", "#dmin-table")
def on_dense_box(event):
    tr = event.target.closest("tr")
    if tr is not None and event.target.hasAttribute("data-dense"):
        S.examine = int(tr.dataset.index)
        set_dense(tr.dataset.tid, bool(event.target.checked))


@when("click", "#dmin-table")
def on_dense_row(event):
    if event.target.tagName == "INPUT":
        return
    tr = event.target.closest("tr[data-index]")
    if tr is not None:
        S.examine = int(tr.dataset.index)
        render_examine()
        render_dmin_table()


def step_examine(delta):
    ids = drained_ids()
    if ids:
        S.examine = (S.examine + delta) % len(ids)
        render_examine()
        render_dmin_table()


@when("click", "#examine-prev")
def on_examine_prev(event):
    step_examine(-1)


@when("click", "#examine-next")
def on_examine_next(event):
    step_examine(1)


def plast_ready():
    return bool(S.adopted and S.points4)


@when("click", "#plast-accept")
def on_plast_accept(event):
    if not plast_ready():
        return
    a = S.adopted
    window.calibrationPlasticity = jsobj({"mtc": a["m"], "n": a["n"], "chi": a["chi"],
                                         "reason": el("plast-reason").value})
    outs = {o.dataset.param: o for o in document.querySelectorAll("#final-params output")}
    outs["mtc"].innerHTML = fmt(a["m"], 4)
    outs["n"].innerHTML = fmt(a["n"], 4)
    outs["chi"].innerHTML = fmt(a["chi"], 4)


def plast_errors():
    a = S.adopted
    return pc.errors(S.points4, a["m"], a["n"], a["chi"])


@when("click", "#plast-download-json")
def on_plast_json(event):
    if plast_ready():
        download(pc.adopted_json(S.ls4, S.adopted, plast_errors(), el("plast-reason").value, {"note": csl_note()}),
                 "norsand_plasticity_parameters.json", "application/json")


@when("click", "#plast-download-csv")
def on_plast_csv(event):
    if plast_ready():
        a = S.adopted
        download(pc.residuals_csv(S.points4, a["m"], a["n"], a["chi"]), "norsand_dense_residuals.csv", "text/csv")


@when("click", "#plast-download-report")
def on_plast_report(event):
    if plast_ready():
        ls = S.ls4
        ls_err = pc.errors(S.points4, ls["Mtc"], ls["N"], ls["chi_tc"])
        warnings = [f"{t}: {w}" for t, v in S.plast.items() for w in v["warnings"]]
        text = pc.report_text(S.points4, ls, ls_err, S.adopted, plast_errors(), el("plast-reason").value,
                              mtc_checks(), warnings, csl_note())
        download(text, "norsand_step4_report.txt")


# --------------------------------------------------------------------------------------
# Step 5: manual NorSand trials (Part III)
# --------------------------------------------------------------------------------------
TRIAL_PLOTS = ["fit-q-eps1", "fit-p-q", "fit-vol", "fit-psi"]


def trial_tests():
    """Included CID and CIU tests with a void ratio, in file order."""
    if not S.built:
        return []
    return [t for t in S.built["tests"]
            if t["n_rows"] and t["has_e"] and S.included.get(t["tid"], True) and t["type"] in ("CID", "CIU")]


def step5_inputs():
    """CSL in NorSandParams form and the step 4 plasticity values, or None if not ready."""
    if not (S.fit and S.fit.get("semi_log")):
        return None
    if S.plast_dirty:
        update_plasticity()
    if not S.adopted or not all(core.finite(S.adopted[k]) for k in ("m", "n", "chi")):
        return None
    return (core.norsand_csl(S.fit, csl_choice()),
            {"mtc": S.adopted["m"], "n": S.adopted["n"], "chi": S.adopted["chi"]})


def read_controls():
    def num(i, default):
        try:
            v = float(el(i).value)
            return v if math.isfinite(v) else default
        except ValueError:
            return default
    return {"gref": num("cal-gref", float("nan")), "h0": num("cal-h0", float("nan")),
            "hy": num("cal-hy", 0.0) if not el("cal-hy").disabled else 0.0,
            "ocr": num("cal-ocr", float("nan")), "nu": num("cal-nu", float("nan")),
            "m": num("cal-m", float("nan")), "s": int(el("cal-s").value)}


def set_box(id_, value):
    box = el(id_)
    box.value = f"{value:g}"
    box.dispatchEvent(window.Event.new("change"))   # recentres the slider


def load_controls(tid):
    t = S.trials.get(tid)
    info = test_info(tid)
    if t is None:
        # A new test starts from the values on screen, with its own OCR from step 1 and S = 0.
        t = {**read_controls(), "ocr": max(1.0, info["ocr"]) if info and core.finite(info["ocr"]) else 1.0, "s": 0}
        if not core.finite(t["gref"]):
            t.update({k: tc.DEFAULT_TRIAL[k] for k in ("gref", "h0", "nu", "m")})
        S.trials[tid] = t
    for key, id_ in (("gref", "cal-gref"), ("h0", "cal-h0"), ("nu", "cal-nu"), ("m", "cal-m"), ("ocr", "cal-ocr")):
        if core.finite(t[key]):
            set_box(id_, t[key])
    radio = document.querySelector(f'input[name="hy-mode"][value="{"yes" if t["hy"] else "no"}"]')
    radio.checked = True
    radio.dispatchEvent(window.Event.new("change"))
    if t["hy"]:
        set_box("cal-hy", t["hy"])
    is_cid = info and info["type"] == "CID"
    el("cal-s").value = "0" if is_cid else str(t["s"])
    el("cal-s").querySelector('option[value="1"]').disabled = bool(is_cid)


def save_controls():
    if S.trial_sel:
        S.trials[S.trial_sel] = read_controls()


def update_trial_page():
    """Refresh the test list, readiness message and H0 hint when step 5 is opened."""
    tests = trial_tests()
    ids = [t["tid"] for t in tests]
    ready = step5_inputs() is not None
    status = el("trial-status")
    if not S.built:
        status.textContent = "Import data in step 1 first."
    elif not ready:
        status.innerHTML = ("Fit the CSL in step 3 and calibrate M<sub>tc</sub>, N and &chi;<sub>tc</sub> "
                            "in step 4 first (at least three dense drained tests).")
    status.hidden = ready or False
    if ready and not ids:
        status.hidden = False
        status.textContent = "No included CID or CIU test has a void ratio to simulate."
    for b in ("trial-run", "trial-run-all", "trial-copy-all"):
        el(b).disabled = not (ready and ids)

    s = S.fit and S.fit.get("semi_log")
    el("h0-suggest").textContent = (f"{4 / s['lambda10']:.0f} (4/λ with λ = {s['lambda10']:.4f} from the semi-log fit)"
                                    if s and s["lambda10"] > 0 else "fit the CSL first")

    sel = el("trial-test")
    if S.trial_sel not in ids:
        S.trial_sel = ids[0] if ids else None
    sel.innerHTML = "".join(
        f'<option value="{esc(t["tid"])}"{" selected" if t["tid"] == S.trial_sel else ""}>'
        f'{esc(t["tid"])} ({t["type"]})</option>' for t in tests) or '<option value="">No tests yet</option>'
    if S.trial_sel:
        load_controls(S.trial_sel)
    render_trial_info()
    render_trial_plots()
    render_state_table()
    render_trial_log()


def render_trial_info():
    info = test_info(S.trial_sel) if S.trial_sel else None
    fn = csl_function()
    if not info or not fn:
        el("trial-test-info").textContent = ""
        return
    g = test_rows(S.trial_sel)
    g = g[core.kept_mask(g, S.excluded.get(S.trial_sel)) & (g["p"] > 0) & g["e"].notna()].sort_values("e1", kind="stable")
    if g.empty:
        el("trial-test-info").textContent = ""
        return
    p0, e0 = float(g["p"].iloc[0]), float(g["e"].iloc[0])
    psi0 = e0 - float(fn(np.array([p0]))[0])
    el("trial-test-info").innerHTML = (f"p&prime;<sub>0</sub> = {p0:.1f} kPa &middot; e<sub>0</sub> = {e0:.4f} &middot; "
                                       f"&psi;<sub>0</sub> = {psi0:+.4f} &middot; last &epsilon;<sub>1</sub> = {g['e1'].max():.2f}%")


@when("change", "#trial-test")
def on_trial_test(event):
    save_controls()
    S.trial_sel = event.target.value or None
    if S.trial_sel:
        load_controls(S.trial_sel)
    render_trial_info()
    render_trial_plots()


def run_one_trial(tid, trial, inputs):
    csl, plast = inputs
    info = test_info(tid)
    g = test_rows(tid)
    g = g[core.kept_mask(g, S.excluded.get(tid)) & (g["p"] > 0) & g["e"].notna()]
    cutoff_text = el("trial-cutoff").value.strip()
    try:
        cutoff = float(cutoff_text) if cutoff_text else None
    except ValueError:
        cutoff = None
    record, model, lab = tc.run_trial(g, info["type"], trial, csl, plast, csl_function(),
                                      cutoff=cutoff, rtol=float(el("trial-rtol").value))
    record["run"] = len(S.runs) + 1
    record["test_id"] = tid
    S.runs.append(record)
    S.latest[tid] = (record, model, lab, cutoff)
    return record


async def run_trials(tids):
    inputs = step5_inputs()
    if S.running or inputs is None:
        return
    S.running = True
    save_controls()
    for b in ("trial-run", "trial-run-all"):
        el(b).disabled = True
    status = el("trial-run-status")
    failed = 0
    try:
        for k, tid in enumerate(tids, 1):
            status.textContent = f"Running NorSand for {tid}" + (f" ({k} of {len(tids)})" if len(tids) > 1 else "") + "…"
            await asyncio.sleep(0.03)  # let the page repaint before the solver blocks
            if tid not in S.trials:
                info = test_info(tid)
                S.trials[tid] = {**read_controls(), "ocr": max(1.0, info["ocr"]) if core.finite(info["ocr"]) else 1.0, "s": 0}
            rec = run_one_trial(tid, S.trials[tid], inputs)
            failed += rec["status"] != "OK"
            if tid == S.trial_sel or len(tids) == 1:
                render_trial_plots()
            render_trial_log()
        last = S.runs[-1]
        if len(tids) == 1:
            status.textContent = (f"Run {last['run']}: q RMSE {fmt(last['q_RMSE_kPa'], 2)} kPa, R² {fmt(last['q_R2'], 3)}."
                                  if last["status"] == "OK" else last["message"])
        else:
            status.textContent = f"Ran {len(tids)} tests" + (f", {failed} failed. See the trial log." if failed else ".")
    finally:
        S.running = False
        for b in ("trial-run", "trial-run-all"):
            el(b).disabled = False
        render_state_table()


@when("click", "#trial-run")
async def on_trial_run(event):
    if S.trial_sel:
        await run_trials([S.trial_sel])


@when("click", "#trial-run-all")
async def on_trial_run_all(event):
    await run_trials([t["tid"] for t in trial_tests()])


@when("click", "#trial-copy-all")
def on_trial_copy_all(event):
    save_controls()
    if not S.trial_sel:
        return
    src = S.trials[S.trial_sel]
    for t in trial_tests():
        tid = t["tid"]
        base = S.trials.get(tid) or {"ocr": max(1.0, t["ocr"]) if core.finite(t["ocr"]) else 1.0, "s": 0}
        S.trials[tid] = {**base, **{k: src[k] for k in ("gref", "h0", "hy", "nu", "m")}}
    el("trial-run-status").textContent = (f"Copied Gref, H0, Hy, ν and m from {S.trial_sel} to all tests. "
                                          "Each test keeps its own OCR and S.")


def render_trial_plots():
    tid = S.trial_sel
    entry = S.latest.get(tid) if tid else None
    if not tid or not csl_function():
        for i in TRIAL_PLOTS:
            clear_plot(i)
        return
    info = test_info(tid)
    if entry:
        record, model, lab, cutoff = entry
    else:
        g = test_rows(tid)
        g = g[core.kept_mask(g, S.excluded.get(tid)) & (g["p"] > 0) & g["e"].notna()].sort_values("e1", kind="stable")
        fn = csl_function()
        lab = pd.DataFrame({"e1": g["e1"], "q": g["q"], "p": g["p"], "psi": g["e"] - fn(g["p"].to_numpy(float)),
                            "other": g["ev"] if info["type"] == "CID" else g["du"]})
        record, model, cutoff = None, None, None
    cid = info["type"] == "CID"
    e1_axis = axis("Axial strain, &epsilon;<sub>1</sub> (%)", rangemode="tozero")
    shapes = [vline(cutoff, MUTED, "dot")] if cutoff is not None else None

    def pair(lx, ly, mx=None, my=None):
        t = [line(lx, ly, "Laboratory", color=INK, width=1.8)]
        if model is not None:
            t.append(line(mx, my, "NorSand trial", color=ACCENT, dash="dash", width=2.2))
        return t

    def tag(key, digits, unit=""):
        if not record or record["status"] != "OK" or record.get(key) is None:
            return ""
        return f" &middot; RMSE {record[key]:.{digits}f}{unit}"

    m = model if model is not None else {}
    title_q = "q &ndash; &epsilon;<sub>1</sub>" + tag("q_RMSE_kPa", 2, " kPa")
    if record and record["status"] == "OK" and record.get("q_R2") is not None:
        title_q += f", R&sup2; {record['q_R2']:.3f}"
    draw("fit-q-eps1", pair(lab["e1"], lab["q"], m.get("eps1_pct"), m.get("q_kpa")), e1_axis,
         axis("Deviator stress, q (kPa)"), title=title_q, shapes=shapes)
    draw("fit-p-q", pair(lab["p"], lab["q"], m.get("p_kpa"), m.get("q_kpa")),
         axis("Mean effective stress, p&prime; (kPa)", rangemode="tozero"),
         axis("Deviator stress, q (kPa)", rangemode="tozero"), title="p&prime; &ndash; q")
    if cid:
        draw("fit-vol", pair(lab["e1"], lab["other"], m.get("eps1_pct"), m.get("epsv_pct")), e1_axis,
             axis("Volumetric strain, &epsilon;<sub>v</sub> (%)"),
             title="&epsilon;<sub>v</sub> &ndash; &epsilon;<sub>1</sub>" + tag("volume_RMSE_pct", 3, "%"), shapes=shapes)
    else:
        draw("fit-vol", pair(lab["e1"], lab["other"], m.get("eps1_pct"), m.get("delta_u_kpa")), e1_axis,
             axis("Excess pore pressure, &Delta;u (kPa)"),
             title="&Delta;u &ndash; &epsilon;<sub>1</sub>" + tag("u_RMSE_kPa", 2, " kPa"), shapes=shapes)
    draw("fit-psi", pair(lab["e1"], lab["psi"], m.get("eps1_pct"), m.get("psi")), e1_axis,
         axis("State parameter, &psi;"), title="&psi; &ndash; &epsilon;<sub>1</sub>" + tag("psi_RMSE", 4),
         shapes=(shapes or []) + [hline(0)])
    if record and record["status"] != "OK":
        el("trial-run-status").textContent = record["message"]


def render_state_table():
    body = el("state-check").querySelector("tbody")
    rows = []
    for t in trial_tests():
        entry = S.latest.get(t["tid"])
        if not entry:
            continue
        r = entry[0]
        other = r["volume_RMSE_pct"] if t["type"] == "CID" else r["u_RMSE_kPa"]
        unit = "%" if t["type"] == "CID" else " kPa"
        peak = (f"{r['q_peak_lab_kPa']:.0f} / {r['q_peak_sim_kPa']:.0f}" if r["status"] == "OK" else "&ndash;")
        rows.append(
            f'<tr><td>{esc(t["tid"])}</td><td>{t["type"]}</td><td class="num">{fmt(r["psi0"], 4)}</td>'
            f'<td class="num">{fmt(r["ocr"], 2)}</td><td class="num">{fmt(r["gref"], 0)}</td><td class="num">{fmt(r["h0"], 1)}</td>'
            f'<td class="num">{fmt(r["hy"], 1)}</td><td class="num">{r["s"]}</td>'
            f'<td class="num">{fmt(r["q_RMSE_kPa"], 2)}</td><td class="num">{fmt(r["q_R2"], 3)}</td>'
            f'<td class="num">{fmt(other, 3)}{unit if core.finite(other) else ""}</td><td class="num">{fmt(r["psi_RMSE"], 4)}</td>'
            f'<td class="num">{peak}</td></tr>')
    body.innerHTML = "".join(rows) or '<tr class="cal-table__empty"><td colspan="13">Results appear here once NorSand has run.</td></tr>'


def render_trial_log():
    body = el("trial-log").querySelector("tbody")
    if not S.runs:
        body.innerHTML = '<tr class="cal-table__empty"><td colspan="13">No runs yet.</td></tr>'
        return
    rows = []
    for r in reversed(S.runs):
        cls = "badge--open" if r["status"] == "OK" else "badge--external"
        note = "" if r["status"] == "OK" else esc(r["message"])
        rows.append(
            f'<tr><td class="num">{r["run"]}</td><td>{esc(r["test_id"])}</td><td><span class="badge {cls}">{r["status"]}</span></td>'
            f'<td class="num">{fmt(r["gref"], 0)}</td><td class="num">{fmt(r["h0"], 1)}</td><td class="num">{fmt(r["hy"], 1)}</td>'
            f'<td class="num">{fmt(r["ocr"], 2)}</td><td class="num">{fmt(r["nu"], 2)}</td><td class="num">{fmt(r["m"], 2)}</td>'
            f'<td class="num">{r["s"]}</td><td class="num">{fmt(r["q_RMSE_kPa"], 2)}</td><td class="num">{fmt(r["q_R2"], 3)}</td>'
            f'<td class="cal-checks">{note}</td></tr>')
    body.innerHTML = "".join(rows)


@when("click", "#trial-download-log")
def on_trial_log_download(event):
    if S.runs:
        download(tc.log_csv(S.runs), "norsand_manual_trials.csv", "text/csv")


@when("click", "#trial-download-sim")
def on_trial_sim_download(event):
    entry = S.latest.get(S.trial_sel) if S.trial_sel else None
    if entry and entry[1] is not None:
        download(tc.model_csv(entry[1]), f"norsand_{pc.safe_name(S.trial_sel)}_run{entry[0]['run']}.csv", "text/csv")


# --------------------------------------------------------------------------------------
# Step 6: common hardening and stiffness, shared run and export (Part 4)
# --------------------------------------------------------------------------------------
LAWS = {"const": "One stiffness", "state": "State-dependent"}
COMMON_PLOTS = ["common-h", "common-g", "common-gp0", "common-gpred"]
FINAL_PLOTS = ["final-q-eps1", "final-p-q", "final-vol", "final-psi"]


def test_start(tid):
    """p0, e0 and psi0 (step 3 CSL) from the first kept row of a test."""
    fn = csl_function()
    g = test_rows(tid)
    g = g[core.kept_mask(g, S.excluded.get(tid)) & (g["p"] > 0) & g["e"].notna()].sort_values("e1", kind="stable")
    if g.empty or fn is None:
        return None
    p0, e0 = float(g["p"].iloc[0]), float(g["e"].iloc[0])
    return {"p0": p0, "e0": e0, "psi0": e0 - float(fn(np.array([p0]))[0])}


def psi_used(tid, start):
    if el("common-vincent-psi").checked and tid in S.vincent_psi:
        return S.vincent_psi[tid]
    return start["psi0"]


def fill_from_step5(only_missing=False):
    n = 0
    for t in trial_tests():
        tid = t["tid"]
        entry = S.latest.get(tid)
        if only_missing and tid in S.common_rows:
            continue
        if entry and entry[0]["status"] == "OK":
            r = entry[0]
            S.common_rows[tid] = {"use": True, "gref": r["gref"], "h": r["h0"] - r["hy"] * r["psi0"],
                                  "ocr": r["ocr"], "s": int(r["s"]), "source": f"step 5, run {r['run']}"}
            n += 1
        elif tid not in S.common_rows:
            ocr = t["ocr"] if core.finite(t["ocr"]) else 1.0
            S.common_rows[tid] = {"use": False, "gref": float("nan"), "h": float("nan"),
                                  "ocr": max(1.0, ocr), "s": 0, "source": "not run in step 5"}
    return n


@when("click", "#common-fill-step5")
def on_common_fill(event):
    n = fill_from_step5()
    el("common-run-status").textContent = (f"Filled {n} test(s) from their latest step 5 run." if n
                                           else "No successful step 5 runs yet. Run the tests in step 5 first.")
    refresh_common()


@when("click", "#common-load-vincent")
def on_common_vincent(event):
    ids = {t["tid"]: t for t in trial_tests()}
    matched = 0
    for r in cmn.VINCENT_FRS.itertuples():
        if r.test_id in ids:
            S.common_rows[r.test_id] = {"use": True, "gref": float(r.Gref_bestfit_kPa), "h": float(r.H_bestfit),
                                        "ocr": float(r.OCR), "s": int(r.S), "source": "Vincent (Seequent 2021)"}
            S.vincent_psi[r.test_id] = float(r.psi0_Vincent)
            matched += 1
    el("common-vincent-psi-field").hidden = matched == 0
    el("common-run-status").textContent = (
        f"Loaded Vincent's values for {matched} of 12 Fraser River sand tests." if matched
        else "None of Vincent's test IDs (CID-D050 … CIU-M400) match the imported tests.")
    refresh_common()


def render_common_inputs():
    body = el("common-inputs").querySelector("tbody")
    rows = []
    for t in trial_tests():
        tid = t["tid"]
        row = S.common_rows.get(tid)
        start = test_start(tid)
        if row is None or start is None:
            continue
        psi = psi_used(tid, start)

        def box(field, v, digits):
            return (f'<input type="number" step="any" data-field="{field}" '
                    f'value="{f"{v:.{digits}f}" if core.finite(v) else ""}">')
        s_opts = "".join(f'<option value="{v}"{" selected" if row["s"] == v else ""}'
                         f'{" disabled" if v == 1 and t["type"] == "CID" else ""}>{v}</option>' for v in (0, 1))
        rows.append(
            f'<tr data-tid="{esc(tid)}"><td><input type="checkbox" data-field="use"{" checked" if row["use"] else ""}></td>'
            f'<td>{esc(tid)}</td><td>{t["type"]}</td><td class="num">{start["p0"]:.1f}</td><td class="num">{psi:+.4f}</td>'
            f'<td>{box("gref", row["gref"], 0)}</td><td>{box("h", row["h"], 1)}</td><td>{box("ocr", row["ocr"], 2)}</td>'
            f'<td><select data-field="s">{s_opts}</select></td><td class="cal-checks">{esc(row["source"])}</td></tr>')
    body.innerHTML = "".join(rows) or '<tr class="cal-table__empty"><td colspan="10">Tests appear here once the CSL and plasticity are calibrated.</td></tr>'


@when("change", "#common-inputs")
def on_common_input(event):
    target = event.target
    tr = target.closest("tr[data-tid]")
    if tr is None or not target.dataset.field:
        return
    row = S.common_rows[tr.dataset.tid]
    field = target.dataset.field
    if field == "use":
        row["use"] = bool(target.checked)
    elif field == "s":
        row["s"] = int(target.value)
    else:
        try:
            row[field] = float(target.value)
        except ValueError:
            row[field] = float("nan")
    if field != "use":
        row["source"] = "edited"
        tr.querySelector("td:last-child").textContent = "edited"
    refit_common()


@when("change", "#common-m, #common-vincent-psi")
def on_common_settings(event):
    refresh_common()


def common_m_nu():
    def num(i, default):
        try:
            v = float(el(i).value)
            return v if math.isfinite(v) else default
        except ValueError:
            return default
    return num("common-m", 0.47), num("common-nu", 0.2)


def refresh_common():
    render_common_inputs()
    refit_common()


def refit_common():
    m, _ = common_m_nu()
    pref = S.fit["pref"] if S.fit else core.PREF_DEFAULT
    el("common-pref").textContent = f"{pref:g} kPa"
    rows = []
    for t in trial_tests():
        tid = t["tid"]
        row = S.common_rows.get(tid)
        start = test_start(tid)
        if row and start and row["use"] and core.finite(row["gref"]) and core.finite(row["h"]):
            rows.append({"tid": tid, "psi0": psi_used(tid, start), "p0": start["p0"], "gref": row["gref"], "h": row["h"]})
    try:
        S.common_fit = cmn.fit_relations(rows, m, pref)
        el("common-fit-status").textContent = f"Fitted to {len(rows)} tests."
    except ValueError as exc:
        S.common_fit = None
        el("common-fit-status").textContent = str(exc) + f" ({len(rows)} usable now.)"
    f = S.common_fit
    outs = {"common-h0": ("H0", 2), "common-hy": ("Hpsi", 2), "common-h-r2": ("H_r2", 3), "common-h-rmse": ("H_rmse", 3),
            "common-h-loo": ("H_loocv_rmse", 3), "common-gc": ("Gref_const", 0), "common-gc-r2": ("Gref_const_r2", 3),
            "common-gc-rmse": ("Gref_const_rmse", 0), "common-ga": ("Gref_a", 0), "common-gb": ("Gref_b", 0),
            "common-gs-r2": ("Gref_state_r2", 3), "common-gs-rmse": ("Gref_state_rmse", 0),
            "common-gs-loo": ("Gref_state_loocv_rmse", 0)}
    for id_, (key, digits) in outs.items():
        el(id_).innerHTML = fmt(f and f[key], digits)
    el("common-run").disabled = not (f and step5_inputs())
    el("common-adopt").disabled = not f
    render_common_plots()


def render_common_plots():
    f = S.common_fit
    if not f:
        for i in COMMON_PLOTS:
            clear_plot(i)
        return
    pts = f["points"]
    psi = [p["psi0"] for p in pts]
    names = [p["tid"] for p in pts]
    xs = np.linspace(min(psi) - 0.03, max(psi) + 0.03, 60)
    psi_axis = axis("Initial state parameter, &psi;<sub>0</sub>")

    def dots(x, y, name="Individual values", **kw):
        return {"type": "scatter", "mode": "markers+text", "x": lst(x), "y": lst(y), "text": names, "name": name,
                "textposition": "top right", "textfont": {"size": 9, "color": MUTED},
                "marker": {"color": INK, "size": 8}, **kw}

    draw("common-h", [dots(psi, [p["h"] for p in pts]),
                      line(xs, f["H0"] - f["Hpsi"] * xs, "H = H<sub>0</sub> &minus; H<sub>y</sub>&psi;<sub>0</sub>", color=ACCENT, width=2.4)],
         psi_axis, axis("Individual H"),
         title=f"H = {f['H0']:.1f} &minus; ({f['Hpsi']:.1f}) &psi;<sub>0</sub> &middot; R&sup2; {f['H_r2']:.3f}", legend=True)
    draw("common-g", [dots(psi, [p["gref"] / 1000 for p in pts]),
                      line(xs, (f["Gref_a"] + f["Gref_b"] * xs) / 1000, "a + b&psi;<sub>0</sub>", color=BLUE, dash="dot", width=2.4),
                      line([xs[0], xs[-1]], [f["Gref_const"] / 1000] * 2, "One G<sub>ref</sub>", color=ACCENT, dash="dash", width=2.4)],
         psi_axis, axis("G<sub>ref</sub> (MPa)"),
         title=f"G<sub>ref</sub> &ndash; &psi;<sub>0</sub> &middot; R&sup2; {f['Gref_state_r2']:.3f} (state), {f['Gref_const_r2']:.3f} (one)", legend=True)
    p0 = [p["p0"] for p in pts]
    px = np.linspace(min(p0) * 0.95, max(p0) * 1.05, 100)
    draw("common-gp0", [
        markers(p0, [p["G_p0_input"] / 1000 for p in pts], "Individual values at p′₀", color=INK, size=8),
        line(px, f["Gref_const"] / 1000 * (px / f["pref"]) ** f["m"], "One G<sub>ref</sub>", color=ACCENT, dash="dash", width=2.4),
        markers(p0, [p["G_p0_state"] / 1000 for p in pts], "State-dependent", color=BLUE, symbol="x", size=9)],
        axis("Initial mean effective stress, p&prime;<sub>0</sub> (kPa)"), axis("G at p&prime;<sub>0</sub> (MPa)"),
        title=f"G at p&prime;<sub>0</sub> with m = {f['m']:g} fixed", legend=True)
    g = [p["gref"] / 1000 for p in pts]
    lim = [min(g + [p["Gref_state_fit"] / 1000 for p in pts]) - 1, max(g + [p["Gref_state_fit"] / 1000 for p in pts]) + 1]
    draw("common-gpred", [
        markers(g, [p["Gref_state_fit"] / 1000 for p in pts], "State-dependent", color=BLUE, size=8, text=names),
        markers(g, [p["Gref_const_fit"] / 1000 for p in pts], "One G<sub>ref</sub>", color=ACCENT, symbol="x", size=9, text=names),
        line(lim, lim, "1:1", color=MUTED, dash="dot", width=1)],
        axis("Individual G<sub>ref</sub> (MPa)"), axis("Predicted G<sub>ref</sub> (MPa)"),
        title="Predicted &ndash; individual G<sub>ref</sub>", legend=True)


def update_common_page():
    inputs = step5_inputs()
    status = el("common-status")
    if not S.built:
        status.textContent = "Import data in step 1 first."
        status.hidden = False
    elif inputs is None:
        status.innerHTML = "Fit the CSL in step 3 and calibrate M<sub>tc</sub>, N and &chi;<sub>tc</sub> in step 4 first."
        status.hidden = False
    else:
        status.hidden = True
        fill_from_step5(only_missing=True)
    refresh_common()
    render_final_table()
    render_final_select()
    render_final_plots()


async def run_common():
    inputs = step5_inputs()
    f = S.common_fit
    if S.running or inputs is None or not f:
        return
    S.running = True
    el("common-run").disabled = True
    status = el("common-run-status")
    m, nu = common_m_nu()
    used = [p["tid"] for p in f["points"]]
    psi_map = {p["tid"]: p["psi0"] for p in f["points"]}
    S.common_runs = {}
    failed = 0
    try:
        for k, tid in enumerate(used, 1):
            row = S.common_rows[tid]
            for law in LAWS:
                status.textContent = f"Running {tid} with {LAWS[law].lower()} ({k} of {len(used)})…"
                await asyncio.sleep(0.02)
                trial = {"gref": cmn.gref_for(f, law, psi_map[tid]), "h0": f["H0"], "hy": f["Hpsi"],
                         "ocr": row["ocr"], "nu": nu, "m": m, "s": row["s"]}
                g = test_rows(tid)
                g = g[core.kept_mask(g, S.excluded.get(tid)) & (g["p"] > 0) & g["e"].notna()]
                record, model, lab = tc.run_trial(g, test_info(tid)["type"], trial, inputs[0], inputs[1], csl_function())
                record["test_id"], record["law"] = tid, law
                S.common_runs[(tid, law)] = (record, model, lab)
                failed += record["status"] != "OK"
        status.textContent = (f"Ran {len(used)} tests with both stiffness laws"
                              + (f"; {failed} run(s) failed, see the table." if failed else "."))
    finally:
        S.running = False
        el("common-run").disabled = False
    if S.final_sel not in used:
        S.final_sel = used[0] if used else None
    render_final_table()
    render_final_select()
    render_final_plots()


@when("click", "#common-run")
async def on_common_run(event):
    await run_common()


def render_final_table():
    body = el("fit-errors").querySelector("tbody")
    rows = []
    for (tid, law), (r, _m, _l) in S.common_runs.items():
        typ = r["type"]
        other = r["volume_RMSE_pct"] if typ == "CID" else r["u_RMSE_kPa"]
        unit = "%" if typ == "CID" else " kPa"
        step5 = S.latest.get(tid)
        step5_q = step5[0]["q_RMSE_kPa"] if step5 and step5[0]["status"] == "OK" else None
        ok = r["status"] == "OK"
        peak = f"{r['q_peak_lab_kPa']:.0f} / {r['q_peak_sim_kPa']:.0f}" if ok else esc(r["message"])
        rows.append(
            f'<tr data-tid="{esc(tid)}" class="cal-row{" is-active" if tid == S.final_sel else ""}"><td>{esc(tid)}</td><td>{typ}</td>'
            f'<td class="num">{fmt(r["psi0"], 4)}</td><td>{LAWS[law]}</td><td class="num">{fmt(r["gref"], 0)}</td>'
            f'<td class="num">{fmt(r["q_RMSE_kPa"], 2)}</td><td class="num">{fmt(r["q_R2"], 3)}</td>'
            f'<td class="num">{fmt(other, 3)}{unit if core.finite(other) else ""}</td><td class="num">{fmt(r["psi_RMSE"], 4)}</td>'
            f'<td class="num">{peak}</td><td class="num">{fmt(step5_q, 2)}</td></tr>')
    body.innerHTML = "".join(rows) or '<tr class="cal-table__empty"><td colspan="11">Results appear here after the shared run.</td></tr>'


def render_final_select():
    tids = list(dict.fromkeys(tid for tid, _ in S.common_runs))
    el("final-test").innerHTML = "".join(
        f'<option value="{esc(t)}"{" selected" if t == S.final_sel else ""}>{esc(t)}</option>' for t in tids) \
        or '<option value="">No results yet</option>'


@when("change", "#final-test")
def on_final_test(event):
    S.final_sel = event.target.value or None
    render_final_table()
    render_final_plots()


@when("click", "#fit-errors")
def on_final_row(event):
    tr = event.target.closest("tr[data-tid]")
    if tr is not None:
        S.final_sel = tr.dataset.tid
        el("final-test").value = S.final_sel
        render_final_table()
        render_final_plots()


def render_final_plots():
    tid = S.final_sel
    runs = {law: S.common_runs.get((tid, law)) for law in LAWS} if tid else {}
    base = next((v for v in runs.values() if v), None)
    if not base:
        for i in FINAL_PLOTS:
            clear_plot(i)
        return
    lab = base[2]
    cid = base[0]["type"] == "CID"
    styles = {"const": (ACCENT, "dash"), "state": (BLUE, "dot")}

    def series(xcol_lab, ycol_lab, xcol, ycol):
        t = [line(lab[xcol_lab], lab[ycol_lab], "Laboratory", color=INK, width=1.8)]
        for law, v in runs.items():
            if v and v[1] is not None:
                color, dash = styles[law]
                t.append(line(v[1][xcol], v[1][ycol], LAWS[law], color=color, dash=dash, width=2.2))
        return t

    e1_axis = axis("Axial strain, &epsilon;<sub>1</sub> (%)", rangemode="tozero")
    draw("final-q-eps1", series("e1", "q", "eps1_pct", "q_kpa"), e1_axis, axis("Deviator stress, q (kPa)"),
         title=f"q &ndash; &epsilon;<sub>1</sub> &middot; {esc(tid)}")
    draw("final-p-q", series("p", "q", "p_kpa", "q_kpa"), axis("Mean effective stress, p&prime; (kPa)", rangemode="tozero"),
         axis("Deviator stress, q (kPa)", rangemode="tozero"), title="p&prime; &ndash; q")
    if cid:
        draw("final-vol", series("e1", "other", "eps1_pct", "epsv_pct"), e1_axis,
             axis("Volumetric strain, &epsilon;<sub>v</sub> (%)"), title="&epsilon;<sub>v</sub> &ndash; &epsilon;<sub>1</sub>")
    else:
        draw("final-vol", series("e1", "other", "eps1_pct", "delta_u_kpa"), e1_axis,
             axis("Excess pore pressure, &Delta;u (kPa)"), title="&Delta;u &ndash; &epsilon;<sub>1</sub>")
    draw("final-psi", series("e1", "psi", "eps1_pct", "psi"), e1_axis, axis("State parameter, &psi;"),
         title="&psi; &ndash; &epsilon;<sub>1</sub>", shapes=[hline(0)])


@when("click", "#common-adopt")
def on_common_adopt(event):
    f = S.common_fit
    if not f:
        return
    law = document.querySelector('input[name="adopt-law"]:checked').value
    m, nu = common_m_nu()
    S.adopted_common = {"law": law, "H0": f["H0"], "Hy": f["Hpsi"], "m": m, "nu": nu, "pref": f["pref"],
                        "Gref": f["Gref_const"] if law == "const" else None,
                        "Gref_a": f["Gref_a"] if law == "state" else None, "Gref_b": f["Gref_b"] if law == "state" else None,
                        "per_test": {tid: {"ocr": r["ocr"], "S": r["s"]} for tid, r in S.common_rows.items() if r["use"]}}
    window.calibrationCommon = jsobj(S.adopted_common)
    outs = {o.dataset.param: o for o in document.querySelectorAll("#final-params output")}
    outs["h0"].innerHTML = fmt(f["H0"], 2)
    outs["hy"].innerHTML = fmt(f["Hpsi"], 2)
    outs["gref"].innerHTML = (fmt(f["Gref_const"], 0) if law == "const"
                              else f"{f['Gref_a']:.0f} + ({f['Gref_b']:.0f})&psi;<sub>0</sub>")
    outs["m"].innerHTML = fmt(m, 3)
    outs["nu"].innerHTML = fmt(nu, 3)
    svals = sorted({r["s"] for r in S.common_rows.values() if r["use"]})
    outs["s"].innerHTML = str(svals[0]) if len(svals) == 1 else "per test (" + ", ".join(map(str, svals)) + ")"
    el("common-adopt-status").textContent = (f"Adopted H0, Hy and the {LAWS[law].lower()} law. "
                                             "Accept the CSL in step 3 and the plasticity in step 4 to complete the set.")


# ---------- downloads ----------
def parameter_set():
    p = {"status": "Educational calibration. Review every step before engineering use.",
         "csl": None, "plasticity": None, "hardening_and_elasticity": S.adopted_common}
    if S.fit and S.fit.get("semi_log"):
        p["csl"] = {**core.norsand_csl(S.fit, csl_choice()), "fitted_to_points": S.fit["n"]}
    if S.adopted:
        p["plasticity"] = {"mtc": S.adopted["m"], "n": S.adopted["n"], "chi_tc": S.adopted["chi"],
                           "reason": el("plast-reason").value}
    return p


@when("click", "#export-params")
def on_export_params(event):
    download(json.dumps(parameter_set(), indent=2, default=str), "norsand_calibrated_parameters.json", "application/json")


@when("click", "#export-results")
def on_export_results(event):
    if S.common_runs:
        rows = [r for (r, _m, _l) in S.common_runs.values()]
        df = pd.DataFrame(rows, columns=["test_id", "law"] + [c for c in tc.LOG_FIELDS if c not in ("run", "test_id")])
        download(df.to_csv(index=False), "norsand_shared_run_results.csv", "text/csv")
    if S.common_fit:
        download(cmn.points_csv(S.common_fit), "norsand_common_relationship_points.csv", "text/csv")


@when("click", "#export-report")
def on_export_report(event):
    lines = ["NORSAND CALIBRATION REPORT", ""]
    if S.built:
        lines += [f"Tests: {len(S.built['tests'])}; rows: {len(S.built['data'])}", ""]
    lines.append("STEP 3: CSL")
    lines.append("  " + csl_note())
    if S.fit and S.fit.get("semi_log"):
        lines.append(f"  Fitted to {S.fit['n']} accepted points.")
    lines += ["", "STEP 4: PLASTICITY"]
    if S.adopted:
        lines.append(f"  Mtc = {S.adopted['m']:.4f}, N = {S.adopted['n']:.4f}, chi_tc = {S.adopted['chi']:.4f}"
                     f" from {len(S.points4)} dense tests. Reason: {el('plast-reason').value or 'least squares'}")
    lines += ["", "STEP 5: MANUAL TRIALS", f"  {len(S.runs)} runs."]
    for tid, (r, *_rest) in S.latest.items():
        lines.append(f"  {tid}: Gref={r['gref']:g}, H0={r['h0']:g}, Hy={r['hy']:g}, OCR={r['ocr']:g}, S={r['s']}, "
                     f"q RMSE={fmt(r['q_RMSE_kPa'], 2)} kPa")
    lines += ["", "STEP 6: COMMON RELATIONSHIPS"]
    if S.common_fit:
        lines += ["  " + s for s in cmn.summary_lines(S.common_fit)]
    for (tid, law), (r, *_rest) in S.common_runs.items():
        lines.append(f"  {tid} [{LAWS[law]}]: Gref={fmt(r['gref'], 0)}, q RMSE={fmt(r['q_RMSE_kPa'], 2)} kPa, "
                     f"R2={fmt(r['q_R2'], 3)}, status {r['status']}")
    lines += ["", "ADOPTED", "  " + json.dumps(parameter_set(), default=str)]
    download("\n".join(lines).replace("&ndash;", "-") + "\n", "norsand_calibration_report.txt")


@when("click", "#export-figures")
def on_export_figures(event):
    for div in document.querySelectorAll("#step-6 .js-plotly-plot"):
        Plotly.downloadImage(div, jsobj({"format": "png", "filename": f"norsand_{div.id}", "width": 900, "height": 600}))


def on_step_shown(event):
    if event.detail == 4 and S.plast_dirty:
        update_plasticity()
    elif event.detail == 5:
        update_trial_page()
    elif event.detail == 6:
        if S.trial_sel:
            save_controls()
        update_common_page()
    elif S.trial_sel:
        save_controls()


# --------------------------------------------------------------------------------------
# Start-up
# --------------------------------------------------------------------------------------
document.addEventListener("cal:step", create_proxy(on_step_shown))
remember_placeholders()
el("engine-status").textContent = "Ready. Choose a file, paste data or enter values."
el("engine-status").classList.add("is-ready")
_files = el("lab-file").files
if _files and _files.length:
    asyncio.ensure_future(read_file_list(_files))

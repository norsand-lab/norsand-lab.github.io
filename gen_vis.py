#Drained dense graphs
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import json
import pandas as pd
from pyscript import document, when, display
from pyodide.ffi import to_js
from js import Plotly
from modules.norsand_core import run_triaxial
from modules.norsand_core import NorSandParams

# outputs shortened list with stepnum items with log scale
def data_choose(data, stepnum):
    filtered = []
    base = np.power(len(data), (1 / (stepnum - 1)))
    for i in range(stepnum):
        filtered.append(data[int(np.round(np.power(base, i))) - 1])
    return filtered

# calculates p and q coordinates for yield curve
def yield_curve(pi, M, n=200):
    p = np.linspace(0, pi * np.e, n)
    q = np.zeros_like(p)
    q[1:] = M * p[1:] * (1 + np.log(pi / p[1:]))
    return p, q

@when("click", "#run-button")
def plot_graph(event):
    PARAMS = NorSandParams(
    name="GeoStudio CID Dense",
    drainage=document.getElementById("test-type").value,
    p0=float(document.getElementById("p0").value),
    psi0=float(document.getElementById("psi0").value),
    ocr=float(document.getElementById("ocr").value),
    eta0=float(document.getElementById("eta0").value),
    csl_type=document.getElementById("csl-form").value,
    pref=float(document.getElementById("pref").value),
    gamma=float(document.getElementById("gamma").value) / 100,
    lambda10=float(document.getElementById("lambda").value),
    ca=float(document.getElementById("ca").value),
    cb=float(document.getElementById("cb").value),
    cc=float(document.getElementById("cc").value),
    mtc=float(document.getElementById("mtc").value),
    n=float(document.getElementById("n_param").value),
    chi=float(document.getElementById("chi-tc").value),
    h0=float(document.getElementById("h0").value),
    hy=float(document.getElementById("hy").value),
    s=float(document.getElementById("s_param").value),
    gref=float(document.getElementById("gref").value),  # kPa at p_ref=100 kPa, as in the GeoStudio table
    nu=float(document.getElementById("nu").value),
    m=float(document.getElementById("m_exp").value),
    max_strain=float(document.getElementById("max-strain").value) / 100,   # matches MATLAB validated driver: MaxStrainForSim = 0.22
    n_steps=int(document.getElementById("max-steps").value),     # matches MATLAB validated history export
    )
    df = run_triaxial(PARAMS)
    data = df.to_dict(orient="list")
    mtc = PARAMS.mtc                # CSL slope
    slide_steps = 18                # number of slider steps
    initial_step = 0                # step that slider starts on
    static_num = 5                  # number of static traces
    step_to_indices = {}            # stores and indexes dynamic traces
    p_path = []                     # loading path p values
    q_path = []                     # loading path q values
    p_max = 0                       # maximum x axis p value, currently 0 but changes based on data
    q_max = max(data["p_kpa"]) * mtc  # maximum y axis q value
    a, b, c = PARAMS.ca, PARAMS.cb, PARAMS.cc     # parameters for curved CSL in void ratio space
    lam, gam = PARAMS.lambda10, PARAMS.gamma      # parameters for semi-log CSL in void ratio space
    pref = PARAMS.pref              # reference pressure for CSL
    pressure = []                   # pore water pressure for isotropic conditions
    for step in data["step"]:
        pressure.append(data["p_kpa"][0] + (data["q_kpa"][step]) / 3 - data["p_kpa"][step])
    main_x, main_y = 1, 1           # layout positions of graphs (row,col)
    a_x, a_y = 1, 2
    b_x, b_y = 2, 2
    c_x, c_y = 3, 2

    # Initialize figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        column_widths=[0.7, 0.3],
        horizontal_spacing=0.08,   # space between columns
        vertical_spacing=0.1,      # space between rows
        specs=[[{"rowspan": 3}, {}],
               [     None,      {}],
               [     None,      {}]],
    )

    # Static traces - always visible
    fig.add_trace(go.Scatter(x=[0, 2.5 * q_max / mtc], y=[0, 2.5 * q_max], visible=True, name="CSL", mode="lines", line=dict(dash='dot', color="blue")), row=main_x, col=main_y)
    fig.add_trace(go.Scatter(x=data["eps1_pct"], y=data["q_kpa"], visible=True, name="Loading Path", showlegend=False, line=dict(color="#282828")), row=a_x, col=a_y)
    fig.add_trace(go.Scatter(x=data["eps1_pct"], y=data["epsv_pct"], visible=True, name="Loading Path", showlegend=False, line=dict(color="#282828")), row=b_x, col=b_y)
    ec = []
    if(PARAMS.csl_type == "curved"):
        for p in data["p_kpa"]:
            ec.append(a - b * pow((p / pref), c))
    elif(PARAMS.csl_type == "semi_log"):
        for p in data["p_kpa"]:
            ec.append(gam - lam * np.log10(p / pref))
    fig.add_trace(go.Scatter(x=data["p_kpa"], y=ec, visible=True, name="CSL", showlegend=False, mode="lines", line=dict(color="blue", dash='dot')), row=c_x, col=c_y)
    fig.add_trace(go.Scatter(x=data["p_kpa"], y=data["e"], visible=True, name="Loading Path", showlegend=False, line=dict(color="#282828")), row=c_x, col=c_y)

    # Slider-controlled traces
    for k, step in enumerate(data_choose(data["step"], slide_steps)):
        p_yield, q_yield = yield_curve(data["p_i_kpa"][step], data["Mi"][step])
        if max(p_yield) > p_max:
            p_max = max(p_yield)
        idx_start = len(fig.data)
        fig.add_trace(go.Scatter(
            visible=False,
            name="Yield Surface",
            line=dict(color="purple"),
            x=p_yield,
            y=q_yield
        ),
            row=main_x, col=main_y
        )

        current_p = data["p_kpa"][step]
        current_q = data["q_kpa"][step]
        for i in range(step):
            p_path = data["p_kpa"][0:i + 1]
            q_path = data["q_kpa"][0:i + 1]
        fig.add_trace(go.Scatter(
            visible=False,
            name="Current State",
            mode="markers",
            marker=dict(size=10, color="red"),
            x=[current_p],
            y=[current_q]
        ),
            row=main_x, col=main_y
        )

        fig.add_trace(go.Scatter(
            visible=False,
            name="Loading Path",
            mode='lines',
            line=dict(color="#282828"),
            x=p_path,
            y=q_path
        ),
            row=main_x, col=main_y
        )

        fig.add_trace(go.Scatter(
            visible=False,
            name="Image Point",
            mode="markers",
            marker=dict(size=10, color="green"),
            x=[data["p_i_kpa"][step]],
            y=[data["p_i_kpa"][step] * data["Mi"][step]]
        ),
            row=main_x, col=main_y
        )

        fig.add_trace(go.Scatter(
            visible=False,
            name="Current State",
            mode="markers",
            marker=dict(
                size=12,
                symbol="circle-open",
                color="red",
                line=dict(width=2)
            ),
            showlegend=False,
            x=[data["eps1_pct"][step]],
            y=[data["q_kpa"][step]]
        ),
            row=a_x, col=a_y
        )

        fig.add_trace(go.Scatter(
            visible=False,
            name="Current State",
            mode="markers",
            marker=dict(
                size=12,
                symbol="circle-open",
                color="red",
                line=dict(width=2)
            ),
            showlegend=False,
            x=[data["eps1_pct"][step]],
            y=[data["epsv_pct"][step]]
        ),
            row=b_x, col=b_y
        )

        fig.add_trace(go.Scatter(
            visible=False,
            name="Current State",
            mode="markers",
            marker=dict(
                size=12,
                symbol="circle-open",
                color="red",
                line=dict(width=2)
            ),
            showlegend=False,
            x=[data["p_kpa"][step]],
            y=[data["e"][step]]
        ),
            row=c_x, col=c_y
        )
        idx_end = len(fig.data)
        step_to_indices[k] = list(range(idx_start, idx_end))

    # Initial visible traces
    for i in step_to_indices[initial_step]:
        fig.data[i].visible = True

    # updates visible traces based on slider
    steps = []
    for k in range(len(step_to_indices)):
        visible = [True] * static_num + [False] * (len(fig.data) - static_num)
        for i in step_to_indices[k]:
            visible[i] = True
        steps.append(dict(method="update", args=[{"visible": visible}], label=f"{fig.data[step_to_indices[k][5]].x[0]:.3f}%"))

    sliders = [dict(
        active=initial_step,
        steps=steps,
        currentvalue={"prefix": "Axial Strain: "}
    )]
    fig.update_layout(
        uirevision="constant",
        sliders=sliders
    )

    # formatting and calculating axis ranges
    q_range = [min(data["q_kpa"]), max(data["q_kpa"])]
    q_buffer = 0.15 * (max(data["q_kpa"]) - min(data["q_kpa"]))
    ec_range = [min(min(ec), min(data["e"])), max(max(ec), max(data["e"]))]
    ec_buffer = 0.15 * (max(max(ec), max(data["e"])) - min(min(ec), min(data["e"])))
    v_range = [min(data["epsv_pct"]), max(data["epsv_pct"])]
    v_buffer = 0.15 * (max(data["epsv_pct"]) - min(data["epsv_pct"]))
    a_range = [min(data["eps1_pct"]), max(data["eps1_pct"])]
    lnp_range = np.log10([min(data["p_kpa"]), max(data["p_kpa"])]) #compensates for plotly only accepting log axis ranges in log10

    fig.update_yaxes(title_text="Deviatoric stress, q (kPa)", range=[0, 2.5 * q_max], row=main_x, col=main_y)
    fig.update_xaxes(title_text="Mean effective stress, p' (kPa)", range=[0, 1.15 * p_max], row=main_x, col=main_y)

    fig.update_yaxes(title_text="q (kPa)", range=[q_range[0], q_range[1] + q_buffer], row=a_x, col=a_y)
    fig.update_yaxes(title_text="Volumetric strain, ε<sub>V</sub> (%)", range=[v_range[0], v_range[1] + v_buffer], row=b_x, col=b_y)
    fig.update_yaxes(title_text="Void ratio, e", range=[ec_range[0] - ec_buffer, ec_range[1] + ec_buffer], row=c_x, col=c_y)

    fig.update_xaxes(title_text="Axial strain, ε<sub>1</sub> (%)", title_standoff=0, range=a_range, row=a_x, col=a_y)
    fig.update_xaxes(title_text="Axial strain, ε<sub>1</sub> (%)", title_standoff=0, range=a_range, row=b_x, col=b_y)
    fig.update_xaxes(title_text="Mean effective stress, p'(kPa, log scale)", type="log", range=lnp_range, title_standoff=0, row=c_x, col=c_y)

    fig.update_layout(
        width=900,
        height=600,
        template="plotly_white",
        font=dict(size=10),
        margin=dict(l=30, r=20, t=30, b=30),
        legend=dict(
            font=dict(size=9),
            x=1.02, y=1,
            xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#ddd",
            borderwidth=1)
    )
    fig.update_xaxes(
        tickfont=dict(size=9),
    )
    fig.update_yaxes(
        tickfont=dict(size=9),
    )
    fig.update_traces(marker=dict(size=7), line=dict(width=2))
    fig_dict = json.loads(fig.to_json())
    Plotly.newPlot("plotDiv", to_js(fig_dict["data"]), to_js(fig_dict["layout"]))
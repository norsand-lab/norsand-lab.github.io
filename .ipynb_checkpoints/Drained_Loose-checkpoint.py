#Drained loose graphs
import json
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np

with open("data/Drained_loose_history.json", "r") as file:
    raw_data = json.load(file)

# outputs shortened list with stepnum items
def data_choose(data, stepnum):  
    filtered = []
    for i in np.linspace(0, len(data)-1, stepnum):
        filtered.append(data[int(np.round(i))])
    return filtered

# calculates p and q coordinates for yield curve
def yield_curve(pi, M, n=200):
    p = np.linspace(0, pi * np.e, n)
    q = np.zeros_like(p)
    q[1:] = M * p[1:] * (1 + np.log(pi / p[1:]))
    return p, q

data= {}
for col in raw_data:            # slices data set
    data.update({col: raw_data[col][:]})          
M = 1.45                        # CSL slope
slide_steps = 24                # number of slider steps
initial_step = 0                # step that slider starts on
static_num = 5                  # number of static traces
step_to_indices = {}            # stores and indexes dynamic traces
p_path = []                     # loading path q values
q_path = []                     # loading path q values
p_max = 0                       # maximum x axis p value
q_max = max(data["p_kPa"]) * M  # maximum y axis q value
a, b, c = 1.01, 0.087, 0.38     # parameters for curved CSL in void ratio space
pref = 100                      # reference pressure for CSL                       
pressure = []                   # pore water pressure for isotropic conditions
for step in data["step"]:
    pressure.append(data["p_kPa"][0] + (data["q_kPa"][step - 1]) / 3 - data["p_kPa"][step - 1])
main_x, main_y = 1, 1           # layout positions of graphs (row,col)
a_x, a_y = 1, 2
b_x, b_y = 2, 2
c_x, c_y = 3, 2

# Initialize figure with subplots
fig = make_subplots(
    rows=3, cols=2,
    column_widths=[0.7, 0.3],
    horizontal_spacing=0.08,   # space between columns
    vertical_spacing=0.1,     # space between rows
    specs=[[{"rowspan": 3}, {}],
           [     None,      {}],
           [     None,      {}]],
    subplot_titles=("Drained Loose: Stress Path, Yield Surface, and Image Point", "", "", "")
)

# Static traces - always visible
fig.add_trace(go.Scatter(x=[0, 2.5 * q_max / M], y=[0, 2.5 * q_max], visible=True, name="CSL", mode = "lines", line = dict(dash='dot', color="blue")), row=main_x, col=main_y)
fig.add_trace(go.Scatter(x=data["axial_strain_percent"], y=data["q_kPa"], visible=True, name="Loading Path", showlegend=False, line=dict(color="#282828")), row=a_x, col=a_y)
fig.add_trace(go.Scatter(x=data["axial_strain_percent"], y=data["volumetric_strain_percent"], visible=True, name="Loading Path", showlegend=False, line=dict(color="#282828")), row=b_x, col=b_y)
ec = []
for p in data["p_kPa"]:
    ec.append(a - b * pow((p / pref), c))
fig.add_trace(go.Scatter(x=data["p_kPa"], y=ec, visible=True, name="CSL", showlegend=False, mode = "lines", line = dict(color = "blue", dash='dot')), row=c_x, col=c_y)
fig.add_trace(go.Scatter(x=data["p_kPa"], y=data["void_ratio"], visible=True, name="Loading Path", showlegend=False, line=dict(color="#282828")), row=c_x, col=c_y)

# Slider-controlled traces
for k, step in enumerate(data_choose(data["step"], slide_steps)):
    p_yield, q_yield = yield_curve(data["pi_kPa"][step - 1], M)
    if max(p_yield) > p_max:
        p_max = max(p_yield)
    idx_start = len(fig.data)
    fig.add_trace(go.Scatter(
        visible = False,
        name = "Yield Surface",
        line=dict(color="purple"),
        x = p_yield,
        y = q_yield
    ),
        row=main_x, col=main_y
    )
    
    current_p = data["p_kPa"][step - 1]
    current_q = data["q_kPa"][step - 1]
    for i in range(step):
        p_path = data["p_kPa"][0:i+1]
        q_path = data["q_kPa"][0:i+1]
    fig.add_trace(go.Scatter(
        visible = False,
        name = "Current State",
        mode="markers",
        marker=dict(size=10, color="red"),
        x = [current_p],
        y = [current_q]
    ),
    row=main_x, col=main_y      
    )
    
    fig.add_trace(go.Scatter(
        visible = False,
        name = "Loading Path",
        mode = 'lines',
        line=dict(color="#282828"),
        x = p_path,
        y = q_path
    ),
    row=main_x, col=main_y      
    )
    
    fig.add_trace(go.Scatter(
        visible = False,
        name = "Image Point",
        mode="markers",
        marker=dict(size=10, color="green"),
        x = [data["pi_kPa"][step - 1]],
        y = [data["pi_kPa"][step - 1] * M]
    ),
    row=main_x, col=main_y             
    )
    
    fig.add_trace(go.Scatter(
        visible = False,
        name = "Current State",
        mode="markers",
        marker=dict(
            size=12,
            symbol="circle-open",
            color="red",       
            line=dict(width=2)
        ),
        showlegend=False,
        x = [data["axial_strain_percent"][step - 1]],
        y = [data["q_kPa"][step - 1]]
    ),
    row=a_x, col=a_y            
    )
    
    fig.add_trace(go.Scatter(
        visible = False,
        name = "Current State",
        mode="markers",
        marker=dict(
            size=12,
            symbol="circle-open",
            color="red",       
            line=dict(width=2)
        ),
        showlegend=False,
        x = [data["axial_strain_percent"][step - 1]],
        y = [data["volumetric_strain_percent"][step - 1]]
    ),
    row=b_x, col=b_y             
    )
    
    fig.add_trace(go.Scatter(
        visible = False,
        name = "Current State",
        mode="markers",
        marker=dict(
            size=12,
            symbol="circle-open",
            color="red",       
            line=dict(width=2)
        ),
        showlegend=False,
        x = [data["p_kPa"][step - 1]],
        y = [data["void_ratio"][step - 1]]
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
    steps.append(dict(method="update", args=[{"visible": visible}], label=f"{fig.data[step_to_indices[k][5]].x[0]:.1f}%"))
    
sliders = [dict(
    active=initial_step, 
    steps=steps, 
    currentvalue={"prefix": "Axial Strain: "}
    )]
fig.update_layout(
    uirevision="constant",
    sliders=sliders
)

#formatting and calculating axis ranges
q_range = [min(data["q_kPa"]), max(data["q_kPa"])]
q_buffer = 0.15 * (max(data["q_kPa"]) - min(data["q_kPa"]))
ec_range = [min(min(ec), min(data["void_ratio"])), max(max(ec), max(data["void_ratio"]))]
ec_buffer = 0.15 * (max(max(ec), max(data["void_ratio"])) - min(min(ec), min(data["void_ratio"])))
v_range = [min(data["volumetric_strain_percent"]), max(data["volumetric_strain_percent"])]
v_buffer = 0.15 * (max(data["volumetric_strain_percent"]) - min(data["volumetric_strain_percent"]))
a_range = [min(data["axial_strain_percent"]), max(data["axial_strain_percent"])]
lnp_range = np.log10([min(data["p_kPa"]), max(data["p_kPa"])])
#a_buffer = 0.15 * (max(data["axial_strain_percent"]) - min(data["axial_strain_percent"]))

fig.update_yaxes(title_text="Deviatoric Stress, q (kPa)", range=[0,  2.5 * q_max], row=main_x, col=main_y)
fig.update_xaxes(title_text="Mean Effective Stress, p' (kPa)", range=[0,  1.15 * p_max], row=main_x, col=main_y)

fig.update_yaxes(title_text="q (kPa)", range=[q_range[0], q_range[1] + q_buffer], row=a_x, col=a_y)  
fig.update_yaxes(title_text="Volumetric Strain, ε<sub>V</sub> (%)", range=[v_range[0], v_range[1] + v_buffer], row=b_x, col=b_y) 
fig.update_yaxes(title_text="Void Ratio, e<sub>c</sub>", range=[ec_range[0] - ec_buffer, ec_range[1] + ec_buffer], row=c_x, col=c_y)  

fig.update_xaxes(title_text="Axial Strain, ε<sub>1</sub> (%)", title_standoff=0, range=a_range, row=a_x, col=a_y)
fig.update_xaxes(title_text="Axial Strain, ε<sub>1</sub> (%)", title_standoff=0, range=a_range, row=b_x, col=b_y)
fig.update_xaxes(title_text="ln p' (kPa)", type="log", range=lnp_range, title_standoff=0, row=c_x, col=c_y)

fig.update_layout(
    width=900,
    height=600,
    template = "plotly_white",
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
fig.show(renderer="browser")
fig.write_html("Drained_Loose.html")
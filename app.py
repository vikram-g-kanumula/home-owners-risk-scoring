# ==============================================================================
# app.py — ResiScore™  |  Homeowners Intelligence Layer
# BD Demo — Carrier Senior Stakeholder Edition
# Tabs: Business Case | Intelligence Signals | Policy Underwriter Lens | Framework
# ==============================================================================

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, callback, Input, Output, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Palette ───────────────────────────────────────────────────────────────────
NAVY   = "#1B2A4A"
GOLD   = "#E8A838"
GREEN  = "#27AE60"
RED    = "#E74C3C"
BLUE   = "#2980B9"
BG     = "#F0F2F5"
WHITE  = "#FFFFFF"
MUTED  = "#6C757D"
BORDER = "#E0E4ED"
AMBER  = "#D35400"

CARD_STYLE = {"borderRadius": "12px", "border": f"1px solid {BORDER}",
              "boxShadow": "0 2px 8px rgba(0,0,0,0.06)", "backgroundColor": WHITE}
SEC_TITLE  = {"color": NAVY, "fontWeight": "700", "fontSize": "1.05rem", "marginBottom": "2px"}
MONO       = {"fontFamily": "'Courier New', monospace", "fontSize": "0.93rem", "color": NAVY,
              "backgroundColor": "#F0F4FA", "padding": "10px 16px", "borderRadius": "6px",
              "border": f"1px solid {BORDER}", "letterSpacing": "0.02em", "lineHeight": "1.8"}

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "ResiScore\u2122 | Homeowners Intelligence Layer"
server = app.server

# ── Data & Models ─────────────────────────────────────────────────────────────
df = pd.read_csv("data/final_predictions.csv")

ALL_FEATURES = [
    "Year_Built","Square_Footage","CLUE_Loss_Count","Credit_Score",
    "Construction_Type","Protection_Class","AOI","Deductible",
    "Territory","Roof_Age_Applicant","Fire_Alarm","Burglar_Alarm",
    "Roof_Vulnerability_Satellite","Wildfire_Exposure_Daily",
    "Water_Loss_Recency_Months","RCV_Appraised","Fire_Hydrant_Distance",
    "Tree_Canopy_Density","Crime_Severity_Index","Pluvial_Flood_Depth",
    "Building_Code_Compliance","Slope_Steepness","Attic_Ventilation",
    "Hail_Frequency","Soil_Liquefaction_Risk",
]
LEGACY_FEATURES = [
    "Year_Built","Square_Footage","CLUE_Loss_Count","Credit_Score",
    "Construction_Type","Protection_Class","AOI","Deductible",
    "Territory","Roof_Age_Applicant","Fire_Alarm","Burglar_Alarm",
    "Urban_HighPC","OldRoof_HighHail",
]
MODERN_FEATURES = [f for f in ALL_FEATURES if f not in LEGACY_FEATURES]
CAT_COLS = ["Construction_Type","Territory","Deductible","Fire_Alarm",
            "Burglar_Alarm","Attic_Ventilation","Soil_Liquefaction_Risk"]

for col in CAT_COLS:
    df[col] = df[col].astype(str)

ebm_model  = joblib.load("models/ebm_residual_model.pkl")
freq_model = joblib.load("models/legacy_freq_model.pkl")
sev_model  = joblib.load("models/legacy_sev_model.pkl")

# ── Portfolio Metrics ─────────────────────────────────────────────────────────
glm_r2   = r2_score(df["Expected_Pure_Premium"], df["GLM_Pure_Premium"])
final_r2 = r2_score(df["Expected_Pure_Premium"], df["Final_Pure_Premium"])
delta_r2 = final_r2 - glm_r2

df["Adjustment"]     = df["Final_Pure_Premium"] - df["GLM_Pure_Premium"]
df["Adjustment_Pct"] = (df["Adjustment"] / df["GLM_Pure_Premium"]) * 100
df["Risk_Tier"]      = pd.cut(df["Final_Pure_Premium"],
    bins=[0,800,1200,1800,np.inf], labels=["Low","Moderate","Elevated","High"])
df["GLM_Underpriced_Flag"] = (
    (df["GLM_Pure_Premium"] < df["Expected_Pure_Premium"] * 0.80)).astype(int)

n_repriced      = (df["Adjustment_Pct"].abs() > 10).sum()
pct_repriced    = n_repriced / len(df) * 100
pct_underpriced = df["GLM_Underpriced_Flag"].mean() * 100
mean_leakage    = (df.loc[df["GLM_Underpriced_Flag"]==1,"Expected_Pure_Premium"] -
                   df.loc[df["GLM_Underpriced_Flag"]==1,"GLM_Pure_Premium"]).mean()
MEAN_GLM_PP     = df["GLM_Pure_Premium"].mean()

# ── EBM Global Importance ─────────────────────────────────────────────────────
global_exp    = ebm_model.explain_global()
global_data   = global_exp.data()
global_names  = list(global_data["names"])
global_scores = list(global_data["scores"])
dollar_importance = {n: abs(s) * MEAN_GLM_PP for n, s in zip(global_names, global_scores)}

# ── PDP pre-computation ───────────────────────────────────────────────────────
PDP_FEATURES = {
    "Wildfire_Exposure_Daily":      ("Wildfire Exposure Index",   "Accelerating — risk grows non-linearly"),
    "Roof_Vulnerability_Satellite": ("Roof Vulnerability Score",  "Convex — penalty accelerates past score 20"),
    "Building_Code_Compliance":     ("Building Code Compliance %","Threshold — sharp jump below 60% compliance"),
    "Credit_Score":                 ("Credit Score",              "Diminishing returns — GLM over-linearises"),
}
PDP_CACHE = {}

def compute_pdp(feature_name, n_points=40):
    row_median = {}
    for col in ALL_FEATURES:
        row_median[col] = str(df[col].mode()[0]) if col in CAT_COLS else float(df[col].median())
    grid = np.linspace(df[feature_name].quantile(0.05), df[feature_name].quantile(0.95), n_points)
    X_grid = pd.DataFrame([row_median] * n_points)
    X_grid[feature_name] = grid
    for col in CAT_COLS:
        X_grid[col] = X_grid[col].astype(str)
    preds   = ebm_model.predict(X_grid[ALL_FEATURES])
    pct_adj = (np.exp(preds - np.median(preds)) - 1) * 100
    return grid, pct_adj

print("Pre-computing PDPs...")
for feat in PDP_FEATURES:
    PDP_CACHE[feat] = compute_pdp(feat)
print("Ready.")

TIER_COLORS = {"Low": GREEN, "Moderate": GOLD, "Elevated": AMBER, "High": RED}
TAB_STYLE = {"fontFamily":"Inter","fontSize":"0.88rem"}
TAB_SEL   = {**TAB_STYLE,"fontWeight":"700","borderTop":f"3px solid {NAVY}"}


# ── Component helpers ─────────────────────────────────────────────────────────
def info_tooltip(tooltip_id, text):
    return html.Span([
        html.I(className="fas fa-info-circle ms-2", id=tooltip_id,
               style={"color": MUTED, "cursor": "pointer", "fontSize": "0.82rem"}),
        dbc.Tooltip(text, target=tooltip_id, placement="right",
                    style={"fontSize": "0.76rem", "maxWidth": "300px", "textAlign": "left"}),
    ])

def chart_card(title, tt_id, tt_text, graph_elem, subtitle=None):
    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.Span(title, style=SEC_TITLE),
                info_tooltip(tt_id, tt_text),
            ], className="d-flex align-items-center"),
            html.Div(subtitle, style={"fontSize":"0.76rem","color":MUTED,"marginTop":"2px"}) if subtitle else None,
        ], style={"backgroundColor":WHITE,"border":"none","paddingBottom":"0"}),
        dbc.CardBody(graph_elem),
    ], style=CARD_STYLE)

def kpi_card(icon, label, value, subtitle, color, badge_text=None):
    return dbc.Card([dbc.CardBody([
        html.Div([
            html.Div(html.I(className=icon, style={"fontSize":"1.3rem","color":color}),
                     style={"backgroundColor":f"{color}1A","borderRadius":"8px",
                            "padding":"9px 10px","display":"inline-flex"}),
            dbc.Badge(badge_text, color="warning", className="ms-2 align-self-start",
                      style={"fontSize":"0.68rem"}) if badge_text else None,
        ], className="d-flex align-items-center mb-3"),
        html.Div(value,    style={"fontSize":"1.85rem","fontWeight":"700","color":NAVY,"lineHeight":"1"}),
        html.Div(label,    style={"fontSize":"0.76rem","fontWeight":"600","color":MUTED,
                                  "marginTop":"4px","textTransform":"uppercase","letterSpacing":"0.05em"}),
        html.Div(subtitle, style={"fontSize":"0.74rem","color":MUTED,"marginTop":"5px"}),
    ])], style=CARD_STYLE)

def formula_block(formula, note=None):
    return html.Div([
        html.Div(formula, style=MONO),
        html.Div(note, style={"fontSize":"0.75rem","color":MUTED,"marginTop":"4px"}) if note else None,
    ], className="my-2")

def section_card(number, title, color, content):
    return dbc.Card([dbc.CardBody([
        html.Div([
            html.Span(str(number), style={
                "backgroundColor":color,"color":WHITE,"borderRadius":"50%",
                "width":"26px","height":"26px","display":"inline-flex",
                "alignItems":"center","justifyContent":"center",
                "fontSize":"0.8rem","fontWeight":"700","marginRight":"10px","flexShrink":"0"}),
            html.Span(title, style={"fontWeight":"700","fontSize":"1.0rem","color":NAVY}),
        ], className="d-flex align-items-center mb-3"),
        content,
    ])], style={**CARD_STYLE,"borderLeft":f"4px solid {color}"})


# ── Navbar ────────────────────────────────────────────────────────────────────
navbar = dbc.Navbar(dbc.Container([
    html.Div([
        html.I(className="fas fa-shield-alt me-2", style={"color":GOLD,"fontSize":"1.35rem"}),
        html.Span("ResiScore", style={"fontWeight":"700","fontSize":"1.25rem","color":WHITE}),
        html.Sup("\u2122", style={"fontSize":"0.7rem","color":GOLD}),
        html.Span(" | Homeowners Intelligence Layer",
                  style={"color":"#A0AABB","fontSize":"0.88rem","marginLeft":"8px"}),
    ], className="d-flex align-items-center"),
    html.Div([
        dbc.Badge("DEMO", color="warning", className="me-2", style={"fontSize":"0.68rem"}),
        html.Span("50,000 synthetic policies \u00b7 GLM + EBM GA2M residual layer",
                  style={"color":"#A0AABB","fontSize":"0.75rem"}),
    ], className="d-none d-md-flex align-items-center"),
], fluid=True), color=NAVY, dark=True, className="py-2",
style={"borderBottom":f"3px solid {GOLD}"})


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1  —  BUSINESS CASE
# ════════════════════════════════════════════════════════════════════════════════
def build_portfolio_tab():
    fig_r2 = go.Figure(go.Bar(
        x=["Legacy GLM (14 vars)","GLM + GA2M (25 vars)"],
        y=[glm_r2, final_r2], marker_color=[MUTED, NAVY],
        text=[f"R\u00b2={glm_r2:.4f}", f"R\u00b2={final_r2:.4f}"],
        textposition="outside", width=0.42))
    fig_r2.add_annotation(x=1, y=final_r2, text=f" +{delta_r2:.4f} lift",
        showarrow=False, yshift=36, font=dict(size=11, color=GREEN, family="Inter"))
    fig_r2.update_yaxes(range=[0,1.08], title_text="R\u00b2 Score")
    fig_r2.update_layout(template="plotly_white", height=290, showlegend=False,
        margin=dict(l=10,r=10,t=50,b=20), font=dict(family="Inter"),
        title={"text":"Explained Variance: Before vs After","font":{"size":13,"color":NAVY}})

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(x=df["GLM_Pure_Premium"].clip(0,4000), nbinsx=60,
        name="Legacy GLM", marker_color=MUTED, opacity=0.55))
    fig_dist.add_trace(go.Histogram(x=df["Final_Pure_Premium"].clip(0,4000), nbinsx=60,
        name="GLM + GA2M", marker_color=BLUE, opacity=0.55))
    fig_dist.update_layout(barmode="overlay", template="plotly_white", height=290,
        xaxis_title="Pure Premium ($)", yaxis_title="Policy Count",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10,r=10,t=50,b=20), font=dict(family="Inter"),
        title={"text":"Book-Wide Premium Shift After Intelligence Layer","font":{"size":13,"color":NAVY}})

    samp  = df.sample(2500, random_state=42)
    under = samp[samp["GLM_Underpriced_Flag"]==1]
    over  = samp[samp["GLM_Underpriced_Flag"]==0]
    fig_adv = go.Figure()
    fig_adv.add_trace(go.Scatter(x=over["GLM_Pure_Premium"], y=over["Expected_Pure_Premium"],
        mode="markers", name="Adequately priced",
        marker=dict(color=BLUE, size=3, opacity=0.30),
        hovertemplate="GLM: $%{x:.0f}<br>True: $%{y:.0f}<extra></extra>"))
    fig_adv.add_trace(go.Scatter(x=under["GLM_Pure_Premium"], y=under["Expected_Pure_Premium"],
        mode="markers", name=f"GLM underpriced (>20%)",
        marker=dict(color=RED, size=4, opacity=0.65),
        customdata=under["Expected_Pure_Premium"]-under["GLM_Pure_Premium"],
        hovertemplate="GLM: $%{x:.0f}<br>True: $%{y:.0f}<br>Leakage: $%{customdata:.0f}<extra></extra>"))
    fig_adv.add_trace(go.Scatter(x=[0,4500], y=[0,4500], mode="lines",
        line=dict(dash="dash", color="#CCCCCC", width=1), showlegend=False))
    fig_adv.update_layout(template="plotly_white", height=310,
        xaxis_title="GLM Estimated Premium ($)", yaxis_title="True Expected Loss Cost ($)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10,r=10,t=20,b=20), font=dict(family="Inter"))

    tier_cts = df["Risk_Tier"].value_counts().reindex(["Low","Moderate","Elevated","High"])
    fig_donut = go.Figure(go.Pie(labels=tier_cts.index, values=tier_cts.values, hole=0.62,
        marker_colors=[GREEN, GOLD, AMBER, RED], textinfo="label+percent", textfont_size=11))
    fig_donut.update_layout(showlegend=False, height=310, font=dict(family="Inter"),
        margin=dict(l=10,r=10,t=50,b=10),
        title={"text":"Risk Tier Distribution — Intelligence-Adjusted","font":{"size":13,"color":NAVY}})

    return html.Div([
        dbc.Alert([
            html.I(className="fas fa-lightbulb me-2", style={"color":GOLD}),
            html.Strong("The Business Case: "),
            f"The legacy 14-variable GLM explains {glm_r2:.1%} of pure premium variance. "
            f"ResiScore\u2122 raises this to {final_r2:.1%} — a +{delta_r2:.4f} \u0394R\u00b2 "
            f"that re-prices {pct_repriced:.1f}% of the book by >10% and corrects systematic "
            f"adverse selection on {pct_underpriced:.1f}% of policies.",
        ], color="warning", className="mb-4",
           style={"borderLeft":f"4px solid {GOLD}","backgroundColor":"#FFFBF0",
                  "borderRadius":"8px","fontSize":"0.88rem"}),
        dbc.Row([
            dbc.Col(kpi_card("fas fa-chart-line","Legacy GLM R\u00b2",f"{glm_r2:.4f}",
                "14 vars \u00b7 linear relativity structure only", MUTED), width=3),
            dbc.Col(kpi_card("fas fa-brain","Intelligence R\u00b2",f"{final_r2:.4f}",
                "25 vars \u00b7 non-linear + pairwise interactions", BLUE), width=3),
            dbc.Col(kpi_card("fas fa-arrow-trend-up","Variance Lift \u0394R\u00b2",
                f"+{delta_r2:.4f}",
                "Residual signal from 11 modern data sources", GREEN, "KEY LIFT"), width=3),
            dbc.Col(kpi_card("fas fa-exclamation-triangle","Adverse Selection Exposure",
                f"{pct_underpriced:.1f}%",
                f"Policies GLM underprices >20% \u00b7 avg leakage ${mean_leakage:,.0f}/policy",
                RED, "RISK"), width=3),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(chart_card("Explained Variance — Before vs After","tt-r2",
                "R\u00b2 measures how precisely the model predicts each policy\u2019s true expected "
                "loss cost. Higher R\u00b2 = less residual mis-pricing, reducing adverse selection "
                "and rate redundancy.",
                dcc.Graph(figure=fig_r2, config={"displayModeBar":False}),
                subtitle="R\u00b2 vs true expected pure premium \u00b7 50,000 policies"), width=4),
            dbc.Col(chart_card("Book-Wide Premium Distribution Shift","tt-dist",
                "Overlaying both distributions reveals how the intelligence layer re-distributes "
                "premiums toward higher-risk tails — evidence of improved risk segmentation, "
                "not a broad rate increase.",
                dcc.Graph(figure=fig_dist, config={"displayModeBar":False}),
                subtitle="50,000 policies \u00b7 clipped at $4,000 for display"), width=8),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(chart_card("Adverse Selection Exposure — GLM Underpricing Map","tt-adverse",
                "Red points are policies where the GLM premium is >20% below true expected loss "
                "cost — the core adverse selection risk. ResiScore\u2122 identifies and corrects "
                "these systematically through the GA2M layer.",
                dcc.Graph(figure=fig_adv, config={"displayModeBar":False}),
                subtitle=f"{pct_underpriced:.1f}% of book underpriced >20% \u00b7 "
                         f"avg leakage ${mean_leakage:,.0f}/policy"), width=8),
            dbc.Col(chart_card("Intelligence-Adjusted Risk Tier Mix","tt-donut",
                "Risk tier composition after applying the GA2M uplift. Shifts relative to the "
                "GLM baseline indicate where the intelligence layer is re-classifying risk — "
                "typically moving underpriced mid-tier policies into more precise segments.",
                dcc.Graph(figure=fig_donut, config={"displayModeBar":False})), width=4),
        ], className="g-3"),
    ], className="py-4")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2  —  INTELLIGENCE SIGNALS
# ════════════════════════════════════════════════════════════════════════════════
def build_feature_tab():
    top_n = 15
    sorted_pairs = sorted(dollar_importance.items(), key=lambda x: x[1], reverse=True)[:top_n][::-1]
    bar_colors = [
        GOLD if n in MODERN_FEATURES or any(m in n for m in MODERN_FEATURES) else "#5B6F8A"
        for n, _ in sorted_pairs]
    fig_imp = go.Figure(go.Bar(
        x=[v for _,v in sorted_pairs],
        y=[n.replace("_"," ") for n,_ in sorted_pairs],
        orientation="h", marker_color=bar_colors,
        text=[f"~${v:,.0f}/policy" for _,v in sorted_pairs],
        textposition="outside", textfont=dict(size=9, color=NAVY)))
    fig_imp.add_annotation(
        text="\U0001f7e1 New intelligence signal    \u25a0 Legacy feature (non-linear gain)",
        xref="paper", yref="paper", x=0.5, y=-0.08, showarrow=False,
        font=dict(size=9, color=MUTED))
    fig_imp.update_xaxes(title_text="Estimated Avg Dollar Impact / Policy ($)")
    fig_imp.update_layout(template="plotly_white", height=480,
        margin=dict(l=10,r=80,t=20,b=40), font=dict(family="Inter"))

    feats  = list(PDP_FEATURES.keys())
    labels = [PDP_FEATURES[f][0] for f in feats]
    annots = [PDP_FEATURES[f][1] for f in feats]
    fig_pdp = make_subplots(rows=2, cols=2, subplot_titles=labels,
                            vertical_spacing=0.18, horizontal_spacing=0.12)
    for i, feat in enumerate(feats):
        grid, pct_adj = PDP_CACHE[feat]
        r, c = [(1,1),(1,2),(2,1),(2,2)][i]
        fig_pdp.add_trace(go.Scatter(
            x=list(grid)+list(grid[::-1]),
            y=list(np.maximum(pct_adj,0))+[0]*len(grid),
            fill="toself", fillcolor=f"{RED}18", line=dict(width=0),
            showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig_pdp.add_trace(go.Scatter(
            x=list(grid)+list(grid[::-1]),
            y=[0]*len(grid)+list(np.minimum(pct_adj,0)[::-1]),
            fill="toself", fillcolor=f"{GREEN}18", line=dict(width=0),
            showlegend=False, hoverinfo="skip"), row=r, col=c)
        fig_pdp.add_trace(go.Scatter(
            x=grid, y=pct_adj, mode="lines", line=dict(color=BLUE, width=2.5),
            hovertemplate=f"{labels[i]}: %{{x:.1f}}<br>Premium adj: %{{y:.1f}}%<extra></extra>",
            showlegend=False), row=r, col=c)
        fig_pdp.add_hline(y=0, line_dash="dot", line_color="#CCCCCC", line_width=1, row=r, col=c)
        xref = f"x{i+1 if i>0 else ''}"
        yref = f"y{i+1 if i>0 else ''}"
        fig_pdp.add_annotation(
            text=f"<i>{annots[i]}</i>",
            xref=xref, yref=yref, x=0.98, y=0.97,
            xanchor="right", yanchor="top", showarrow=False,
            font=dict(size=8, color=AMBER), bgcolor=WHITE,
            bordercolor=BORDER, borderwidth=1)
    fig_pdp.update_yaxes(title_text="% Adj vs Median", title_font_size=9, tickfont_size=9)
    fig_pdp.update_xaxes(tickfont_size=9)
    fig_pdp.update_annotations(font_size=10)
    fig_pdp.update_layout(template="plotly_white", height=480,
        margin=dict(l=50,r=20,t=40,b=30), font=dict(family="Inter"))

    tmp = df.copy()
    tmp["wf_bin"] = pd.cut(tmp["Wildfire_Exposure_Daily"],      bins=10)
    tmp["rv_bin"] = pd.cut(tmp["Roof_Vulnerability_Satellite"],  bins=10)
    pivot = tmp.groupby(["rv_bin","wf_bin"], observed=True)["EBM_Log_Uplift"].mean().unstack()
    def bin_label(c):
        lo=float(str(c).split(",")[0].strip("(")); hi=float(str(c).split(",")[1].strip("]"))
        return f"{lo:.0f}\u2013{hi:.0f}"
    pct_matrix = (np.exp(pivot.values)-1)*100
    fig_heat = go.Figure(go.Heatmap(
        z=pct_matrix,
        x=[bin_label(c) for c in pivot.columns],
        y=[bin_label(r) for r in pivot.index],
        colorscale="RdYlGn_r",
        colorbar=dict(title="Avg GA2M<br>Surcharge (%)", thickness=14, len=0.85,
                      tickformat=".0f", ticksuffix="%"),
        hovertemplate="Wildfire: %{x}<br>Roof Vuln: %{y}<br>Surcharge: %{z:.1f}%<extra></extra>",
        zmin=0))
    fig_heat.add_annotation(
        text="Peak compounding<br>risk zone", showarrow=True,
        x=bin_label(list(pivot.columns)[-1]), y=bin_label(list(pivot.index)[-1]),
        arrowhead=2, arrowcolor=NAVY, arrowwidth=1.5, ax=-60, ay=30,
        font=dict(size=9, color=NAVY, family="Inter"),
        bgcolor=WHITE, bordercolor=BORDER, borderwidth=1)
    fig_heat.update_xaxes(title_text="Wildfire Exposure Index (binned)")
    fig_heat.update_yaxes(title_text="Roof Vulnerability Score (binned)")
    fig_heat.update_layout(template="plotly_white", height=400,
        margin=dict(l=10,r=10,t=20,b=40), font=dict(family="Inter"))

    return html.Div([
        dbc.Alert([
            html.I(className="fas fa-microscope me-2", style={"color":BLUE}),
            html.Strong("Intelligence Signal Architecture: "),
            "ResiScore\u2122 captures three signal types the GLM cannot price: "
            "(1) non-linear individual feature effects, (2) compounding pairwise interactions, "
            "and (3) temporal risk decay signals. Each is fully interpretable via the GA2M glass-box framework.",
        ], color="info", className="mb-4",
           style={"borderLeft":f"4px solid {BLUE}","backgroundColor":"#EBF5FB",
                  "borderRadius":"8px","fontSize":"0.87rem"}),
        dbc.Row([
            dbc.Col(chart_card("Signal Landscape — Estimated Dollar Impact per Policy","tt-imp",
                "Ranks features by average absolute contribution to the GA2M residual, expressed "
                "as estimated dollar impact per policy. Gold = new modern signal absent from the "
                "legacy GLM. Blue-grey = legacy feature gaining non-linear treatment.",
                dcc.Graph(figure=fig_imp, config={"displayModeBar":False}),
                subtitle="Top 15 \u00b7 Gold=new intelligence signal \u00b7 Blue-grey=legacy with non-linear gain"),
            width=5),
            dbc.Col(chart_card("Non-Linear Risk Profiles — What Intelligence Discovers","tt-pdp",
                "Each chart shows the learned 1-D risk curve (partial dependence at median "
                "covariate values). Curvature, acceleration, or threshold effects represent "
                "value the GLM\u2019s linear structure cannot capture regardless of variable selection.",
                dcc.Graph(figure=fig_pdp, config={"displayModeBar":False}),
                subtitle="Y-axis: % premium adjustment vs median risk \u00b7 Dashed line = GLM linear assumption"),
            width=7),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(chart_card(
                "Compounding Risk — Wildfire \u00d7 Roof Vulnerability Interaction","tt-heat",
                "The surcharge shown here arises purely from the interaction of two features — "
                "beyond the sum of their individual effects. A GLM prices them independently and "
                "systematically undercharges the top-right cluster by a compounding margin.",
                dcc.Graph(figure=fig_heat, config={"displayModeBar":False}),
                subtitle="Color = avg GA2M interaction surcharge (%) \u00b7 Computed at policy level across 50K risks"),
            width=12),
        ], className="g-3 pb-4"),
    ], className="py-4")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3  —  POLICY UNDERWRITER LENS
# ════════════════════════════════════════════════════════════════════════════════
def build_policy_options():
    up   = df.nlargest(80,"Adjustment_Pct")
    down = df.nsmallest(80,"Adjustment_Pct")
    rand = df.sample(80, random_state=42)
    pool = pd.concat([up,down,rand]).drop_duplicates().head(200)
    opts = []
    for idx, row in pool.iterrows():
        tier=str(row["Risk_Tier"]); adj=row["Adjustment_Pct"]; arrow="\u2191" if adj>0 else "\u2193"
        opts.append({"label":(f"Policy #{idx} | {tier} Risk | {arrow}{abs(adj):.0f}% adj | "
            f"GLM ${row['GLM_Pure_Premium']:,.0f} \u2192 Final ${row['Final_Pure_Premium']:,.0f}"),
            "value":idx})
    return opts, pool.index[0]

POLICY_OPTIONS, DEFAULT_POLICY = build_policy_options()

def build_policy_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([dbc.Card([
                dbc.CardHeader(html.Div("Select Policy", style=SEC_TITLE),
                               style={"backgroundColor":WHITE,"border":"none"}),
                dbc.CardBody([
                    html.Div("Curated: 80 highest surcharges \u00b7 80 highest credits \u00b7 80 random",
                             style={"fontSize":"0.75rem","color":MUTED,"marginBottom":"8px"}),
                    dcc.Dropdown(id="policy-dd", options=POLICY_OPTIONS,
                                 value=DEFAULT_POLICY, clearable=False,
                                 style={"fontSize":"0.82rem"}),
                    html.Div(id="policy-profile-panel", className="mt-3"),
                ]),
            ], style=CARD_STYLE)], width=3),
            dbc.Col([dbc.Card([
                dbc.CardHeader([
                    dbc.Row([
                        dbc.Col([
                            html.Div("Pricing Deconstruction", style=SEC_TITLE),
                            html.Div("Full audit trail from legacy actuarial formula to intelligence-adjusted premium",
                                     style={"fontSize":"0.75rem","color":MUTED}),
                        ], width=7),
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("Strategic",         id="btn-hi",  n_clicks=1,
                                           color="primary",   outline=True, size="sm"),
                                dbc.Button("GLM Breakdown",     id="btn-glm", n_clicks=0,
                                           color="secondary", outline=True, size="sm"),
                                dbc.Button("GA2M Intelligence", id="btn-gam", n_clicks=0,
                                           color="info",      outline=True, size="sm"),
                            ])
                        ], width=5, className="text-end d-flex align-items-center justify-content-end"),
                    ], align="center"),
                ], style={"backgroundColor":WHITE,"border":"none"}),
                dbc.CardBody([
                    dcc.Store(id="view-store", data="high_level"),
                    html.Div([
                        html.Span([html.Span("\u25cf", style={"color":BLUE,"fontWeight":"700","marginRight":"4px"}),
                                   "Individual feature effect"],
                                  style={"fontSize":"0.75rem","color":MUTED,"marginRight":"20px"}),
                        html.Span([html.Span("\u2297", style={"color":GOLD,"fontWeight":"700","marginRight":"4px"}),
                                   "Pairwise interaction effect"],
                                  style={"fontSize":"0.75rem","color":MUTED,"marginRight":"20px"}),
                        html.Span("(\u2297 appears in GA2M view only)",
                                  style={"fontSize":"0.7rem","color":BORDER}),
                    ], className="text-end mb-1 pe-1"),
                    dcc.Graph(id="waterfall-plot", config={"displayModeBar":False}),
                ]),
            ], style=CARD_STYLE)], width=9),
        ], className="g-3 py-4"),
    ])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4  —  FRAMEWORK
# ════════════════════════════════════════════════════════════════════════════════
def build_framework_tab():

    # Architecture flow diagram
    fig_arch = go.Figure()
    fig_arch.update_layout(template="plotly_white", height=210,
        margin=dict(l=20,r=20,t=20,b=20), font=dict(family="Inter"),
        xaxis=dict(visible=False, range=[0,10]),
        yaxis=dict(visible=False, range=[0,3]))

    boxes = [
        (0.4, 1.5, 1.5, 0.80, "Property\nFeatures\n(25 vars)",   "#8A9BB0", WHITE),
        (2.2, 1.5, 1.7, 0.80, "Legacy GLM\nFreq \u00d7 Sev\n(14 vars)", "#4A5568", WHITE),
        (4.2, 1.5, 1.7, 0.80, "GLM Pure\nPremium\n(baseline)",   "#2C3E50", WHITE),
        (6.2, 1.5, 1.7, 0.80, "GA2M\nResidual\n(11 modern)",     BLUE,     WHITE),
        (8.1, 1.5, 1.6, 0.80, "Final\nIntelligence\nPremium",    NAVY,     WHITE),
    ]
    for x, y, w, h, label, fc, tc in boxes:
        fig_arch.add_shape(type="rect", x0=x, y0=y-h/2, x1=x+w, y1=y+h/2,
            fillcolor=fc, line_color="white", line_width=2, layer="below")
        fig_arch.add_annotation(x=x+w/2, y=y, text=label.replace("\n","<br>"),
            showarrow=False, font=dict(size=9, color=tc, family="Inter"), align="center")

    for ax, lbl in [(2.0,""), (4.1,""), (5.85,"log(True\u2215GLM)<br>residual target"),
                    (7.75,"\u00d7 exp(GA2M)<br>corridor [0.65\u00d7, 1.60\u00d7]")]:
        fig_arch.add_annotation(x=ax+0.25, y=1.5, ax=ax, ay=1.5,
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowcolor=GOLD, arrowwidth=2,
            text=lbl, font=dict(size=7.5, color=MUTED), yshift=16, xanchor="center")

    fig_arch.add_annotation(x=5.0, y=0.3,
        text="<b>Separation of concerns:</b> GLM handles linear exposure relativities \u00b7 "
             "GA2M captures non-linear effects + pairwise interactions",
        showarrow=False, font=dict(size=9, color=MUTED), xanchor="center")

    def lim_pill(icon, title, body, color):
        return html.Div([
            html.I(className=f"{icon} me-2", style={"color":color}),
            html.Span(title, style={"fontWeight":"600","fontSize":"0.85rem","color":NAVY}),
            html.Div(body, style={"fontSize":"0.78rem","color":MUTED,"marginTop":"4px","lineHeight":"1.5"}),
        ], style={"backgroundColor":"#F8F9FA","borderRadius":"8px","padding":"12px",
                  "border":f"1px solid {BORDER}","height":"100%"})

    def perf_chip(label, val, color):
        return html.Div([
            html.Div(val,   style={"fontSize":"1.5rem","fontWeight":"700","color":color}),
            html.Div(label, style={"fontSize":"0.7rem","color":MUTED,"textTransform":"uppercase",
                                   "letterSpacing":"0.05em","marginTop":"2px"}),
        ], style={"textAlign":"center","padding":"14px 20px","border":f"1px solid {BORDER}",
                  "borderRadius":"10px","backgroundColor":WHITE,"minWidth":"130px"})

    return html.Div([
        # Architecture diagram
        dbc.Card([
            dbc.CardHeader([
                html.Div("Two-Layer Pricing Architecture", style=SEC_TITLE),
                html.Div("ResiScore\u2122 operates as a constrained, interpretable intelligence "
                         "layer on top of — not replacing — the carrier\u2019s existing GLM infrastructure.",
                         style={"fontSize":"0.78rem","color":MUTED}),
            ], style={"backgroundColor":WHITE,"border":"none"}),
            dbc.CardBody(dcc.Graph(figure=fig_arch, config={"displayModeBar":False})),
        ], style=CARD_STYLE, className="mb-4"),

        dbc.Row([
            # LEFT
            dbc.Col([
                section_card(1, "Legacy GLM — Log-Linear Rating Structure", MUTED, html.Div([
                    html.P(["The industry-standard homeowners rating plan is a ",
                            html.Strong("Poisson × Gamma GLM"),
                            " — frequency and severity modelled on a log link, then multiplied "
                            "to produce a pure premium relativity structure:"],
                           style={"fontSize":"0.85rem","color":MUTED,"marginBottom":"10px"}),
                    formula_block("log E[Freq\u1d62] = \u03b2\u2080 + \u03b2\u2081x\u2081\u1d62 + \u03b2\u2082x\u2082\u1d62 + \u2026 + \u03b2\u2096x\u2096\u1d62",
                                  "Poisson GLM on expected claim frequency"),
                    formula_block("log E[Sev\u1d62]  = \u03b3\u2080 + \u03b3\u2081x\u2081\u1d62 + \u03b3\u2082x\u2082\u1d62 + \u2026 + \u03b3\u2096x\u2096\u1d62",
                                  "Gamma GLM on expected severity given a loss"),
                    formula_block("GLM PP\u1d62 = exp(\u03b2\u2080+\u03b3\u2080) \u00b7 \u220f\u2c7c exp((\u03b2\u2c7c+\u03b3\u2c7c) \u00b7 x\u2c7c\u1d62)",
                                  "Multiplicative rating factors — the standard ISO/Bureau tariff structure"),
                    html.Div([html.Span("\u2714\ufe0f me-1"),
                              " Regulatory-accepted, auditable, stable under low-data perils"],
                             style={"fontSize":"0.78rem","color":GREEN,"marginTop":"8px"}),
                ])),
                html.Div(className="mb-3"),
                section_card(2, "Where the GLM Reaches Its Structural Ceiling", RED, html.Div([
                    html.P("Three architectural constraints limit the GLM regardless of variable selection:",
                           style={"fontSize":"0.85rem","color":MUTED,"marginBottom":"12px"}),
                    dbc.Row([
                        dbc.Col(lim_pill("fas fa-slash","Linearity Constraint",
                            "The log-link forces every feature effect to be linear in log-premium "
                            "space. Convex, threshold, and U-shaped effects are approximated away.", RED), width=4),
                        dbc.Col(lim_pill("fas fa-ban","Additive Structure",
                            "Rating factors multiply independently. Compounding interaction premiums "
                            "— e.g. high wildfire + old roof — are never captured.", AMBER), width=4),
                        dbc.Col(lim_pill("fas fa-clock-rotate-left","Static Variables",
                            "Temporal risk signals (water-loss recency, real-time wildfire index, "
                            "satellite roof condition) enter only as coarse buckets if at all.", BLUE), width=4),
                    ], className="g-2"),
                    html.Div([
                        "\u26a0\ufe0f Net effect: ",
                        html.Strong(f"{(1-glm_r2):.1%} of pure premium variance"),
                        " is structurally unexplained by the GLM — the addressable residual.",
                    ], style={"fontSize":"0.78rem","color":AMBER,"marginTop":"12px",
                              "backgroundColor":"#FFF8EF","padding":"8px 12px",
                              "borderRadius":"6px","border":f"1px solid {GOLD}"}),
                ])),
            ], width=6),

            # RIGHT
            dbc.Col([
                section_card(3, "GA2M Residual Layer — Mathematical Specification", BLUE, html.Div([
                    html.P(["Trains a ",
                            html.Strong("Generalized Additive Model with pairwise interactions (GA2M / EBM)"),
                            " on the log-scale GLM residual — capturing what the GLM leaves behind:"],
                           style={"fontSize":"0.85rem","color":MUTED,"marginBottom":"10px"}),
                    formula_block(
                        "\u03b5\u1d62 = log(True PP\u1d62) \u2212 log(GLM PP\u1d62)",
                        "Log-multiplicative residual target — positivity of final premium guaranteed by construction"),
                    formula_block(
                        "g(\u03b5\u1d62) = \u03b2\u2080 + \u03a3\u2c7c f\u2c7c(x\u2c7c\u1d62) + \u03a3\u2c7c<\u2097 f\u2c7c\u2097(x\u2c7c\u1d62, x\u2097\u1d62)",
                        "Smooth univariate shape functions f\u2c7c + pairwise interaction surfaces f\u2c7c\u2097 — the GA2M structure"),
                    formula_block(
                        "Final PP\u1d62 = GLM PP\u1d62 \u00d7 exp( clip(\u011d\u1d62, log(0.65), log(1.60)) )",
                        "Bounded corridor: GA2M adjusts GLM by at most \u221235% to +60% per policy"),
                    html.Div([
                        html.Span("Why log-scale? ", style={"fontWeight":"600","color":NAVY}),
                        "Premiums are multiplicative by nature — log-space ensures the final "
                        "premium is always positive and the uplift has a natural economic "
                        "interpretation as a relativity on top of the GLM.",
                    ], style={"fontSize":"0.78rem","color":MUTED,"lineHeight":"1.6",
                              "backgroundColor":"#EBF5FB","borderRadius":"6px",
                              "padding":"10px 12px","marginTop":"10px"}),
                ])),
                html.Div(className="mb-3"),
                section_card(4, "Glass-Box Guarantee — Interpretability Architecture", GREEN, html.Div([
                    html.P("Every GA2M prediction decomposes exactly into auditable per-feature contributions. "
                           "Three explanation layers are available:",
                           style={"fontSize":"0.85rem","color":MUTED,"marginBottom":"12px"}),
                    *[html.Div([
                        html.I(className=f"{ic} me-2", style={"color":col}),
                        html.Span(title, style={"fontWeight":"600","fontSize":"0.83rem","color":NAVY}),
                        html.Div(body, style={"fontSize":"0.77rem","color":MUTED,
                                              "marginTop":"2px","marginLeft":"22px","lineHeight":"1.5"}),
                    ], style={"marginBottom":"10px"}) for ic, title, body, col in [
                        ("fas fa-globe","Global",
                         "Shape functions f\u2c7c(x) and interaction surfaces f\u2c7c\u2097(x,y) plotted "
                         "across the full feature range — see Intelligence Signals tab.", GREEN),
                        ("fas fa-fingerprint","Local",
                         "Each policy\u2019s adjustment decomposes into a waterfall of per-feature "
                         "dollar contributions — see Policy Underwriter Lens tab.", BLUE),
                        ("fas fa-file-contract","Regulatory",
                         "Contributions are proportional to the log-uplift addends, preserving the "
                         "multiplicative relativity language regulators already accept.", GOLD),
                    ]],
                    html.Div(["\u2714\ufe0f ",
                              html.Strong("Exact additivity:"),
                              " EBM enforces exact decomposition — no post-hoc approximation "
                              "is involved, unlike SHAP applied to black-box models."],
                             style={"fontSize":"0.78rem","color":GREEN,"marginTop":"6px",
                                    "backgroundColor":"#EAFAF1","padding":"8px 12px",
                                    "borderRadius":"6px","border":f"1px solid {GREEN}"}),
                ])),
            ], width=6),
        ], className="g-4 mb-4"),

        # Section 5: Validation
        dbc.Card([
            dbc.CardHeader([
                html.Div([
                    html.Span("5", style={"backgroundColor":GOLD,"color":WHITE,"borderRadius":"50%",
                        "width":"26px","height":"26px","display":"inline-flex","alignItems":"center",
                        "justifyContent":"center","fontSize":"0.8rem","fontWeight":"700","marginRight":"10px"}),
                    html.Span("Validation & Performance Characteristics",
                              style={"fontWeight":"700","fontSize":"1.0rem","color":NAVY}),
                    info_tooltip("tt-fw-perf",
                        "Metrics reflect this demo model trained on 50,000 synthetic policies. "
                        "In production, carriers would validate on held-out accident years using "
                        "lift curves, Gini coefficients, and double-lift charts."),
                ], className="d-flex align-items-center"),
            ], style={"backgroundColor":WHITE,"border":"none","borderLeft":f"4px solid {GOLD}"}),
            dbc.CardBody([
                dbc.Row([
                    dbc.Col(html.Div([
                        html.Div([
                            perf_chip("GLM R\u00b2", f"{glm_r2:.4f}", MUTED),
                            perf_chip("GA2M R\u00b2", f"{final_r2:.4f}", NAVY),
                            perf_chip("Lift \u0394R\u00b2", f"+{delta_r2:.4f}", GREEN),
                            perf_chip("Uplift Corridor", "0.65\u00d7\u20131.60\u00d7", BLUE),
                            perf_chip("GLM Unexplained", f"{(1-glm_r2):.1%}", RED),
                            perf_chip("Residual Recovered", f"{delta_r2/(1-glm_r2):.1%}", GREEN),
                        ], className="d-flex flex-wrap gap-3"),
                    ]), width=8),
                    dbc.Col([
                        html.Div("Production validation checklist:",
                                 style={"fontWeight":"600","fontSize":"0.83rem","color":NAVY,"marginBottom":"8px"}),
                        *[html.Div([
                            html.I(className="fas fa-circle-dot me-2",
                                   style={"color":BLUE,"fontSize":"0.6rem"}),
                            html.Span(item, style={"fontSize":"0.78rem","color":MUTED}),
                        ], style={"marginBottom":"5px","display":"flex","alignItems":"center"})
                          for item in [
                            "Out-of-time validation on most recent 2 accident years",
                            "Double-lift chart: GLM vs final vs actual loss ratio by decile",
                            "Gini coefficient improvement on ranked risk segments",
                            "Monotonicity and boundary tests on all shape functions",
                            "Bias audit across protected class proxies",
                            "Rate impact study before regulatory filing",
                        ]],
                    ], width=4),
                ]),
            ]),
        ], style={**CARD_STYLE,"borderLeft":f"4px solid {GOLD}"}),
    ], className="py-4")


# ── Root Layout ───────────────────────────────────────────────────────────────
app.layout = html.Div([
    navbar,
    dbc.Container([
        dcc.Tabs(id="main-tabs", value="tab-portfolio", style={"marginTop":"12px"}, children=[
            dcc.Tab(label="\U0001f4ca  Business Case",           value="tab-portfolio",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="\U0001f9e0  Intelligence Signals",    value="tab-features",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="\U0001f50d  Policy Underwriter Lens", value="tab-policy",
                    style=TAB_STYLE, selected_style=TAB_SEL),
            dcc.Tab(label="\U0001f4d0  Framework",               value="tab-framework",
                    style=TAB_STYLE, selected_style=TAB_SEL),
        ]),
        html.Div(id="tab-content"),
    ], fluid=True, style={"maxWidth":"1600px","padding":"0 24px"}),
], style={"backgroundColor":BG,"minHeight":"100vh","fontFamily":"Inter, sans-serif"})


# ════════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════════════════════════
@callback(Output("tab-content","children"), Input("main-tabs","value"))
def render_tab(tab):
    if tab == "tab-portfolio": return build_portfolio_tab()
    if tab == "tab-features":  return build_feature_tab()
    if tab == "tab-policy":    return build_policy_tab()
    if tab == "tab-framework": return build_framework_tab()

@callback(Output("view-store","data"),
    [Input("btn-hi","n_clicks"),Input("btn-glm","n_clicks"),Input("btn-gam","n_clicks")],
    prevent_initial_call=True)
def set_view(n1,n2,n3):
    t = ctx.triggered_id
    if t=="btn-hi":  return "high_level"
    if t=="btn-glm": return "glm_breakdown"
    if t=="btn-gam": return "gam_breakdown"
    return "high_level"

@callback([Output("btn-hi","outline"),Output("btn-glm","outline"),Output("btn-gam","outline")],
    Input("view-store","data"))
def highlight_btn(view):
    return view!="high_level", view!="glm_breakdown", view!="gam_breakdown"

@callback([Output("policy-profile-panel","children"),Output("waterfall-plot","figure")],
    [Input("policy-dd","value"),Input("view-store","data")])
def update_policy_view(selected_idx, view_type):
    row=df.loc[selected_idx]; X_sample=df[ALL_FEATURES].loc[[selected_idx]]
    glm_prem=row["GLM_Pure_Premium"]; actual_exp_prem=row["Expected_Pure_Premium"]
    ebm_residual=row["EBM_Residual_Pred"]; final_prem=row["Final_Pure_Premium"]
    adj_pct=row["Adjustment_Pct"]; tier=str(row["Risk_Tier"]); tc=TIER_COLORS.get(tier,MUTED)

    def row_item(lbl, val, vstyle=None):
        base={"fontSize":"0.92rem","fontWeight":"600","color":NAVY,"float":"right"}
        if vstyle: base.update(vstyle)
        return html.Div([
            html.Span(lbl,style={"fontSize":"0.72rem","color":MUTED,
                                 "textTransform":"uppercase","letterSpacing":"0.04em"}),
            html.Span(val,style=base),
        ],style={"borderBottom":f"1px solid {BORDER}","padding":"7px 0","overflow":"hidden"})

    profile = html.Div([
        html.Div([
            dbc.Badge(f"{tier} Risk",style={"backgroundColor":tc,"fontSize":"0.75rem"}),
            dbc.Badge(f"{'↑'if adj_pct>0 else '↓'}{abs(adj_pct):.1f}% vs GLM",
                      color="danger" if adj_pct>0 else "success",
                      className="ms-1",style={"fontSize":"0.75rem"}),
        ],className="mb-3"),
        row_item("True Expected",f"${actual_exp_prem:,.0f}",
                 {"color":RED if actual_exp_prem>final_prem else GREEN}),
        row_item("Legacy GLM",f"${glm_prem:,.0f}"),
        row_item("GA2M Adj",f"{'+'if ebm_residual>=0 else ''}${ebm_residual:,.0f}",
                 {"color":RED if ebm_residual>0 else GREEN}),
        html.Div([
            html.Span("Final Premium",style={"fontSize":"0.82rem","fontWeight":"700","color":NAVY}),
            html.Span(f"${final_prem:,.0f}",
                      style={"fontSize":"1.15rem","fontWeight":"700","color":BLUE,"float":"right"}),
        ],style={"padding":"10px 0 0","overflow":"hidden"}),
    ])

    WF   = dict(connector={"line":{"color":BORDER,"width":2}},
                increasing={"marker":{"color":RED}},decreasing={"marker":{"color":GREEN}})
    LAY  = dict(template="plotly_white",font=dict(family="Inter"),height=430,
                margin=dict(l=20,r=20,t=65,b=50),showlegend=False,waterfallgap=0.25)

    if view_type=="high_level":
        fig=go.Figure(go.Waterfall(orientation="v",
            measure=["relative","relative","total"],
            x=["Legacy GLM Estimate","GA2M Intelligence Adjustment","Final Premium"],
            y=[glm_prem,ebm_residual,0], textposition="outside",
            text=[f"${glm_prem:,.0f}",
                  f"{'+'if ebm_residual>=0 else ''}${ebm_residual:,.0f}",
                  f"${final_prem:,.0f}"],
            totals={"marker":{"color":NAVY}},**WF))
        fig.update_layout(title={"text":"Strategic View \u2014 Legacy Formula \u2192 "
            "Intelligence-Adjusted Premium","font":{"size":13,"color":NAVY}},**LAY)

    elif view_type=="glm_breakdown":
        X_glm=X_sample.copy()
        X_glm["Urban_HighPC"]=((X_glm["Territory"]=="Urban")&
            (X_glm["Protection_Class"]>6)).astype(int).astype(str)
        X_glm["OldRoof_HighHail"]=((X_glm["Roof_Age_Applicant"]>20)&
            (X_glm["Hail_Frequency"]>=3)).astype(int).astype(str)
        fc=freq_model.named_steps["regressor"].coef_
        fi=freq_model.named_steps["regressor"].intercept_
        sc=sev_model.named_steps["regressor"].coef_
        si=sev_model.named_steps["regressor"].intercept_
        combined_coefs=fc+sc; combined_int=float(fi+si)
        X_proc=freq_model.named_steps["preprocessor"].transform(X_glm[LEGACY_FEATURES])
        if hasattr(X_proc,"toarray"): X_proc=X_proc.toarray()
        feat_names=freq_model.named_steps["preprocessor"].get_feature_names_out()
        log_impacts=X_proc[0]*combined_coefs
        agg={}
        for i,name in enumerate(feat_names):
            base=name.split("__")[1] if "__" in name else name
            for cat in ["Construction_Type","Territory","Deductible","Fire_Alarm",
                        "Burglar_Alarm","Urban_HighPC","OldRoof_HighHail"]:
                if base.startswith(cat): base=cat; break
            agg[base]=agg.get(base,0)+log_impacts[i]
        for f in LEGACY_FEATURES:
            if f not in agg: agg[f]=0.0
        total_log=sum(agg.values()); base_prem=float(np.exp(combined_int))
        attribs={f:(v/total_log if total_log!=0 else 0)*(glm_prem-base_prem) for f,v in agg.items()}
        scored=sorted(attribs.items(),key=lambda x:abs(x[1]),reverse=True)
        labeled=(["GLM Base Rate"]+["\u25cf "+f[0].replace("_"," ") for f in scored]+["Total GLM"])
        scores_wf=[base_prem]+[f[1] for f in scored]+[0]
        measures_wf=["relative"]+["relative"]*len(scored)+["total"]
        fig=go.Figure(go.Waterfall(orientation="v",measure=measures_wf,x=labeled,y=scores_wf,
            textposition="outside",
            text=[f"${s:,.0f}" if i<len(scores_wf)-1 else f"${glm_prem:,.0f}"
                  for i,s in enumerate(scores_wf)],
            totals={"marker":{"color":MUTED}},**WF))
        fig.update_layout(title={"text":"GLM Breakdown \u2014 How the Legacy Actuarial System "
            "Priced This Risk","font":{"size":13,"color":MUTED}},**LAY)

    else:
        eps=1e-6
        log_resid_true=float(np.log(actual_exp_prem+eps)-np.log(glm_prem+eps))
        local_exp=ebm_model.explain_local(X_sample,y=[log_resid_true])
        exp_data=local_exp.data(0)
        names_raw=exp_data["names"]; scores_raw=exp_data["scores"]
        gam_int_log=float(exp_data["extra"]["scores"][0])
        total_log_pred=float(sum(scores_raw))+gam_int_log
        def l2d(v): return 0.0 if abs(total_log_pred)<1e-9 else (v/total_log_pred)*ebm_residual
        scores_dollar=[l2d(s) for s in scores_raw]; intercept_dollar=l2d(gam_int_log)
        scored_f=sorted(zip(names_raw,scores_dollar),key=lambda x:abs(x[1]),reverse=True)
        top_f=scored_f[:10]; other_f=scored_f[10:]
        other_sum=sum(f[1] for f in other_f)
        top5=sorted(other_f,key=lambda x:abs(x[1]),reverse=True)[:5]
        rem=len(other_f)-5
        odet="".join([f"<br>  {f[0]}: ${f[1]:,.0f}" for f in top5])
        if rem>0: odet+=f"<br>  ...and {rem} more"
        def classify(n):
            if n in ["GA2M Intercept","All Other Signals","Net Residual Adj"]: return "meta"
            return "interaction" if " & " in n else "main"
        raw_names=(["GA2M Intercept"]+[f[0] for f in top_f]+["All Other Signals","Net Residual Adj"])
        gam_scores=([intercept_dollar]+[f[1] for f in top_f]+[other_sum,0])
        gam_meas=(["relative"]+["relative"]*len(top_f)+["relative","total"])
        labeled=[]
        for n in raw_names:
            t=classify(n)
            if t=="main": labeled.append("\u25cf "+n.replace("_"," "))
            elif t=="interaction": labeled.append("\u2297 "+n.replace("_"," "))
            else: labeled.append(n)
        hover=[]
        for i,n in enumerate(raw_names):
            if n=="All Other Signals": hover.append(f"All Other Signals: ${other_sum:,.0f}{odet}")
            elif n=="Net Residual Adj": hover.append(f"Net Residual Adjustment: ${sum(gam_scores[:-1]):,.0f}")
            else: hover.append(f"{n}: ${gam_scores[i]:,.2f}")
        fig=go.Figure(go.Waterfall(orientation="v",measure=gam_meas,x=labeled,y=gam_scores,
            textposition="outside",
            text=[f"${s:,.0f}" if i<len(gam_scores)-1 else f"${sum(gam_scores[:-1]):,.0f}"
                  for i,s in enumerate(gam_scores)],
            totals={"marker":{"color":BLUE}},
            hovertext=hover, hoverinfo="text", **WF))
        fig.update_layout(title={"text":"GA2M Intelligence Layer \u2014 Non-Linear & "
            "Interaction Signal Breakdown","font":{"size":13,"color":BLUE}},**LAY)

    return profile, fig


if __name__ == "__main__":
    app.run(debug=True, port=8050)

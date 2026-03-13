# ==============================================================================
# app.py — ResiScore™ Homeowners Intelligence Layer
# Compatible with revised data_simulation.py / baseline_glm.py / residual_model.py
# ==============================================================================

import pandas as pd
import numpy as np
import dash
from dash import dcc, html, callback, Input, Output, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import joblib
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings("ignore")

# ── Color Palette ─────────────────────────────────────────────────────────────
NAVY   = "#1B2A4A"
GOLD   = "#E8A838"
GREEN  = "#27AE60"
RED    = "#E74C3C"
BLUE   = "#2980B9"
BG     = "#F0F2F5"
WHITE  = "#FFFFFF"
MUTED  = "#6C757D"
BORDER = "#E0E4ED"

CARD_STYLE = {
    "borderRadius": "12px",
    "border": f"1px solid {BORDER}",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.06)",
    "backgroundColor": WHITE,
}
SEC_TITLE = {"color": NAVY, "fontWeight": "700", "fontSize": "1.05rem", "marginBottom": "2px"}

# ── App Init ──────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
    ],
    suppress_callback_exceptions=True,
)
app.title = "ResiScore\u2122 | Homeowners Intelligence Layer"
server = app.server

# ── Data & Model Loading ──────────────────────────────────────────────────────
df = pd.read_csv("data/final_predictions.csv")

ALL_FEATURES = [
    "Year_Built", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
    "Construction_Type", "Protection_Class", "AOI", "Deductible",
    "Territory", "Roof_Age_Applicant", "Fire_Alarm", "Burglar_Alarm",
    "Roof_Vulnerability_Satellite", "Wildfire_Exposure_Daily",
    "Water_Loss_Recency_Months", "RCV_Appraised", "Fire_Hydrant_Distance",
    "Tree_Canopy_Density", "Crime_Severity_Index", "Pluvial_Flood_Depth",
    "Building_Code_Compliance", "Slope_Steepness", "Attic_Ventilation",
    "Hail_Frequency", "Soil_Liquefaction_Risk",
]
LEGACY_FEATURES = [
    "Year_Built", "Square_Footage", "CLUE_Loss_Count", "Credit_Score",
    "Construction_Type", "Protection_Class", "AOI", "Deductible",
    "Territory", "Roof_Age_Applicant", "Fire_Alarm", "Burglar_Alarm",
    "Urban_HighPC", "OldRoof_HighHail",   # engineered actuary interactions
]
MODERN_FEATURES = [f for f in ALL_FEATURES if f not in LEGACY_FEATURES]
CAT_COLS = [
    "Construction_Type", "Territory", "Deductible",
    "Fire_Alarm", "Burglar_Alarm", "Attic_Ventilation", "Soil_Liquefaction_Risk",
]

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
df["Risk_Tier"]      = pd.cut(
    df["Final_Pure_Premium"],
    bins=[0, 800, 1200, 1800, np.inf],
    labels=["Low", "Moderate", "Elevated", "High"],
)
n_repriced   = (df["Adjustment_Pct"].abs() > 10).sum()
pct_repriced = n_repriced / len(df) * 100

# ── EBM Global Explanations ───────────────────────────────────────────────────
global_exp    = ebm_model.explain_global()
global_data   = global_exp.data()
global_names  = global_data["names"]
global_scores = global_data["scores"]
g_pairs       = sorted(zip(global_names, global_scores), key=lambda x: x[1])

TIER_COLORS = {"Low": GREEN, "Moderate": GOLD, "Elevated": "#E67E22", "High": RED}

SHAPE_FEATURES = [
    "Wildfire_Exposure_Daily", "Roof_Vulnerability_Satellite", "Tree_Canopy_Density",
    "Hail_Frequency", "Crime_Severity_Index", "Pluvial_Flood_Depth",
    "Water_Loss_Recency_Months", "Building_Code_Compliance", "Slope_Steepness",
    "Fire_Hydrant_Distance", "CLUE_Loss_Count", "Credit_Score", "Protection_Class",
    "Roof_Age_Applicant", "Year_Built", "Square_Footage",
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def kpi_card(icon, label, value, subtitle, color, badge_text=None):
    return dbc.Card([
        dbc.CardBody([
            html.Div([
                html.Div(
                    html.I(className=icon, style={"fontSize": "1.3rem", "color": color}),
                    style={"backgroundColor": f"{color}1A", "borderRadius": "8px",
                           "padding": "9px 10px", "display": "inline-flex"},
                ),
                dbc.Badge(badge_text, color="warning", className="ms-2 align-self-start",
                          style={"fontSize": "0.68rem"}) if badge_text else None,
            ], className="d-flex align-items-center mb-3"),
            html.Div(value,    style={"fontSize": "1.9rem", "fontWeight": "700",
                                      "color": NAVY, "lineHeight": "1"}),
            html.Div(label,    style={"fontSize": "0.78rem", "fontWeight": "600", "color": MUTED,
                                      "marginTop": "4px", "textTransform": "uppercase",
                                      "letterSpacing": "0.05em"}),
            html.Div(subtitle, style={"fontSize": "0.75rem", "color": MUTED, "marginTop": "5px"}),
        ])
    ], style=CARD_STYLE)


# ── Navbar ────────────────────────────────────────────────────────────────────
navbar = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.I(className="fas fa-shield-alt me-2",
                   style={"color": GOLD, "fontSize": "1.35rem"}),
            html.Span("ResiScore",
                      style={"fontWeight": "700", "fontSize": "1.25rem", "color": WHITE}),
            html.Sup("\u2122", style={"fontSize": "0.7rem", "color": GOLD}),
            html.Span(" | Homeowners Intelligence Layer",
                      style={"color": "#A0AABB", "fontSize": "0.88rem", "marginLeft": "8px"}),
        ], className="d-flex align-items-center"),
        html.Div([
            dbc.Badge("DEMO", color="warning", className="me-2",
                      style={"fontSize": "0.68rem"}),
            html.Span("50,000 synthetic policies \u00b7 GLM + EBM (GA2M) residual layer",
                      style={"color": "#A0AABB", "fontSize": "0.75rem"}),
        ], className="d-none d-md-flex align-items-center"),
    ], fluid=True),
    color=NAVY, dark=True, className="py-2",
    style={"borderBottom": f"3px solid {GOLD}"},
)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════════
def build_portfolio_tab():
    fig_r2 = go.Figure(go.Bar(
        x=["Legacy GLM (14 vars)", "GLM + GA2M (25 vars)"],
        y=[glm_r2, final_r2],
        marker_color=[MUTED, NAVY],
        text=[f"R\u00b2={glm_r2:.4f}", f"R\u00b2={final_r2:.4f}"],
        textposition="outside",
        width=0.45,
    ))
    fig_r2.add_annotation(
        x=1, y=final_r2, text=f" +{delta_r2:.4f} lift",
        showarrow=False, yshift=36,
        font=dict(size=11, color=GREEN, family="Inter"),
    )
    fig_r2.update_yaxes(range=[0, 1.05], title_text="R\u00b2 Score")
    fig_r2.update_layout(
        title={"text": "Variance Explained: GLM vs Intelligence Layer",
               "font": {"size": 13, "color": NAVY}},
        template="plotly_white", height=290,
        margin=dict(l=10, r=10, t=50, b=20), showlegend=False,
        font=dict(family="Inter, sans-serif"),
    )

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=df["GLM_Pure_Premium"].clip(0, 4000), nbinsx=60,
        name="Legacy GLM", marker_color=MUTED, opacity=0.55,
    ))
    fig_dist.add_trace(go.Histogram(
        x=df["Final_Pure_Premium"].clip(0, 4000), nbinsx=60,
        name="GLM + GA2M", marker_color=BLUE, opacity=0.55,
    ))
    fig_dist.update_layout(
        barmode="overlay",
        title={"text": "Premium Distribution: Legacy vs Intelligence-Adjusted",
               "font": {"size": 13, "color": NAVY}},
        xaxis_title="Pure Premium ($)", yaxis_title="Policy Count",
        template="plotly_white", height=290,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=50, b=20),
        font=dict(family="Inter, sans-serif"),
    )

    samp = df.sample(2500, random_state=42)
    fig_scat = go.Figure()
    fig_scat.add_trace(go.Scatter(
        x=samp["GLM_Pure_Premium"], y=samp["Final_Pure_Premium"],
        mode="markers",
        marker=dict(
            color=samp["Adjustment_Pct"],
            colorscale="RdYlGn_r", size=4, opacity=0.55,
            colorbar=dict(title="Adj %", thickness=12, len=0.8),
            cmin=-30, cmax=30,
        ),
        hovertemplate="GLM: $%{x:.0f}<br>Final: $%{y:.0f}<extra></extra>",
    ))
    fig_scat.add_trace(go.Scatter(
        x=[0, 4500], y=[0, 4500], mode="lines",
        line=dict(dash="dash", color="#CCCCCC", width=1), showlegend=False,
    ))
    fig_scat.update_layout(
        title={"text": "Where the GA2M Diverges from GLM (each dot = 1 policy)",
               "font": {"size": 13, "color": NAVY}},
        xaxis_title="GLM Premium ($)", yaxis_title="Final Intelligence Premium ($)",
        template="plotly_white", height=310,
        margin=dict(l=10, r=10, t=50, b=20),
        font=dict(family="Inter, sans-serif"),
    )

    tier_cts = df["Risk_Tier"].value_counts().reindex(["Low", "Moderate", "Elevated", "High"])
    fig_donut = go.Figure(go.Pie(
        labels=tier_cts.index, values=tier_cts.values, hole=0.62,
        marker_colors=[GREEN, GOLD, "#E67E22", RED],
        textinfo="label+percent", textfont_size=11,
    ))
    fig_donut.update_layout(
        title={"text": "Portfolio Risk Tier (Intelligence-Adjusted)",
               "font": {"size": 13, "color": NAVY}},
        showlegend=False, height=310,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, sans-serif"),
    )

    return html.Div([
        dbc.Alert([
            html.I(className="fas fa-lightbulb me-2", style={"color": GOLD}),
            html.Strong("The Opportunity: "),
            f"The legacy 14-variable GLM explains {glm_r2:.1%} of premium variance. "
            f"Adding 11 modern non-linear signals via GA2M lifts this to {final_r2:.1%} \u2014 "
            f"a +{delta_r2:.4f} \u0394R\u00b2 that re-prices {pct_repriced:.1f}% of the book by >10%.",
        ], color="warning", className="mb-4",
           style={"borderLeft": f"4px solid {GOLD}", "backgroundColor": "#FFFBF0",
                  "borderRadius": "8px", "fontSize": "0.88rem"}),
        dbc.Row([
            dbc.Col(kpi_card("fas fa-chart-line",         "Legacy GLM R\u00b2",
                             f"{glm_r2:.4f}",
                             "14 vars (12 legacy + 2 actuary interactions)", MUTED), width=3),
            dbc.Col(kpi_card("fas fa-brain",               "Intelligence R\u00b2",
                             f"{final_r2:.4f}",
                             "25 vars \u00b7 non-linear + pairwise interactions", BLUE), width=3),
            dbc.Col(kpi_card("fas fa-arrow-trend-up",      "Variance Lift (\u0394R\u00b2)",
                             f"+{delta_r2:.4f}",
                             "Residual signal from 11 modern data sources", GREEN, "KEY LIFT"), width=3),
            dbc.Col(kpi_card("fas fa-exclamation-triangle","Re-priced Risks",
                             f"{pct_repriced:.1f}%",
                             f"Policies adjusted >10% vs GLM ({n_repriced:,} policies)", GOLD), width=3),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_r2, config={"displayModeBar": False})), style=CARD_STYLE), width=4),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_dist, config={"displayModeBar": False})), style=CARD_STYLE), width=8),
        ], className="g-3 mb-4"),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_scat, config={"displayModeBar": False})), style=CARD_STYLE), width=8),
            dbc.Col(dbc.Card(dbc.CardBody(
                dcc.Graph(figure=fig_donut, config={"displayModeBar": False})), style=CARD_STYLE), width=4),
        ], className="g-3"),
    ], className="py-4")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — FEATURE INTELLIGENCE
# ════════════════════════════════════════════════════════════════════════════════
def build_feature_tab():
    top15     = g_pairs[-15:]
    bar_colors = [
        GOLD if (n in MODERN_FEATURES or any(m in n for m in MODERN_FEATURES)) else MUTED
        for n, s in top15
    ]
    fig_global = go.Figure(go.Bar(
        x=[s for n, s in top15],
        y=[n.replace("_", " ") for n, s in top15],
        orientation="h",
        marker_color=bar_colors,
        text=[f"${s:,.2f}" for n, s in top15],
        textposition="outside",
        textfont_size=10,
    ))
    fig_global.add_annotation(
        text="\U0001f7e1 Modern feature    \u25a1 Legacy feature",
        xref="paper", yref="paper", x=0.5, y=-0.06, showarrow=False,
        font=dict(size=10, color=MUTED),
    )
    fig_global.update_xaxes(title_text="Avg Abs Contribution (log uplift)")
    fig_global.update_layout(
        title={"text": "GA2M Global Importance \u2014 Avg Abs Contribution to Residual",
               "font": {"size": 13, "color": NAVY}},
        template="plotly_white", height=480,
        margin=dict(l=10, r=60, t=50, b=40),
        font=dict(family="Inter, sans-serif"),
    )

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div("Global Feature Importance", style=SEC_TITLE),
                        html.Div("Averaged across all 50K policies \u2014 "
                                 "what the GA2M learns from log-scale GLM residuals",
                                 style={"fontSize": "0.78rem", "color": MUTED}),
                    ], style={"backgroundColor": WHITE, "border": "none", "paddingBottom": "0"}),
                    dbc.CardBody(dcc.Graph(figure=fig_global, config={"displayModeBar": False})),
                ], style=CARD_STYLE),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div("EBM Shape Function Viewer", style=SEC_TITLE),
                        html.Div("Glass-box: each feature\u2019s exact learned 1-D effect "
                                 "on log-uplift residual",
                                 style={"fontSize": "0.78rem", "color": MUTED}),
                        dcc.Dropdown(
                            id="shape-feature-dd",
                            options=[{"label": f.replace("_", " "), "value": f}
                                     for f in SHAPE_FEATURES],
                            value="Wildfire_Exposure_Daily",
                            clearable=False,
                            className="mt-2",
                            style={"fontSize": "0.83rem"},
                        ),
                    ], style={"backgroundColor": WHITE, "border": "none", "paddingBottom": "0"}),
                    dbc.CardBody(dcc.Graph(id="shape-fn-plot",
                                          config={"displayModeBar": False})),
                ], style=CARD_STYLE),
            ], width=6),
        ], className="g-3 py-4"),
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.Div(
                            "Top Interaction \u2014 Wildfire Exposure \u00d7 Roof Vulnerability",
                            style=SEC_TITLE),
                        html.Div(
                            "Non-additive risk the GLM cannot see: old roofs in high-wildfire "
                            "zones carry combined penalties beyond the sum of individual effects. "
                            "Color = avg GA2M log-uplift.",
                            style={"fontSize": "0.78rem", "color": MUTED}),
                    ], style={"backgroundColor": WHITE, "border": "none"}),
                    dbc.CardBody(dcc.Graph(id="interaction-heatmap",
                                          config={"displayModeBar": False})),
                ], style=CARD_STYLE),
            ], width=12),
        ], className="g-3 pb-4"),
    ])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — POLICY UNDERWRITER LENS
# ════════════════════════════════════════════════════════════════════════════════
def build_policy_options():
    up   = df.nlargest(80, "Adjustment_Pct")
    down = df.nsmallest(80, "Adjustment_Pct")
    rand = df.sample(80, random_state=42)
    pool = pd.concat([up, down, rand]).drop_duplicates().head(200)
    opts = []
    for idx, row in pool.iterrows():
        tier  = str(row["Risk_Tier"])
        adj   = row["Adjustment_Pct"]
        arrow = "\u2191" if adj > 0 else "\u2193"
        opts.append({
            "label": (f"Policy #{idx} | {tier} Risk | {arrow}{abs(adj):.0f}% adj | "
                      f"GLM ${row['GLM_Pure_Premium']:,.0f} \u2192 "
                      f"Final ${row['Final_Pure_Premium']:,.0f}"),
            "value": idx,
        })
    return opts, pool.index[0]


POLICY_OPTIONS, DEFAULT_POLICY = build_policy_options()


def build_policy_tab():
    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.Div("Select Policy", style=SEC_TITLE),
                                   style={"backgroundColor": WHITE, "border": "none"}),
                    dbc.CardBody([
                        html.Div(
                            "Curated: 80 highest surcharges \u00b7 80 highest credits \u00b7 80 random",
                            style={"fontSize": "0.75rem", "color": MUTED, "marginBottom": "8px"}),
                        dcc.Dropdown(
                            id="policy-dd", options=POLICY_OPTIONS,
                            value=DEFAULT_POLICY, clearable=False,
                            style={"fontSize": "0.82rem"},
                        ),
                        html.Div(id="policy-profile-panel", className="mt-3"),
                    ]),
                ], style=CARD_STYLE),
            ], width=3),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        dbc.Row([
                            dbc.Col(html.Div("Pricing Deconstruction", style=SEC_TITLE), width=6),
                            dbc.Col([
                                dbc.ButtonGroup([
                                    dbc.Button("Strategic",         id="btn-hi",  n_clicks=1,
                                               color="primary",   outline=True, size="sm"),
                                    dbc.Button("GLM Breakdown",     id="btn-glm", n_clicks=0,
                                               color="secondary", outline=True, size="sm"),
                                    dbc.Button("GA2M Intelligence", id="btn-gam", n_clicks=0,
                                               color="info",      outline=True, size="sm"),
                                ]),
                            ], width=6, className="text-end"),
                        ], align="center"),
                    ], style={"backgroundColor": WHITE, "border": "none"}),
                    dbc.CardBody([
                        dcc.Store(id="view-store", data="high_level"),
                        # Dash legend row — clear space, no overlap with chart ticks
                        html.Div([
                            html.Span([
                                html.Span("\u25cf", style={"color": BLUE, "fontWeight": "700",
                                                           "marginRight": "4px"}),
                                "Individual feature effect",
                            ], style={"fontSize": "0.75rem", "color": MUTED, "marginRight": "20px"}),
                            html.Span([
                                html.Span("\u2297", style={"color": GOLD, "fontWeight": "700",
                                                           "marginRight": "4px"}),
                                "Pairwise interaction effect",
                            ], style={"fontSize": "0.75rem", "color": MUTED, "marginRight": "20px"}),
                            html.Span("(\u2297 appears in GA2M view only)",
                                      style={"fontSize": "0.7rem", "color": BORDER}),
                        ], className="text-end mb-1 pe-1"),
                        dcc.Graph(id="waterfall-plot", config={"displayModeBar": False}),
                    ]),
                ], style=CARD_STYLE),
            ], width=9),
        ], className="g-3 py-4"),
    ])


# ── Root Layout ───────────────────────────────────────────────────────────────
app.layout = html.Div([
    navbar,
    dbc.Container([
        dcc.Tabs(
            id="main-tabs", value="tab-portfolio",
            children=[
                dcc.Tab(label="\U0001f4ca  Portfolio Intelligence",  value="tab-portfolio",
                        style={"fontFamily": "Inter", "fontSize": "0.88rem"},
                        selected_style={"fontFamily": "Inter", "fontSize": "0.88rem",
                                        "fontWeight": "700",
                                        "borderTop": f"3px solid {NAVY}"}),
                dcc.Tab(label="\U0001f52c  Feature Intelligence",    value="tab-features",
                        style={"fontFamily": "Inter", "fontSize": "0.88rem"},
                        selected_style={"fontFamily": "Inter", "fontSize": "0.88rem",
                                        "fontWeight": "700",
                                        "borderTop": f"3px solid {NAVY}"}),
                dcc.Tab(label="\U0001f50d  Policy Underwriter Lens", value="tab-policy",
                        style={"fontFamily": "Inter", "fontSize": "0.88rem"},
                        selected_style={"fontFamily": "Inter", "fontSize": "0.88rem",
                                        "fontWeight": "700",
                                        "borderTop": f"3px solid {NAVY}"}),
            ],
            style={"marginTop": "12px"},
        ),
        html.Div(id="tab-content"),
    ], fluid=True, style={"maxWidth": "1600px", "padding": "0 24px"}),
], style={"backgroundColor": BG, "minHeight": "100vh", "fontFamily": "Inter, sans-serif"})


# ════════════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ════════════════════════════════════════════════════════════════════════════════

@callback(Output("tab-content", "children"), Input("main-tabs", "value"))
def render_tab(tab):
    if tab == "tab-portfolio": return build_portfolio_tab()
    if tab == "tab-features":  return build_feature_tab()
    if tab == "tab-policy":    return build_policy_tab()


@callback(Output("shape-fn-plot", "figure"), Input("shape-feature-dd", "value"))
def update_shape(feature_name):
    try:
        idx   = list(ebm_model.feature_names_in_).index(feature_name)
        fdata = global_exp.data(idx)
        x_vals = fdata.get("names", [])
        y_vals = fdata.get("scores", [])
        if x_vals and (isinstance(x_vals[0], (list, tuple)) or
                       (hasattr(x_vals[0], "__len__") and not isinstance(x_vals[0], str))):
            fig = go.Figure()
            fig.add_annotation(text="Select a univariate feature.",
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        fig = go.Figure()
        fig.add_hline(y=0, line_dash="dash", line_color="#CCCCCC", line_width=1)
        fig.add_trace(go.Scatter(
            x=x_vals, y=y_vals, mode="lines+markers",
            line=dict(color=BLUE, width=2.5), marker=dict(size=5),
            fill="tozeroy", fillcolor=f"{BLUE}18",
        ))
        fig.update_xaxes(title_text=feature_name.replace("_", " "))
        fig.update_yaxes(title_text="GA2M Contribution (log uplift)")
        fig.update_layout(
            title={"text": f"Shape Function: {feature_name.replace('_', ' ')}",
                   "font": {"size": 13, "color": NAVY}},
            template="plotly_white", height=480,
            margin=dict(l=10, r=10, t=50, b=30),
            font=dict(family="Inter, sans-serif"), showlegend=False,
        )
        return fig
    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Unavailable: {e}", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=MUTED))
        return fig


@callback(Output("interaction-heatmap", "figure"), Input("main-tabs", "value"))
def update_heatmap(tab):
    if tab != "tab-features":
        return go.Figure()
    tmp             = df.copy()
    tmp["wf_bin"]   = pd.cut(tmp["Wildfire_Exposure_Daily"],     bins=10)
    tmp["rv_bin"]   = pd.cut(tmp["Roof_Vulnerability_Satellite"], bins=10)
    pivot = tmp.groupby(["rv_bin", "wf_bin"], observed=True)["EBM_Log_Uplift"].mean().unstack()

    def bin_label(c):
        lo = float(str(c).split(",")[0].strip("("))
        hi = float(str(c).split(",")[1].strip("]"))
        return f"{lo:.0f}\u2013{hi:.0f}"

    x_labels = [bin_label(c) for c in pivot.columns]
    y_labels  = [bin_label(r) for r in pivot.index]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=x_labels, y=y_labels,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Avg GA2M<br>Log Uplift", thickness=14, len=0.8),
        hovertemplate=(
            "Wildfire: %{x}<br>Roof Vuln: %{y}<br>"
            "Avg Log Uplift: %{z:.4f}<extra></extra>"
        ),
    ))
    fig.update_xaxes(title_text="Wildfire Exposure (binned)")
    fig.update_yaxes(title_text="Roof Vulnerability (binned)")
    fig.update_layout(
        template="plotly_white", height=380,
        margin=dict(l=10, r=10, t=20, b=30),
        font=dict(family="Inter, sans-serif"),
    )
    return fig


@callback(
    Output("view-store", "data"),
    [Input("btn-hi", "n_clicks"), Input("btn-glm", "n_clicks"), Input("btn-gam", "n_clicks")],
    prevent_initial_call=True,
)
def set_view(n1, n2, n3):
    triggered = ctx.triggered_id
    if triggered == "btn-hi":  return "high_level"
    if triggered == "btn-glm": return "glm_breakdown"
    if triggered == "btn-gam": return "gam_breakdown"
    return "high_level"


@callback(
    [Output("btn-hi",  "outline"),
     Output("btn-glm", "outline"),
     Output("btn-gam", "outline")],
    Input("view-store", "data"),
)
def highlight_active_button(view):
    return (
        view != "high_level",
        view != "glm_breakdown",
        view != "gam_breakdown",
    )


@callback(
    [Output("policy-profile-panel", "children"),
     Output("waterfall-plot",       "figure")],
    [Input("policy-dd",  "value"),
     Input("view-store", "data")],
)
def update_policy_view(selected_idx, view_type):
    row             = df.loc[selected_idx]
    X_sample        = df[ALL_FEATURES].loc[[selected_idx]]
    glm_prem        = row["GLM_Pure_Premium"]
    actual_exp_prem = row["Expected_Pure_Premium"]
    ebm_residual    = row["EBM_Residual_Pred"]   # dollar adj (Final - GLM)
    final_prem      = row["Final_Pure_Premium"]
    adj_pct         = row["Adjustment_Pct"]
    tier            = str(row["Risk_Tier"])
    tier_color      = TIER_COLORS.get(tier, MUTED)

    # ── Profile panel ──────────────────────────────────────────────────────
    def row_item(label, value, vstyle=None):
        base = {"fontSize": "0.92rem", "fontWeight": "600", "color": NAVY, "float": "right"}
        if vstyle: base.update(vstyle)
        return html.Div([
            html.Span(label, style={"fontSize": "0.72rem", "color": MUTED,
                                    "textTransform": "uppercase", "letterSpacing": "0.04em"}),
            html.Span(value, style=base),
        ], style={"borderBottom": f"1px solid {BORDER}", "padding": "7px 0",
                  "overflow": "hidden"})

    profile_panel = html.Div([
        html.Div([
            dbc.Badge(f"{tier} Risk",
                      style={"backgroundColor": tier_color, "fontSize": "0.75rem"}),
            dbc.Badge(f"{'↑' if adj_pct > 0 else '↓'}{abs(adj_pct):.1f}% vs GLM",
                      color="danger" if adj_pct > 0 else "success",
                      className="ms-1", style={"fontSize": "0.75rem"}),
        ], className="mb-3"),
        row_item("True Expected", f"${actual_exp_prem:,.0f}",
                 {"color": RED if actual_exp_prem > final_prem else GREEN}),
        row_item("Legacy GLM",   f"${glm_prem:,.0f}"),
        row_item("GA2M Adj",
                 f"{'+'if ebm_residual>=0 else ''}${ebm_residual:,.0f}",
                 {"color": RED if ebm_residual > 0 else GREEN}),
        html.Div([
            html.Span("Final Premium",
                      style={"fontSize": "0.82rem", "fontWeight": "700", "color": NAVY}),
            html.Span(f"${final_prem:,.0f}",
                      style={"fontSize": "1.15rem", "fontWeight": "700",
                             "color": BLUE, "float": "right"}),
        ], style={"padding": "10px 0 0", "overflow": "hidden"}),
    ])

    # ── Shared waterfall style ─────────────────────────────────────────────
    WF_COMMON = dict(
        connector={"line": {"color": BORDER, "width": 2}},
        increasing={"marker": {"color": RED}},
        decreasing={"marker": {"color": GREEN}},
    )
    LAYOUT_BASE = dict(
        template="plotly_white",
        font=dict(family="Inter, sans-serif"),
        height=430,
        margin=dict(l=20, r=20, t=65, b=50),
        showlegend=False,
        waterfallgap=0.25,
    )

    # ── Strategic View ─────────────────────────────────────────────────────
    if view_type == "high_level":
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Legacy GLM Estimate", "GA2M Intelligence Adjustment", "Final Premium"],
            y=[glm_prem, ebm_residual, 0],
            textposition="outside",
            text=[f"${glm_prem:,.0f}",
                  f"{'+'if ebm_residual>=0 else ''}${ebm_residual:,.0f}",
                  f"${final_prem:,.0f}"],
            totals={"marker": {"color": NAVY}},
            **WF_COMMON,
        ))
        fig.update_layout(
            title={"text": "Strategic View \u2014 Legacy Formula \u2192 Intelligence-Adjusted Premium",
                   "font": {"size": 13, "color": NAVY}},
            **LAYOUT_BASE,
        )

    # ── GLM Breakdown View ─────────────────────────────────────────────────
    elif view_type == "glm_breakdown":
        # Re-engineer the two interaction features for the GLM preprocessor
        X_glm = X_sample.copy()
        X_glm["Urban_HighPC"]     = (
            (X_glm["Territory"] == "Urban") &
            (X_glm["Protection_Class"] > 6)).astype(int).astype(str)
        X_glm["OldRoof_HighHail"] = (
            (X_glm["Roof_Age_Applicant"] > 20) &
            (X_glm["Hail_Frequency"] >= 3)).astype(int).astype(str)

        freq_coef = freq_model.named_steps["regressor"].coef_
        freq_int  = freq_model.named_steps["regressor"].intercept_
        sev_coef  = sev_model.named_steps["regressor"].coef_
        sev_int   = sev_model.named_steps["regressor"].intercept_
        combined_coefs = freq_coef + sev_coef
        combined_int   = float(freq_int + sev_int)

        X_proc = freq_model.named_steps["preprocessor"].transform(X_glm[LEGACY_FEATURES])
        if hasattr(X_proc, "toarray"): X_proc = X_proc.toarray()
        feat_names  = freq_model.named_steps["preprocessor"].get_feature_names_out()
        log_impacts = X_proc[0] * combined_coefs

        agg = {}
        for i, name in enumerate(feat_names):
            base = name.split("__")[1] if "__" in name else name
            for cat in ["Construction_Type", "Territory", "Deductible",
                        "Fire_Alarm", "Burglar_Alarm",
                        "Urban_HighPC", "OldRoof_HighHail"]:
                if base.startswith(cat): base = cat; break
            agg[base] = agg.get(base, 0) + log_impacts[i]
        for f in LEGACY_FEATURES:
            if f not in agg: agg[f] = 0.0

        total_log  = sum(agg.values())
        base_prem  = float(np.exp(combined_int))
        attribs    = {f: (v / total_log if total_log != 0 else 0) * (glm_prem - base_prem)
                      for f, v in agg.items()}
        scored = sorted(attribs.items(), key=lambda x: abs(x[1]), reverse=True)

        # Label bars with ● prefix (all GLM terms are individual linear effects)
        labeled_names = (
            ["GLM Base Rate"] +
            ["\u25cf " + f[0].replace("_", " ") for f in scored] +
            ["Total GLM"]
        )
        scores_wf    = [base_prem] + [f[1] for f in scored] + [0]
        measures_wf  = ["relative"] + ["relative"] * len(scored) + ["total"]

        fig = go.Figure(go.Waterfall(
            orientation="v", measure=measures_wf,
            x=labeled_names, y=scores_wf,
            textposition="outside",
            text=[f"${s:,.0f}" if i < len(scores_wf) - 1 else f"${glm_prem:,.0f}"
                  for i, s in enumerate(scores_wf)],
            totals={"marker": {"color": MUTED}},
            **WF_COMMON,
        ))
        fig.update_layout(
            title={"text": "GLM Breakdown \u2014 How the Legacy Actuarial System Priced This Risk",
                   "font": {"size": 13, "color": MUTED}},
            **LAYOUT_BASE,
        )

    # ── GA2M Intelligence View ─────────────────────────────────────────────
    else:
        eps = 1e-6
        log_resid_true = float(
            np.log(actual_exp_prem + eps) - np.log(glm_prem + eps))

        local_exp     = ebm_model.explain_local(X_sample, y=[log_resid_true])
        exp_data      = local_exp.data(0)
        names_raw     = exp_data["names"]
        scores_raw    = exp_data["scores"]        # log-uplift units
        gam_int_log   = float(exp_data["extra"]["scores"][0])

        # ── Convert log-uplift contributions → dollar space ───────────────
        # Proportional rescaling: each bar's dollar value =
        #   (log_contribution / total_log_pred) × total_dollar_adjustment
        total_log_pred = float(sum(scores_raw)) + gam_int_log
        def log_to_dollar(log_val):
            if abs(total_log_pred) < 1e-9: return 0.0
            return (log_val / total_log_pred) * ebm_residual

        scores_dollar  = [log_to_dollar(s) for s in scores_raw]
        intercept_dollar = log_to_dollar(gam_int_log)

        # Sort and split top-10 vs rest
        scored_f  = sorted(zip(names_raw, scores_dollar),
                            key=lambda x: abs(x[1]), reverse=True)
        top_f     = scored_f[:10]
        other_f   = scored_f[10:]
        other_sum = sum(f[1] for f in other_f)

        # Cap "All Other Signals" hover to 5 features to keep tooltip short
        top5_others   = sorted(other_f, key=lambda x: abs(x[1]), reverse=True)[:5]
        remaining_cnt = len(other_f) - 5
        other_detail  = "".join(
            [f"<br>  {f[0]}: ${f[1]:,.0f}" for f in top5_others])
        if remaining_cnt > 0:
            other_detail += f"<br>  ...and {remaining_cnt} more signals"

        def classify(name):
            if name in ["GA2M Intercept", "All Other Signals", "Net Residual Adj"]:
                return "meta"
            return "interaction" if " & " in name else "main"

        raw_names   = (["GA2M Intercept"] +
                       [f[0] for f in top_f] +
                       ["All Other Signals", "Net Residual Adj"])
        gam_scores  = ([intercept_dollar] +
                       [f[1] for f in top_f] +
                       [other_sum, 0])
        gam_measures = (["relative"] +
                        ["relative"] * len(top_f) +
                        ["relative", "total"])

        # Prefix labels
        labeled_names = []
        for n in raw_names:
            t = classify(n)
            if t == "main":        labeled_names.append("\u25cf " + n.replace("_", " "))
            elif t == "interaction": labeled_names.append("\u2297 " + n.replace("_", " "))
            else:                  labeled_names.append(n)

        # Hover strings (short to avoid Plotly tooltip drop)
        hover = []
        for i, n in enumerate(raw_names):
            if n == "All Other Signals":
                hover.append(f"All Other Signals: ${other_sum:,.0f}{other_detail}")
            elif n == "Net Residual Adj":
                hover.append(f"Net Residual Adjustment: ${sum(gam_scores[:-1]):,.0f}")
            else:
                hover.append(f"{n}: ${gam_scores[i]:,.2f}")

        fig = go.Figure(go.Waterfall(
            orientation="v", measure=gam_measures,
            x=labeled_names, y=gam_scores,
            textposition="outside",
            text=[f"${s:,.0f}" if i < len(gam_scores) - 1
                  else f"${sum(gam_scores[:-1]):,.0f}"
                  for i, s in enumerate(gam_scores)],
            totals={"marker": {"color": BLUE}},
            hovertext=hover,
            hoverinfo="text",
            **WF_COMMON,
        ))
        fig.update_layout(
            title={"text": "GA2M Intelligence Layer \u2014 Non-Linear & Interaction Signal Breakdown",
                   "font": {"size": 13, "color": BLUE}},
            **LAYOUT_BASE,
        )

    return profile_panel, fig


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)

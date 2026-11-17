"""
CRICKET PRESSURE PERFORMANCE INTELLIGENCE SYSTEM
================================================
Interactive Scouting Dashboard & Player Analytics Suite
Author: Antigravity Senior Sports Analytics Team
Framework: Dash & Plotly with Dash Bootstrap Components
Scope: 352,625 deliveries across 7 competitions (IPL, T20I, BBL, SA20, MLC, ILT20, LPL: 2022-2026)
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc

# ════════════════════════════════════════════════════════════════
# 1. LOAD PRE-COMPUTED DATASETS
# ════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

bat_df = pd.read_csv(os.path.join(DATA_DIR, 'dashboard_batters.csv'))
bow_df = pd.read_csv(os.path.join(DATA_DIR, 'dashboard_bowlers.csv'))
cs_df = pd.read_csv(os.path.join(DATA_DIR, 'comp_season_stats.csv'))

with open(os.path.join(DATA_DIR, 'player_registry_lookup.json'), 'r', encoding='utf-8') as f:
    player_registry = json.load(f)

with open(os.path.join(DATA_DIR, 'ml_model_intelligence.json'), 'r', encoding='utf-8') as f:
    ml_intel = json.load(f)

# Quick lookup indexes
bat_dict = bat_df.set_index('player_name').to_dict(orient='index')
bow_dict = bow_df.set_index('player_name').to_dict(orient='index')

ALL_PLAYERS = sorted(list(player_registry.keys()))
FIXED_19_NAMES = [p for p in ALL_PLAYERS if player_registry[p].get('is_fixed19')]
COMPETITIONS = ['All Competitions', 'IPL', 'T20I', 'Big Bash League', 'SA20', 'Major League Cricket', 'International League T20', 'Lanka Premier League']
SEASONS = ['All Seasons', '2022', '2023', '2024', '2025', '2026']

# ════════════════════════════════════════════════════════════════
# 2. COLOR PALETTES & PLOTLY DARK TEMPLATE
# ════════════════════════════════════════════════════════════════
THEME = {
    'bg': '#0a0e17',
    'card': '#141d2e',
    'border': '#1f2d44',
    'cyan': '#00e5ff',
    'blue': '#3b82f6',
    'purple': '#8b5cf6',
    'coral': '#ff3366',
    'gold': '#f59e0b',
    'emerald': '#10b981',
    'muted': '#94a3b8',
    'white': '#f8fafc'
}

COMPARISON_COLORS = [
    {'line': '#00e5ff', 'fill': 'rgba(0, 229, 255, 0.22)'},
    {'line': '#ff3366', 'fill': 'rgba(255, 51, 102, 0.22)'},
    {'line': '#f59e0b', 'fill': 'rgba(245, 158, 11, 0.22)'},
    {'line': '#10b981', 'fill': 'rgba(16, 185, 129, 0.22)'},
]

# ════════════════════════════════════════════════════════════════
# 3. HELPER FUNCTIONS FOR RADAR CHARTS
# ════════════════════════════════════════════════════════════════
def build_radar_chart(categories, values_dict, title="Scouting Radar", is_bowling=False):
    """
    Builds a professional football scouting-style radar pentagon.
    categories: list of 5 dimension names
    values_dict: {player_label: [val1, val2, val3, val4, val5], ...}
    """
    fig = go.Figure()
    cats_closed = categories + [categories[0]]
    
    idx = 0
    for player_name, vals in values_dict.items():
        vals_closed = vals + [vals[0]]
        color_scheme = COMPARISON_COLORS[idx % len(COMPARISON_COLORS)]
        
        # Override default color if single player
        if len(values_dict) == 1:
            line_col = THEME['purple'] if is_bowling else THEME['cyan']
            fill_col = 'rgba(139, 92, 246, 0.25)' if is_bowling else 'rgba(0, 229, 255, 0.25)'
        else:
            line_col = color_scheme['line']
            fill_col = color_scheme['fill']

        fig.add_trace(go.Scatterpolar(
            r=vals_closed,
            theta=cats_closed,
            fill='toself',
            fillcolor=fill_col,
            name=player_name,
            line=dict(color=line_col, width=2.5),
            marker=dict(size=7, color=line_col, symbol='circle'),
            hovertemplate="<b>%{theta}</b><br>Score: %{r:.1f} / 100<extra>" + player_name + "</extra>"
        ))
        idx += 1

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=['20', '40', '60', '80', '100'],
                tickfont=dict(size=9, color=THEME['muted']),
                linecolor='rgba(255, 255, 255, 0.1)',
                gridcolor='rgba(255, 255, 255, 0.1)',
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color=THEME['white'], family='Inter, sans-serif'),
                linecolor='rgba(255, 255, 255, 0.15)',
                gridcolor='rgba(255, 255, 255, 0.12)',
            ),
            bgcolor='rgba(15, 23, 42, 0.65)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=45, r=45, t=35, b=35),
        showlegend=(len(values_dict) > 1),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(color=THEME['white'], size=11)
        ),
        height=380,
    )
    return fig

# ════════════════════════════════════════════════════════════════
# 4. INITIALIZE DASH APPLICATION
# ════════════════════════════════════════════════════════════════
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Cricket Pressure Performance Intelligence Dashboard",
    suppress_callback_exceptions=True
)
server = app.server

# ════════════════════════════════════════════════════════════════
# 5. DASHBOARD LAYOUT
# ════════════════════════════════════════════════════════════════
app.layout = html.Div(style={'backgroundColor': THEME['bg'], 'minHeight': '100vh', 'paddingBottom': '40px'}, children=[
    dcc.Location(id='url', refresh=False),
    
    # ── Top Navbar / Header ──────────────────────────────────────
    html.Div(style={'background': 'linear-gradient(180deg, #101626 0%, #0c121e 100%)', 'borderBottom': f'1px solid {THEME["border"]}', 'padding': '16px 28px', 'marginBottom': '20px'}, children=[
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Span("⚡ ANTIGRAVITY SPORTS ANALYTICS", style={'color': THEME['cyan'], 'fontSize': '0.75rem', 'fontWeight': '700', 'letterSpacing': '0.12em'}),
                    html.H3("PRESSURE PERFORMANCE INTELLIGENCE DASHBOARD", style={'color': THEME['white'], 'fontWeight': '800', 'margin': '2px 0 0 0', 'fontSize': '1.35rem'}),
                    html.P("Multi-League Empirical Pressure Analysis & GPU-Accelerated ML Intelligence (2022–2026)", style={'color': THEME['muted'], 'fontSize': '0.8rem', 'margin': 0})
                ])
            ], md=7, xs=12),
            dbc.Col([
                html.Div(style={'display': 'flex', 'gap': '10px', 'justifyContent': 'flex-end', 'alignItems': 'center', 'height': '100%'}, children=[
                    html.Div(className="badge-pill-cyan", children="352,625 Balls"),
                    html.Div(className="badge-pill-purple", children="7 Competitions"),
                    html.Div(className="badge-pill-gold", children="NVIDIA RTX GPU"),
                    html.Div(className="badge-pill-emerald", children="1,635 Players")
                ])
            ], md=5, xs=12)
        ])
    ]),

    # ── Master Controls Card (Global Filters) ───────────────────
    dbc.Container(fluid=True, style={'padding': '0 28px'}, children=[
        dbc.Card(className="dash-card mb-4", style={'padding': '16px 20px'}, children=[
            dbc.Row([
                # Player Search Dropdown
                dbc.Col([
                    html.Label("Search & Select Player (1,635 Total):", style={'fontSize': '0.75rem', 'fontWeight': '600', 'color': THEME['muted'], 'textTransform': 'uppercase'}),
                    dcc.Dropdown(
                        id='player-select-dropdown',
                        options=[
                            {'label': f"★ {p} [Challenge 19]" if player_registry[p].get('is_fixed19') else p, 'value': p}
                            for p in ALL_PLAYERS
                        ],
                        value='Virat Kohli',
                        clearable=False,
                        style={'backgroundColor': THEME['card'], 'color': '#000'}
                    )
                ], md=4, xs=12),
                
                # Universe Toggle (All Players vs Fixed 19)
                dbc.Col([
                    html.Label("Player Universe Scope:", style={'fontSize': '0.75rem', 'fontWeight': '600', 'color': THEME['muted'], 'textTransform': 'uppercase'}),
                    dbc.RadioItems(
                        id='universe-scope-toggle',
                        options=[
                            {'label': ' All Players (1,635)', 'value': 'all'},
                            {'label': ' Challenge Shortlist Only (19)', 'value': 'fixed19'}
                        ],
                        value='all',
                        inline=True,
                        style={'fontSize': '0.85rem', 'marginTop': '6px', 'color': THEME['white']}
                    )
                ], md=3, xs=12),

                # Role Mode Selector
                dbc.Col([
                    html.Label("Analysis Mode:", style={'fontSize': '0.75rem', 'fontWeight': '600', 'color': THEME['muted'], 'textTransform': 'uppercase'}),
                    dbc.RadioItems(
                        id='role-mode-toggle',
                        options=[
                            {'label': ' 🏏 Batting', 'value': 'batting'},
                            {'label': ' 🎯 Bowling', 'value': 'bowling'},
                            {'label': ' ⚡ Dual / All-Rounder', 'value': 'dual'}
                        ],
                        value='batting',
                        inline=True,
                        style={'fontSize': '0.85rem', 'marginTop': '6px', 'color': THEME['white']}
                    )
                ], md=3, xs=12),

                # Quick Reset
                dbc.Col([
                    html.Label("Quick Reset:", style={'fontSize': '0.75rem', 'fontWeight': '600', 'color': THEME['muted'], 'textTransform': 'uppercase'}),
                    html.Div([
                        dbc.Button("Reset Filters", id='reset-filters-btn', size="sm", outline=True, color="secondary", style={'width': '100%'})
                    ])
                ], md=2, xs=12)
            ])
        ]),

        # ── Main Dashboard Tabs ──────────────────────────────────
        dbc.Tabs(id="dashboard-tabs", active_tab="tab-profile", className="nav-pills mb-4", children=[
            dbc.Tab(label="👤 Player Scouting Profile", tab_id="tab-profile"),
            dbc.Tab(label="⚔️ Head-to-Head Comparison", tab_id="tab-comparison"),
            dbc.Tab(label="⚡ Pressure vs Normal Context", tab_id="tab-context"),
            dbc.Tab(label="🌍 Competition Dynamics", tab_id="tab-competitions"),
            dbc.Tab(label="🤖 GPU ML Intelligence", tab_id="tab-ml"),
            dbc.Tab(label="🏆 Leaderboards & 19 Roster", tab_id="tab-leaderboards"),
        ]),

        # ── Tab Content Container ────────────────────────────────
        html.Div(id="tab-content-area")
    ])
])

# ════════════════════════════════════════════════════════════════
# 6. CALLBACKS
# ════════════════════════════════════════════════════════════════

# Callback 0: Sync URL query params with Active Tab
@app.callback(
    Output('dashboard-tabs', 'active_tab'),
    Input('url', 'search')
)
def sync_url_tab(search):
    if search and 'tab=' in search:
        for param in search.lstrip('?').split('&'):
            if param.startswith('tab='):
                val = param.split('=')[1]
                if val in ['tab-profile', 'tab-comparison', 'tab-context', 'tab-competitions', 'tab-ml', 'tab-leaderboards']:
                    return val
    return 'tab-profile'

# Callback 1: Filter Dropdown based on Universe Scope
@app.callback(
    Output('player-select-dropdown', 'options'),
    Input('universe-scope-toggle', 'value')
)
def update_player_dropdown(scope):
    if scope == 'fixed19':
        candidates = FIXED_19_NAMES
    else:
        candidates = ALL_PLAYERS
    return [
        {'label': f"★ {p} [Challenge 19]" if player_registry[p].get('is_fixed19') else p, 'value': p}
        for p in candidates
    ]

# Callback 2: Main Tab Switching
@app.callback(
    Output('tab-content-area', 'children'),
    [Input('dashboard-tabs', 'active_tab'),
     Input('player-select-dropdown', 'value'),
     Input('role-mode-toggle', 'value')]
)
def render_tab_content(active_tab, selected_player, role_mode):
    if not selected_player or selected_player not in player_registry:
        selected_player = 'Virat Kohli'
        
    p_meta = player_registry.get(selected_player, {})
    bat_data = bat_dict.get(selected_player, {})
    bow_data = bow_dict.get(selected_player, {})
    
    # Check dual-role suitability
    is_dual = p_meta.get('is_dual_role', False)
    
    # ── TAB 1: PLAYER SCOUTING PROFILE ──────────────────────────
    if active_tab == "tab-profile":
        return render_player_profile_tab(selected_player, p_meta, bat_data, bow_data, role_mode, is_dual)
        
    # ── TAB 2: HEAD-TO-HEAD COMPARISON ─────────────────────────
    elif active_tab == "tab-comparison":
        return render_comparison_tab(selected_player, role_mode)

    # ── TAB 3: PRESSURE VS NORMAL CONTEXT ───────────────────────
    elif active_tab == "tab-context":
        return render_context_tab(selected_player, role_mode)

    # ── TAB 4: COMPETITION DYNAMICS ─────────────────────────────
    elif active_tab == "tab-competitions":
        return render_competitions_tab(selected_player, role_mode)

    # ── TAB 5: GPU MACHINE LEARNING INTELLIGENCE ────────────────
    elif active_tab == "tab-ml":
        return render_ml_tab()

    # ── TAB 6: LEADERBOARDS & 19 ROSTER ─────────────────────────
    elif active_tab == "tab-leaderboards":
        return render_leaderboards_tab()

    return html.Div("Tab not found.")


# ════════════════════════════════════════════════════════════════
# 7. TAB RENDERING IMPLEMENTATIONS
# ════════════════════════════════════════════════════════════════

def render_player_profile_tab(player_name, meta, bat, bow, role_mode, is_dual):
    """Renders the comprehensive Player Scouting Profile with 5-axis Radar."""
    is_f19 = meta.get('is_fixed19', False)
    role = meta.get('primary_role', 'Player')
    nat = meta.get('nationality', 'International')
    comps = meta.get('competitions', [])
    teams = meta.get('teams', ['Multi-Franchise'])
    fielding = meta.get('fielding', None)
    
    # Determine effective mode
    if role_mode == 'dual' and not is_dual:
        effective_mode = 'bowling' if (bow and bow.get('p_balls', 0) > bat.get('p_balls', 0)) else 'batting'
    else:
        effective_mode = role_mode
        
    # Build Top Header Banner
    badge_list = [
        html.Span(f"Nationality: {nat}", className="badge-pill-purple me-2"),
        html.Span(f"Role: {role}", className="badge-pill-cyan me-2"),
    ]
    if is_f19:
        badge_list.append(html.Span("★ Challenge 19 Shortlist", className="badge-pill-gold me-2"))
    if is_dual:
        badge_list.append(html.Span("⚡ Verified All-Rounder (Dual Eligible)", className="badge-pill-emerald me-2"))
        
    header_card = dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
        dbc.Row([
            dbc.Col([
                html.Div(badge_list, className="mb-2"),
                html.H2(player_name, style={'color': THEME['white'], 'fontWeight': '800', 'margin': 0}),
                html.P(f"Competitions: {', '.join(comps) if comps else 'None'} | Teams: {', '.join(teams[:3])}", style={'color': THEME['muted'], 'fontSize': '0.85rem', 'marginTop': '4px'})
            ], md=8),
            dbc.Col([
                html.Div(style={'textAlign': 'right'}, children=[
                    html.Div("OVERALL PRESSURE RATING (BPR)", style={'fontSize': '0.75rem', 'color': THEME['muted'], 'fontWeight': '600'}),
                    html.Div(
                        f"{bat.get('bpr', 0.0):.1f}" if effective_mode == 'batting' else f"{bow.get('bpr', 0.0):.1f}",
                        style={'fontSize': '2.4rem', 'fontWeight': '900', 'color': THEME['cyan'] if effective_mode == 'batting' else THEME['purple'], 'fontFamily': 'JetBrains Mono'}
                    ),
                    html.Div(
                        f"Batting Rank: #{int(bat.get('fixed19_rank'))}" if (is_f19 and bat.get('fixed19_rank') and effective_mode == 'batting') else
                        (f"Bowling Rank: #{int(bow.get('fixed19_rank'))}" if (is_f19 and bow.get('fixed19_rank') and effective_mode == 'bowling') else "Dataset Universal Pool"),
                        style={'fontSize': '0.75rem', 'color': THEME['gold'] if is_f19 else THEME['muted']}
                    )
                ])
            ], md=4)
        ])
    ])

    # Check for Pure Batter exclusion in Bowling
    if effective_mode == 'bowling' and bow.get('fixed19_status') == 'Non-Bowler (Excluded)':
        exclusion_warning = dbc.Alert([
            html.H5("⚠️ Role Exclusion Notice", className="alert-heading", style={'fontWeight': '700'}),
            html.P(bow.get('why_selected', 'Excluded from bowling analysis: Primary role is pure batter/keeper.')),
            html.Hr(),
            html.P("To view this player's verified pressure performance, switch the Analysis Mode toggle to 🏏 Batting.", className="mb-0", style={'fontSize': '0.85rem'})
        ], color="warning", className="mb-4")
        return html.Div([header_card, exclusion_warning])

    # ── Build Radar Charts ──────────────────────────────────────
    if effective_mode == 'batting' or (effective_mode == 'dual' and not is_dual):
        categories = ['Batting\nEfficiency', 'Dot-Ball\nControl', 'Pressure\nContribution', 'Consistency', 'Chase & Match\nImpact']
        values = [
            bat.get('radar_dim1_efficiency', 0.0),
            bat.get('radar_dim2_dot_control', 0.0),
            bat.get('radar_dim3_contribution', 0.0),
            bat.get('radar_dim4_consistency', 0.0),
            bat.get('radar_dim5_chase_impact', 0.0)
        ]
        radar_fig = build_radar_chart(categories, {player_name: values}, is_bowling=False)
        radar_title = "BATTER SCOUTING RADAR (5 DIMENSIONS)"
        dim_explanations = [
            ("1. Batting Efficiency", values[0], f"Pressure SR ({bat.get('p_sr', 0.0)}) + Scoring Rate ({bat.get('p_runs', 0)}/{bat.get('p_balls', 0)}b) + ML RAE ({bat.get('rae_per_ball', 0.0):+.2f}/ball)"),
            ("2. Dot-Ball Control", values[1], f"Pressure Dot% ({bat.get('p_dot_pct', 0.0)}%) inverted against T20 reference baseline (fewer dots = higher score)"),
            ("3. Pressure Contribution", values[2], f"Run volume ({bat.get('p_runs', 0)} runs) + SR Delta ({bat.get('delta_sr', 0.0):+.1f} vs Normal) + Extreme pressure share"),
            ("4. Consistency", values[3], f"Match-to-match stability (Coefficient of Variation of innings SR across {bat.get('p_inns', 0)} pressure innings)"),
            ("5. Chase & Match Impact", values[4], f"Innings 2 chase pressure SR ({bat.get('chase_p_sr', 0.0)}) and performance in high RRR contexts")
        ]
        stat_cards = render_batting_kpis(bat)

    elif effective_mode == 'bowling':
        categories = ['Economy\nControl', 'Dot-Ball\nAbility', 'Wicket\nThreat', 'Consistency', 'Pressure\nImpact']
        values = [
            bow.get('radar_dim1_economy', 0.0),
            bow.get('radar_dim2_dot_ability', 0.0),
            bow.get('radar_dim3_wicket_threat', 0.0),
            bow.get('radar_dim4_consistency', 0.0),
            bow.get('radar_dim5_pressure_impact', 0.0)
        ]
        radar_fig = build_radar_chart(categories, {player_name: values}, is_bowling=True)
        radar_title = "BOWLER SCOUTING RADAR (5 DIMENSIONS)"
        dim_explanations = [
            ("1. Economy Control", values[0], f"Pressure Economy ({bow.get('p_economy', 0.0):.2f}) inverted against format baseline of 9.60 rpo (lower eco = higher score)"),
            ("2. Dot-Ball Ability", values[1], f"Pressure Dot% ({bow.get('p_dot_pct', 0.0):.1f}%) scaled against death overs reference distribution"),
            ("3. Wicket Threat", values[2], f"Pressure strike rate ({bow.get('p_bowling_sr', 0.0):.1f} balls/wkt) + Wickets Above Expected WAE ({bow.get('wae_wkts', 0.0):+.1f})"),
            ("4. Consistency", values[3], f"Spell-to-spell stability (CV of match economies across {bow.get('p_inns', 0)} pressure spells)"),
            ("5. Pressure Impact", values[4], f"Runs Saved vs ML Expected ({bow.get('runs_saved', 0.0):+.1f} runs) + Extreme pressure suppression ({bow.get('extreme_economy', 0.0):.2f} rpo)")
        ]
        stat_cards = render_bowling_kpis(bow)

    else: # DUAL ALL-ROUNDER VIEW
        # Return two side-by-side radars!
        bat_cats = ['Batting\nEfficiency', 'Dot-Ball\nControl', 'Pressure\nContribution', 'Consistency', 'Chase & Match\nImpact']
        bat_vals = [bat.get('radar_dim1_efficiency', 0.0), bat.get('radar_dim2_dot_control', 0.0), bat.get('radar_dim3_contribution', 0.0), bat.get('radar_dim4_consistency', 0.0), bat.get('radar_dim5_chase_impact', 0.0)]
        bat_fig = build_radar_chart(bat_cats, {f"{player_name} (Batting)": bat_vals}, is_bowling=False)
        
        bow_cats = ['Economy\nControl', 'Dot-Ball\nAbility', 'Wicket\nThreat', 'Consistency', 'Pressure\nImpact']
        bow_vals = [bow.get('radar_dim1_economy', 0.0), bow.get('radar_dim2_dot_ability', 0.0), bow.get('radar_dim3_wicket_threat', 0.0), bow.get('radar_dim4_consistency', 0.0), bow.get('radar_dim5_pressure_impact', 0.0)]
        bow_fig = build_radar_chart(bow_cats, {f"{player_name} (Bowling)": bow_vals}, is_bowling=True)
        
        return html.Div([
            header_card,
            dbc.Row([
                dbc.Col([
                    dbc.Card(className="dash-card mb-4", style={'padding': '16px'}, children=[
                        html.H5("🏏 BATTING RADAR PROFILE", style={'color': THEME['cyan'], 'fontWeight': '700', 'fontSize': '0.95rem'}),
                        dcc.Graph(figure=bat_fig, config={'displayModeBar': False}),
                        render_batting_kpis(bat)
                    ])
                ], md=6),
                dbc.Col([
                    dbc.Card(className="dash-card mb-4", style={'padding': '16px'}, children=[
                        html.H5("🎯 BOWLING RADAR PROFILE", style={'color': THEME['purple'], 'fontWeight': '700', 'fontSize': '0.95rem'}),
                        dcc.Graph(figure=bow_fig, config={'displayModeBar': False}),
                        render_bowling_kpis(bow)
                    ])
                ], md=6)
            ]),
            render_fielding_card(fielding)
        ])

    # Sample warning if small sample
    sample_conf = bat.get('sample_confidence') if effective_mode == 'batting' else bow.get('sample_confidence')
    sample_alert = None
    if 'Insufficient' in str(sample_conf) or 'Provisional' in str(sample_conf):
        sample_alert = dbc.Alert([
            html.Span("ℹ️ Sample Size Notice: ", style={'fontWeight': '700'}),
            html.Span(f"{sample_conf}. Radar dimensions represent preliminary indications and should be evaluated with confidence penalties.")
        ], color="info", className="mb-3", style={'fontSize': '0.82rem'})

    return html.Div([
        header_card,
        sample_alert if sample_alert else html.Div(),
        dbc.Row([
            # Left: Radar Chart
            dbc.Col([
                dbc.Card(className="dash-card mb-4", style={'padding': '18px'}, children=[
                    html.H5(radar_title, style={'color': THEME['white'], 'fontWeight': '700', 'fontSize': '0.95rem', 'marginBottom': '10px'}),
                    dcc.Graph(figure=radar_fig, config={'displayModeBar': False})
                ])
            ], md=6, xs=12),

            # Right: Dimension Breakdown with explanations
            dbc.Col([
                dbc.Card(className="dash-card mb-4", style={'padding': '18px'}, children=[
                    html.H5("RADAR DIMENSION BREAKDOWN (0 - 100 SCALE)", style={'color': THEME['white'], 'fontWeight': '700', 'fontSize': '0.95rem', 'marginBottom': '14px'}),
                    html.Div([
                        html.Div(className="mb-3", children=[
                            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '4px'}, children=[
                                html.Span(dim[0], style={'fontWeight': '600', 'fontSize': '0.85rem', 'color': THEME['white']}),
                                html.Span(f"{dim[1]:.1f} / 100", style={'fontWeight': '700', 'fontSize': '0.85rem', 'color': THEME['cyan'] if effective_mode == 'batting' else THEME['purple'], 'fontFamily': 'JetBrains Mono'})
                            ]),
                            dbc.Progress(value=dim[1], max=100, color="info" if effective_mode == 'batting' else "warning", style={'height': '7px', 'backgroundColor': '#1f293d'}),
                            html.Small(dim[2], style={'color': THEME['muted'], 'fontSize': '0.72rem', 'display': 'block', 'marginTop': '3px'})
                        ]) for dim in dim_explanations
                    ])
                ])
            ], md=6, xs=12)
        ]),

        # Statistics Cards
        stat_cards,
        
        # Fielding Card
        render_fielding_card(fielding)
    ])


def render_batting_kpis(bat):
    """Renders batting KPI card grid."""
    return html.Div([
        html.H5("BATTING PERFORMANCE & ML RESIDUALS", style={'color': THEME['white'], 'fontWeight': '700', 'fontSize': '0.95rem', 'marginBottom': '12px'}),
        dbc.Row([
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("PRESSURE RUNS", className="kpi-title"),
                    html.Div(f"{bat.get('p_runs', 0):,}", className="kpi-value"),
                    html.Div(f"Faced {bat.get('p_balls', 0)} balls ({bat.get('p_inns', 0)} inns)", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("PRESSURE STRIKE RATE", className="kpi-title"),
                    html.Div(f"{bat.get('p_sr', 0.0):.1f}", className="kpi-value", style={'color': THEME['cyan']}),
                    html.Div(f"Normal SR: {bat.get('n_sr', 0.0):.1f} (Δ {bat.get('delta_sr', 0.0):+.1f})", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("PRESSURE AVERAGE", className="kpi-title"),
                    html.Div(f"{bat.get('p_avg', 0.0):.1f}", className="kpi-value"),
                    html.Div(f"Normal Avg: {bat.get('n_avg', 0.0):.1f}", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("DOT % / BOUNDARY %", className="kpi-title"),
                    html.Div(f"{bat.get('p_dot_pct', 0.0):.1f}% / {bat.get('p_bdry_pct', 0.0):.1f}%", className="kpi-value", style={'fontSize': '1.35rem'}),
                    html.Div("Under High & Extreme Pressure", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("ML RUNS ABOVE EXPECTED (RAE)", className="kpi-title"),
                    html.Div(f"{bat.get('rae_runs', 0.0):+.1f}", className="kpi-value", style={'color': THEME['emerald'] if bat.get('rae_runs', 0.0) >= 0 else THEME['coral']}),
                    html.Div(f"Actual {bat.get('p_runs', 0)} vs Expected {bat.get('exp_p_runs', 0.0):.1f} xRuns", className="kpi-subtitle")
                ])
            ], md=4, xs=12),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("EXTREME PRESSURE METRICS", className="kpi-title"),
                    html.Div(f"{bat.get('extreme_runs', 0)} Runs @ {bat.get('extreme_sr', 0.0):.1f} SR", className="kpi-value", style={'fontSize': '1.35rem'}),
                    html.Div(f"Extreme sample: {bat.get('extreme_balls', 0)} balls", className="kpi-subtitle")
                ])
            ], md=4, xs=12),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("INNINGS 2 CHASE PRESSURE", className="kpi-title"),
                    html.Div(f"{bat.get('chase_p_runs', 0)} Runs @ {bat.get('chase_p_sr', 0.0):.1f} SR", className="kpi-value", style={'fontSize': '1.35rem'}),
                    html.Div(f"Chase sample: {bat.get('chase_p_balls', 0)} balls", className="kpi-subtitle")
                ])
            ], md=4, xs=12),
        ])
    ])


def render_bowling_kpis(bow):
    """Renders bowling KPI card grid."""
    return html.Div([
        html.H5("BOWLING PERFORMANCE & ML RESIDUALS", style={'color': THEME['white'], 'fontWeight': '700', 'fontSize': '0.95rem', 'marginBottom': '12px'}),
        dbc.Row([
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("PRESSURE OVERS & BALLS", className="kpi-title"),
                    html.Div(f"{bow.get('p_overs', 0.0):.1f} ov", className="kpi-value"),
                    html.Div(f"{bow.get('p_balls', 0)} legal balls ({bow.get('p_inns', 0)} spells)", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("PRESSURE ECONOMY", className="kpi-title"),
                    html.Div(f"{bow.get('p_economy', 0.0):.2f}", className="kpi-value", style={'color': THEME['purple']}),
                    html.Div(f"Normal Eco: {bow.get('n_economy', 0.0):.2f} (Δ {bow.get('delta_eco', 0.0):+.2f})", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("PRESSURE WICKETS", className="kpi-title"),
                    html.Div(f"{int(bow.get('p_wickets', 0))} wkts", className="kpi-value"),
                    html.Div(f"SR: {bow.get('p_bowling_sr', 0.0):.1f} balls/wkt", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("PRESSURE DOT% / BDRY%", className="kpi-title"),
                    html.Div(f"{bow.get('p_dot_pct', 0.0):.1f}% / {bow.get('p_bdry_pct', 0.0):.1f}%", className="kpi-value", style={'fontSize': '1.35rem'}),
                    html.Div(f"Total conceded: {bow.get('p_runs_conceded', 0)} runs", className="kpi-subtitle")
                ])
            ], md=3, xs=6),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("ML RUNS SAVED VS EXPECTED", className="kpi-title"),
                    html.Div(f"{bow.get('runs_saved', 0.0):+.1f}", className="kpi-value", style={'color': THEME['emerald'] if bow.get('runs_saved', 0.0) >= 0 else THEME['coral']}),
                    html.Div(f"Conceded {bow.get('p_runs_conceded', 0)} vs Expected {bow.get('exp_p_runs', 0.0):.1f} xRuns", className="kpi-subtitle")
                ])
            ], md=4, xs=12),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("WICKETS ABOVE EXPECTED (WAE)", className="kpi-title"),
                    html.Div(f"{bow.get('wae_wkts', 0.0):+.2f} Wkts", className="kpi-value", style={'fontSize': '1.35rem'}),
                    html.Div(f"Actual {bow.get('p_wickets', 0)} vs Expected {bow.get('exp_p_wkts', 0.0):.2f}", className="kpi-subtitle")
                ])
            ], md=4, xs=12),
            dbc.Col([
                html.Div(className="kpi-card mb-3", children=[
                    html.Div("EXTREME PRESSURE SUPPRESSION", className="kpi-title"),
                    html.Div(f"{bow.get('extreme_economy', 0.0):.2f} RPO", className="kpi-value", style={'fontSize': '1.35rem'}),
                    html.Div(f"{bow.get('extreme_balls', 0)} balls, {bow.get('extreme_wickets', 0)} wkts", className="kpi-subtitle")
                ])
            ], md=4, xs=12),
        ])
    ])


def render_fielding_card(fielding):
    """Renders fielding stats or honest missing data disclosure."""
    if fielding:
        content = html.Div([
            html.Div(f"Catches Recorded: {fielding['catches']} | Catches Per Match (CPM): {fielding['cpm']:.2f} | IPL Rank: #{fielding['rank']}", style={'color': THEME['emerald'], 'fontWeight': '600'})
        ])
    else:
        content = html.Div([
            html.Span("ℹ️ Multi-League Delivery Note: ", style={'fontWeight': '600', 'color': THEME['muted']}),
            html.Span("Individual fielder identities for catches and run-outs are not recorded in standard Cricsheet ball-by-ball formats. Fielding data presented where official competition summaries exist.", style={'color': THEME['muted'], 'fontSize': '0.8rem'})
        ])
        
    return dbc.Card(className="dash-card mb-4", style={'padding': '14px 20px'}, children=[
        html.H6("FIELDING CONTRIBUTIONS (AUDIT CHECK)", style={'color': THEME['white'], 'fontWeight': '700', 'fontSize': '0.85rem', 'marginBottom': '6px'}),
        content
    ])


# ── TAB 2: HEAD-TO-HEAD COMPARISON ─────────────────────────────
def render_comparison_tab(default_player, role_mode):
    is_bowling = (role_mode == 'bowling')
    default_compare = ['Virat Kohli', 'Rohit Sharma', 'Dinesh Karthik'] if not is_bowling else ['Dwayne Bravo', 'Ravindra Jadeja', 'Rashid Khan']
    
    return html.Div([
        dbc.Card(className="dash-card mb-4", style={'padding': '18px'}, children=[
            html.H5("⚔️ MULTI-PLAYER HEAD-TO-HEAD SCOUTING COMPARISON", style={'color': THEME['white'], 'fontWeight': '700'}),
            html.P("Select up to 4 players to overlay their radar pentagons and compare side-by-side performance dimensions on a normalized 0-100 scale.", style={'color': THEME['muted'], 'fontSize': '0.85rem'}),
            dbc.Row([
                dbc.Col([
                    dcc.Dropdown(
                        id='compare-players-dropdown',
                        options=[{'label': f"★ {p}" if player_registry[p].get('is_fixed19') else p, 'value': p} for p in ALL_PLAYERS],
                        value=default_compare,
                        multi=True,
                        style={'backgroundColor': THEME['card'], 'color': '#000'}
                    )
                ], md=9, xs=12),
                dbc.Col([
                    dbc.RadioItems(
                        id='compare-mode-toggle',
                        options=[
                            {'label': ' 🏏 Compare Batting', 'value': 'batting'},
                            {'label': ' 🎯 Compare Bowling', 'value': 'bowling'}
                        ],
                        value='bowling' if is_bowling else 'batting',
                        inline=True,
                        style={'color': THEME['white'], 'marginTop': '6px', 'fontSize': '0.85rem'}
                    )
                ], md=3, xs=12)
            ])
        ]),

        html.Div(id='comparison-results-area')
    ])

@app.callback(
    Output('comparison-results-area', 'children'),
    [Input('compare-players-dropdown', 'value'),
     Input('compare-mode-toggle', 'value')]
)
def update_comparison_view(selected_players, compare_mode):
    if not selected_players:
        return dbc.Alert("Please select at least one player to compare.", color="secondary")
        
    is_bow = (compare_mode == 'bowling')
    dataset = bow_dict if is_bow else bat_dict
    
    categories = ['Economy Control', 'Dot-Ball Ability', 'Wicket Threat', 'Consistency', 'Pressure Impact'] if is_bow else \
                 ['Batting Efficiency', 'Dot-Ball Control', 'Pressure Contribution', 'Consistency', 'Chase & Match Impact']
                 
    values_dict = {}
    table_rows = []
    
    for p in selected_players[:4]:
        record = dataset.get(p, {})
        if is_bow:
            vals = [
                record.get('radar_dim1_economy', 0.0),
                record.get('radar_dim2_dot_ability', 0.0),
                record.get('radar_dim3_wicket_threat', 0.0),
                record.get('radar_dim4_consistency', 0.0),
                record.get('radar_dim5_pressure_impact', 0.0)
            ]
            stat_summary = f"{record.get('p_overs', 0.0):.1f} ov | {record.get('p_economy', 0.0):.2f} eco | {int(record.get('p_wickets', 0))} wkts | RS: {record.get('runs_saved', 0.0):+.1f}"
        else:
            vals = [
                record.get('radar_dim1_efficiency', 0.0),
                record.get('radar_dim2_dot_control', 0.0),
                record.get('radar_dim3_contribution', 0.0),
                record.get('radar_dim4_consistency', 0.0),
                record.get('radar_dim5_chase_impact', 0.0)
            ]
            stat_summary = f"{record.get('p_runs', 0)} runs @ {record.get('p_sr', 0.0):.1f} SR | Dot: {record.get('p_dot_pct', 0.0):.1f}% | RAE: {record.get('rae_runs', 0.0):+.1f}"
            
        values_dict[p] = vals
        bpr = record.get('bpr', 0.0)
        
        table_rows.append(html.Tr([
            html.Td(p, style={'fontWeight': '700', 'color': THEME['white']}),
            html.Td(f"{bpr:.1f}", style={'fontWeight': '800', 'color': THEME['cyan'] if not is_bow else THEME['purple'], 'fontFamily': 'JetBrains Mono'}),
            html.Td(f"{vals[0]:.1f}"),
            html.Td(f"{vals[1]:.1f}"),
            html.Td(f"{vals[2]:.1f}"),
            html.Td(f"{vals[3]:.1f}"),
            html.Td(f"{vals[4]:.1f}"),
            html.Td(stat_summary, style={'fontSize': '0.78rem', 'color': THEME['muted']})
        ]))

    radar_fig = build_radar_chart(categories, values_dict, is_bowling=is_bow)
    
    # Differential Bar Chart
    df_diff = pd.DataFrame()
    for p, vals in values_dict.items():
        sub_df = pd.DataFrame({'Player': p, 'Dimension': categories, 'Score': vals})
        df_diff = pd.concat([df_diff, sub_df])
        
    bar_fig = px.bar(
        df_diff, x='Dimension', y='Score', color='Player', barmode='group',
        color_discrete_sequence=['#00e5ff', '#ff3366', '#f59e0b', '#10b981'],
        height=380
    )
    bar_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.6)',
        font=dict(color=THEME['white']),
        margin=dict(l=40, r=40, t=30, b=40),
        legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center")
    )

    return html.Div([
        dbc.Row([
            dbc.Col([
                dbc.Card(className="dash-card mb-4", style={'padding': '18px'}, children=[
                    html.H6("OVERLAID SCOUTING RADARS", style={'color': THEME['white'], 'fontWeight': '700'}),
                    dcc.Graph(figure=radar_fig, config={'displayModeBar': False})
                ])
            ], md=6, xs=12),
            dbc.Col([
                dbc.Card(className="dash-card mb-4", style={'padding': '18px'}, children=[
                    html.H6("DIMENSION METRIC COMPARISON", style={'color': THEME['white'], 'fontWeight': '700'}),
                    dcc.Graph(figure=bar_fig, config={'displayModeBar': False})
                ])
            ], md=6, xs=12)
        ]),

        # Head-to-Head Table
        dbc.Card(className="dash-card mb-4", style={'padding': '18px'}, children=[
            html.H6("HEAD-TO-HEAD STATISTICAL BREAKDOWN", style={'color': THEME['white'], 'fontWeight': '700', 'marginBottom': '12px'}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Player Name"),
                    html.Th("Overall BPR"),
                    html.Th(categories[0]),
                    html.Th(categories[1]),
                    html.Th(categories[2]),
                    html.Th(categories[3]),
                    html.Th(categories[4]),
                    html.Th("Key Pressure Summary")
                ])),
                html.Tbody(table_rows)
            ], responsive=True, hover=True, className="table-dark")
        ])
    ])


# ── TAB 3: PRESSURE VS NORMAL CONTEXT ──────────────────────────
def render_context_tab(selected_player, role_mode):
    is_bow = (role_mode == 'bowling')
    
    if is_bow:
        # Scatter: Normal Economy vs Pressure Economy
        valid = bow_df[(bow_df['p_balls'] >= 30) & (bow_df['n_balls'] >= 30) & (bow_df['fixed19_status'] != 'Non-Bowler (Excluded)')].copy()
        fig = px.scatter(
            valid, x='n_economy', y='p_economy', size='p_balls', color='bpr',
            hover_name='player_name',
            hover_data={'n_economy': ':.2f', 'p_economy': ':.2f', 'p_balls': True, 'runs_saved': ':.1f', 'bpr': ':.1f'},
            color_continuous_scale='Purples',
            title="Normal Economy vs Pressure Economy (Lower Pressure Eco = Elite Death Suppression)",
            labels={'n_economy': 'Normal Baseline Economy (RPO)', 'p_economy': 'Pressure Economy (RPO)', 'bpr': 'BPR /100'},
            height=460
        )
        # Add diagonal y=x line
        fig.add_trace(go.Scatter(x=[5, 14], y=[5, 14], mode='lines', line=dict(dash='dash', color='rgba(255,255,255,0.3)'), name='Parity (y=x)'))
        
        # Highlight selected player
        p_row = bow_df[bow_df['player_name'] == selected_player]
        if len(p_row) > 0 and p_row.iloc[0]['p_balls'] >= 5:
            r0 = p_row.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r0['n_economy']], y=[r0['p_economy']], mode='markers+text',
                marker=dict(size=16, color=THEME['coral'], symbol='star', line=dict(color='#fff', width=2)),
                text=[f"★ {selected_player}"], textposition="top center", name=selected_player
            ))
            
    else:
        # Scatter: Normal SR vs Pressure SR
        valid = bat_df[(bat_df['p_balls'] >= 30) & (bat_df['n_balls'] >= 30)].copy()
        fig = px.scatter(
            valid, x='n_sr', y='p_sr', size='p_balls', color='bpr',
            hover_name='player_name',
            hover_data={'n_sr': ':.1f', 'p_sr': ':.1f', 'p_balls': True, 'rae_runs': ':.1f', 'bpr': ':.1f'},
            color_continuous_scale='Viridis',
            title="Normal Strike Rate vs Pressure Strike Rate (Points Above Diagonal = Pressure Uplift)",
            labels={'n_sr': 'Normal Baseline Strike Rate', 'p_sr': 'Pressure Strike Rate', 'bpr': 'BPR /100'},
            height=460
        )
        fig.add_trace(go.Scatter(x=[80, 220], y=[80, 220], mode='lines', line=dict(dash='dash', color='rgba(255,255,255,0.3)'), name='Parity (y=x)'))
        
        p_row = bat_df[bat_df['player_name'] == selected_player]
        if len(p_row) > 0 and p_row.iloc[0]['p_balls'] >= 5:
            r0 = p_row.iloc[0]
            fig.add_trace(go.Scatter(
                x=[r0['n_sr']], y=[r0['p_sr']], mode='markers+text',
                marker=dict(size=16, color=THEME['coral'], symbol='star', line=dict(color='#fff', width=2)),
                text=[f"★ {selected_player}"], textposition="top center", name=selected_player
            ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(15, 23, 42, 0.7)',
        font=dict(color=THEME['white']),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    return html.Div([
        dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
            html.H5("⚡ CONTEXTUAL QUADRANT ANALYSIS: PRESSURE VS NORMAL BASELINE", style={'color': THEME['white'], 'fontWeight': '700'}),
            html.P("Empirical validation of how players respond when match context shifts from Low/Moderate to High/Extreme pressure.", style={'color': THEME['muted'], 'fontSize': '0.85rem'}),
            dcc.Graph(figure=fig, config={'displayModeBar': False})
        ])
    ])


# ── TAB 4: COMPETITION DYNAMICS ─────────────────────────────────
def render_competitions_tab(selected_player, role_mode):
    role_type = 'Bowler' if role_mode == 'bowling' else 'Batter'
    p_slices = cs_df[(cs_df['player_name'] == selected_player) & (cs_df['role_type'] == role_type)].copy()
    
    if len(p_slices) == 0:
        return dbc.Alert(f"No multi-league tournament slices recorded for {selected_player} as a {role_type}.", color="info")

    comp_fig = px.bar(
        p_slices, x='competition', y='p_runs' if role_type == 'Batter' else 'runs',
        color='season', barmode='group',
        title=f"{selected_player} — Deliveries & Performance Across Leagues (2022-2026)",
        labels={'competition': 'Competition', 'p_runs': 'Pressure Runs', 'runs': 'Runs Conceded'},
        height=380
    )
    comp_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15, 23, 42, 0.65)',
        font=dict(color=THEME['white']), margin=dict(l=40, r=40, t=40, b=40)
    )

    # Detailed Table
    table_rows = []
    for _, row in p_slices.iterrows():
        table_rows.append(html.Tr([
            html.Td(row['competition'], style={'fontWeight': '600', 'color': THEME['white']}),
            html.Td(str(row['season'])),
            html.Td(str(row['balls'])),
            html.Td(str(row['runs'])),
            html.Td(f"{row['sr']:.1f}" if role_type == 'Batter' else f"{row['sr']:.2f}"),
            html.Td(str(row['p_balls'])),
            html.Td(str(row['p_runs'])),
            html.Td(f"{row['p_sr']:.1f}" if role_type == 'Batter' else f"{row['p_sr']:.2f}"),
            html.Td(f"{row['exp_runs']:.1f}", style={'color': THEME['cyan']})
        ]))

    return html.Div([
        dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
            html.H5(f"🌍 {selected_player.upper()} — MULTI-LEAGUE BREAKDOWN", style={'color': THEME['white'], 'fontWeight': '700'}),
            dcc.Graph(figure=comp_fig, config={'displayModeBar': False})
        ]),
        dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
            html.H6("DETAILED LEAGUE & SEASON SLICES", style={'color': THEME['white'], 'fontWeight': '700', 'marginBottom': '12px'}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Competition"), html.Th("Season"), html.Th("Total Balls"), html.Th("Total Runs"),
                    html.Th("Overall SR/Eco"), html.Th("Pressure Balls"), html.Th("Pressure Runs"),
                    html.Th("Pressure SR/Eco"), html.Th("ML Expected Runs")
                ])),
                html.Tbody(table_rows)
            ], responsive=True, hover=True, className="table-dark")
        ])
    ])


# ── TAB 5: GPU MACHINE LEARNING INTELLIGENCE ───────────────────
def render_ml_tab():
    feat_df = pd.DataFrame(ml_intel['top_features'])
    feat_fig = px.bar(
        feat_df, x='importance', y='label', orientation='h',
        title="NVIDIA GPU XGBoost — Feature Importance (Gain Metric)",
        labels={'importance': 'Feature Gain Importance', 'label': 'Feature Name'},
        color='importance', color_continuous_scale='Tealgrn', height=420
    )
    feat_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(15, 23, 42, 0.65)',
        font=dict(color=THEME['white']), yaxis=dict(autorange="reversed"),
        margin=dict(l=40, r=40, t=40, b=40)
    )

    models_list = ml_intel.get('models', [])
    model_cards = []
    for m in models_list:
        model_cards.append(dbc.Col([
            dbc.Card(className="kpi-card mb-3", children=[
                html.Div(m['task'], className="kpi-title", style={'color': THEME['cyan']}),
                html.Div(f"RMSE: {m.get('test_rmse', m.get('brier_score', 'N/A'))}", className="kpi-value", style={'fontSize': '1.35rem'}),
                html.Div(m['algorithm'], className="kpi-subtitle", style={'marginTop': '4px'}),
                html.Small(m.get('rmse_improvement', f"Log-Loss: {m.get('log_loss', '')}"), style={'color': THEME['emerald'], 'display': 'block', 'marginTop': '4px'})
            ])
        ], md=4, xs=12))

    return html.Div([
        dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
            html.H5("🤖 GPU-ACCELERATED MACHINE LEARNING ARCHITECTURE", style={'color': THEME['white'], 'fontWeight': '700'}),
            html.P(f"Hardware: {ml_intel.get('hardware')} | Scope: {ml_intel.get('dataset_scope')}", style={'color': THEME['muted'], 'fontSize': '0.85rem'}),
            dbc.Row(model_cards)
        ]),
        dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
            html.H5("MODEL INTERPRETABILITY & SHAP FEATURE IMPORTANCE", style={'color': THEME['white'], 'fontWeight': '700'}),
            dcc.Graph(figure=feat_fig, config={'displayModeBar': False})
        ])
    ])


# ── TAB 6: MASTER LEADERBOARDS & CHALLENGE 19 ROSTER ───────────
def render_leaderboards_tab():
    # 19 Nominated Challenge Table
    bat_f19 = bat_df[bat_df['is_fixed19']].sort_values('fixed19_rank').reset_index(drop=True)
    bow_f19 = bow_df[bow_df['is_fixed19']].sort_values('fixed19_rank').reset_index(drop=True)

    bat_f19_rows = []
    for _, r in bat_f19.iterrows():
        bat_f19_rows.append(html.Tr([
            html.Td(f"#{int(r['fixed19_rank'])}" if pd.notna(r['fixed19_rank']) else "-", style={'fontWeight': '700', 'color': THEME['cyan']}),
            html.Td(r['player_name'], style={'fontWeight': '700', 'color': THEME['white']}),
            html.Td(r['fixed19_status'], style={'color': THEME['emerald'] if r['fixed19_status'] == 'Eligible' else THEME['gold']}),
            html.Td(str(r['p_balls'])),
            html.Td(str(r['p_runs'])),
            html.Td(f"{r['p_sr']:.1f}"),
            html.Td(f"{r['rae_runs']:+.1f}", style={'color': THEME['emerald'] if r['rae_runs'] >= 0 else THEME['coral']}),
            html.Td(f"{r['bpr']:.2f}", style={'fontWeight': '800', 'color': THEME['cyan'], 'fontFamily': 'JetBrains Mono'}),
            html.Td(r.get('why_selected', ''), style={'fontSize': '0.75rem', 'color': THEME['muted']})
        ]))

    bow_f19_rows = []
    for _, r in bow_f19.iterrows():
        bow_f19_rows.append(html.Tr([
            html.Td(f"#{int(r['fixed19_rank'])}" if pd.notna(r['fixed19_rank']) else "-", style={'fontWeight': '700', 'color': THEME['purple']}),
            html.Td(r['player_name'], style={'fontWeight': '700', 'color': THEME['white']}),
            html.Td(r['fixed19_status'], style={'color': THEME['emerald'] if r['fixed19_status'] == 'Eligible' else (THEME['coral'] if 'Excluded' in str(r['fixed19_status']) else THEME['gold'])}),
            html.Td(f"{r['p_overs']:.1f}"),
            html.Td(str(int(r['p_wickets']))),
            html.Td(f"{r['p_economy']:.2f}"),
            html.Td(f"{r['runs_saved']:+.1f}", style={'color': THEME['emerald'] if r['runs_saved'] >= 0 else THEME['coral']}),
            html.Td(f"{r['bpr']:.2f}", style={'fontWeight': '800', 'color': THEME['purple'], 'fontFamily': 'JetBrains Mono'}),
            html.Td(r.get('why_selected', ''), style={'fontSize': '0.75rem', 'color': THEME['muted']})
        ]))

    return html.Div([
        dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
            html.H5("★ CHALLENGE SHORTLIST: TOP 10 BATTERS UNDER PRESSURE", style={'color': THEME['cyan'], 'fontWeight': '700'}),
            html.P("Locked 19-player evaluation strictly verified against research paper and Excel deliverable 01_Top_10_Batsmen.xlsx.", style={'color': THEME['muted'], 'fontSize': '0.85rem'}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Rank"), html.Th("Player Name"), html.Th("Status"), html.Th("Pressure Balls"),
                    html.Th("Pressure Runs"), html.Th("Pressure SR"), html.Th("ML RAE"), html.Th("BPR /100"), html.Th("Scouting Justification")
                ])),
                html.Tbody(bat_f19_rows)
            ], responsive=True, hover=True, className="table-dark")
        ]),

        dbc.Card(className="dash-card mb-4", style={'padding': '20px'}, children=[
            html.H5("★ CHALLENGE SHORTLIST: TOP 10 BOWLERS UNDER PRESSURE", style={'color': THEME['purple'], 'fontWeight': '700'}),
            html.P("Role-gated bowling evaluation verified against 02_Top_10_Bowlers.xlsx (Pure batters excluded from bowling rankings).", style={'color': THEME['muted'], 'fontSize': '0.85rem'}),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Rank"), html.Th("Bowler Name"), html.Th("Status"), html.Th("Pressure Overs"),
                    html.Th("Wkts"), html.Th("Pressure Economy"), html.Th("Runs Saved"), html.Th("BPR /100"), html.Th("Scouting Justification")
                ])),
                html.Tbody(bow_f19_rows)
            ], responsive=True, hover=True, className="table-dark")
        ])
    ])


# ════════════════════════════════════════════════════════════════
# 8. WSGI EXPORTS FOR VERCEL, GUNICORN & PRODUCTION RUNTIMES
# ════════════════════════════════════════════════════════════════
dash_app = app
server = app.server
handler = server
app = server  # for Vercel WSGI entrypoint

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    print(f"\n⚡ Starting Cricket Pressure Intelligence Dashboard on http://127.0.0.1:{port}")
    dash_app.run(debug=False, host='127.0.0.1', port=port)

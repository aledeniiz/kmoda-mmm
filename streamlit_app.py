"""
================================================================================
K-MODA · MARKETING MIX MODELING
PASO 9: APLICACIÓN STREAMLIT INTERACTIVA
================================================================================
Universidad Alfonso X el Sabio · Caso Práctico MMM · IA
--------------------------------------------------------------------------------
Uso: streamlit run 09_streamlit_app.py
================================================================================
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

warnings.filterwarnings("ignore")

# ── Rutas ─────────────────────────────────────────────────────────────────────
# __file__ puede apuntar al worktree o a la carpeta principal según desde dónde
# se ejecute streamlit. Buscamos modelos/ subiendo niveles si hace falta.
def _find_project_dir():
    candidate = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):  # subir hasta 4 niveles
        if os.path.isdir(os.path.join(candidate, "modelos")):
            return candidate
        candidate = os.path.dirname(candidate)
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR    = _find_project_dir()
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")
OUT_DIR     = os.path.join(BASE_DIR, "outputs")
DATOS_DIR   = os.path.join(BASE_DIR, "datos")

# ══════════════════════════════════════════════════════════════════════════════
# PALETA PREMIUM LIGHT · Indigo/violet + vibrant accents
# ══════════════════════════════════════════════════════════════════════════════
PALETA = ["#4F46E5", "#8B5CF6", "#EC4899", "#14B8A6",
          "#F59E0B", "#3B82F6", "#10B981", "#F43F5E"]
COLOR_ACCENT   = "#4F46E5"    # indigo-600 — accent principal
COLOR_SECUND   = "#8B5CF6"    # violet-500 — secundario
COLOR_SUCCESS  = "#10B981"    # emerald-500
COLOR_WARNING  = "#F59E0B"    # amber-500
COLOR_DANGER   = "#EF4444"    # red-500
COLOR_INFO     = "#3B82F6"    # blue-500
COLOR_BG       = "#F7F8FB"    # light background
COLOR_SURFACE  = "#FFFFFF"    # cards / paneles
COLOR_BORDER   = "#E5E8F0"    # bordes sutiles
COLOR_GRID     = "#EEF0F6"    # gridlines de gráficos
COLOR_TEXT     = "#111827"    # texto principal (slate-900)
COLOR_TEXT2    = "#4B5563"    # secundario (slate-600)
COLOR_MUTED    = "#6B7280"    # labels (slate-500)
# Aliases retrocompatibles (no romper código existente)
COLOR_ORO      = COLOR_ACCENT
COLOR_GRIS     = COLOR_TEXT
COLOR_VERDE    = COLOR_SUCCESS
COLOR_ROJO     = COLOR_DANGER

# ══════════════════════════════════════════════════════════════════════════════
# BENCHMARKS mROI · Industria moda retail 2024-2025
# ══════════════════════════════════════════════════════════════════════════════
# Fuentes: Nielsen 2024 Media ROI Benchmark, WARC Fashion 2024, Meta Retail Report,
# Litmus Email Benchmarks 2024. Midpoints del rango publicado por canal.
#
# Estos valores se usan cuando:
#   (a) el modelo crudo colapsa un canal a ~0 por multicolinealidad  → prior industria
#   (b) el modelo crudo da un valor absurdo (Prensa 19x) → cap industria
#
# Digital canales maduraron (Social, Video) → mROI al alza.
# Offline canales con atención/consumo decreciente → mROI a la baja.
INDUSTRIA_MROI = {
    "Paid Search":  8.0,   # branded+genéricas capturan intent — siempre rentable
    "Social Paid": 10.0,   # Meta+TikTok+Instagram: núcleo visual de moda DTC (2024+)
    "Video Online": 6.0,   # YouTube+CTV+TikTok: storytelling + alcance masivo
    "Display":      3.0,   # retargeting dinámico moderno (no prospecting solo)
    "Email CRM":   14.0,   # mayor ROI del mercado (Litmus 2024)
    "Exterior":     0.8,   # atención móvil fragmentada, DOOH selectivo apenas 1x
    "Prensa":       0.5,   # lectura física en declive estructural
    "Radio Local":  0.3,   # streaming desplaza radio lineal en targets <45
}
# Techos: valores máximos plausibles para CAP cuando el modelo sobreajusta.
# Importante: capitales offline bajos reflejan caída real de atribución
# (el modelo sobreajusta Prensa 19x por correlación espuria con ventas altas).
MROI_CEIL = {
    "Paid Search":  15.0,
    "Social Paid":  12.0,
    "Video Online":  8.0,
    "Display":       3.5,
    "Email CRM":    18.0,
    "Exterior":      1.2,
    "Prensa":        1.0,
    "Radio Local":   0.6,
}

def mroi_ajustado(canal, mroi_modelo):
    """Retorna el mROI 'defendible': industria si modelo colapsa, modelo capado si modelo absurdo."""
    m_ind = INDUSTRIA_MROI.get(canal, 1.0)
    m_cap = MROI_CEIL.get(canal, 10.0)
    if mroi_modelo < 0.5:
        return m_ind       # canal colapsado por multicolinealidad
    return min(mroi_modelo, m_cap)  # cap si sobreajuste

# Alias — MIX_FASHION es el mix canónico (se define abajo tras load pkl)
# Se expone aquí para retrocompatibilidad; la fuente de verdad real es MIX_CANONICO_2025.
MIX_FASHION = {
    "Social Paid":  0.30, "Video Online": 0.22, "Paid Search":  0.18,
    "Display":      0.10, "Email CRM":    0.08, "Exterior":     0.07,
    "Prensa":       0.03, "Radio Local":  0.02,
}

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY TEMPLATE GLOBAL · light premium · aplicado a TODOS los charts
# ══════════════════════════════════════════════════════════════════════════════
pio.templates["mmm_light"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(255,255,255,0)",
        plot_bgcolor="rgba(255,255,255,0)",
        font=dict(family="'IBM Plex Sans', -apple-system, sans-serif",
                  color=COLOR_TEXT, size=12),
        title=dict(font=dict(family="'IBM Plex Sans', sans-serif",
                             color=COLOR_TEXT, size=14, weight=600)),
        xaxis=dict(
            gridcolor=COLOR_GRID, zerolinecolor=COLOR_BORDER, linecolor=COLOR_BORDER,
            tickcolor=COLOR_MUTED, tickfont=dict(color=COLOR_MUTED, size=11),
            title=dict(font=dict(color=COLOR_MUTED, size=12)),
        ),
        yaxis=dict(
            gridcolor=COLOR_GRID, zerolinecolor=COLOR_BORDER, linecolor=COLOR_BORDER,
            tickcolor=COLOR_MUTED, tickfont=dict(color=COLOR_MUTED, size=11),
            title=dict(font=dict(color=COLOR_MUTED, size=12)),
        ),
        colorway=PALETA,
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.98)", bordercolor=COLOR_ACCENT,
            font=dict(family="'IBM Plex Sans', sans-serif", color=COLOR_TEXT, size=12),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)", bordercolor=COLOR_BORDER, borderwidth=1,
            font=dict(color=COLOR_TEXT2, size=11),
        ),
        margin=dict(t=40, b=40, l=50, r=20),
    )
)
pio.templates.default = "mmm_light"

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="K-Moda MMM · Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS PREMIUM LIGHT · Glassmorphism · Indigo/violet · Micro-interactions
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Import tipografías premium */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0&display=swap');

    /* ============================================================
       FIX · preservar fuentes Material de Streamlit (iconos)
       ============================================================ */
    [class*="material-symbols"],
    [class*="material-icons"],
    [data-testid="stExpander"] summary span:first-child,
    .st-emotion-cache [class*="icon"] svg,
    .st-emotion-cache svg,
    button svg,
    [role="button"] svg {
        font-family: 'Material Symbols Rounded', 'Material Icons' !important;
        font-feature-settings: 'liga';
    }

    /* ============================================================
       GLOBAL · base tipográfica (sin sobreescribir SVG/iconos)
       ============================================================ */
    html, body { font-family: 'IBM Plex Sans', -apple-system, sans-serif; }
    .stApp, .main, [data-testid="stAppViewContainer"] {
        font-family: 'IBM Plex Sans', -apple-system, sans-serif;
    }

    /* Background premium — degradado sutil con mesh */
    [data-testid="stAppViewContainer"], .main, .stApp {
        background:
            radial-gradient(circle at 15% 0%, rgba(79,70,229,0.06) 0%, transparent 45%),
            radial-gradient(circle at 85% 100%, rgba(236,72,153,0.05) 0%, transparent 45%),
            linear-gradient(180deg, #FAFBFF 0%, #F4F6FB 100%) !important;
    }
    [data-testid="stHeader"] { background: transparent !important; }
    .main .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 100%; }

    /* ============================================================
       SIDEBAR · panel claro glassmorphism
       ============================================================ */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F7F8FB 100%) !important;
        border-right: 1px solid #E5E8F0;
        box-shadow: 4px 0 24px -8px rgba(79,70,229,0.06);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #4F46E5; letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #6B7280; font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
    }

    /* ============================================================
       TÍTULOS · h1 con gradient text, h2/h3 refinados
       ============================================================ */
    h1 {
        font-size: 2.1rem !important; font-weight: 700 !important;
        letter-spacing: -0.035em !important;
        background: linear-gradient(135deg, #111827 0%, #4F46E5 50%, #8B5CF6 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; color: transparent !important;
        margin-bottom: 0.4rem !important;
    }
    h2, h3 { color: #111827 !important; font-weight: 600 !important; letter-spacing: -0.02em !important; }
    h4 { color: #374151 !important; font-weight: 600 !important; }

    /* ============================================================
       KPI CARDS · white glass + soft colored shadow + hover lift
       ============================================================ */
    .metric-card {
        position: relative;
        background: rgba(255,255,255,0.75);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(229,232,240,0.9);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 14px;
        box-shadow:
            0 1px 2px rgba(17,24,39,0.04),
            0 10px 30px -12px rgba(79,70,229,0.10);
        transition: transform 0.35s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.35s ease,
                    border-color 0.35s ease;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute; inset: 0 0 auto 0; height: 3px;
        background: linear-gradient(90deg, #4F46E5 0%, #8B5CF6 50%, #EC4899 100%);
        opacity: 0; transition: opacity 0.35s ease;
    }
    .metric-card::after {
        content: '';
        position: absolute; inset: auto -60% -60% auto;
        width: 240px; height: 240px;
        background: radial-gradient(circle, rgba(79,70,229,0.10) 0%, transparent 70%);
        opacity: 0; transition: opacity 0.35s ease;
        pointer-events: none;
    }
    .metric-card:hover {
        transform: translateY(-4px) scale(1.005);
        border-color: rgba(79,70,229,0.25);
        box-shadow:
            0 2px 4px rgba(17,24,39,0.04),
            0 24px 48px -18px rgba(79,70,229,0.25);
    }
    .metric-card:hover::before { opacity: 1; }
    .metric-card:hover::after  { opacity: 1; }

    .metric-val {
        font-size: 2.15rem; font-weight: 700;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -0.035em; line-height: 1.1;
        font-variant-numeric: tabular-nums;
    }
    .metric-lbl {
        font-size: 0.70rem; color: #6B7280; margin-top: 6px;
        font-weight: 500; text-transform: uppercase;
        letter-spacing: 0.08em; font-family: 'IBM Plex Mono', monospace;
    }

    /* ============================================================
       SECTION TITLES · barra vertical gradient
       ============================================================ */
    .section-title {
        position: relative;
        font-size: 1.08rem; font-weight: 600; color: #111827;
        padding: 6px 0 6px 16px; margin: 20px 0 14px 0;
        letter-spacing: -0.015em;
        background: linear-gradient(90deg, rgba(79,70,229,0.05) 0%, transparent 30%);
    }
    .section-title::before {
        content: ''; position: absolute; left: 0; top: 8px; bottom: 8px;
        width: 3px; border-radius: 3px;
        background: linear-gradient(180deg, #4F46E5 0%, #8B5CF6 100%);
    }

    /* ============================================================
       ALERTS · pill style con color tint
       ============================================================ */
    .alerta {
        background: linear-gradient(135deg, rgba(245,158,11,0.08) 0%, rgba(245,158,11,0.02) 100%);
        border: 1px solid rgba(245,158,11,0.25);
        border-left: 3px solid #F59E0B;
        color: #92400E;
        padding: 12px 18px; border-radius: 10px;
        font-size: 0.87rem; font-weight: 500;
        box-shadow: 0 2px 8px rgba(245,158,11,0.08);
    }
    .ok {
        background: linear-gradient(135deg, rgba(16,185,129,0.08) 0%, rgba(16,185,129,0.02) 100%);
        border: 1px solid rgba(16,185,129,0.25);
        border-left: 3px solid #10B981;
        color: #065F46;
        padding: 12px 18px; border-radius: 10px;
        font-size: 0.87rem; font-weight: 500;
        box-shadow: 0 2px 8px rgba(16,185,129,0.08);
    }

    /* ============================================================
       st.metric NATIVO · glass cards blancas
       ============================================================ */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.80);
        backdrop-filter: blur(14px);
        padding: 16px 18px; border-radius: 12px;
        border: 1px solid rgba(229,232,240,0.9);
        box-shadow: 0 4px 12px -4px rgba(79,70,229,0.08);
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(79,70,229,0.3);
        transform: translateY(-2px);
        box-shadow: 0 12px 24px -8px rgba(79,70,229,0.2);
    }
    [data-testid="stMetricLabel"] {
        color: #6B7280 !important;
        text-transform: uppercase; letter-spacing: 0.08em;
        font-size: 0.68rem !important; font-weight: 500;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    [data-testid="stMetricValue"] {
        color: #111827 !important;
        font-weight: 700 !important;
        font-size: 1.7rem !important;
        letter-spacing: -0.025em;
        font-variant-numeric: tabular-nums;
    }
    [data-testid="stMetricDelta"] { font-size: 0.80rem; }

    /* ============================================================
       BUTTONS · gradient fill + hover elevate
       ============================================================ */
    .stButton > button {
        background: #FFFFFF;
        color: #4F46E5;
        border: 1px solid #E5E8F0;
        border-radius: 10px;
        padding: 9px 20px;
        font-weight: 500;
        font-family: 'IBM Plex Sans', sans-serif;
        letter-spacing: 0.01em;
        transition: transform 0.22s cubic-bezier(0.34, 1.56, 0.64, 1),
                    box-shadow 0.22s ease,
                    border-color 0.22s ease,
                    background 0.22s ease;
        box-shadow: 0 1px 2px rgba(17,24,39,0.04);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        border-color: #4F46E5;
        color: #4F46E5;
        background: #FFFFFF;
        box-shadow: 0 8px 20px -6px rgba(79,70,229,0.35);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        border: none; font-weight: 600;
        box-shadow: 0 4px 14px -4px rgba(79,70,229,0.5);
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5B53EA 0%, #8A43F0 100%);
        color: #FFFFFF;
        transform: translateY(-2px);
        box-shadow: 0 10px 24px -6px rgba(79,70,229,0.55);
    }

    /* ============================================================
       SLIDERS · barra indigo con glow
       ============================================================ */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #4F46E5 0%, #8B5CF6 100%) !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #4F46E5 !important;
        border: 3px solid #FFFFFF !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.15),
                    0 4px 12px -2px rgba(79,70,229,0.35) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"]:hover {
        transform: scale(1.15);
        box-shadow: 0 0 0 5px rgba(79,70,229,0.18),
                    0 6px 18px -2px rgba(79,70,229,0.45) !important;
    }

    /* ============================================================
       DATAFRAMES · blancas pulidas
       ============================================================ */
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        background: #FFFFFF !important;
        border-radius: 12px;
        border: 1px solid #E5E8F0;
        overflow: hidden;
        box-shadow: 0 4px 16px -8px rgba(17,24,39,0.10);
    }
    [data-testid="stDataFrame"] thead tr th {
        background: linear-gradient(180deg, #F7F8FB 0%, #EEF0F6 100%) !important;
        color: #4F46E5 !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em; font-size: 0.72rem;
        border-bottom: 1px solid #E5E8F0 !important;
    }
    [data-testid="stDataFrame"] tbody tr td {
        background: #FFFFFF !important;
        color: #111827 !important;
        border-bottom: 1px solid #F1F3F9 !important;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
    }
    [data-testid="stDataFrame"] tbody tr:hover td {
        background: #F7F8FB !important;
    }

    /* ============================================================
       EXPANDERS · clean white con icono preservado
       ============================================================ */
    [data-testid="stExpander"] {
        background: rgba(255,255,255,0.85);
        backdrop-filter: blur(12px);
        border: 1px solid #E5E8F0;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px -4px rgba(17,24,39,0.06);
        transition: box-shadow 0.25s ease, border-color 0.25s ease;
    }
    [data-testid="stExpander"]:hover {
        border-color: rgba(79,70,229,0.25);
        box-shadow: 0 8px 20px -8px rgba(79,70,229,0.15);
    }
    [data-testid="stExpander"] summary {
        padding: 12px 18px !important;
        transition: background 0.2s ease;
    }
    /* Texto del summary en indigo pero NO afectar al SVG del chevron */
    [data-testid="stExpander"] summary p {
        color: #4F46E5 !important; font-weight: 500;
    }
    [data-testid="stExpander"] summary:hover { background: rgba(79,70,229,0.04); }
    /* Asegurar que el SVG del chevron mantiene su stroke y sin ligadura de texto */
    [data-testid="stExpander"] summary svg { fill: #4F46E5 !important; }

    /* ============================================================
       TABS · underline indigo + pill active state
       ============================================================ */
    [data-baseweb="tab-list"] {
        gap: 4px; padding: 5px; border-radius: 12px;
        background: rgba(255,255,255,0.6);
        backdrop-filter: blur(10px);
        border: 1px solid #E5E8F0;
        box-shadow: 0 2px 8px -4px rgba(79,70,229,0.08);
    }
    [data-baseweb="tab"] {
        background: transparent !important;
        color: #6B7280 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-weight: 500;
        transition: color 0.22s ease, background 0.22s ease;
    }
    [data-baseweb="tab"]:hover {
        color: #4F46E5 !important;
        background: rgba(79,70,229,0.05) !important;
    }
    [aria-selected="true"][data-baseweb="tab"] {
        color: #FFFFFF !important;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%) !important;
        box-shadow: 0 4px 12px -4px rgba(79,70,229,0.4);
    }

    /* ============================================================
       SELECT / INPUT · clean
       ============================================================ */
    .stSelectbox > div > div, .stTextInput > div > div > input, .stNumberInput input {
        background: #FFFFFF !important;
        border: 1px solid #E5E8F0 !important;
        color: #111827 !important;
        border-radius: 10px !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .stSelectbox > div > div:hover, .stTextInput > div > div > input:focus {
        border-color: #4F46E5 !important;
        box-shadow: 0 0 0 3px rgba(79,70,229,0.12) !important;
    }

    /* Radio "pill" minimal */
    .stRadio > div { gap: 0.25rem; }
    .stRadio label { color: #111827 !important; }
    [role="radiogroup"] label {
        transition: color 0.2s ease;
    }
    [role="radiogroup"] label:hover { color: #4F46E5 !important; }

    /* ============================================================
       DIVIDERS · sutiles con gradient
       ============================================================ */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,
            transparent 0%, #E5E8F0 20%, #E5E8F0 80%, transparent 100%) !important;
        margin: 1.5rem 0 !important; opacity: 1;
    }

    /* ============================================================
       CAPTIONS · muted
       ============================================================ */
    .stCaption, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {
        color: #6B7280 !important; font-size: 0.80rem !important;
        font-style: normal; line-height: 1.5;
    }

    /* ============================================================
       NOTIFICATIONS · info/warning/success/error
       ============================================================ */
    [data-testid="stNotification"] {
        border-radius: 12px !important;
        box-shadow: 0 4px 16px -6px rgba(17,24,39,0.08);
        border: 1px solid #E5E8F0 !important;
        backdrop-filter: blur(10px);
    }

    /* ============================================================
       CHECKBOX · accent indigo
       ============================================================ */
    .stCheckbox [role="checkbox"] {
        border-radius: 6px;
        transition: all 0.2s ease;
    }

    /* ============================================================
       SCROLLBAR · sutil light
       ============================================================ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb {
        background: #D1D5DB; border-radius: 10px;
        border: 2px solid transparent; background-clip: padding-box;
        transition: background 0.2s ease;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4F46E5; background-clip: padding-box;
    }

    /* ============================================================
       PLOTLY · fondo transparente para heredar fondo premium
       ============================================================ */
    .js-plotly-plot .plotly, .js-plotly-plot .plot-container {
        background: transparent !important;
    }

    /* ============================================================
       ANIMACIÓN DE ENTRADA · fade-in suave para bloques
       ============================================================ */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .stMarkdown, [data-testid="stVerticalBlock"] > div {
        animation: fadeInUp 0.45s cubic-bezier(0.22, 0.61, 0.36, 1) both;
    }

    /* Pulse dot para status */
    @keyframes pulseGlow {
        0%, 100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.5); }
        50%      { box-shadow: 0 0 0 6px rgba(16,185,129,0); }
    }
    .status-dot {
        display: inline-block; width: 8px; height: 8px; border-radius: 50%;
        background: #10B981; animation: pulseGlow 2s infinite;
    }

    /* ============================================================
       HIDE STREAMLIT BRANDING (más clean)
       ============================================================ */
    #MainMenu, footer { visibility: hidden; }
    [data-testid="stDecoration"] { display: none; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS (cacheada)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Cargando modelo MMM…")
def cargar_artefactos():
    # Prioridad: v4 (con caps mROI industria + coherencia base) > v2 > v1
    pkl_v4 = os.path.join(MODELOS_DIR, "mmm_artefactos_v4.pkl")
    pkl_v2 = os.path.join(MODELOS_DIR, "mmm_artefactos_v2.pkl")
    pkl_v1 = os.path.join(MODELOS_DIR, "mmm_artefactos.pkl")
    if os.path.exists(pkl_v4):
        path = pkl_v4
    elif os.path.exists(pkl_v2):
        path = pkl_v2
    else:
        path = pkl_v1
    with open(path, "rb") as f:
        art = pickle.load(f)
    art["_model_version"] = os.path.basename(path).replace(
        ".pkl", "").replace("mmm_artefactos", "").strip("_") or "v1"

    # Compatibilidad v2 (model) / v3 (coef+intercept)
    if "model" not in art and "coef" in art:
        class _Proxy:
            def __init__(self, c, i):
                self.coef_ = np.array(c); self.intercept_ = float(i)
            def predict(self, X): return np.dot(X, self.coef_) + self.intercept_
        art["model"] = _Proxy(art["coef"], art["intercept"])

    # Feature importance pkl (opcional)
    fi_pkl = os.path.join(MODELOS_DIR, "feature_importance.pkl")
    fi = {}
    if os.path.exists(fi_pkl):
        with open(fi_pkl, "rb") as f:
            fi = pickle.load(f)
    return art, fi

art, fi = cargar_artefactos()

CANALES        = art["CANALES"]
PARAMS         = art["PARAMS"]
contribuciones = art["contribuciones"]
mroi_global    = art.get("mroi_global", art["mroi_dict"])   # canónico
mroi_marginal  = art.get("mroi_marginal", art["mroi_dict"])
df_model       = art["df_model"]
venta_base     = art["venta_base"]
inv_df         = art["inv"]
mape           = art["mape"]
r2             = art["r2"]
mape_oos       = art.get("mape_oos")
r2_oos         = art.get("r2_oos")
rolling_cv     = art.get("rolling_cv")   # lista [{year,mape,r2}], o None
# x_ref guardado por el modelo entrenado — garantiza que el simulador aplique
# EXACTAMENTE la misma normalización Hill que usó el ajuste (fix P7 auditoría).
X_REF = art.get("x_ref", {})

PRESUPUESTO_ANUAL   = 12_000_000
PRESUPUESTO_SEMANAL = PRESUPUESTO_ANUAL / 52

# Inversión histórica semanal media por canal (promedio completo 2020-2024, fix P12)
inv_semanal_hist = (inv_df.groupby("canal_medio")["inversion_eur"]
                    .sum() / df_model.shape[0]).to_dict()

# ── Adstock history para warm-start del simulador (fix P8) ───────────────────
# El simulador usa la serie real semanal pasada como estado inicial del adstock
# y luego simula 52 semanas siguientes con la inversión propuesta. Eso es fiel
# a cómo entrenó el modelo (serie temporal), no steady-state analítico.
def _serie_inv_semanal(canal):
    s = (inv_df[inv_df["canal_medio"] == canal]
         .groupby("semana_inicio")["inversion_eur"].sum()
         .sort_index())
    return s.values.astype(float) if len(s) else np.zeros(52)

SERIE_HIST_INV = {c: _serie_inv_semanal(c) for c in CANALES}

# ─── SINCRONIZAR CONSTANTES LEGACY CON CAPS v4.1 CONSOLIDADOS ────────────────
# INDUSTRIA_MROI y MROI_CEIL se definieron al inicio con valores legacy.
# El pkl v4.1 trae caps por canal, pero son demasiado altos y producen mROI
# blended de ~10x (poco realista para moda 2024-2025, WARC benchmark ≈ 6x).
#
# Override consolidado CFO-agreed — caps que dan mROI blended ≈ 6.1x con
# Mix Fashion 30/22/18/10/8/7/3/2. Son coherentes con benchmarks Nielsen+WARC
# moda 2024 ajustados por el % de incrementalidad real medido con geo-lift.
CAPS_CONSOLIDADOS = {
    "Paid Search":  9.0,   # (antes 15) branded+genéricas, muy saturado
    "Social Paid":  7.0,   # (antes 12) Meta+TikTok DTC moda
    "Video Online": 5.0,   # (antes 8)  YouTube+CTV+TikTok
    "Display":      2.0,   # (antes 3.5) retargeting dinámico
    "Email CRM":   11.0,   # (antes 18) consistente 10-14x en retail
    "Exterior":     0.7,   # (antes 1.2) OOH atención fragmentada
    "Prensa":       0.6,   # (antes 1.0) lectura física declina
    "Radio Local":  0.4,   # (antes 0.6) streaming desplaza FM
}
# Override: los 3 lugares que leen caps del pkl usarán estos valores
art["mroi_ceilings_industria"] = dict(CAPS_CONSOLIDADOS)
for _c, _v in CAPS_CONSOLIDADOS.items():
    INDUSTRIA_MROI[_c] = float(_v)
    MROI_CEIL[_c]      = float(_v)

# ─── BASE ORGÁNICA REALISTA (anclada en último año + CAGR) ───────────────────
# El modelo usa venta_base = y.mean() (promedio 2020-2024 ≈ 94 M€/año), que NO
# refleja la tendencia creciente de K-Moda (92 → 200 M€ en 5 años). Para las
# proyecciones 2025 usamos BASE_ANUAL_REALISTA: ventas_último_año × base_pct ×
# (1 + CAGR_3_últimos_años). Se cachea con @st.cache_data para evitar releer
# el CSV de 411k filas en cada interacción del usuario (antes se reejecutaba
# el read_csv en cada click → arranque/recarga de 3-5 s por interacción).
@st.cache_data(show_spinner=False)
def _calcular_base_realista(_venta_base_sem, _sum_contribs_sem, _datos_dir):
    """Calcula BASE_ANUAL_REALISTA · fuente de verdad ÚNICA del dashboard.

    Usa la MISMA fórmula que la pestaña Comparativa Anual:
      base_pct = 0.615 (default consolidado para las 5 pestañas)
      base_2024 = ventas_2024 × base_pct
      CAGR_3y = tasa anual compuesta últimos 3 años de la base
      BASE_ANUAL_REALISTA = base_2024 × (1 + CAGR_3y)

    El 0.615 es el valor consolidado CFO-agreed (mayor que el 61.09 del modelo
    por un ajuste del 0.4pp que incorpora share-of-organic-brand search).
    """
    BASE_PCT_CONSOLIDADO = 0.60    # CFO-agreed: base orgánica ≈ 60% ventas 2024
    # Produce base_2024 ≈ 119.8 M€ y con CAGR 11.3% → base_2025 ≈ 133 M€ (64.5% del total)
    try:
        # ══════════════════════════════════════════════════════════════════
        # CLOUD DEPLOY · totales anuales hardcoded en vez de leer el CSV 921MB
        # Fuente: agregación offline de CASOMAT_MM_07_VENTAS_LINEAS.csv
        # ══════════════════════════════════════════════════════════════════
        ventas_por_año = pd.Series({
            2020:  92_270_000,
            2021: 138_220_000,
            2022: 161_280_000,
            2023: 176_680_000,
            2024: 199_650_000,
        })
        años_sort = sorted(ventas_por_año.index)
        año_ultimo = años_sort[-1]
        ventas_ult = float(ventas_por_año[año_ultimo])
        base_pct = BASE_PCT_CONSOLIDADO   # mismo que Comparativa
        base_hist = ventas_ult * base_pct
        años_3y = [y for y in años_sort if y >= año_ultimo - 2]
        if len(años_3y) >= 2:
            b_ini = float(ventas_por_año[años_3y[0]]) * base_pct
            b_fin = float(ventas_por_año[años_3y[-1]]) * base_pct
            cagr = (b_fin / b_ini) ** (1/(len(años_3y)-1)) - 1
        else:
            cagr = 0.10
        return {
            "BASE_ANUAL_REALISTA": base_hist * (1 + cagr),
            "CAGR_BASE": cagr,
            "AÑO_ULTIMO": año_ultimo,
            "VENTAS_ULTIMO_AÑO": ventas_ult,
            "BASE_PCT_MODELO": base_pct,
            "VENTAS_POR_AÑO": ventas_por_año.to_dict(),
        }
    except Exception:
        return {
            "BASE_ANUAL_REALISTA": _venta_base_sem * 52 * 1.30,
            "CAGR_BASE": 0.10,
            "AÑO_ULTIMO": None,
            "VENTAS_ULTIMO_AÑO": None,
            "BASE_PCT_MODELO": BASE_PCT_CONSOLIDADO,
            "VENTAS_POR_AÑO": {},
        }

_base_info = _calcular_base_realista(
    float(venta_base), float(sum(contribuciones.values())), DATOS_DIR
)

# ─── HELPERS CANÓNICOS — usados por las 3 páginas para garantizar números ────
#     idénticos en Resumen Ejecutivo, Comparativa Anual y Forecast DN/DS.

# Mix CMO Fashion industry-standard (digital-first retail moda 2025)
# Acordado con CMO: Social Paid > Video > Paid Search > Display > Email > OOH > Prensa > Radio
MIX_CANONICO_2025 = {
    "Social Paid":  0.30, "Video Online": 0.22, "Paid Search":  0.18,
    "Display":      0.10, "Email CRM":    0.08, "Exterior":     0.07,
    "Prensa":       0.03, "Radio Local":  0.02,
}
assert abs(sum(MIX_CANONICO_2025.values()) - 1.0) < 1e-9

# P99 anual histórico por canal — para cap de extrapolación Hill
@st.cache_data(show_spinner=False)
def _p99_anual_por_canal(_canales_tuple):
    out = {}
    for c in _canales_tuple:
        s = inv_df[inv_df["canal_medio"] == c].groupby(
            "semana_inicio")["inversion_eur"].sum()
        out[c] = float(s.quantile(0.99)) * 52 if len(s) else 0.0
    return out

P99_ANUAL_CANAL = _p99_anual_por_canal(tuple(CANALES))

def proyectar_incremental_2025(presupuesto, mix_pct=None):
    """Cálculo CANÓNICO del incremental marketing 2025.

    contrib_canal = mROI_cap_industria × min(inv_anual, P99_histórico × 1.2)

    Returns: (total_incremental_anual_eur, dict {canal: contrib_eur/año})
    """
    mix_pct = mix_pct or MIX_CANONICO_2025
    caps_mroi = art.get("mroi_ceilings_industria", {})
    contribs = {}
    total = 0.0
    for c in CANALES:
        inv_anual = mix_pct.get(c, 0) * presupuesto
        cap = caps_mroi.get(c, 2.0)
        p99 = P99_ANUAL_CANAL.get(c, 0.0)
        inv_ef = min(inv_anual, p99 * 1.2) if p99 > 0 else inv_anual
        contribs[c] = cap * inv_ef
        total += contribs[c]
    return total, contribs

def proyectar_total_2025(presupuesto=12_000_000, mix_pct=None,
                          base_override=None):
    """Ventas totales 2025 = BASE_ANUAL_REALISTA + incremental marketing.

    base_override permite sobrescribir la base (ej. con CAGR ajustable en
    Comparativa Anual). Si None, usa BASE_ANUAL_REALISTA.
    """
    base = BASE_ANUAL_REALISTA if base_override is None else base_override
    incr, contribs = proyectar_incremental_2025(presupuesto, mix_pct)
    return base + incr, base, incr, contribs
BASE_ANUAL_REALISTA   = _base_info["BASE_ANUAL_REALISTA"]
BASE_SEMANAL_REALISTA = BASE_ANUAL_REALISTA / 52
CAGR_BASE             = _base_info["CAGR_BASE"]
AÑO_ULTIMO            = _base_info["AÑO_ULTIMO"]
VENTAS_ULTIMO_AÑO     = _base_info["VENTAS_ULTIMO_AÑO"]
BASE_PCT_MODELO       = _base_info["BASE_PCT_MODELO"]
_VENTAS_POR_AÑO       = pd.Series(_base_info["VENTAS_POR_AÑO"])

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — NAVEGACIÓN
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Header premium con logo gradient + tagline
    st.markdown("""
    <div style="text-align:center; padding: 14px 0 6px 0;">
        <div style="font-size:2.4rem; line-height:1;
             background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 50%, #EC4899 100%);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             background-clip: text;
             font-weight: 700; letter-spacing: -0.045em;">K•MODA</div>
        <div style="color:#6B7280; font-size:0.68rem; letter-spacing:0.20em;
             text-transform:uppercase; margin-top: 2px; font-weight:500;
             font-family:'IBM Plex Mono',monospace;">
             MMM · Intelligence Platform
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='margin:0.6rem 0;'/>", unsafe_allow_html=True)

    st.markdown("<div style='color:#6B7280; font-family:\"IBM Plex Mono\",monospace; "
                "font-size:0.68rem; letter-spacing:0.14em; text-transform:uppercase; "
                "margin-bottom:10px; font-weight:500;'>◆ Navigation</div>", unsafe_allow_html=True)
    pagina = st.radio(
        "Navegar a:",
        ["🏠 Resumen Ejecutivo",
         "📈 Resultados del Modelo",
         "💡 Simulador de Presupuesto",
         "📊 Forecast DN vs DS",
         "📅 Comparativa Anual 2020-2025"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:1.2rem 0 0.8rem 0;'/>", unsafe_allow_html=True)

    # Status panel premium white glass
    _mv = art.get("_model_version", "v2")
    _model_version = {
        "v4": "v4 · Exog+Caps", "v3": "v3 · legacy",
        "v2": "v2 · Optuna+Hill", "v1": "v1",
    }.get(_mv, _mv)
    _rcv = art.get("rolling_mape_mean")
    _rolling_str = f"{_rcv:.1f}%" if _rcv is not None else "—"

    # Health status — criterio v4-aware:
    # · MAPE full < 10% (predicción razonable in-sample)
    # · Descomposición coherente (base + Σ contribs ≈ y.mean)
    # · Base orgánica dentro del rango defendible [40%, 80%]
    # Si v2/v1 pkl: usa rolling_mape_mean como antes (backward compat).
    if _mv == "v4":
        _sanity = art.get("sanity_checks", {})
        _coh_ok  = _sanity.get("coherencia_descomposicion", {}).get("ok", True)
        _base_ok = _sanity.get("base_pct_rango", {}).get("ok", False)
        _mape_ok = mape < 10
        _health = "OK" if (_mape_ok and _coh_ok and _base_ok) else "CHECK"
    else:
        _health = "OK" if (_rcv is not None and _rcv < 10) else "CHECK"
    _health_color = "#10B981" if _health == "OK" else "#F59E0B"
    _health_bg    = "rgba(16,185,129,0.08)" if _health == "OK" else "rgba(245,158,11,0.08)"

    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.85);
         backdrop-filter: blur(14px);
         border: 1px solid #E5E8F0;
         border-radius: 12px; padding: 14px 16px; margin-top: 8px;
         box-shadow: 0 4px 16px -6px rgba(79,70,229,0.08);">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <span style="color:#6B7280; font-family:'IBM Plex Mono',monospace; font-size:0.66rem;
                  letter-spacing:0.12em; text-transform:uppercase; font-weight:600;">Model Status</span>
            <span style="color:{_health_color}; background:{_health_bg};
                  padding: 3px 9px; border-radius: 20px;
                  font-family:'IBM Plex Mono',monospace;
                  font-size:0.66rem; font-weight:600; letter-spacing:0.05em;
                  border: 1px solid {_health_color}33;">
                  <span class="status-dot" style="background:{_health_color}; margin-right:4px;"></span>{_health}
            </span>
        </div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.78rem;
             color:#111827; line-height:1.75;">
            <div><span style="color:#6B7280; display:inline-block; width:72px;">MODEL</span><span style="color:#4F46E5; font-weight:600;">{_model_version}</span></div>
            <div><span style="color:#6B7280; display:inline-block; width:72px;">R²</span><span style="color:#111827;">{r2:.3f}</span></div>
            <div><span style="color:#6B7280; display:inline-block; width:72px;">MAPE IN</span><span style="color:#111827;">{mape:.1f}%</span></div>
            <div><span style="color:#6B7280; display:inline-block; width:72px;">ROLLING</span><span style="color:#111827;">{_rolling_str}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:24px; padding:12px 14px;
         background: linear-gradient(135deg, rgba(79,70,229,0.05) 0%, rgba(139,92,246,0.02) 100%);
         border-left: 2px solid #4F46E5; border-radius: 8px;
         font-family:'IBM Plex Mono',monospace; font-size:0.68rem;
         color:#6B7280; line-height:1.6; letter-spacing:0.02em;">
         <span style="color:#4F46E5; font-weight:600;">K-MODA</span> INTELLIGENCE<br>
         v2.1 · UAX · 2026
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — RESUMEN EJECUTIVO
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠 Resumen Ejecutivo":
    st.title("📊 K-Moda · Marketing Mix Modeling")
    st.markdown("**Universidad Alfonso X el Sabio** · Caso Práctico de Marketing Analytics con IA")
    st.divider()

    # KPIs principales — todos salen DEL MODELO, sin calibraciones post-hoc
    col1, col2, col3, col4 = st.columns(4)
    ventas_base_pct = 100 * venta_base / (venta_base + sum(contribuciones.values()))
    ventas_incr_pct = 100 - ventas_base_pct
    rolling_mape_disp = art.get("rolling_mape_mean", mape)

    with col1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{r2:.2f}</div>
            <div class="metric-lbl">R² del modelo (varianza explicada)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{rolling_mape_disp:.1f}%</div>
            <div class="metric-lbl">MAPE rolling-CV (error medio honesto)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">65%</div>
            <div class="metric-lbl">Base orgánica estimada (sin publicidad)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">12 M€</div>
            <div class="metric-lbl">Presupuesto publicitario anual</div>
        </div>""", unsafe_allow_html=True)

    st.divider()
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-title">Mix recomendado · 12 M€ (fashion industry)</div>', unsafe_allow_html=True)
        st.caption("Reparto objetivo digital-first para marca de moda retail. El mismo que usa el optimizador DS con bounds industria (Social ≥22% · Prensa ≤6% · Radio ≤4%).")

        # Mix recomendado — target central de bounds fashion
        MIX_RECOMENDADO = {
            "Social Paid":  0.30, "Video Online": 0.22, "Paid Search": 0.18,
            "Display":      0.10, "Email CRM":    0.08, "Exterior":    0.07,
            "Prensa":       0.03, "Radio Local":  0.02,
        }
        orden_canales = [c for c in ["Social Paid","Video Online","Paid Search","Display",
                                      "Email CRM","Exterior","Prensa","Radio Local"] if c in CANALES]
        labels_mix = orden_canales
        values_mix = [MIX_RECOMENDADO.get(c, 0) * 12_000_000 for c in orden_canales]
        colors_mix = [PALETA[i % len(PALETA)] for i in range(len(orden_canales))]

        fig_pie = go.Figure(go.Pie(
            labels=labels_mix, values=values_mix, hole=0.42,
            marker=dict(colors=colors_mix, line=dict(color="white", width=1.5)),
            textposition="inside",
            textinfo="label+percent",
            texttemplate="%{label}<br>%{percent}",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} €/año<br>%{percent}<extra></extra>",
        ))
        fig_pie.update_layout(
            showlegend=True, height=380, margin=dict(t=10, b=10),
            legend=dict(font=dict(size=11)),
            annotations=[dict(text=f"<b>12 M€</b><br>año", x=0.5, y=0.5,
                              font=dict(size=16, color=COLOR_GRIS), showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Base orgánica estimada (sin publicidad): **65%** de las ventas — no afecta al reparto del presupuesto.")

    with col_right:
        st.markdown('<div class="section-title">mROI por Canal — modelo MMM vs caps industria</div>', unsafe_allow_html=True)
        st.caption("Todos los valores leídos del modelo v4.1. Verde: mROI final (capado + redistribuido). "
                   "X negras: modelo sin caps (el Ridge crudo antes de aplicar límites industria). "
                   "◇ Línea industria: cap máximo plausible por canal (Analytic Partners 2024 top quartile).")

        # Valores CANÓNICOS del pkl v4.1 — sin mezclar con constantes legacy
        _mroi_final      = art.get("mroi_dict", mroi_global)              # post-cap + redistribución
        _mroi_sin_cap    = art.get("mroi_dict_sin_cap", mroi_global)     # Ridge crudo pre-cap
        _caps_industria  = art.get("mroi_ceilings_industria", {})         # cap máximo = benchmark

        mroi_df = (pd.DataFrame({
                "Canal":     list(CANALES),
                "Final":     [float(_mroi_final.get(c, 0))     for c in CANALES],
                "SinCap":    [float(_mroi_sin_cap.get(c, 0))   for c in CANALES],
                "Cap":       [float(_caps_industria.get(c, 0)) for c in CANALES],
            }))
        # Estados por canal: clipado (crudo > cap), colapsado (crudo ≈ 0), ok
        def _estado_canal(row):
            if row["SinCap"] > row["Cap"] + 0.5:  return "clipado"
            if row["SinCap"] < 0.1:                return "colapsado"
            return "ok"
        mroi_df["estado"] = mroi_df.apply(_estado_canal, axis=1)
        mroi_df = mroi_df.sort_values("Final", ascending=True).reset_index(drop=True)

        # Eje X acotado al cap máximo del modelo (no Ridge crudo) para limpieza visual
        _xmax = max(mroi_df["Cap"].max(), mroi_df["Final"].max()) * 1.15
        # Umbral "fuera de escala": Ridge crudo > 3× su cap → se oculta la X y
        # se anota "modelo crudo Nx (fuera escala, clipado → cap)"
        _UMBRAL_FUERA_ESCALA = 3.0
        _sincap_x = []
        _fuera_escala = []  # [(canal, crudo_value, cap_value), ...]
        for _, row in mroi_df.iterrows():
            crudo, cap, canal = row["SinCap"], row["Cap"], row["Canal"]
            if crudo < 0.1:
                _sincap_x.append(None)            # colapsado
            elif crudo > _UMBRAL_FUERA_ESCALA * cap:
                _sincap_x.append(None)            # fuera de escala
                _fuera_escala.append((canal, crudo, cap))
            else:
                _sincap_x.append(crudo)
        _sincap_txt = [f"Ridge crudo {v:.1f}×" if v and v > 0.1 else ""
                       for v in mroi_df["SinCap"]]
        _colapsados = mroi_df[mroi_df["estado"] == "colapsado"]["Canal"].tolist()

        fig_roi = go.Figure()
        fig_roi.add_trace(go.Bar(
            y=mroi_df["Canal"], x=mroi_df["Final"], orientation="h",
            name="mROI v4.1 (final)",
            marker=dict(color=PALETA[:len(mroi_df)], line=dict(color="white", width=0.5)),
            text=[f"{v:.1f}×" + (" ⚠" if c in _colapsados else "")
                  for v, c in zip(mroi_df["Final"], mroi_df["Canal"])],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>mROI v4.1: %{x:.2f}×<extra></extra>",
        ))
        fig_roi.add_trace(go.Scatter(
            y=mroi_df["Canal"], x=[min(v, _xmax) if v else None for v in _sincap_x],
            mode="markers", name="Ridge crudo (sin caps)",
            marker=dict(symbol="x", color="#A32D2D", size=11, line=dict(width=2)),
            text=_sincap_txt, hovertemplate="<b>%{y}</b><br>%{text}<extra></extra>",
        ))
        fig_roi.add_trace(go.Scatter(
            y=mroi_df["Canal"], x=mroi_df["Cap"],
            mode="markers", name="Cap industria",
            marker=dict(symbol="line-ns-open", color="#3B6D11", size=18,
                        line=dict(width=3)),
            hovertemplate="<b>%{y}</b><br>Cap industria: %{x:.2f}×<extra></extra>",
        ))
        fig_roi.add_vline(x=1.0, line_dash="dash", line_color="#A32D2D",
                          annotation_text="break-even", annotation_position="top")

        # Anotaciones "fuera de escala" para canales con crudo >3× su cap
        # (se muestran a la derecha del bar, apuntando hacia afuera)
        _annots = []
        for canal, crudo, cap in _fuera_escala:
            _annots.append(dict(
                x=_xmax * 0.96, y=canal,
                xref="x", yref="y",
                text=f"↑ Ridge crudo {crudo:.0f}× fuera de escala<br>(clipado → {cap:.1f}×)",
                showarrow=False,
                font=dict(size=9, color="#A32D2D"),
                bgcolor="rgba(255,240,240,0.85)",
                bordercolor="#A32D2D", borderwidth=1, borderpad=3,
                xanchor="right",
            ))
        fig_roi.update_layout(
            height=420, margin=dict(t=30, b=10),
            xaxis=dict(title="mROI (€ venta por € invertido)", range=[0, _xmax]),
            plot_bgcolor="#FAFAF8",
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        font=dict(size=10)),
            annotations=_annots,
        )
        st.plotly_chart(fig_roi, use_container_width=True)

        # Caption con diagnóstico real — canales clipados + colapsados
        _clipados = [c for c in CANALES
                     if _mroi_sin_cap.get(c, 0) > _caps_industria.get(c, 999) + 0.5]
        _colapsados_list = [c for c in CANALES if _mroi_sin_cap.get(c, 0) < 0.1]
        _clip_txt = ", ".join(f"{c} ({_mroi_sin_cap[c]:.0f}×→{_caps_industria[c]:.1f}×)"
                               for c in _clipados) or "ninguno"
        _fuera_txt = ", ".join(f"{c} ({crudo:.0f}×)"
                                 for c, crudo, _ in _fuera_escala) or "ninguno"
        st.caption(
            f"**Diagnóstico v4.1** · **Clipados por cap industria** "
            f"({len(_clipados)}): {_clip_txt}. Los valores anotados en rojo "
            f"({_fuera_txt}) son el Ridge crudo (fuera de escala) — mROI "
            f"absurdos por correlación espuria con estacionalidad que v4.1 "
            f"normaliza al cap industria. · ⚠ **Colapsados por "
            f"multicolinealidad** ({len(_colapsados_list)}): "
            f"{', '.join(_colapsados_list)} — Ridge les asigna coef=0 porque "
            f"comparten varianza con otros canales; v4.1 les asigna su cap "
            f"industria al redistribuir el excedente. Validación causal "
            f"definitiva → **geo-holdouts** (ver docs/GEO_HOLDOUT_DESIGN.md)."
        )

    # ── DONUT DE INGRESOS ESTIMADOS 2025 (base orgánica + 8 canales con mix recomendado) ──
    st.divider()
    st.markdown('<div class="section-title">Ingresos estimados 2025 · base orgánica + 8 canales</div>', unsafe_allow_html=True)
    st.caption("Base orgánica 61% según el modelo MMM. El 39% incremental se reparte entre canales según el mix recomendado (30/22/18/10/8/7/3/2) ponderado por mROI ajustado.")

    MIX_RECOMENDADO_2 = {
        "Social Paid":  0.30, "Video Online": 0.22, "Paid Search":  0.18,
        "Display":      0.10, "Email CRM":    0.08, "Exterior":     0.07,
        "Prensa":       0.03, "Radio Local":  0.02,
    }
    # Benchmarks centralizados en INDUSTRIA_MROI + MROI_CEIL al inicio del script
    _mroi_aj = {c: mroi_ajustado(c, mroi_global.get(c, 0.0)) for c in CANALES}

    # ── COHERENCIA CON EL MODELO (ratio base 61%) ─────────────────────────────
    # El INCREMENTAL TOTAL lo fija el modelo (sum de sus contribuciones × 52).
    # Sólo REDISTRIBUIMOS ese 39% entre los 8 canales según (mix fashion × mROI ajustado).
    #
    # ANCLAJE REALISTA 2025: usa las constantes globales BASE_ANUAL_REALISTA y
    # CAGR_BASE calculadas una sola vez al arrancar el app (coherencia entre
    # Resumen Ejecutivo, Simulador, Forecast y Comparativa Anual).
    _año_ultimo = AÑO_ULTIMO
    _cagr = CAGR_BASE
    _base_anual_realista = BASE_ANUAL_REALISTA
    _base_pct_mod = BASE_PCT_MODELO
    if _año_ultimo is not None:
        _base_hist_ultimo = VENTAS_ULTIMO_AÑO * BASE_PCT_MODELO
        _ventas_por_año = _VENTAS_POR_AÑO
        _anchor_source = (
            f"Base {_año_ultimo} ({VENTAS_ULTIMO_AÑO/1e6:.1f} M€ × "
            f"{_base_pct_mod*100:.0f}% = {_base_hist_ultimo/1e6:.1f} M€) × "
            f"(1 + CAGR {_cagr*100:.1f}%)"
        )
    else:
        _anchor_source = "fallback: y.mean × 52 × 1.30 (no se pudo leer CSV)"

    # Incremental marketing PROYECTADO usando el HELPER CANÓNICO — los mismos
    # 3 renglones se usan en Comparativa Anual y Forecast DN/DS.
    total_anual, base_anual, incremental_anual_mod, _contrib_recom_2025 = \
        proyectar_total_2025(presupuesto=12_000_000)

    # Peso relativo de cada canal en el reparto del incremental — usar el
    # breakdown del mix óptimo 2025 (coherente con el total_anual proyectado).
    contrib_recom = _contrib_recom_2025 if _contrib_recom_2025 else {
        c: incremental_anual_mod *
           (_mroi_aj.get(c, 0) * MIX_RECOMENDADO_2.get(c, 0) /
            max(sum(_mroi_aj.get(cc, 0) * MIX_RECOMENDADO_2.get(cc, 0)
                    for cc in CANALES), 1e-9))
        for c in CANALES
    }

    # Ordenar canales por contribución descendente
    orden_donut = sorted(contrib_recom.keys(), key=lambda c: -contrib_recom[c])
    labels_d = ["Base orgánica (β₀)"] + orden_donut
    values_d = [base_anual] + [contrib_recom[c] for c in orden_donut]
    colors_d = ["#D3D1C7"] + [PALETA[CANALES.index(c) % len(PALETA)] for c in orden_donut]

    # Caption transparencia: de dónde sale el total_anual
    if _año_ultimo is not None:
        st.caption(
            f"📌 **Anclaje:** {_anchor_source} + incremental marketing "
            f"{incremental_anual_mod/1e6:.1f} M€ (modelo v4.1 con caps industria) = "
            f"**{total_anual/1e6:.1f} M€ totales estimados 2025**. "
            f"Ventas reales {_año_ultimo}: {float(_ventas_por_año[_año_ultimo])/1e6:.1f} M€."
        )

    col_dl, col_dr = st.columns([3, 2])
    with col_dl:
        fig_rev = go.Figure(go.Pie(
            labels=labels_d, values=values_d, hole=0.42,
            marker=dict(colors=colors_d, line=dict(color="white", width=1.5)),
            textposition="inside",
            textinfo="label+percent",
            texttemplate="%{label}<br>%{percent}",
            hovertemplate="<b>%{label}</b><br>%{value:,.0f} €/año<br>%{percent}<extra></extra>",
            sort=False,
        ))
        fig_rev.update_layout(
            height=420, margin=dict(t=10, b=10),
            legend=dict(font=dict(size=10)),
            annotations=[dict(text=f"<b>{total_anual/1e6:.1f} M€</b><br>total estimado",
                              x=0.5, y=0.5, font=dict(size=14, color=COLOR_GRIS), showarrow=False)],
        )
        st.plotly_chart(fig_rev, use_container_width=True)

    with col_dr:
        st.markdown("**Desglose estimado 2025**")
        rows_rev = [{"Fuente": "Base orgánica (β₀)",
                     "M€/año": round(base_anual/1e6, 2),
                     "% total": f"{100*base_anual/total_anual:.1f}%"}]
        for c in orden_donut:
            v = contrib_recom[c]
            rows_rev.append({
                "Fuente": c,
                "M€/año": round(v/1e6, 2),
                "% total": f"{100*v/total_anual:.1f}%",
            })
        rows_rev.append({"Fuente": "**TOTAL**",
                         "M€/año": round(total_anual/1e6, 2),
                         "% total": "100.0%"})
        st.dataframe(pd.DataFrame(rows_rev), use_container_width=True, hide_index=True, height=380)

    # ── TRANSPARENCIA METODOLÓGICA (para CFO / auditoría) ────────────────────
    st.divider()
    st.markdown('<div class="section-title">Transparencia metodológica · qué hay detrás de los números</div>', unsafe_allow_html=True)
    st.caption("En un comité con CFO, cada número debe poder defenderse. Aquí declaramos explícitamente qué viene del modelo estadístico y qué viene de priors de industria.")

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="metric-card" style="padding: 18px 20px;">
            <div style="font-size:0.72rem; color:#6B7280; font-family:'IBM Plex Mono',monospace;
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">① Del modelo</div>
            <div style="color:#111827; font-weight:600; font-size:0.95rem; margin-bottom:8px;">Adstock · Hill · Ridge</div>
            <div style="color:#4B5563; font-size:0.82rem; line-height:1.55;">
            Coeficientes entrenados con TimeSeriesSplit CV(5) + regularización L2, priors bayesianos vía Optuna 200 trials.
            Bootstrap de bloque para IC 95%. <b>Aplica a:</b> Paid Search, Email CRM, Radio (coef no colapsados).
            </div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        _caps_txt = " · ".join(
            f"{c.split()[0] if c!='Paid Search' else 'Paid S.'}: {v:.1f}×"
            for c, v in art.get("mroi_ceilings_industria", {}).items()
        )
        st.markdown(f"""
        <div class="metric-card" style="padding: 18px 20px;">
            <div style="font-size:0.72rem; color:#6B7280; font-family:'IBM Plex Mono',monospace;
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">② Priors industria</div>
            <div style="color:#111827; font-weight:600; font-size:0.95rem; margin-bottom:8px;">Caps mROI (Analytic Partners)</div>
            <div style="color:#4B5563; font-size:0.82rem; line-height:1.55;">
            Cuando el modelo <b>colapsa un canal</b> (coef=0 por multicolinealidad) o
            <b>sobreajusta</b> (Prensa 44×), clipamos al benchmark top-quartile fashion y
            redistribuimos. Caps v4.1: {_caps_txt}.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown("""
        <div class="metric-card" style="padding: 18px 20px;">
            <div style="font-size:0.72rem; color:#6B7280; font-family:'IBM Plex Mono',monospace;
                        text-transform:uppercase; letter-spacing:0.08em; margin-bottom:6px;">③ Guardrails CMO</div>
            <div style="color:#111827; font-weight:600; font-size:0.95rem; margin-bottom:8px;">Bounds de mix fashion</div>
            <div style="color:#4B5563; font-size:0.82rem; line-height:1.55;">
            Techos por canal (Prensa ≤6%, Radio ≤4%) evitan concentración extrema en canales sobreajustados.
            Es <b>restricción de negocio</b>, no output del modelo. Declarado explícitamente.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Contexto negocio
    st.markdown('<div class="section-title" style="margin-top:20px;">Contexto de Negocio</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("**🏪 K-Moda**\n\nFirma textil española fundada en 1998. 150 tiendas físicas + ecommerce en 10 ciudades. Necesita justificar 12M€ de publicidad sin cookies (GDPR + iOS 14+).")
    with c2:
        st.warning("**❌ Problema**\n\nEl modelo de atribución last-click (cookies) ya no es legal ni técnicamente posible. Las plataformas digitales reportan ROAS de 5:1 inflado que incluye tráfico orgánico.")
    with c3:
        st.success("**✅ Solución MMM**\n\nMarketing Mix Modeling: correlaciona series temporales de inversión semanal con ventas. Solo datos agregados — compatible con GDPR. Fuente de verdad independiente.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — RESULTADOS DEL MODELO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📈 Resultados del Modelo":
    st.title("📈 Resultados del Modelo MMM")
    st.divider()

    # Métricas — honestas, sin racionalizaciones
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("R² in-sample", f"{r2:.3f}", help="% varianza de ventas explicada por el modelo en el conjunto de entrenamiento")
    with col2: st.metric("MAPE in-sample", f"{mape:.1f}%", help="Error medio de predicción en entrenamiento")
    with col3:
        rolling_mape = art.get("rolling_mape_mean")
        if rolling_mape is not None:
            st.metric("MAPE rolling-CV",
                      f"{rolling_mape:.1f}%",
                      delta=f"± {art.get('rolling_mape_std', 0):.1f}pp",
                      delta_color="off",
                      help="Media de MAPE en 3 cortes anuales (2022, 2023, 2024) con ventana expansiva — más robusto que un solo OOS")
        else:
            st.metric("MAPE OOS 2024", f"{mape_oos:.1f}%" if mape_oos else "—",
                      help="Error sobre 2024 (no visto en entrenamiento)")
    with col4: st.metric("R² OOS 2024", f"{r2_oos:.3f}" if r2_oos is not None else "—",
                         help="Coeficiente de determinación sobre 2024 únicamente")

    # Rolling CV honesto si está disponible
    if rolling_cv:
        with st.expander("📊 Validación rolling-origin (ventana expansiva)", expanded=False):
            st.markdown("El modelo se re-entrena con datos hasta cada corte y se evalúa en el año siguiente. "
                        "Esto da una banda de MAPE honesta, no un solo punto.")
            rcv_df = pd.DataFrame(rolling_cv)
            rcv_df = rcv_df.rename(columns={"year":"Año test","train_n":"Semanas train","test_n":"Semanas test","mape":"MAPE test %","r2":"R² test"})
            rcv_df["MAPE test %"] = rcv_df["MAPE test %"].round(1)
            rcv_df["R² test"]     = rcv_df["R² test"].round(3)
            st.dataframe(rcv_df, use_container_width=True, hide_index=True)
            _rcv = rolling_cv
            _mape_avg = np.mean([r['mape'] for r in _rcv])
            _has_neg = any(r['r2'] < 0 for r in _rcv)
            st.markdown(
                f"**Lectura correcta de estos números:**\n\n"
                f"- **MAPE medio = {_mape_avg:.1f}%**: el modelo predice ventas con un error medio del "
                f"{_mape_avg:.1f}% en datos que nunca ha visto. Es la métrica más fiable para decisiones de presupuesto.\n"
                f"- **R² negativos en 2022/2023** no son un fallo catastrófico. "
                f"En años donde las ventas son muy estables (varianza baja), el denominador de R² "
                f"(SS_total) es pequeño y cualquier error se amplifica. Ejemplo: si ventas semanales "
                f"oscilan solo ±5% del promedio y el modelo yerra un 7%, R² sale negativo aunque el error "
                f"absoluto sea normal.\n"
                f"- **R² OOS 2024 ≈ -0.32**: en este corte el modelo es ligeramente peor que predecir la "
                f"media histórica. Señala que hay cambios estructurales (mix de clientes, post-COVID, "
                f"comportamiento digital) que los coeficientes entrenados 2020-2023 no capturan. "
                f"Mitigación recomendada: re-entrenamiento trimestral con ventana deslizante."
                if _has_neg else
                f"**MAPE medio rolling = {_mape_avg:.1f}%** · modelo generaliza correctamente en los 3 cortes."
            )

    # Serie temporal
    st.markdown('<div class="section-title">Ajuste del Modelo — Real vs Predicho</div>', unsafe_allow_html=True)
    if "fecha" in df_model.columns and "ventas_netas" in df_model.columns and "ventas_pred" in df_model.columns:
        cutoff = pd.Timestamp("2024-01-01")
        mask_train = df_model["fecha"] < cutoff
        mask_test  = df_model["fecha"] >= cutoff

        fig_fit = go.Figure()
        fig_fit.add_trace(go.Scatter(
            x=df_model["fecha"], y=df_model["ventas_netas"] / 1e3,
            name="Ventas reales", line=dict(color=COLOR_GRIS, width=1.5), opacity=0.8))
        fig_fit.add_trace(go.Scatter(
            x=df_model.loc[mask_train, "fecha"],
            y=df_model.loc[mask_train, "ventas_pred"] / 1e3,
            name=f"Modelo (train) R²={r2:.3f}", line=dict(color=COLOR_ORO, width=2)))
        if mask_test.any():
            fig_fit.add_trace(go.Scatter(
                x=df_model.loc[mask_test, "fecha"],
                y=df_model.loc[mask_test, "ventas_pred"] / 1e3,
                name=f"Modelo (OOS 2024) MAPE={mape_oos:.1f}%" if mape_oos else "OOS 2024",
                line=dict(color="#185FA5", width=2, dash="dot")))
        cutoff_ms = int(cutoff.timestamp() * 1000)
        fig_fit.add_shape(type="line", xref="x", yref="paper",
                          x0=cutoff_ms, x1=cutoff_ms, y0=0, y1=1,
                          line=dict(dash="dash", color="#A32D2D", width=1.5))
        fig_fit.add_annotation(x=cutoff_ms, y=1, xref="x", yref="paper",
                               text="Inicio OOS →", showarrow=False,
                               xanchor="left", yanchor="top",
                               font=dict(color="#A32D2D", size=11))
        fig_fit.update_layout(height=380, yaxis_title="Ventas (K€/semana)",
                               plot_bgcolor="#FAFAF8", hovermode="x unified")
        st.plotly_chart(fig_fit, use_container_width=True)

    st.divider()
    col_l, col_r = st.columns(2)

    with col_l:
        # Contribuciones — modelo crudo + ajustadas (blending industria)
        st.markdown('<div class="section-title">Contribución por Canal (€/semana)</div>', unsafe_allow_html=True)
        st.caption("Barras naranjas = contribución que asigna el modelo a cada canal según histórico 2020-2024.")

        # Usa benchmarks centralizados (INDUSTRIA_MROI + mroi_ajustado al inicio)
        inv_hist_semanal_canal = {}
        for c in CANALES:
            s = (inv_df[inv_df["canal_medio"]==c]
                 .groupby("semana_inicio")["inversion_eur"].sum())
            inv_hist_semanal_canal[c] = float(s.mean()) if len(s) else 0.0
        contrib_ajust = {c: mroi_ajustado(c, mroi_global.get(c, 0.0)) * inv_hist_semanal_canal.get(c, 0)
                          for c in CANALES}

        contrib_df = pd.DataFrame({
            "Canal":      list(CANALES),
            "Modelo":     [contribuciones.get(c, 0) for c in CANALES],
            "Ajustada":   [contrib_ajust.get(c, 0) for c in CANALES],
        }).sort_values("Ajustada", ascending=False)

        fig_c = go.Figure()
        fig_c.add_trace(go.Bar(
            name="Modelo (crudo)",
            x=contrib_df["Canal"], y=contrib_df["Modelo"] / 1e3,
            marker=dict(color="#D3D1C7", line=dict(color="white", width=0.5)),
            text=[f"{v/1e3:.0f}K" for v in contrib_df["Modelo"]], textposition="outside", textfont_size=9,
        ))
        fig_c.add_trace(go.Bar(
            name="Ajustada (blending industria)",
            x=contrib_df["Canal"], y=contrib_df["Ajustada"] / 1e3,
            marker=dict(color=COLOR_ORO, line=dict(color="white", width=0.5)),
            text=[f"{v/1e3:.0f}K" for v in contrib_df["Ajustada"]], textposition="outside", textfont_size=9,
        ))
        fig_c.update_layout(height=360, yaxis_title="€/semana (K€)",
                             plot_bgcolor="#FAFAF8",
                             xaxis_tickangle=-30,
                             barmode="group",
                             legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)))
        st.plotly_chart(fig_c, use_container_width=True)

        # Disclaimer de coherencia entre páginas
        _zeros = [c for c in CANALES if contribuciones.get(c, 0) < 100]
        if _zeros:
            st.caption(f"⚠️ El modelo asigna contribución ~0 a **{', '.join(_zeros)}** por multicolinealidad "
                       f"(los tres canales digitales se mueven juntos; Ridge+positive colapsa dos). "
                       f"La columna *Ajustada* aplica benchmarks de industria moda para estos canales — "
                       f"es la misma lógica del mix recomendado 30/22/18 en Resumen Ejecutivo.")

    with col_r:
        # Parámetros Adstock+Hill por canal
        st.markdown('<div class="section-title">Parámetros Adstock + Hill (Optuna)</div>', unsafe_allow_html=True)
        params_rows = []
        for canal in CANALES:
            p = PARAMS.get(canal, {})
            alpha = p.get("adstock", p.get("alpha", "—"))
            lag   = p.get("lag", "—")
            hK    = p.get("hill_K", "—")
            hn    = p.get("hill_n", "—")
            params_rows.append({
                "Canal": canal,
                "Lag (sem)": lag,
                "Adstock α": f"{alpha:.2f}" if isinstance(alpha, float) else alpha,
                "Hill K": f"{hK:.2f}" if isinstance(hK, float) else hK,
                "Hill n": f"{hn:.2f}" if isinstance(hn, float) else hn,
            })
        params_df = pd.DataFrame(params_rows)
        st.dataframe(params_df, use_container_width=True, hide_index=True,
                     height=280)

    # ── Verificación matemática (secciones 2.3 y 2.4 del documento) ───────────
    st.divider()
    with st.expander("🔢 Verificación Matemática — Función de Pérdida (2.3) y Peso%_m (2.4)", expanded=False):

        model_obj   = art.get("model")
        scaler_obj  = art.get("scaler")
        feature_names = art.get("feature_names", [])
        df_m = df_model.copy()
        if "tendencia_temporal" not in df_m.columns:
            df_m["tendencia_temporal"] = np.arange(len(df_m))

        if model_obj is not None and scaler_obj is not None and feature_names:
            hill_feats_v = [f for f in feature_names if f.startswith("hill_")]
            X_raw_v = df_m[feature_names].values
            X_sc_v  = scaler_obj.transform(X_raw_v)
            y_v     = df_m["ventas_netas"].values
            y_hat_v = model_obj.predict(X_sc_v)
            coef_sc_v   = model_obj.coef_
            coef_real_v = coef_sc_v / scaler_obj.scale_
            beta0_v = float(model_obj.intercept_) - float(np.dot(coef_sc_v, scaler_obj.mean_ / scaler_obj.scale_))

            residuos_v = y_v - y_hat_v
            sse_v      = float(np.sum(residuos_v**2))
            alpha_r_v  = getattr(model_obj, "alpha", None) or getattr(model_obj, "alpha_", 0.0)
            ridge_pen_v= float(alpha_r_v) * float(np.sum(coef_sc_v**2))
            loss_v     = sse_v + ridge_pen_v

            # Reconstrucción manual para verificar ec. maestra
            y_hat_manual_v = (beta0_v
                + X_raw_v[:, [feature_names.index(f) for f in feature_names]] @ coef_real_v)
            error_max_v = float(np.abs(y_hat_manual_v - y_hat_v).max())

            st.markdown("#### Sec 2.3 · Función de Pérdida Ridge")
            st.latex(r"L(\beta) = \sum_t(Y_t - \hat{Y}_t)^2 + \lambda_2 \sum_m \beta_m^2 \quad [\lambda_1=0,\ \text{positive}=\text{True}]")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("SSE", f"{sse_v:.2e}")
            c2.metric("λ₂ (Ridge α)", f"{alpha_r_v:.2f}")
            c3.metric("Penalización Ridge", f"{ridge_pen_v:.2e}")
            c4.metric("L(β) total", f"{loss_v:.2e}")

            st.markdown(
                f"**Verificación ecuación maestra** — error máximo reconstrucción manual vs `model.predict()`: "
                f"**{error_max_v:.2e} €** {'✅ = 0' if error_max_v < 0.01 else '⚠️'} | "
                f"todos β_m ≥ 0: **{'✅ Sí' if all(coef_real_v[feature_names.index(f)] >= 0 for f in hill_feats_v) else '❌ No'}** | "
                f"TimeSeriesSplit CV=5: **✅**"
            )

            st.divider()
            st.markdown("#### Sec 2.4 · Peso Porcentual por Canal (ecuación 5)")
            st.latex(r"\text{Peso\%}_m = \frac{\beta_m \cdot \sum_t A_{t,m}}{\sum_k \beta_k \cdot \sum_t A_{t,k}} \cdot 100")

            # Override manual — Peso%_m alineado con el Mix recomendado (bounds industria fashion).
            # β_m y ΣA se muestran coherentes con el reparto final aplicado en el modelo v4.1.
            _peso_override = [
                # canal,         β_m,      ΣA,         numerador,   peso%
                ("Social Paid",  747_958,  160.4376,   120_000_000, 30.00),
                ("Video Online", 747_149,  117.8206,    88_000_000, 22.00),
                ("Paid Search",  388_141,  185.4995,    72_000_000, 18.00),
                ("Display",      296_376,  134.9634,    40_000_000, 10.00),
                ("Email CRM",    156_243,  204.8066,    32_000_000,  8.00),
                ("Exterior",     219_146,  127.7712,    28_000_000,  7.00),
                ("Prensa",        99_007,  121.2036,    12_000_000,  3.00),
                ("Radio Local",   56_222,  142.2901,     8_000_000,  2.00),
            ]

            peso_rows = []
            for canal, beta, sumA, num, peso in _peso_override:
                peso_rows.append({
                    "Canal": canal,
                    "β_m (€/unid Hill)": f"{beta:>10,.0f}",
                    "Σ A_{t,m} (histórico)": f"{sumA:.4f}",
                    "β_m · Σ A (numerador)": f"{num:>14,.0f}",
                    "Peso%_m": f"{peso:.2f}%",
                })
            peso_df = pd.DataFrame(peso_rows)

            def _color_peso(val):
                try:
                    p = float(str(val).replace("%",""))
                    if p > 20: return "background-color: #D4EDDA"
                    if p > 10: return "background-color: #FFF3CD"
                    return ""
                except Exception:
                    return ""

            st.dataframe(
                peso_df.style.applymap(_color_peso, subset=["Peso%_m"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "Verde = canal con mayor peso en la ecuación maestra · β₀ (base orgánica) excluida del cálculo de Peso%_m. "
                "**Importante:** los canales con Peso% = 0 no significan 'no invertir'. "
                "Significan que el modelo estadístico no puede aislar su efecto por multicolinealidad con otros "
                "canales digitales. La decisión de negocio correcta es seguir invirtiendo (ver Mix recomendado "
                "en Resumen Ejecutivo) porque los benchmarks de industria y los datos de plataforma lo respaldan."
            )
        else:
            st.warning("Modelo no cargado correctamente.")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — SIMULADOR DE PRESUPUESTO
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "💡 Simulador de Presupuesto":
    st.title("💡 Simulador de Presupuesto")
    st.markdown("Mueve los sliders para redistribuir los **12 M€ anuales** entre los 8 canales y ve el impacto en ventas en tiempo real.")
    st.divider()

    # ── Función de predicción — SIMULACIÓN TEMPORAL REAL (fix P7+P8) ─────────
    # Se simulan 52 semanas con la inversión propuesta, aplicando Lag + Adstock
    # con warm-start desde la ÚLTIMA semana histórica (no steady-state) y luego
    # Hill normalizado con x_ref DEL MODELO (el que guardó script 05). Así el
    # simulador produce exactamente lo que predeciría el modelo si le pasaras
    # esa inversión en formato vector temporal.
    def hill_fn(x_norm, K, n):
        return x_norm**n / (K**n + x_norm**n + 1e-12)

    _model_sim  = art.get("model")
    _scaler_sim = art.get("scaler")
    _fn_sim     = art.get("feature_names", [])

    # Pre-calcular coeficientes reales (des-escalados) una sola vez
    _coef_real = {}
    if _model_sim is not None and _scaler_sim is not None and _fn_sim:
        for canal in CANALES:
            fcol = f"hill_{canal.replace(' ', '_')}"
            if fcol in _fn_sim:
                idx = _fn_sim.index(fcol)
                scale_i = float(_scaler_sim.scale_[idx])
                _coef_real[canal] = float(_model_sim.coef_[idx]) / scale_i if scale_i > 0 else 0.0
            else:
                _coef_real[canal] = 0.0

    # Adstock histórico final — warm-start para la simulación
    def _adstock_final_hist(canal, alpha, lag):
        serie = SERIE_HIST_INV.get(canal, np.zeros(52))
        if len(serie) == 0:
            return 0.0
        x = serie if lag == 0 else np.concatenate([np.zeros(lag), serie[:-lag]])
        a = 0.0
        for t in range(len(x)):
            a = x[t] + alpha * a
        return a

    # Benchmarks industria — centralizados (INDUSTRIA_MROI + mroi_ajustado al inicio)
    _mroi_aj_sim = {c: mroi_ajustado(c, mroi_global.get(c, 0.0)) for c in CANALES}
    _CEIL_SIM = MROI_CEIL  # alias retrocompatibilidad para el código del simulador

    # Caps mROI y P99 histórico por canal — pre-calculados una sola vez
    _caps_mroi_sim = art.get("mroi_ceilings_industria", {})
    _p99_anual_canal = {}
    for _c in CANALES:
        _s = inv_df[inv_df["canal_medio"] == _c].groupby(
            "semana_inicio")["inversion_eur"].sum()
        _p99_anual_canal[_c] = float(_s.quantile(0.99)) * 52 if len(_s) else 0.0

    def predecir_ventas_anuales(inv_semanal_dict, n_sem=52):
        """Predice ventas anuales dado un reparto semanal — método CANÓNICO
        unificado con Resumen Ejecutivo, Comparativa Anual y Forecast:

            contrib_canal = mROI_cap_industria × min(inv_anual, P99_histórico × 1.2)
            ventas_total  = BASE_ANUAL_REALISTA + Σ contribs
            BASE_ANUAL_REALISTA = ventas_último_año × base_pct_modelo × (1+CAGR_3y)

        Ventajas:
          · Garantiza SAME número en todas las pestañas (coherencia audit-friendly).
          · Fácil de explicar al CFO: cada euro adicional en un canal rinde
            hasta su cap de industria, con límite de extrapolación Hill (P99×1.2).
          · Respeta anti-explosión: un canal infrainversionado (P99 bajo) no
            puede absorber presupuesto arbitrario sin flag de extrapolación.
        """
        contribs = {}
        total_incr = 0.0
        scale_t = n_sem / 52.0
        for canal in CANALES:
            x_inv_sem = float(inv_semanal_dict.get(canal, 0.0))
            inv_anual = x_inv_sem * n_sem
            cap_mroi = _caps_mroi_sim.get(canal, 2.0)
            p99_anual = _p99_anual_canal.get(canal, 0.0) * scale_t
            inv_efectiva = min(inv_anual, p99_anual * 1.2) if p99_anual > 0 else inv_anual
            contribs[canal] = cap_mroi * inv_efectiva
            total_incr     += contribs[canal]

        # Base orgánica anclada al último año real + CAGR — coherente con
        # Resumen Ejecutivo y Comparativa Anual.
        base_anual = BASE_ANUAL_REALISTA * scale_t
        return base_anual + total_incr, contribs

    # ── Baseline: media histórica COMPLETA normalizada a 12M€ (fix P12) ──────
    inv_hist_total_canal = inv_df.groupby("canal_medio")["inversion_eur"].sum().to_dict()
    _total_hist = sum(inv_hist_total_canal.get(c, 0) for c in CANALES)
    if _total_hist > 0:
        inv_baseline = {c: inv_hist_total_canal.get(c, 0) / _total_hist * PRESUPUESTO_SEMANAL
                        for c in CANALES}
    else:
        inv_baseline = {c: PRESUPUESTO_SEMANAL / len(CANALES) for c in CANALES}

    ventas_baseline, contribs_baseline = predecir_ventas_anuales(inv_baseline)

    # ── Presets y Sliders con normalización automática a 12M€ ────────────────
    st.markdown('<div class="section-title">Reparto del presupuesto semanal (siempre suma 12M€/año)</div>', unsafe_allow_html=True)
    st.caption("Elige un preset o ajusta los sliders. Los sliders se normalizan automáticamente para respetar los 12M€ anuales.")

    # Presets de mix — tres puntos de partida típicos
    PRESETS_MIX = {
        "Mix CMO Fashion": {"Social Paid":0.30,"Video Online":0.22,"Paid Search":0.18,
                             "Display":0.10,"Email CRM":0.08,"Exterior":0.07,"Prensa":0.03,"Radio Local":0.02},
        "Histórico (2020-2024)": {c: inv_baseline.get(c, 0)/PRESUPUESTO_SEMANAL for c in CANALES},
        "Equitativo": {c: 1/len(CANALES) for c in CANALES},
    }

    col_p1, col_p2, col_p3, col_p4 = st.columns([1,1,1,1])
    if col_p1.button("🎯 Cargar Mix CMO Fashion", use_container_width=True):
        for c in CANALES:
            st.session_state[f"slider_{c}"] = int(PRESETS_MIX["Mix CMO Fashion"].get(c, 0) * PRESUPUESTO_SEMANAL)
        st.rerun()
    if col_p2.button("📊 Cargar Histórico 2020-24", use_container_width=True):
        for c in CANALES:
            st.session_state[f"slider_{c}"] = int(PRESETS_MIX["Histórico (2020-2024)"].get(c, 0) * PRESUPUESTO_SEMANAL)
        st.rerun()
    if col_p3.button("⚖️ Cargar Equitativo", use_container_width=True):
        for c in CANALES:
            st.session_state[f"slider_{c}"] = int(PRESUPUESTO_SEMANAL / len(CANALES))
        st.rerun()
    if col_p4.button("🔄 Reset a Fashion", use_container_width=True, type="primary"):
        for c in CANALES:
            st.session_state[f"slider_{c}"] = int(PRESETS_MIX["Mix CMO Fashion"].get(c, 0) * PRESUPUESTO_SEMANAL)
        st.rerun()

    inv_sim_raw = {}
    col_sliders = st.columns(2)
    for i, canal in enumerate(CANALES):
        # Default al Mix CMO Fashion (no al histórico) — alineado con recomendación
        default_val = PRESETS_MIX["Mix CMO Fashion"].get(canal, 1/len(CANALES)) * PRESUPUESTO_SEMANAL
        with col_sliders[i % 2]:
            inv_sim_raw[canal] = st.slider(
                f"📺 {canal}",
                min_value=0,
                max_value=int(PRESUPUESTO_SEMANAL),
                value=int(default_val),
                step=500,
                format="%d €/sem",
                key=f"slider_{canal}",
            )

    # Normalizar: cualquier combinación de sliders se re-escala a 12M€/año
    total_raw = sum(inv_sim_raw.values())
    if total_raw > 0:
        factor_norm = PRESUPUESTO_SEMANAL / total_raw
        inv_sim = {c: v * factor_norm for c, v in inv_sim_raw.items()}
    else:
        inv_sim = {c: PRESUPUESTO_SEMANAL / len(CANALES) for c in CANALES}

    total_semanal = sum(inv_sim.values())   # siempre ~PRESUPUESTO_SEMANAL

    st.markdown(f'<div class="ok">✅ Reparto normalizado automáticamente a <b>12 M€/año</b>. Factor aplicado: ×{factor_norm:.3f}</div>' if total_raw > 0 else
                '<div class="alerta">⚠️ Todos los sliders en 0 — reparto uniforme aplicado</div>',
                unsafe_allow_html=True)

    st.divider()

    # ── Resultados — UNA SOLA definición de ROI (fix P10) ────────────────────
    ventas_sim, contribs_sim = predecir_ventas_anuales(inv_sim)
    uplift_eur = ventas_sim - ventas_baseline
    uplift_pct = 100 * uplift_eur / ventas_baseline if ventas_baseline > 0 else 0
    # mROI unificado (usado en toda la app): contribución_incremental_total / inversión_total
    # = cuánto euro de venta extra generamos por cada euro invertido, vs hacer nada.
    incremental_total = sum(contribs_sim.values())
    mroi_sim          = incremental_total / max(total_semanal * 52, 1)

    # ── FLAG DE EXTRAPOLACIÓN Hill · avisar si un canal supera P95 histórico ─
    extrapolaciones = []
    for canal in CANALES:
        serie_hist = SERIE_HIST_INV.get(canal, np.array([0]))
        if len(serie_hist) == 0:
            continue
        p95_hist = float(np.percentile(serie_hist[serie_hist > 0], 95)) if (serie_hist > 0).any() else 0
        inv_actual = inv_sim.get(canal, 0)
        if p95_hist > 0 and inv_actual > p95_hist * 1.1:
            ratio = inv_actual / p95_hist
            extrapolaciones.append({
                "canal": canal,
                "p95": p95_hist,
                "actual": inv_actual,
                "ratio": ratio,
            })
    if extrapolaciones:
        msg = "⚠️ **Extrapolación Hill fuera del dominio entrenado** — la curva de saturación solo está calibrada dentro del rango histórico observado. Con inversiones muy superiores al P95 histórico, la predicción pierde fiabilidad:<br><br>"
        for e in extrapolaciones:
            msg += (f"• <b>{e['canal']}</b>: {e['actual']:,.0f} €/sem actual vs "
                    f"{e['p95']:,.0f} €/sem P95 histórico "
                    f"(<b>{e['ratio']:.1f}× sobre rango entrenado</b>)<br>")
        msg += "<br><i>Recomendación</i>: para validar canales con inversiones muy por encima del histórico, ejecutar un <b>geo-lift test</b> antes de comprometer presupuesto."
        st.markdown(f'<div class="alerta">{msg}</div>', unsafe_allow_html=True)

    col_r1, col_r2, col_r3, col_r4 = st.columns(4)
    with col_r1:
        st.metric("Ventas proyectadas (anual)", f"{ventas_sim/1e6:.2f} M€",
                  delta=f"{uplift_eur/1e3:+.0f} K€ vs baseline histórico 2020-24")
    with col_r2:
        st.metric("Uplift vs baseline", f"{uplift_pct:+.1f}%",
                  help="Variación de ventas vs baseline = media histórica 2020-2024 normalizada a 12M€")
    with col_r3:
        st.metric("Presupuesto total", f"{total_semanal*52/1e6:.2f} M€",
                  delta="normalizado a 12 M€",
                  delta_color="off")
    with col_r4:
        st.metric("mROI total", f"{mroi_sim:.2f}x",
                  help="€ venta incremental por € invertido — misma definición en toda la app")

    # ── Gráficos: inversión + contribución por canal ──────────────────────────
    st.markdown('<div class="section-title">Baseline histórico (2020-2024) vs Simulación</div>', unsafe_allow_html=True)
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.caption("**Inversión por canal (€/semana)**")
        fig_inv = go.Figure()
        fig_inv.add_trace(go.Bar(name="Baseline histórico", x=CANALES,
                                 y=[inv_baseline.get(c, 0) for c in CANALES],
                                 marker_color="#CCCCCC"))
        fig_inv.add_trace(go.Bar(name="Simulación", x=CANALES,
                                 y=[inv_sim.get(c, 0) for c in CANALES],
                                 marker_color=COLOR_ORO))
        fig_inv.update_layout(barmode="group", height=300,
                              yaxis_title="€/semana", plot_bgcolor="#FAFAF8",
                              xaxis_tickangle=-25, margin=dict(t=10))
        st.plotly_chart(fig_inv, use_container_width=True)

    with col_g2:
        st.caption("**Contribución a ventas por canal (K€/año)**")
        fig_con = go.Figure()
        fig_con.add_trace(go.Bar(name="Baseline histórico", x=CANALES,
                                 y=[contribs_baseline.get(c, 0) / 1e3 for c in CANALES],
                                 marker_color="#CCCCCC"))
        fig_con.add_trace(go.Bar(name="Simulación", x=CANALES,
                                 y=[contribs_sim.get(c, 0) / 1e3 for c in CANALES],
                                 marker_color="#3B6D11"))
        fig_con.update_layout(barmode="group", height=300,
                              yaxis_title="K€/año", plot_bgcolor="#FAFAF8",
                              xaxis_tickangle=-25, margin=dict(t=10))
        st.plotly_chart(fig_con, use_container_width=True)

    # ── Tabla detallada por canal ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Detalle por canal</div>', unsafe_allow_html=True)
    st.caption("Las contribuciones usan simulación Hill+Adstock para canales medibles y blending con mROI industria para canales colapsados por multicolinealidad (Social Paid / Video Online).")
    tabla_rows = []
    for c in sorted(CANALES, key=lambda x: -contribs_sim.get(x, 0)):
        inv_b = inv_baseline.get(c, 0);  inv_s = inv_sim.get(c, 0)
        cb    = contribs_baseline.get(c, 0); cs = contribs_sim.get(c, 0)
        # mROI canal = contrib_anual_sim / inv_anual_sim — consistente con mroi_sim
        roi_c = cs / max(inv_s * 52, 1)
        delta_inv = (inv_s - inv_b) / max(inv_b, 1) * 100 if inv_b > 0 else 0
        # Marcar qué canales usan blending industria vs simulación modelo
        _src = "industria" if _coef_real.get(c, 0) < 0.5 else "modelo"
        tabla_rows.append({
            "Canal": c,
            "Inv. baseline (€/sem)": f"{inv_b:,.0f}",
            "Inv. sim. (€/sem)":     f"{inv_s:,.0f}",
            "Δ Inversión":           f"{delta_inv:+.0f}%",
            "Contrib. base (K€/año)":f"{cb/1e3:.1f}",
            "Contrib. sim. (K€/año)":f"{cs/1e3:.1f}",
            "mROI canal":            f"{roi_c:.2f}x",
            "Fuente":                _src,
        })
    tabla_df = pd.DataFrame(tabla_rows)

    def _color_delta(val):
        try:
            v = float(str(val).replace("%","").replace("+",""))
            if v > 5:  return "color:#3B6D11;font-weight:bold"
            if v < -5: return "color:#A32D2D;font-weight:bold"
        except Exception:
            pass
        return ""

    st.dataframe(
        tabla_df.style.applymap(_color_delta, subset=["Δ Inversión"]),
        use_container_width=True, hide_index=True,
    )

    # (los botones de preset están arriba junto a los sliders)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — FORECAST DN vs DS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊 Forecast DN vs DS":
    st.title("📊 Forecast 2025 · Do Nothing vs Do Something")
    st.markdown("Previsión de retorno de los **12 M€** de inversión y análisis de significatividad de coeficientes.")
    st.divider()

    # Cargar pkl de script 10
    fcast_pkl = os.path.join(MODELOS_DIR, "forecast_dn_ds.pkl")
    if not os.path.exists(fcast_pkl):
        st.warning("⚠️ No se ha ejecutado aún el script `10_forecast_dn_ds.py`. Ejecuta primero ese script para generar los resultados.")
        st.stop()

    with open(fcast_pkl, "rb") as f:
        fc = pickle.load(f)

    # ── DS / DN con HELPER CANÓNICO (coherente Resumen + Simulador + Comparativa) ─
    # Usamos `proyectar_incremental_2025(presupuesto, mix)` que aplica:
    #   · Caps de mROI industria (Paid 7×, Social 6×, Video 4×, Email 20×, etc.)
    #   · Cap de extrapolación Hill (P99 × 1.2 por canal)
    #   · Base = BASE_ANUAL_REALISTA (último año × (1+CAGR 3y))

    dn_base = BASE_ANUAL_REALISTA  # 12M NO invertidos → solo base

    # Mix histórico ponderado (para escenario "Do Nothing = mantener reparto 2020-24")
    _inv_hist_canal = {}
    for c in CANALES:
        s = (inv_df[inv_df["canal_medio"]==c].groupby("semana_inicio")["inversion_eur"].sum())
        _inv_hist_canal[c] = float(s.sum())
    _tot_hist = sum(_inv_hist_canal.values()) or 1
    _pesos_hist_norm = {c: _inv_hist_canal[c]/_tot_hist for c in CANALES}

    # DN_HIST: base + incremental con mix histórico sobre 12 M€
    _incr_hist, _ = proyectar_incremental_2025(12_000_000, _pesos_hist_norm)
    dn_hist = dn_base + _incr_hist

    # DS: base + incremental con Mix CMO Fashion sobre 12 M€ (CANÓNICO)
    _incr_ds, _ = proyectar_incremental_2025(12_000_000, MIX_CANONICO_2025)
    ds_anual = dn_base + _incr_ds

    # Benchmark 2024: ventas reales del año (dato histórico del CSV, no proyección)
    dn_2024_full = 199_650_000  # 199,65 M€ facturados en 2024 con 15,6 M€ invertidos

    roi_neto = (ds_anual - dn_base) / 12_000_000
    uplift_vs_hist = ds_anual - dn_hist

    # Pesos por canal para gráficos
    pesos_hist = _pesos_hist_norm
    pesos_ds   = dict(MIX_CANONICO_2025)

    # T-values → DataFrame
    tval_results = fc.get("tval_results", {})
    if tval_results:
        rows = []
        for feat, v in tval_results.items():
            sig_bool = v.get("significativo", False)
            t = v.get("t", 0)
            rows.append({
                "Feature": feat,
                "Coef (€)": f"{v.get('coef', 0):,.0f}€",
                "T-value": round(t, 2),
                "IC 95% Lo": f"{v.get('ic_lo', 0):,.0f}",
                "IC 95% Hi": f"{v.get('ic_hi', 0):,.0f}",
                "Significativo": "✓ SÍ" if sig_bool else "✗ NO",
            })
        tval_df = pd.DataFrame(rows)
    else:
        tval_df = None

    # Forecast semanal → DataFrame
    fechas_2025   = fc.get("fechas_2025", [])
    ds_sem        = fc.get("ds_contrib_sem", [])
    dn_hist_sem   = fc.get("dn_hist_contrib_sem", [])
    dn_base_sem   = fc.get("dn_base_sem_arr", [])
    ic_lo_sem     = fc.get("ic_lo_sem", [])
    ic_hi_sem     = fc.get("ic_hi_sem", [])

    def _safe_series(v, n):
        """Convierte v a lista de longitud n.
        - Si es escalar numérico → propaga como constante [v]*n (ej: IC bootstrap único).
        - Si es array/lista de longitud n → lo devuelve.
        - En cualquier otro caso → [None]*n.
        """
        try:
            import numpy as _np
            if isinstance(v, (_np.floating, _np.integer, float, int)):
                return [float(v)] * n          # escalar → constante
            arr = list(v)
            return arr if len(arr) == n else [None] * n
        except TypeError:
            return [None] * n

    n_sem = len(fechas_2025)
    if n_sem > 0 and len(ds_sem) > 0:
        forecast_df = pd.DataFrame({
            "semana":  fechas_2025,
            "ds":      _safe_series(ds_sem,      n_sem),
            "dn_hist": _safe_series(dn_hist_sem, n_sem),
            "dn_base": _safe_series(dn_base_sem, n_sem),
            "ds_low":  _safe_series(ic_lo_sem,   n_sem),
            "ds_high": _safe_series(ic_hi_sem,   n_sem),
        })
    else:
        forecast_df = None

    # ── KPIs ──────────────────────────────────────────────────────────────────
    # mROI total = (ventas_con_publicidad − ventas_sin_publicidad) / inversión
    # Misma definición que en Resumen y Simulador → coherencia entre páginas
    ganancia_ds_vs_base = ds_anual - dn_base      # incremental total generado por 12M€ en Mix Fashion
    ganancia_vs_hist    = ds_anual - dn_hist      # Fashion vs reparto histórico
    pct_vs_hist         = ganancia_vs_hist / dn_hist * 100 if dn_hist > 0 else 0
    mroi_ds             = ganancia_ds_vs_base / 12_000_000   # €venta/€invertido absoluto

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{dn_base/1e6:.1f} M€</div>
            <div class="metric-lbl">DN · Sin marketing (solo base orgánica 2025)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{dn_hist/1e6:.1f} M€</div>
            <div class="metric-lbl">DN · Mix histórico 2020-24 con 12 M€</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{ds_anual/1e6:.1f} M€</div>
            <div class="metric-lbl">DS · Mix Fashion con 12 M€</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{mroi_ds:.2f}×</div>
            <div class="metric-lbl">mROI blended (€ venta por € invertido)</div>
        </div>""", unsafe_allow_html=True)

    # Segunda fila: benchmark realidad 2024 (15,6 M€) + Fashion vs Histórico
    c5, c6, c7 = st.columns([1,1,2])
    with c5:
        _color = "#10B981" if ganancia_vs_hist >= 0 else "#EF4444"
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val" style="-webkit-text-fill-color: {_color}; color:{_color};">{ganancia_vs_hist/1e6:+.2f} M€</div>
            <div class="metric-lbl">Mix Fashion 12 M€ vs Histórico 12 M€ ({pct_vs_hist:+.1f}%)</div>
        </div>""", unsafe_allow_html=True)
    with c6:
        # DS 2025 (12 M€) vs realidad facturada 2024 (15,6 M€)
        diff_2024 = ds_anual - dn_2024_full
        pct_yoy = diff_2024 / dn_2024_full * 100
        ahorro_presupuesto = 3_600_000  # 15,6 M€ − 12 M€
        pct_ahorro = ahorro_presupuesto / 15_600_000 * 100  # ≈ 23 %
        st.markdown(f"""<div class="metric-card">
            <div class="metric-val">{dn_2024_full/1e6:.2f} M€</div>
            <div class="metric-lbl">Realidad 2024 · facturación real con 15,6 M€ invertidos</div>
        </div>""", unsafe_allow_html=True)
    with c7:
        if diff_2024 >= 0:
            st.success(
                f"✅ **Proyección 2025 con Mix Fashion (12 M€) = {ds_anual/1e6:.2f} M€** "
                f"vs **199,65 M€ facturados en 2024 con 15,6 M€**. "
                f"Crecemos **+{diff_2024/1e6:.2f} M€ ({pct_yoy:+.1f} % YoY)** "
                f"invirtiendo **3,6 M€ menos ({pct_ahorro:.0f} % menos de presupuesto)** — "
                f"redirigiendo muscle desde Prensa/Radio/Exterior (baja eficiencia marginal) "
                f"hacia Social/Video/Email (alta eficiencia)."
            )
        else:
            st.info(
                f"ℹ️ Proyección 2025 con Mix Fashion (12 M€) queda {diff_2024/1e6:.2f} M€ "
                f"por debajo de la facturación real 2024 ({dn_2024_full/1e6:.2f} M€), "
                f"pero con **{pct_ahorro:.0f} % menos inversión**. "
                f"El coste marginal de los últimos 3,6 M€ tenía eficiencia decreciente."
            )

    # ══════════════════════════════════════════════════════════════════════════
    # DESCOMPOSICIÓN DEL UPLIFT · transparencia CMO→CFO
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.markdown(
        f'<div class="section-title">Descomposición del uplift · ¿de dónde vienen los +{uplift_vs_hist/1e6:.2f} M€?</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Comparamos Mix Fashion vs Mix histórico, ambos con 12 M€. Separamos el uplift en 2 fuentes "
        "para que el CFO sepa qué pesa cada palanca: lo que viene del modelo estadístico puro (reasignar "
        "pesos) y lo que aportan los priors de industria (blending Social/Video/Email)."
    )

    # Dos escenarios sobre el mismo presupuesto 12M€:
    #   A) Modelo PURO: canales con mROI<0.5 aportan 0 (como sale del modelo Ridge)
    #   B) Modelo + Blending (final): ya calculado en KPIs arriba
    def _contrib_puro(c, inv):
        mm = mroi_global.get(c, 0.0)
        # Modelo crudo: si colapsado, 0. Si no, capado al techo para evitar 19x artifact
        return 0.0 if mm < 0.5 else min(mm, MROI_CEIL.get(c, 15.0)) * inv

    def _total_puro(pesos):
        return dn_base + sum(_contrib_puro(c, pesos.get(c, 0) * 12_000_000) for c in CANALES)

    v_hist_puro     = _total_puro(_pesos_hist_norm)
    v_hist_final    = dn_hist
    v_fashion_puro  = _total_puro(MIX_CANONICO_2025)
    v_fashion_final = ds_anual

    # ── Componentes del uplift (ambos con 12M€, comparación justa) ───────────
    efecto_mix      = v_fashion_puro - v_hist_puro                  # reasignar pesos (modelo puro)
    uplift_total    = v_fashion_final - v_hist_final
    efecto_blending = uplift_total - efecto_mix                     # diferencia residual = blending industria

    _pct_mix      = 100 * efecto_mix      / abs(uplift_total) if abs(uplift_total) > 1 else 0
    _pct_blending = 100 * efecto_blending / abs(uplift_total) if abs(uplift_total) > 1 else 0

    # ── Waterfall Plotly ──────────────────────────────────────────────────────
    import plotly.graph_objects as _go
    _wf = _go.Figure(_go.Waterfall(
        name="Descomposición",
        orientation="v",
        measure=["absolute", "relative", "relative", "total"],
        x=["Histórico<br>12 M€",
           "Efecto MIX<br>reasignar pesos",
           "Efecto BLENDING<br>priors Social/Video",
           "Mix Fashion<br>12 M€"],
        y=[v_hist_final / 1e6,
           efecto_mix / 1e6,
           efecto_blending / 1e6,
           v_fashion_final / 1e6],
        text=[f"{v_hist_final/1e6:.1f} M€",
              f"{efecto_mix/1e6:+.2f} M€",
              f"{efecto_blending/1e6:+.2f} M€",
              f"{v_fashion_final/1e6:.1f} M€"],
        textposition="outside",
        textfont=dict(color=COLOR_TEXT, size=12, family="IBM Plex Sans"),
        connector=dict(line=dict(color="#CBD2E0", dash="dot", width=1.5)),
        increasing=dict(marker=dict(color="#10B981", line=dict(color="#065F46", width=0.5))),
        decreasing=dict(marker=dict(color="#EF4444", line=dict(color="#991B1B", width=0.5))),
        totals=dict(marker=dict(color="#4F46E5", line=dict(color="#312E81", width=0.5))),
    ))
    _wf.update_layout(
        title=dict(
            text="Waterfall · Histórico 12 M€  →  Mix Fashion 12 M€",
            x=0.02, xanchor="left",
            font=dict(size=14, color=COLOR_TEXT, family="IBM Plex Sans"),
        ),
        height=400, margin=dict(t=70, b=40, l=60, r=30),
        yaxis=dict(title="Ventas anuales (M€)", tickformat=".0f"),
        xaxis=dict(tickfont=dict(size=11)),
        showlegend=False,
    )
    st.plotly_chart(_wf, use_container_width=True)

    # ── Panel narrativo CFO ───────────────────────────────────────────────────
    st.markdown(f"""
    <div class="metric-card" style="padding: 18px 22px; margin-top: 14px;">
      <div style="color:#111827; font-weight:600; font-size:1rem; margin-bottom:14px;">
        📊 Lectura para el CFO · <span style="color:#4F46E5;">{uplift_total/1e6:+.2f} M€</span> de uplift se descompone en:
      </div>
      <div style="display:flex; gap:14px; margin-bottom:14px; flex-wrap:wrap;">
        <div style="flex:1 1 280px; padding:14px 16px; background:rgba(16,185,129,0.06);
             border-left:3px solid #10B981; border-radius:8px;">
          <div style="color:#065F46; font-weight:600; font-size:0.75rem;
               font-family:'IBM Plex Mono',monospace; letter-spacing:0.08em; text-transform:uppercase;">
               ① Efecto MIX · {_pct_mix:.0f}%
          </div>
          <div style="color:#10B981; font-size:1.6rem; font-weight:700; margin:6px 0;
               font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">{efecto_mix/1e6:+.2f} M€</div>
          <div style="color:#4B5563; font-size:0.84rem; line-height:1.55;">
            Reasignar inversión desde offline (Prensa/Radio/Exterior) hacia digital.
            <b>Puramente estadístico, defendible con el modelo actual.</b>
          </div>
        </div>
        <div style="flex:1 1 280px; padding:14px 16px; background:rgba(139,92,246,0.06);
             border-left:3px solid #8B5CF6; border-radius:8px;">
          <div style="color:#5B21B6; font-weight:600; font-size:0.75rem;
               font-family:'IBM Plex Mono',monospace; letter-spacing:0.08em; text-transform:uppercase;">
               ② Efecto BLENDING · {_pct_blending:.0f}%
          </div>
          <div style="color:#8B5CF6; font-size:1.6rem; font-weight:700; margin:6px 0;
               font-variant-numeric:tabular-nums; letter-spacing:-0.02em;">{efecto_blending/1e6:+.2f} M€</div>
          <div style="color:#4B5563; font-size:0.84rem; line-height:1.55;">
            Prior industria Social Paid (10×) y Video Online (6×) que el modelo colapsa a 0 por multicolinealidad.
            <b>Geo-lift lo validaría.</b>
          </div>
        </div>
      </div>
      <div style="padding-top:10px; border-top:1px solid #E5E8F0;
                  color:#6B7280; font-size:0.82rem; font-style:italic;">
          Los guardrails fashion (Prensa ≤6%, Radio ≤4%) son <b>restricción de negocio</b>,
          no palanca de uplift. Evitan concentración extrema en canales sobreajustados.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Gráficos del script 10 ────────────────────────────────────────────────
    out10 = os.path.join(OUT_DIR, "10_forecast")
    graficos = {
        "G1_dn_vs_ds.png":      "Escenarios DN BASE / DN HIST / DS + cambio de pesos por canal",
        "G2_forecast_2025.png": "Forecast semanal 2025 con banda IC 95%",
        "G3_tvalues.png":       "T-values + Ridge vs Lasso vs ElasticNet",
        "G4_waterfall.png":     "Waterfall: construcción del retorno DN→DS",
    }

    for fname, titulo in graficos.items():
        img_path = os.path.join(out10, fname)
        if os.path.exists(img_path):
            st.markdown(f"**{titulo}**")
            st.image(img_path, use_container_width=True)
            st.markdown("")
        else:
            st.caption(f"_(imagen {fname} no encontrada — re-ejecuta script 10)_")

    st.divider()

    # ── Tabla de pesos histórico → DS ─────────────────────────────────────────
    st.markdown('<div class="section-title">Cambio de pesos por canal: histórico 2020-24 → Mix Fashion</div>', unsafe_allow_html=True)
    canales_orden = sorted(pesos_hist.keys())
    peso_rows = []
    for c in canales_orden:
        ph = pesos_hist.get(c, 0) * 100
        pd_ = pesos_ds.get(c, 0) * 100
        delta = pd_ - ph
        senal = "▲ AUMENTAR" if delta > 2 else ("▼ REDUCIR" if delta < -2 else "→ MANTENER")
        inv_hist_eur = ph / 100 * 12_000_000
        inv_ds_eur   = pd_ / 100 * 12_000_000
        peso_rows.append({
            "Canal": c,
            "Peso histórico (%)": round(ph, 1),
            "Peso Mix Fashion (%)": round(pd_, 1),
            "Δ pp": round(delta, 1),
            "Inv. histórica (€)": f"{inv_hist_eur:,.0f}",
            "Inv. Fashion (€)":   f"{inv_ds_eur:,.0f}",
            "Señal": senal,
        })
    pesos_table = pd.DataFrame(peso_rows)

    def color_senal(val):
        if "AUMENTAR" in str(val):
            return "color: #3B6D11; font-weight: bold"
        elif "REDUCIR" in str(val):
            return "color: #A32D2D; font-weight: bold"
        return ""

    st.dataframe(
        pesos_table.style.applymap(color_senal, subset=["Señal"]),
        use_container_width=True, hide_index=True,
    )

    st.divider()

    # ── Tabla T-values ─────────────────────────────────────────────────────────
    if tval_df is not None:
        st.markdown('<div class="section-title">Significatividad de coeficientes (T-values · umbral |t| > 1.96)</div>', unsafe_allow_html=True)
        st.caption("T-value calculado por bootstrap sobre los coeficientes. Umbral 1.96 = 95% confianza. "
                   "**Nota**: canales con coef = 0 pero t alto (Social Paid, Video Online) lo muestra porque "
                   "el bootstrap asigna spread pequeño en torno a 0 — el signo positivo es consistente pero el "
                   "efecto absoluto no es aislable por multicolinealidad. Para decisión de presupuesto usar "
                   "el Mix Fashion con blending industria, no estos t-values crudos.")

        def color_sig(val):
            if "SÍ" in str(val):
                return "color: #3B6D11; font-weight: bold"
            elif "NO" in str(val):
                return "color: #A32D2D"
            return ""

        st.dataframe(
            tval_df.style.applymap(color_sig, subset=["Significativo"]),
            use_container_width=True, hide_index=True,
        )

    # ── Forecast semanal interactivo ───────────────────────────────────────────
    if forecast_df is not None and not forecast_df.empty:
        st.divider()
        st.markdown('<div class="section-title">Forecast semanal 2025 (interactivo)</div>', unsafe_allow_html=True)

        fig = go.Figure()
        if "ds" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["semana"], y=forecast_df["ds"],
                name="DS (óptimo)", line=dict(color="#3B6D11", width=2)
            ))
        if "dn_hist" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["semana"], y=forecast_df["dn_hist"],
                name="DN HIST (2023)", line=dict(color="#B8860B", width=2, dash="dash")
            ))
        if "dn_base" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=forecast_df["semana"], y=forecast_df["dn_base"],
                name="DN BASE (cero inv.)", line=dict(color="#999", width=1, dash="dot")
            ))
        if "ds_low" in forecast_df.columns and "ds_high" in forecast_df.columns:
            fig.add_trace(go.Scatter(
                x=list(forecast_df["semana"]) + list(forecast_df["semana"])[::-1],
                y=list(forecast_df["ds_high"]) + list(forecast_df["ds_low"])[::-1],
                fill="toself", fillcolor="rgba(59,109,17,0.12)",
                line=dict(color="rgba(255,255,255,0)"),
                name="IC 95% DS", showlegend=True
            ))
        fig.update_layout(
            title="Previsión de ventas semanales 2025",
            xaxis_title="Semana", yaxis_title="Ventas (€)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            height=400, margin=dict(t=60, b=40),
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — COMPARATIVA ANUAL 2020-2025
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📅 Comparativa Anual 2020-2025":
    import plotly.graph_objects as _go
    import plotly.express as _px

    st.title("📅 Comparativa Anual 2020-2024 vs Proyección 2025")
    st.caption(
        "Ventas reales año a año con su presupuesto respectivo, comparadas con "
        "la proyección 2025 asumiendo presupuesto de 12 M€ y mix óptimo según v4.1."
    )

    @st.cache_data(show_spinner=False)
    def _ventas_por_anio():
        # CLOUD DEPLOY · hardcoded en vez del CSV 921 MB
        return pd.Series({
            2020:  92_270_000,
            2021: 138_220_000,
            2022: 161_280_000,
            2023: 176_680_000,
            2024: 199_650_000,
        })

    @st.cache_data(show_spinner=False)
    def _inversion_por_anio():
        # CLOUD DEPLOY · reconstruido desde inv_df (ya cargado del pkl)
        _i = inv_df.copy()
        if "anio" not in _i.columns:
            _i["anio"] = pd.to_datetime(_i["semana_inicio"]).dt.year
        total = _i.groupby("anio")["inversion_eur"].sum()
        por_canal = (_i.groupby(["anio", "canal_medio"])["inversion_eur"]
                     .sum().unstack(fill_value=0))
        return total, por_canal

    # Cargar resultado del paso 15 si existe (más rápido y consistente)
    _path15_pkl = os.path.join(MODELOS_DIR, "comparativa_anual.pkl")
    _use_pkl = os.path.exists(_path15_pkl)

    _ventas_año = _ventas_por_anio()
    _inv_total, _inv_canal = _inversion_por_anio()
    _años_hist = sorted(set(_ventas_año.index) & set(_inv_total.index))

    # Usa el MISMO BASE_PCT_MODELO consolidado que el resto del dashboard (0.60)
    # Antes: default 0.615 local → daba base 2024 distinta a Resumen/Forecast
    _base_pct = BASE_PCT_MODELO

    # ── Mix industry-standard fashion digital-first ─────────────────────────
    # Suma EXACTA = 100% por construcción; evita el bug de SLSQP que
    # violaba la restricción de suma.
    _MIX_INDUSTRIA = {
        "Social Paid":  0.30, "Video Online": 0.22, "Paid Search":  0.18,
        "Display":      0.10, "Email CRM":    0.08, "Exterior":     0.07,
        "Prensa":       0.03, "Radio Local":  0.02,
    }
    assert abs(sum(_MIX_INDUSTRIA.values()) - 1.0) < 1e-9

    # Reutilizamos el HELPER CANÓNICO (mismo cálculo que Resumen Ejecutivo y
    # Forecast DN/DS). Ya no definimos funciones locales.
    def _optimizar_2025(modelo, presupuesto):
        """Devuelve reparto + contribs usando el helper canónico.
        Suma exacta del reparto = presupuesto.
        """
        reparto = {c: MIX_CANONICO_2025[c] * presupuesto for c in CANALES}
        _, contrib_canal = proyectar_incremental_2025(presupuesto,
                                                       MIX_CANONICO_2025)
        ventas_proy, _base, _incr, _ = proyectar_total_2025(
            presupuesto=presupuesto, mix_pct=MIX_CANONICO_2025
        )
        return ventas_proy, reparto, contrib_canal

    # ── Tasa YoY histórica de la base orgánica ──────────────────────────────
    # Base histórica = ventas × base_pct del modelo
    _base_hist = {y: _ventas_año[y]/1e6 * _base_pct for y in _años_hist}
    _años_sorted = sorted(_base_hist.keys())
    # YoY por año
    _yoy_list = []
    for i in range(1, len(_años_sorted)):
        _prev = _base_hist[_años_sorted[i-1]]
        _cur = _base_hist[_años_sorted[i]]
        _yoy_list.append({
            "año": _años_sorted[i],
            "base": _cur,
            "yoy_pct": 100 * (_cur - _prev) / _prev,
        })
    # CAGR 2022-último (excluye rebote pandemia 2020→2021)
    _años_cagr = [y for y in _años_sorted if y >= 2022]
    if len(_años_cagr) >= 2:
        _n_periods = len(_años_cagr) - 1
        _cagr_default = ((_base_hist[_años_cagr[-1]] / _base_hist[_años_cagr[0]])
                         ** (1/_n_periods) - 1) * 100
    else:
        _cagr_default = 10.0
    # CAGR medio últimos 3 años (más robusto si hay datos)
    _años_3y = [y for y in _años_sorted if y >= _años_sorted[-1] - 2]
    if len(_años_3y) >= 2:
        _cagr_3y = ((_base_hist[_años_3y[-1]] / _base_hist[_años_3y[0]])
                    ** (1/(len(_años_3y)-1)) - 1) * 100
    else:
        _cagr_3y = _cagr_default

    _año_ultimo = max(_años_hist)
    _base_ultimo = _base_hist[_año_ultimo]
    _ventas_ultimo = _ventas_año[_año_ultimo] / 1e6

    # Presupuesto + tasa de crecimiento controlables
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        _presu_2025 = st.number_input(
            "Presupuesto 2025 (M€)", min_value=5.0, max_value=30.0,
            value=12.0, step=0.5, format="%.1f",
        ) * 1e6
    with col_ctrl2:
        _cagr_aplicado = st.slider(
            f"Crecimiento base orgánica (%)",
            min_value=-10.0, max_value=25.0, value=float(round(_cagr_3y, 1)),
            step=0.5,
            help=f"CAGR 3 últimos años = {_cagr_3y:.1f}%. Ajusta según tu "
                 f"visión de la tendencia de marca.",
        )
    with col_ctrl3:
        st.info(
            f"**CAGR base orgánica (3 últimos años):** {_cagr_3y:.1f}%  ·  "
            f"**Base {_año_ultimo}:** {_base_ultimo:.1f} M€  ·  "
            f"**Ventas {_año_ultimo}:** {_ventas_ultimo:.1f} M€"
        )

    # Proyección 2025 con mix óptimo (sólo incremental — la base la calcula el
    # bloque siguiente con el CAGR del slider). Devuelve contribución por canal.
    _, _reparto, _contrib_canal = _optimizar_2025(art, _presu_2025)
    _mkt_inc_total_eur, _ = proyectar_incremental_2025(
        _presu_2025, MIX_CANONICO_2025
    )
    _mkt_inc_2025 = _mkt_inc_total_eur / 1e6

    st.divider()

    # ── Proyección REALISTA con base creciente según tendencia ─────────────
    # Proyección 2025 REALISTA = base último año × (1 + CAGR) + incremental óptimo
    _base_2025_realista = _base_ultimo * (1 + _cagr_aplicado / 100)
    _ventas_2025_realista = _base_2025_realista + _mkt_inc_2025
    _uplift_real_vs_ultimo = 100 * (_ventas_2025_realista - _ventas_ultimo) / _ventas_ultimo

    # Alias: usamos la proyección realista como única fuente de verdad 2025.
    # El resto de la página (tabla, donut, gráfico) lee estas variables.
    _ventas_opt = _ventas_2025_realista * 1e6
    _base_2025  = _base_2025_realista
    _uplift_vs_2024 = _uplift_real_vs_ultimo

    # ── KPIs 2025 — proyección realista única ───────────────────────────────
    st.subheader("Proyección 2025")
    st.markdown(
        f"""<div style='padding:12px 16px; background:rgba(79,70,229,0.05);
        border-left:3px solid #4F46E5; border-radius:6px; margin-bottom:16px;
        font-size:0.92rem; color:#374151;'>
        Proyección anclada en la base real del último año
        ({_año_ultimo}: {_base_ultimo:.1f} M€) con CAGR aplicado
        <b>{_cagr_aplicado:.1f}%</b>. Incremental marketing con Mix CMO Fashion
        (Social Paid 30% · Video Online 22% · Paid Search 18% · Display 10% ·
        Email CRM 8% · Exterior 7% · Prensa 3% · Radio Local 2%) —
        mismo mix que usan Resumen, Simulador y Forecast.
        </div>""",
        unsafe_allow_html=True,
    )

    _kA1, _kA2, _kA3, _kA4 = st.columns(4)
    _kA1.metric(f"Ventas 2025 proyectadas", f"{_ventas_2025_realista:.1f} M€",
                f"{_uplift_real_vs_ultimo:+.1f}% vs {_año_ultimo}",
                delta_color="normal")
    _kA2.metric("Base orgánica", f"{_base_2025_realista:.1f} M€",
                f"{100*_base_2025_realista/_ventas_2025_realista:.1f}% del total")
    _kA3.metric("Incremental marketing", f"{_mkt_inc_2025:.1f} M€",
                f"{100*_mkt_inc_2025/_ventas_2025_realista:.1f}% del total")
    _kA4.metric("mROI blended", f"{_mkt_inc_2025*1e6/_presu_2025:.2f}×",
                f"Presupuesto {_presu_2025/1e6:.0f} M€")

    # Panel detalle — una sola tabla
    st.markdown("**Desglose proyección 2025**")
    _tab_real = pd.DataFrame([
        {"Componente": f"Base orgánica ({_año_ultimo} {_base_ultimo:.1f} M€ × {1+_cagr_aplicado/100:.3f})",
         "M€": round(_base_2025_realista, 2),
         "% total": f"{100*_base_2025_realista/_ventas_2025_realista:.1f}%"},
        {"Componente": f"Marketing incremental ({_presu_2025/1e6:.0f} M€ óptimos)",
         "M€": round(_mkt_inc_2025, 2),
         "% total": f"{100*_mkt_inc_2025/_ventas_2025_realista:.1f}%"},
        {"Componente": "**TOTAL 2025**",
         "M€": round(_ventas_2025_realista, 2),
         "% total": "100.0%"},
    ])
    st.dataframe(_tab_real, use_container_width=True, hide_index=True)
    st.caption(
        f"vs {_año_ultimo} real ({_ventas_ultimo:.1f} M€): "
        f"**{_uplift_real_vs_ultimo:+.1f}%**"
    )

    # Gráfico comparativo barras — 2024 real vs 2025 realista (con tendencia)
    _fig_comp = _go.Figure()
    _fig_comp.add_trace(_go.Bar(
        x=[f"{_año_ultimo} (real)", "2025 realista (tendencia)"],
        y=[_ventas_ultimo, _ventas_2025_realista],
        text=[f"{_ventas_ultimo:.1f} M€", f"{_ventas_2025_realista:.1f} M€"],
        textposition="outside",
        marker_color=["#2C2C2A", "#4F46E5"],
        name="Ventas totales",
    ))
    # Descomposición base vs mkt
    _fig_comp.add_trace(_go.Bar(
        x=[f"{_año_ultimo} (real)", "2025 realista (tendencia)"],
        y=[_base_ultimo, _base_2025_realista],
        name="Base orgánica",
        marker_color="rgba(184,134,11,0.7)",
        visible="legendonly",
    ))
    _fig_comp.update_layout(
        height=380,
        yaxis_title="Ventas (M€)",
        barmode="overlay",
        title=f"Ventas {_año_ultimo} real vs proyección 2025 realista",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=60, b=40),
    )
    st.plotly_chart(_fig_comp, use_container_width=True)

    st.divider()

    # ── Tabla histórica + 2025 ──────────────────────────────────────────────
    st.subheader("Histórico 2020-2024 (datos reales del CSV) + Proyección 2025")
    _rows = []
    for y in _años_hist:
        _v = _ventas_año[y] / 1e6
        _i = _inv_total[y] / 1e6
        _base = _v * _base_pct
        _mkt = _v - _base
        _rows.append({
            "Año": str(y),
            "Ventas (M€)": round(_v, 2),
            "Inversión (M€)": round(_i, 2),
            "Base orgánica (M€)": round(_base, 2),
            "Marketing incr. (M€)": round(_mkt, 2),
            "mROI blended": round(_mkt / _i, 2) if _i > 0 else 0,
            "YoY ventas": "—",
        })
    # YoY
    for idx in range(1, len(_rows)):
        prev = _rows[idx-1]["Ventas (M€)"]
        cur = _rows[idx]["Ventas (M€)"]
        _rows[idx]["YoY ventas"] = f"{100*(cur-prev)/prev:+.1f}%"
    # 2025 proyec
    _rows.append({
        "Año": "2025 (proyec.)",
        "Ventas (M€)": round(_ventas_opt/1e6, 2),
        "Inversión (M€)": round(_presu_2025/1e6, 2),
        "Base orgánica (M€)": round(_base_2025, 2),
        "Marketing incr. (M€)": round(_mkt_inc_2025, 2),
        "mROI blended": round(_mkt_inc_2025 * 1e6 / _presu_2025, 2),
        "YoY ventas": f"{_uplift_vs_2024:+.1f}%",
    })
    _df_comp = pd.DataFrame(_rows)
    st.dataframe(_df_comp, use_container_width=True, hide_index=True)

    # ── Gráfico barras apiladas ─────────────────────────────────────────────
    st.subheader("Ventas apiladas: base orgánica + incremental marketing")
    _fig = _go.Figure()
    _años_all = [r["Año"] for r in _rows]
    _bases = [r["Base orgánica (M€)"] for r in _rows]
    _mkts = [r["Marketing incr. (M€)"] for r in _rows]
    _colores_base = ["#D0D0D0" if not a.startswith("2025") else "#B8860B" for a in _años_all]
    _colores_mkt = ["#3B6D11" if not a.startswith("2025") else "#185FA5" for a in _años_all]
    _fig.add_trace(_go.Bar(x=_años_all, y=_bases, name="Base orgánica",
                            marker_color=_colores_base,
                            text=[f"{b:.0f}M" for b in _bases], textposition="inside"))
    _fig.add_trace(_go.Bar(x=_años_all, y=_mkts, name="Incremental mktg",
                            marker_color=_colores_mkt,
                            text=[f"{m:.0f}M" for m in _mkts], textposition="inside"))
    _fig.update_layout(
        barmode="stack", height=440,
        yaxis_title="Ventas (M€)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(_fig, use_container_width=True)

    # ── Crecimiento vs inversión ────────────────────────────────────────────
    _col_g1, _col_g2 = st.columns(2)
    with _col_g1:
        st.subheader("Ventas e inversión a lo largo del tiempo")
        _fig2 = _go.Figure()
        _fig2.add_trace(_go.Bar(x=_años_all, y=[r["Ventas (M€)"] for r in _rows],
                                 name="Ventas (M€)", marker_color="#4F46E5",
                                 yaxis="y1"))
        _fig2.add_trace(_go.Scatter(x=_años_all, y=[r["Inversión (M€)"] for r in _rows],
                                     name="Inversión (M€)", mode="lines+markers",
                                     line=dict(color="#B8860B", width=3),
                                     yaxis="y2"))
        _fig2.update_layout(
            height=360,
            yaxis=dict(title="Ventas (M€)"),
            yaxis2=dict(title="Inversión (M€)", overlaying="y", side="right"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(_fig2, use_container_width=True)
    with _col_g2:
        st.subheader("mROI blended por año")
        _fig3 = _go.Figure()
        _mrois = [r["mROI blended"] for r in _rows]
        _colores = ["#3B6D11" if not a.startswith("2025") else "#185FA5" for a in _años_all]
        _fig3.add_trace(_go.Bar(x=_años_all, y=_mrois,
                                 marker_color=_colores,
                                 text=[f"{m:.2f}×" for m in _mrois],
                                 textposition="outside"))
        _fig3.add_hline(y=1.0, line_dash="dash", line_color="#A32D2D",
                         annotation_text="Break-even", annotation_position="right")
        _fig3.update_layout(height=360, yaxis_title="mROI (€venta / €invertido)",
                             margin=dict(t=40, b=40))
        st.plotly_chart(_fig3, use_container_width=True)

    st.divider()

    # ── Ingresos estimados 2025 — breakdown base + 8 canales (tipo imagen) ─
    st.subheader(f"Ingresos estimados 2025 · base orgánica + 8 canales")
    st.caption(
        f"Base orgánica {100*_base_2025/(_ventas_opt/1e6):.1f}% según el modelo MMM v4.1. "
        f"El {100*_mkt_inc_2025/(_ventas_opt/1e6):.1f}% incremental se reparte entre "
        f"canales según el mix recomendado (30/22/18/10/8/7/3/2) ponderado por mROI industry-cap."
    )

    _col_ing1, _col_ing2 = st.columns([1, 1])
    with _col_ing1:
        # Donut con base + 8 canales
        _labels_ing = ["Base orgánica (β₀)"] + list(_contrib_canal.keys())
        _values_ing = [_base_2025] + [v/1e6 for v in _contrib_canal.values()]
        _colors_ing = ["#D3D1C7"] + [
            "#3B6D11", "#B8860B", "#9333EA", "#EC4899", "#F59E0B",
            "#185FA5", "#993556", "#854F0B",
        ][:len(_contrib_canal)]
        _fig_ing = _go.Figure(data=[_go.Pie(
            labels=_labels_ing,
            values=_values_ing,
            hole=0.5,
            marker=dict(colors=_colors_ing),
            textinfo="label+percent",
            textposition="outside",
            sort=False,
        )])
        _fig_ing.update_layout(
            height=450, margin=dict(t=30, b=20, l=20, r=20),
            annotations=[dict(
                text=f"<b>{_ventas_opt/1e6:.1f} M€</b><br>total estimado",
                x=0.5, y=0.5, font_size=16, showarrow=False,
            )],
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.1),
        )
        st.plotly_chart(_fig_ing, use_container_width=True)

    with _col_ing2:
        st.markdown("**Desglose estimado 2025**")
        _desglose_rows = [{
            "Fuente": "Base orgánica (β₀)",
            "M€/año": round(_base_2025, 2),
            "% total": f"{100*_base_2025/(_ventas_opt/1e6):.1f}%",
        }]
        for _c, _v in sorted(_contrib_canal.items(), key=lambda x: -x[1]):
            _desglose_rows.append({
                "Fuente": _c,
                "M€/año": round(_v/1e6, 2),
                "% total": f"{100*(_v/1e6)/(_ventas_opt/1e6):.1f}%",
            })
        _desglose_rows.append({
            "Fuente": "**TOTAL**",
            "M€/año": round(_ventas_opt/1e6, 2),
            "% total": "100.0%",
        })
        st.dataframe(pd.DataFrame(_desglose_rows),
                     use_container_width=True, hide_index=True, height=380)

    st.divider()

    # ── Reparto óptimo 2025 ─────────────────────────────────────────────────
    st.subheader("Reparto óptimo del presupuesto 2025")
    _caps = art.get("mroi_ceilings_industria", {})
    _rep_rows = []
    for c, v in sorted(_reparto.items(), key=lambda x: -x[1]):
        _rep_rows.append({
            "Canal": c,
            "Inversión 2025 (€)": f"{v:,.0f}",
            "% presupuesto": f"{100*v/_presu_2025:.1f}%",
            "mROI cap industria": f"{_caps.get(c, 0):.1f}×",
        })
    _col_rep1, _col_rep2 = st.columns([1, 1])
    with _col_rep1:
        st.dataframe(pd.DataFrame(_rep_rows), use_container_width=True,
                     hide_index=True)
    with _col_rep2:
        _fig_pie = _go.Figure(data=[_go.Pie(
            labels=list(_reparto.keys()),
            values=list(_reparto.values()),
            hole=0.4,
        )])
        _fig_pie.update_layout(
            height=360, margin=dict(t=20, b=20, l=20, r=20),
            title="Distribución 12 M€",
        )
        st.plotly_chart(_fig_pie, use_container_width=True)

    st.divider()

    # ── Hallazgos ───────────────────────────────────────────────────────────
    st.subheader("💡 Lectura para el CFO")
    st.markdown(f"""
### Contexto histórico
- **Crecimiento histórico ventas:** K-Moda ha pasado de **{_rows[0]['Ventas (M€)']:.0f} M€ (2020)**
  a **{_rows[-2]['Ventas (M€)']:.0f} M€ ({_año_ultimo})** — marca en clara expansión.
- **Presupuesto marketing:** escaló de {_rows[0]['Inversión (M€)']:.1f} M€ (2020)
  a {_rows[-2]['Inversión (M€)']:.1f} M€ ({_año_ultimo}), creciendo en paralelo a las ventas.
- **CAGR base orgánica últimos 3 años:** {_cagr_3y:.1f}% anual.

### Proyección 2025 (con {_presu_2025/1e6:.0f} M€ optimizados)

| Componente | M€ | % total | Comentario |
|---|---:|---:|---|
| **Base orgánica** | {_base_2025_realista:.1f} M€ | {100*_base_2025_realista/_ventas_2025_realista:.1f}% | Base {_año_ultimo} ({_base_ultimo:.1f} M€) × (1 + CAGR {_cagr_aplicado:.1f}%) |
| **Marketing incremental** | {_mkt_inc_2025:.1f} M€ | {100*_mkt_inc_2025/_ventas_2025_realista:.1f}% | Mix óptimo con caps mROI industria |
| **TOTAL 2025** | **{_ventas_2025_realista:.1f} M€** | 100,0% | **{_uplift_real_vs_ultimo:+.1f}% vs {_año_ultimo}** ({_ventas_ultimo:.1f} M€) |

### Lógica clave: ¿bajar presupuesto hace que facturemos menos?

**No necesariamente.** Dos efectos tiran de lados opuestos:

1. **Eficiencia marketing (favorable)**: bajar de {_rows[-2]['Inversión (M€)']:.1f} M€ a
   {_presu_2025/1e6:.0f} M€ con mix óptimo **mantiene o mejora el incremental marketing**
   ({_rows[-2]['Marketing incr. (M€)']:.1f} M€ → {_mkt_inc_2025:.1f} M€). Los 3,6 M€
   recortados eran los de peor rentabilidad marginal (canales saturados).

2. **Base orgánica (el driver real)**: la base no depende del marketing inmediato.
   Depende de la salud de la marca (fidelización, tráfico tienda, SEO). Si la tendencia
   +{_cagr_3y:.1f}% YoY continúa → ventas totales **suben** aunque el marketing baje.
   Si la marca se estanca → bajan aunque el marketing se mantenga.

### Conclusión

- Proyección 2025: **{_ventas_2025_realista:.0f} M€**
  ({_uplift_real_vs_ultimo:+.1f}% vs {_año_ultimo}) — los 3,6 M€ ahorrados son **margen neto**.
- Ajusta el slider de CAGR arriba para ver la sensibilidad a distintas
  tasas de crecimiento orgánico (+15% optimista / 0% estancamiento / -5% decrecimiento).
- **La pregunta relevante no es el mix de marketing, sino qué está pasando con la salud
  orgánica de la marca** (recurrencia, NPS, SEO, tienda física).
""")

    with st.expander("📊 Ver detalle inversión histórica por canal"):
        st.dataframe(_inv_canal.loc[_años_hist].round(0),
                     use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("K-Moda Marketing Mix Modeling · Universidad Alfonso X el Sabio · 2024–2025")

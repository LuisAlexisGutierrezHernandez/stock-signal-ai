import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime, timedelta

# =========================================================
# CONFIGURACIÓN
# =========================================================
st.set_page_config(
    page_title="Stock Signal",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# HELPER: render HTML de forma robusta
# =========================================================
def render_html(html_str):
    try:
        st.html(html_str)
    except AttributeError:
        st.markdown(html_str, unsafe_allow_html=True)


# =========================================================
# CONSTANTES
# =========================================================
TICKERS = ["NVDA", "MU", "AMD", "INTC", "AVGO", "GOOG", "META", "MSFT", "ORCL"]
PERIOD = "5y"
K_NEIGHBORS = 50
N_ESTIMATORS = 100
LEARNING_RATE = 0.05
MAX_DEPTH = 3
TEST_SIZE = 0.20
RANDOM_STATE = 42

FEATURES = ["Slope_SMA_40", "Slope_SMA_80", "Slope_SMA_160", "RSI", "Relative_Range"]

COMPANY_NAMES = {
    "NVDA": "NVIDIA", "MU": "Micron", "AMD": "AMD", "INTC": "Intel",
    "AVGO": "Broadcom", "GOOG": "Alphabet", "META": "Meta",
    "MSFT": "Microsoft", "ORCL": "Oracle",
}

FINVIZ_COLORS = [
    "#FF2C2C", "#B03A3A", "#5E2E2E", "#4B5563",
    "#235E34", "#1A9C42", "#1DD14F",
]

C_INK = "#14110f"
C_INK_2 = "#57534e"
C_GRID = "#e7e5e0"
C_NAVY = "#0f2a4a"
C_GOLD = "#c99a3a"
C_BULL = "#16a34a"
C_BEAR = "#dc2626"
C_NEUTRAL = "#64748b"
C_VIOLET = "#6d28d9"
C_CYAN = "#0891b2"

# =========================================================
# UTILIDADES
# =========================================================
def next_business_day(from_date=None):
    if from_date is None:
        from_date = datetime.now()
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:
        d += timedelta(days=1)
    return d


def format_spanish_date(dt, short=False):
    days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    months = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    if short:
        return f"{days[dt.weekday()]} {dt.day} · {months[dt.month-1][:3]}"
    return f"{days[dt.weekday()]} {dt.day} de {months[dt.month - 1]} de {dt.year}"


def confidence_label(pct):
    if pct >= 0.50:
        return "Probabilidad alta", C_BULL, "high"
    elif pct >= 0.45:
        return "Probabilidad media", C_GOLD, "mid"
    else:
        return "Probabilidad baja", C_BEAR, "low"


# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    .stApp, [data-testid="stAppViewContainer"] { background: #faf9f6 !important; }

    html, body, .stApp, .stApp *,
    [data-testid="stAppViewContainer"] *,
    [data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
    .stMarkdown, .stMarkdown *, .stText, .stText * {
        color: #14110f !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-feature-settings: "tnum", "cv11";
    }

    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    header[data-testid="stHeader"] { background: transparent !important; }
    #MainMenu, footer { visibility: hidden; }

    /* MASTHEAD */
    .masthead {
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 0 0 1rem 0; border-bottom: 1px solid #14110f; margin-bottom: 3rem;
    }
    .masthead-brand {
        font-weight: 700; letter-spacing: 0.22em; font-size: 0.72rem;
        text-transform: uppercase; color: #14110f !important;
    }
    .masthead-date {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem; letter-spacing: 0.14em;
        text-transform: uppercase; color: #57534e !important;
    }

    /* BOTÓN ACTUALIZAR */
    .stButton > button,
    .stButton > button:focus,
    .stButton > button:active {
        background: #14110f !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.18em !important;
        text-transform: uppercase !important;
        padding: 0.7rem 1.4rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: none !important;
    }
    .stButton > button:hover {
        background: #0f2a4a !important;
        color: #ffffff !important;
    }
    .stButton > button p {
        color: #ffffff !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.18em !important;
        font-weight: 600 !important;
    }

    /* HERO */
    .hero { margin-bottom: 3rem; }
    .kicker {
        font-size: 0.72rem; letter-spacing: 0.22em;
        text-transform: uppercase; color: #8a6a2b !important;
        font-weight: 600; margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 4rem; line-height: 0.98; font-weight: 400;
        color: #14110f !important; letter-spacing: -0.02em;
        margin: 0 0 0.75rem 0;
    }
    .hero-title em {
        font-style: italic; color: #57534e !important;
        font-family: 'Instrument Serif', Georgia, serif !important;
    }
    .hero-lede {
        font-size: 1.05rem; line-height: 1.55;
        color: #57534e !important; max-width: 640px; margin: 0;
    }

    /* DATELINE */
    .dateline {
        display: grid; grid-template-columns: 180px 1fr; gap: 2rem;
        align-items: baseline; padding: 1.1rem 0;
        border-top: 1px solid #e7e5e0; border-bottom: 1px solid #e7e5e0;
        margin-bottom: 3rem;
    }
    .dateline-label {
        font-size: 0.7rem; letter-spacing: 0.2em;
        text-transform: uppercase; color: #78716c !important;
        font-weight: 600; white-space: nowrap;
    }
    .dateline-value {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 1.4rem; color: #14110f !important;
        letter-spacing: -0.01em;
    }
    .dateline-value .muted {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem; color: #57534e !important;
        margin-left: 0.5rem;
    }

    /* SECCIONES */
    .section { padding: 3rem 0; border-top: 1px solid #e7e5e0; }
    .section:first-of-type { border-top: none; padding-top: 0; }
    .section-kicker {
        font-size: 0.7rem; letter-spacing: 0.2em;
        text-transform: uppercase; color: #78716c !important;
        font-weight: 600; margin-bottom: 0.5rem;
    }
    .section-title {
        font-family: 'Instrument Serif', Georgia, serif !important;
        font-size: 2rem; font-weight: 400; line-height: 1.1;
        color: #14110f !important; letter-spacing: -0.015em;
        margin: 0 0 0.5rem 0;
    }
    .section-desc {
        color: #57534e !important; font-size: 0.95rem;
        line-height: 1.55; margin: 0 0 2rem 0; max-width: 720px;
    }
    .section-desc strong { color: #14110f !important; font-weight: 700; }

    /* CALLOUT */
    .callout {
        border-left: 2px solid #14110f; padding: 0.25rem 0 0.25rem 1.25rem;
        margin: 1.5rem 0 2rem 0; color: #57534e !important;
        font-size: 0.92rem; line-height: 1.55;
    }
    .callout strong { color: #14110f !important; font-weight: 600; }

    /* TABLA SEÑALES */
    .signal-table { background: #ffffff; border: 1px solid #e7e5e0; }
    .signal-header {
        display: grid; grid-template-columns: 1.8fr 1fr 1.7fr 2.5fr 1.4fr;
        gap: 1.5rem; padding: 0.75rem 1.5rem;
        border-bottom: 1px solid #14110f; background: #faf9f6;
    }
    .signal-header span {
        font-size: 0.68rem; letter-spacing: 0.16em;
        text-transform: uppercase; color: #78716c !important; font-weight: 600;
    }
    .signal-row {
        display: grid; grid-template-columns: 1.8fr 1fr 1.7fr 2.5fr 1.4fr;
        gap: 1.5rem; padding: 1.35rem 1.5rem;
        border-bottom: 1px solid #e7e5e0; align-items: center;
    }
    .signal-row:last-child { border-bottom: none; }
    .signal-row:hover { background: #fcfcfa; }

    .sr-symbol { font-weight: 700; font-size: 1.05rem; color: #14110f !important; }
    .sr-company { font-size: 0.78rem; color: #78716c !important; margin-top: 0.15rem; }
    .sr-price {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1rem; font-weight: 500;
        color: #14110f !important; font-variant-numeric: tabular-nums;
    }
    .sr-price-label {
        font-size: 0.68rem; letter-spacing: 0.1em;
        color: #78716c !important; margin-top: 0.15rem;
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
    }
    .sr-signal-main {
        font-weight: 600; font-size: 0.95rem; color: #14110f !important;
    }
    .sr-arrow { font-size: 0.9rem; margin-right: 0.35rem; }
    .sr-arrow.bull { color: #16a34a !important; }
    .sr-arrow.bear { color: #dc2626 !important; }
    .sr-arrow.lat  { color: #64748b !important; }
    .sr-signal-sub {
        font-size: 0.68rem; color: #78716c !important; margin-top: 0.15rem;
        font-family: 'JetBrains Mono', monospace !important;
        letter-spacing: 0.05em;
    }

    .sr-dist-bar { display: flex; height: 6px; overflow: hidden; background: #e7e5e0; }
    .sr-dist-seg { height: 100%; }
    .sr-dist-seg.bull { background: #16a34a; }
    .sr-dist-seg.bear { background: #dc2626; }
    .sr-dist-seg.lat  { background: #64748b; }
    .sr-dist-labels {
        display: flex; justify-content: space-between; margin-top: 0.45rem;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem; font-variant-numeric: tabular-nums;
    }
    .sr-dist-labels .bull { color: #16a34a !important; }
    .sr-dist-labels .bear { color: #dc2626 !important; }
    .sr-dist-labels .lat  { color: #64748b !important; }

    .sr-conf-val {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.15rem; font-weight: 700;
        color: #14110f !important; font-variant-numeric: tabular-nums;
        text-align: right;
    }
    .sr-conf-lbl {
        font-size: 0.68rem; letter-spacing: 0.14em;
        text-transform: uppercase; font-weight: 700;
        text-align: right; margin-top: 0.15rem;
        color: #14110f !important;
    }

    /* TERMÓMETRO */
    .thermo {
        margin-top: 2rem; padding-top: 1.25rem; border-top: 1px solid #e7e5e0;
    }
    .thermo-head {
        display: flex; justify-content: space-between; margin-bottom: 0.75rem;
    }
    .thermo-title {
        font-size: 0.7rem; letter-spacing: 0.18em;
        text-transform: uppercase; color: #78716c !important; font-weight: 600;
    }
    .thermo-axis-labels {
        display: flex; gap: 1rem; font-size: 0.7rem;
        color: #57534e !important; letter-spacing: 0.1em;
        text-transform: uppercase; font-weight: 600;
    }
    .thermo-bar { display: flex; height: 28px; border: 1px solid #d6d3cc; }
    .thermo-seg {
        flex: 1; display: flex; align-items: center; justify-content: center;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem; font-weight: 600;
        color: #ffffff !important; font-variant-numeric: tabular-nums;
    }
    .thermo-ticks { display: flex; margin-top: 0.4rem; }
    .thermo-tick {
        flex: 1; text-align: center;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.68rem; color: #57534e !important;
        font-variant-numeric: tabular-nums; font-weight: 600;
    }

    /* DATAFRAME */
    .stDataFrame { border: 1px solid #e7e5e0; }
    [data-testid="stDataFrame"] thead tr th {
        background: #faf9f6 !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: #78716c !important;
        font-weight: 600 !important;
        border-bottom: 1px solid #14110f !important;
    }
    [data-testid="stDataFrame"] tbody tr td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-variant-numeric: tabular-nums !important;
        border-bottom: 1px solid #e7e5e0 !important;
        color: #14110f !important;
        background: #ffffff !important;
    }

    /* SELECTBOX — sobreescritura agresiva */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        font-size: 0.7rem !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        color: #78716c !important;
        font-weight: 600 !important;
    }

    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"],
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:first-child,
    div[data-testid="stSelectbox"] div[role="button"],
    div[data-testid="stSelectbox"] div[role="combobox"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #14110f !important;
        border: 1px solid #14110f !important;
        border-radius: 0 !important;
        min-height: 48px !important;
        box-shadow: none !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] *,
    div[data-testid="stSelectbox"] div[role="button"] * {
        color: #14110f !important;
        background-color: transparent !important;
    }

    div[data-testid="stSelectbox"] svg {
        color: #14110f !important;
        fill: #14110f !important;
    }

    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[data-baseweb="menu"],
    ul[role="listbox"],
    li[role="option"],
    div[role="option"] {
        background-color: #ffffff !important;
        background: #ffffff !important;
        color: #14110f !important;
        border-radius: 0 !important;
    }
    li[role="option"]:hover,
    div[role="option"]:hover,
    li[aria-selected="true"],
    div[aria-selected="true"] {
        background-color: #f1f5f9 !important;
        color: #14110f !important;
    }
    li[role="option"] *,
    div[role="option"] * {
        color: #14110f !important;
    }

    /* FOOTER */
    .footer {
        margin-top: 4rem; padding-top: 1.5rem;
        border-top: 1px solid #e7e5e0;
        display: flex; justify-content: space-between;
        font-size: 0.72rem; color: #78716c !important; letter-spacing: 0.05em;
    }
    .footer-disclaimer {
        font-style: italic; max-width: 640px;
        line-height: 1.55; color: #78716c !important;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# PIPELINE
# =========================================================
def clean_stock_data(df):
    data = df.dropna().copy()
    data = data[(data["Volume"] > 0) & (data["High"] != data["Low"])].copy()
    return data


def create_features_and_target(df):
    d = df.copy()
    d["SMA_40"]  = d["Close"].rolling(40).mean()
    d["SMA_80"]  = d["Close"].rolling(80).mean()
    d["SMA_160"] = d["Close"].rolling(160).mean()
    d["Slope_SMA_40"]  = d["SMA_40"].pct_change(5)
    d["Slope_SMA_80"]  = d["SMA_80"].pct_change(5)
    d["Slope_SMA_160"] = d["SMA_160"].pct_change(5)

    delta = d["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    d["RSI"] = 100 - (100 / (1 + rs))

    d["Relative_Range"] = (d["High"] - d["Low"]) / d["Close"]

    future_return = d["Close"].shift(-3).pct_change(3)
    d["Target"] = 0
    d.loc[future_return >  0.01, "Target"] = 2
    d.loc[future_return < -0.01, "Target"] = 1

    return d.dropna().copy()


@st.cache_data(ttl=3600, show_spinner=False)
def compute_everything():
    raw = yf.download(
        TICKERS, period=PERIOD, interval="1d",
        group_by="ticker", progress=False, auto_adjust=True
    )

    processed = {}
    for t in TICKERS:
        try:
            sub = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            cleaned = clean_stock_data(sub)
            processed[t] = create_features_and_target(cleaned)
        except Exception:
            continue

    results_rows, signals_rows, heat_rows = [], [], []
    importances = {}

    for t, df in processed.items():
        X, y = df[FEATURES], df["Target"]
        split = int(len(df) * (1 - TEST_SIZE))
        X_tr, X_te = X.iloc[:split], X.iloc[split:]
        y_tr, y_te = y.iloc[:split], y.iloc[split:]

        knn = KNeighborsClassifier(n_neighbors=K_NEIGHBORS).fit(X_tr, y_tr)
        acc_knn = accuracy_score(y_te, knn.predict(X_te))

        xgb = XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, random_state=RANDOM_STATE,
            verbosity=0, eval_metric="mlogloss"
        ).fit(X_tr, y_tr)
        acc_xgb = accuracy_score(y_te, xgb.predict(X_te))

        results_rows.append({
            "Ticker": t,
            "KNN": acc_knn,
            "XGBoost": acc_xgb,
            "Mejor": "KNN" if acc_knn >= acc_xgb else "XGBoost",
            "MejorPrecision": max(acc_knn, acc_xgb),
        })

        knn_all = KNeighborsClassifier(n_neighbors=K_NEIGHBORS).fit(X, y)
        latest = X.iloc[[-1]]
        pred = int(knn_all.predict(latest)[0])
        proba = knn_all.predict_proba(latest)[0]
        prob_dict = {int(c): float(p) for c, p in zip(knn_all.classes_, proba)}

        signals_rows.append({
            "Ticker": t,
            "Señal": pred,
            "Confianza": float(proba.max()),
            "P_subida": prob_dict.get(2, 0.0),
            "P_bajada": prob_dict.get(1, 0.0),
            "P_sin_cambio": prob_dict.get(0, 0.0),
            "Precio": float(df["Close"].iloc[-1]),
            "CloseDate": df["Close"].index[-1],
        })

        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        change_pct = (last_close - prev_close) / prev_close * 100
        last_vol = float(df["Volume"].iloc[-1])

        heat_rows.append({
            "Ticker": t,
            "Company": COMPANY_NAMES.get(t, t),
            "Change": change_pct,
            "Price": last_close,
            "Size": last_close * last_vol,
        })

        xgb_full = XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, random_state=RANDOM_STATE,
            verbosity=0, eval_metric="mlogloss"
        ).fit(X, y)
        importances[t] = pd.Series(xgb_full.feature_importances_, index=FEATURES).to_dict()

    return (processed, pd.DataFrame(results_rows),
            pd.DataFrame(signals_rows), pd.DataFrame(heat_rows), importances)


# =========================================================
# MASTHEAD (con botón actualizar)
# =========================================================
forecast_day = next_business_day()
today = datetime.now()

mh_col1, mh_col2, mh_col3 = st.columns([3, 1, 1.4])

with mh_col1:
    render_html('<div class="masthead-brand" style="padding-top:0.8rem;">Stock Signal · Análisis de mercado</div>')

with mh_col2:
    if st.button("⟳ Actualizar"):
        st.cache_data.clear()
        st.rerun()

with mh_col3:
    render_html(f'<div class="masthead-date" style="text-align:right; padding-top:0.8rem;">{today.strftime("%d · %m · %Y")} — Edición diaria</div>')

render_html('<div style="border-bottom:1px solid #14110f; margin-bottom:3rem; margin-top:-0.5rem;"></div>')

# =========================================================
# HERO + DATELINE
# =========================================================
render_html(f'''
<div class="hero">
    <div class="kicker">Informe de inteligencia de mercado</div>
    <h1 class="hero-title">Nueve acciones.<br><em>Una señal por acción.</em></h1>
    <p class="hero-lede">
        Análisis técnico asistido por inteligencia artificial sobre las principales
        compañías de semiconductores y software. Cada predicción se construye con
        cinco años de historia bursátil y dos modelos independientes.
    </p>
</div>
''')

render_html(f'''
<div class="dateline">
    <div class="dateline-label">Próximo día hábil</div>
    <div class="dateline-value">
        {format_spanish_date(forecast_day).capitalize()}
        <span class="muted">· hoy es {format_spanish_date(today)}</span>
    </div>
</div>
''')

# =========================================================
# CARGA
# =========================================================
with st.spinner("Procesando 5 años de datos de mercado…"):
    try:
        processed, summary_df, signals_df, heat_df, importances = compute_everything()
    except Exception as e:
        st.error(f"No se pudieron obtener los datos: {e}")
        st.stop()

if not processed:
    st.error("No se pudieron obtener datos. Recarga la página.")
    st.stop()


# =========================================================
# SECCIÓN 1 — MAPA DE CALOR
# =========================================================
render_html('''
<div class="section">
    <div class="section-kicker">Panorama</div>
    <h2 class="section-title">Cómo cerró el mercado hoy</h2>
    <p class="section-desc">
        Tamaño del recuadro proporcional al volumen negociado en dólares.
        Color según variación porcentual del cierre.
    </p>
</div>
''')

heat_sorted = heat_df.sort_values("Change", ascending=False).reset_index(drop=True)

labels = [
    f"<b>{r.Ticker}</b><br><span style='font-size:1.05em'>{r.Change:+.2f}%</span>"
    for r in heat_sorted.itertuples()
]

customdata = np.stack([
    heat_sorted["Company"].values,
    heat_sorted["Change"].values,
    heat_sorted["Price"].values,
], axis=-1)

fig_heat = go.Figure(go.Treemap(
    labels=labels,
    parents=[""] * len(heat_sorted),
    values=heat_sorted["Size"].tolist(),
    customdata=customdata,
    marker=dict(
        colors=heat_sorted["Change"].tolist(),
        colorscale=[
            [0.00, FINVIZ_COLORS[0]], [0.17, FINVIZ_COLORS[1]],
            [0.33, FINVIZ_COLORS[2]], [0.50, FINVIZ_COLORS[3]],
            [0.67, FINVIZ_COLORS[4]], [0.83, FINVIZ_COLORS[5]],
            [1.00, FINVIZ_COLORS[6]],
        ],
        cmid=0, cmin=-3, cmax=3,
        line=dict(width=2, color="#faf9f6"),
    ),
    textposition="middle center",
    textfont=dict(size=15, color="white", family="Inter"),
    hovertemplate=(
        "<b>%{label}</b><br>"
        "%{customdata[0]}<br>"
        "Cambio: %{customdata[1]:+.2f}%<br>"
        "Precio: $%{customdata[2]:.2f}"
        "<extra></extra>"
    ),
    tiling=dict(pad=2),
    sort=True,
))

fig_heat.update_layout(
    height=500,
    paper_bgcolor="#faf9f6",
    plot_bgcolor="#faf9f6",
    margin=dict(l=0, r=0, t=0, b=0),
)

st.plotly_chart(fig_heat, use_container_width=True)

# Termómetro
seg_labels = ["-3%", "-2%", "-1%", "0%", "+1%", "+2%", "+3%"]
ticks = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]

segs_html = "".join(
    f'<div class="thermo-seg" style="background:{c};">{lbl}</div>'
    for c, lbl in zip(FINVIZ_COLORS, seg_labels)
)
ticks_html = "".join(f'<div class="thermo-tick">{t}</div>' for t in ticks)

render_html(
    '<div class="thermo">'
    '<div class="thermo-head">'
    '<div class="thermo-title">Escala de variación diaria</div>'
    '<div class="thermo-axis-labels"><span>Negativo</span><span>Positivo</span></div>'
    '</div>'
    f'<div class="thermo-bar">{segs_html}</div>'
    f'<div class="thermo-ticks">{ticks_html}</div>'
    '</div>'
)


# =========================================================
# SECCIÓN 2 — SELECTOR + ANÁLISIS INDIVIDUAL
# =========================================================
render_html('''
<div class="section">
    <div class="section-kicker">Detalle</div>
    <h2 class="section-title">Análisis individual</h2>
    <p class="section-desc">
        Seleccione un emisor para revisar su evolución reciente, indicadores técnicos
        y la importancia relativa de cada variable en el modelo.
    </p>
</div>
''')

ticker_sel = st.selectbox(
    "Emisor",
    options=list(processed.keys()),
    format_func=lambda t: f"{t}  ·  {COMPANY_NAMES.get(t, t)}",
)

df_sel = processed[ticker_sel]

# --- Precio + Medias ---
render_html(f'''
<div style="margin-top:2rem; margin-bottom:1rem;">
    <div class="section-kicker">Serie temporal · {ticker_sel}</div>
    <h3 style="font-family:'Instrument Serif',serif; font-size:1.4rem; font-weight:400; margin:0; color:#14110f;">
        Precio de cierre y medias móviles
    </h3>
</div>
''')

fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["Close"], name="Cierre",
                               line=dict(color="#14110f", width=1.6)))
fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_40"], name="SMA 40",
                               line=dict(color="#0891b2", width=1.3)))
fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_80"], name="SMA 80",
                               line=dict(color="#c99a3a", width=1.3)))
fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_160"], name="SMA 160",
                               line=dict(color="#6d28d9", width=1.3)))
fig_price.update_layout(
    height=380,
    paper_bgcolor="#faf9f6", plot_bgcolor="#faf9f6",
    font=dict(color="#14110f", family="Inter", size=11),
    margin=dict(l=0, r=0, t=20, b=0),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11, color="#14110f")),
    xaxis=dict(gridcolor="#e7e5e0", zeroline=False,
               tickfont=dict(color="#57534e", size=10)),
    yaxis=dict(gridcolor="#e7e5e0", zeroline=False,
               tickfont=dict(color="#57534e", size=10)),
)
st.plotly_chart(fig_price, use_container_width=True)

col_a, col_b = st.columns(2)

# --- RSI ---
with col_a:
    render_html('''
    <div style="margin-top:1rem; margin-bottom:0.75rem;">
        <div class="section-kicker">RSI (14)</div>
        <h4 style="font-family:'Instrument Serif',serif; font-size:1.15rem; font-weight:400; margin:0; color:#14110f;">
            Índice de fuerza relativa
        </h4>
    </div>
    ''')

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=df_sel.index, y=df_sel["RSI"],
                                 line=dict(color="#0891b2", width=1.4), name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dot", line_color="#dc2626", line_width=1)
    fig_rsi.add_hline(y=30, line_dash="dot", line_color="#16a34a", line_width=1)
    fig_rsi.update_layout(
        height=280,
        paper_bgcolor="#faf9f6", plot_bgcolor="#faf9f6",
        font=dict(color="#14110f", family="Inter", size=11),
        margin=dict(l=0, r=0, t=20, b=0),
        showlegend=False,
        xaxis=dict(gridcolor="#e7e5e0", zeroline=False,
                   tickfont=dict(color="#57534e", size=10)),
        yaxis=dict(gridcolor="#e7e5e0", zeroline=False, range=[0, 100],
                   tickfont=dict(color="#57534e", size=10)),
    )
    st.plotly_chart(fig_rsi, use_container_width=True)

# --- Distribución ---
with col_b:
    render_html('''
    <div style="margin-top:1rem; margin-bottom:0.75rem;">
        <div class="section-kicker">Frecuencia histórica</div>
        <h4 style="font-family:'Instrument Serif',serif; font-size:1.15rem; font-weight:400; margin:0; color:#14110f;">
            Distribución de resultados
        </h4>
    </div>
    ''')

    dist = df_sel["Target"].value_counts(normalize=True).sort_index()
    labels_short = {2: "Subió", 1: "Bajó", 0: "Sin cambio"}
    colors_short = {2: "#16a34a", 1: "#dc2626", 0: "#64748b"}
    x_labels = [labels_short[i] for i in dist.index]
    x_colors = [colors_short[i] for i in dist.index]

    fig_dist = go.Figure(go.Bar(
        x=x_labels, y=dist.values,
        marker_color=x_colors,
        text=[f"{v*100:.0f}%" for v in dist.values],
        textposition="outside",
        textfont=dict(color="#14110f", size=13, family="JetBrains Mono"),
    ))
    fig_dist.update_layout(
        height=280,
        paper_bgcolor="#faf9f6", plot_bgcolor="#faf9f6",
        font=dict(color="#14110f", family="Inter", size=11),
        margin=dict(l=0, r=0, t=20, b=0),
        yaxis=dict(tickformat=".0%", gridcolor="#e7e5e0", zeroline=False,
                   range=[0, max(dist.values) * 1.25],
                   tickfont=dict(color="#57534e", size=10)),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False,
                   tickfont=dict(color="#14110f", size=12)),
        showlegend=False,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# --- Importancia ---
render_html(f'''
<div style="margin-top:2rem; margin-bottom:0.75rem;">
    <div class="section-kicker">Pesos del modelo · {ticker_sel}</div>
    <h3 style="font-family:'Instrument Serif',serif; font-size:1.4rem; font-weight:400; margin:0; color:#14110f;">
        Importancia de cada indicador
    </h3>
</div>
''')

imp = pd.Series(importances[ticker_sel]).sort_values()
nice_names = {
    "RSI": "RSI · Fuerza relativa",
    "Relative_Range": "Rango relativo · Volatilidad",
    "Slope_SMA_40": "Pendiente SMA 40 · Corto plazo",
    "Slope_SMA_80": "Pendiente SMA 80 · Medio plazo",
    "Slope_SMA_160": "Pendiente SMA 160 · Largo plazo",
}
imp.index = [nice_names.get(i, i) for i in imp.index]

fig_imp = go.Figure(go.Bar(
    x=imp.values, y=imp.index, orientation="h",
    marker=dict(
        color=imp.values,
        colorscale=[[0, "#0891b2"], [0.5, "#0f2a4a"], [1, "#6d28d9"]],
        showscale=False,
    ),
    text=[f"{v:.3f}" for v in imp.values],
    textposition="outside",
    textfont=dict(color="#14110f", size=11, family="JetBrains Mono"),
))
fig_imp.update_layout(
    height=280,
    paper_bgcolor="#faf9f6", plot_bgcolor="#faf9f6",
    font=dict(color="#14110f", family="Inter", size=11),
    margin=dict(l=0, r=60, t=20, b=0),
    xaxis=dict(gridcolor="#e7e5e0", zeroline=False,
               tickfont=dict(color="#57534e", size=10)),
    yaxis=dict(gridcolor="rgba(0,0,0,0)", zeroline=False,
               tickfont=dict(color="#14110f", size=12)),
    showlegend=False,
)
st.plotly_chart(fig_imp, use_container_width=True)


# =========================================================
# SECCIÓN 3 — PRONÓSTICO PERSONALIZADO
# =========================================================
sel_row = signals_df[signals_df["Ticker"] == ticker_sel].iloc[0]
sel_conf = sel_row["Confianza"]
sel_signal = int(sel_row["Señal"])
sel_company = COMPANY_NAMES.get(ticker_sel, ticker_sel)
sel_conf_text, _, _ = confidence_label(sel_conf)

sel_p_sub = sel_row["P_subida"] * 100
sel_p_baj = sel_row["P_bajada"] * 100
sel_p_lat = sel_row["P_sin_cambio"] * 100

if sel_signal == 2:
    pred_verb = "una <strong>subida mayor al 1%</strong>"
elif sel_signal == 1:
    pred_verb = "una <strong>bajada mayor al 1%</strong>"
else:
    pred_verb = "un <strong>movimiento lateral</strong> (sin cambio significativo)"

forecast_str = format_spanish_date(forecast_day)

render_html(f'''
<div class="section">
    <div class="section-kicker">Pronóstico · {ticker_sel}</div>
    <h2 class="section-title">Señal para el {forecast_str}</h2>
    <p class="section-desc">
        Para <strong>{ticker_sel}</strong> ({sel_company}), el modelo anticipa {pred_verb}
        durante el próximo día hábil. La probabilidad asignada a este escenario es del
        <strong>{sel_conf:.0%}</strong> ({sel_conf_text.lower()}). El detalle completo de
        los tres escenarios posibles aparece en la tabla inferior.
    </p>
</div>
''')

render_html('''
<div class="callout">
    <strong>Nota.</strong> Una probabilidad del 45% no significa certeza.
    El modelo reparte el 100% de probabilidad entre tres escenarios posibles:
    subida mayor a 1%, bajada mayor a 1%, o movimiento lateral.
</div>
''')

# --- Header de la tabla ---
render_html(
    '<div class="signal-table">'
    '<div class="signal-header">'
    '<span>Emisor</span>'
    '<span>Cierre</span>'
    '<span>Señal</span>'
    '<span>Distribución</span>'
    '<span style="text-align:right;">Probabilidad</span>'
    '</div>'
    '</div>'
)

# --- Filas ---
rows_html = ""
for _, row in signals_df.sort_values("Confianza", ascending=False).iterrows():
    t = row["Ticker"]
    conf = row["Confianza"]
    p_sub = row["P_subida"] * 100
    p_baj = row["P_bajada"] * 100
    p_lat = row["P_sin_cambio"] * 100
    company = COMPANY_NAMES.get(t, t)

    if row["Señal"] == 2:
        arrow, arrow_class, label = "▲", "bull", "Subida esperada"
    elif row["Señal"] == 1:
        arrow, arrow_class, label = "▼", "bear", "Bajada esperada"
    else:
        arrow, arrow_class, label = "—", "lat", "Sin cambio esperado"

    conf_text, _, _ = confidence_label(conf)

    close_date_str = row["CloseDate"].strftime("%d · %m · %Y")
    forecast_date_str = forecast_day.strftime("%d · %m · %Y")

    rows_html += (
        f'<div class="signal-row">'
        f'<div>'
        f'<div class="sr-symbol">{t}</div>'
        f'<div class="sr-company">{company}</div>'
        f'</div>'
        f'<div>'
        f'<div class="sr-price">${row["Precio"]:.2f}</div>'
        f'<div class="sr-price-label">{close_date_str}</div>'
        f'</div>'
        f'<div>'
        f'<div class="sr-signal-main"><span class="sr-arrow {arrow_class}">{arrow}</span>{label}</div>'
        f'<div class="sr-signal-sub">Válida para {forecast_date_str}</div>'
        f'</div>'
        f'<div>'
        f'<div class="sr-dist-bar">'
        f'<div class="sr-dist-seg bull" style="width:{p_sub:.1f}%;"></div>'
        f'<div class="sr-dist-seg bear" style="width:{p_baj:.1f}%;"></div>'
        f'<div class="sr-dist-seg lat" style="width:{p_lat:.1f}%;"></div>'
        f'</div>'
        f'<div class="sr-dist-labels">'
        f'<span class="bull">{p_sub:.0f}% ↑</span>'
        f'<span class="bear">{p_baj:.0f}% ↓</span>'
        f'<span class="lat">{p_lat:.0f}% →</span>'
        f'</div>'
        f'</div>'
        f'<div>'
        f'<div class="sr-conf-val">{conf:.0%}</div>'
        f'<div class="sr-conf-lbl">{conf_text}</div>'
        f'</div>'
        f'</div>'
    )

render_html(f'<div class="signal-table" style="border-top:none;">{rows_html}</div>')


# =========================================================
# SECCIÓN 4 — CÓMO INTERPRETAR LA PROBABILIDAD
# =========================================================
render_html('''
<div class="section">
    <div class="section-kicker">Lectura de la señal</div>
    <h2 class="section-title">Cómo interpretar la probabilidad</h2>
    <p class="section-desc">
        La probabilidad indica qué tan seguro está el modelo del escenario ganador.
        Los mercados son ruidosos: incluso una probabilidad alta puede fallar.
    </p>
</div>
''')

render_html('''
<div class="callout">
    <strong>· Probabilidad alta (≥ 50%)</strong> — Señal sólida. El modelo tiene un favorito claro
    entre los tres escenarios. Es el rango donde el pronóstico merece más atención.<br><br>
    <strong>· Probabilidad media (45% – 50%)</strong> — Buena señal, con riesgo presente.
    El escenario ganador se impone, pero los otros dos siguen siendo plausibles.<br><br>
    <strong>· Probabilidad baja (&lt; 45%)</strong> — Señal moderada, riesgo elevado. La probabilidad
    está repartida y el modelo no encuentra un patrón dominante. Trátese como referencia, no como guía.
</div>
''')


# =========================================================
# SECCIÓN 5 — MODELO
# =========================================================
render_html('''
<div class="section">
    <div class="section-kicker">Metodología</div>
    <h2 class="section-title">Rendimiento de los modelos</h2>
    <p class="section-desc">
        Precisión sobre el 20% de los datos reservados para validación,
        con cinco años de historia bursátil y validación temporal.
    </p>
</div>
''')

render_html('''
<div class="callout">
    <strong>KNN.</strong> Busca los 50 días históricos más parecidos al día actual y
    clasifica según lo que ocurrió después. &nbsp;
    <strong>XGBoost.</strong> Ensamblado de árboles de decisión con boosting que
    construye reglas no lineales sobre los cinco indicadores técnicos.
</div>
''')

display_df = summary_df.copy()
display_df = display_df.rename(columns={
    "KNN": "KNN",
    "XGBoost": "XGBoost",
    "Mejor": "Mejor modelo",
    "MejorPrecision": "Máxima precisión",
})
for c in ["KNN", "XGBoost", "Máxima precisión"]:
    display_df[c] = display_df[c].apply(lambda x: f"{x*100:.1f}%")

st.dataframe(display_df, use_container_width=True, hide_index=True)

render_html('<div style="height:2rem;"></div>')

fig_cmp = go.Figure()
fig_cmp.add_trace(go.Bar(
    name="KNN", x=summary_df["Ticker"], y=summary_df["KNN"],
    marker_color="#0f2a4a",
))
fig_cmp.add_trace(go.Bar(
    name="XGBoost", x=summary_df["Ticker"], y=summary_df["XGBoost"],
    marker_color="#c99a3a",
))
fig_cmp.update_layout(
    barmode="group",
    height=400,
    paper_bgcolor="#faf9f6",
    plot_bgcolor="#faf9f6",
    font=dict(color="#14110f", family="Inter", size=11),
    yaxis=dict(tickformat=".0%", title=None,
               gridcolor="#e7e5e0", zerolinecolor="#e7e5e0",
               tickfont=dict(color="#57534e", size=10)),
    xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None,
               tickfont=dict(color="#14110f", size=11)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11, color="#14110f")),
    margin=dict(l=0, r=0, t=40, b=0),
    bargap=0.35,
)
st.plotly_chart(fig_cmp, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================
render_html(f'''
<div class="footer">
    <div class="footer-disclaimer">
        Documento con fines exclusivamente informativos y educativos.
        No constituye asesoría financiera ni recomendación de inversión.
        Los mercados son impredecibles.
    </div>
    <div style="text-align:right;">
        Stock Signal<br>
        {today.strftime("%d · %m · %Y")}
    </div>
</div>
''')

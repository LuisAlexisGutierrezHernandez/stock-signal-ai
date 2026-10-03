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


def confidence_level(pct):
    if pct >= 0.60: return "Fuerte"
    if pct >= 0.45: return "Moderada"
    return "Débil"


# =========================================================
# CSS — DISEÑO EDITORIAL EJECUTIVO
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --paper: #faf9f6;
        --surface: #ffffff;
        --ink: #14110f;
        --ink-2: #57534e;
        --ink-3: #a8a29e;
        --hairline: #e7e5e0;
        --hairline-2: #d6d3cc;
        --accent: #0f2a4a;
        --bull: #0f7a3e;
        --bear: #b91c1c;
        --neutral: #6b7280;
        --gold: #8a6a2b;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
        color: var(--ink);
        font-feature-settings: "tnum", "cv11";
    }

    .stApp { background: var(--paper); }

    .block-container {
        max-width: 1240px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* Ocultar elementos de Streamlit que restan sofisticación */
    header[data-testid="stHeader"] { background: transparent; }
    #MainMenu, footer { visibility: hidden; }

    /* ============ MASTHEAD ============ */
    .masthead {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        padding: 0 0 1rem 0;
        border-bottom: 1px solid var(--ink);
        margin-bottom: 2.5rem;
    }
    .masthead-brand {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        letter-spacing: 0.22em;
        font-size: 0.72rem;
        text-transform: uppercase;
        color: var(--ink);
    }
    .masthead-date {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ink-2);
    }

    /* ============ HERO ============ */
    .hero { margin-bottom: 2.5rem; }
    .kicker {
        font-size: 0.72rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--gold);
        font-weight: 600;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 4rem;
        line-height: 0.98;
        font-weight: 400;
        color: var(--ink);
        letter-spacing: -0.02em;
        margin: 0 0 0.75rem 0;
    }
    .hero-title em { font-style: italic; color: var(--ink-2); }
    .hero-lede {
        font-size: 1.05rem;
        line-height: 1.55;
        color: var(--ink-2);
        max-width: 640px;
        margin: 0;
    }

    /* ============ DATELINE ============ */
    .dateline {
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 2rem;
        align-items: baseline;
        padding: 1.1rem 0;
        border-top: 1px solid var(--hairline);
        border-bottom: 1px solid var(--hairline);
        margin-bottom: 2.5rem;
    }
    .dateline-label {
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--ink-3);
        font-weight: 600;
        white-space: nowrap;
    }
    .dateline-value {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.35rem;
        color: var(--ink);
        letter-spacing: -0.01em;
    }
    .dateline-value .muted {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: var(--ink-2);
        margin-left: 0.5rem;
    }

    /* ============ SECCIÓN ============ */
    .section-head { margin: 0 0 1.5rem 0; }
    .section-kicker {
        font-size: 0.7rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--ink-3);
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .section-title {
        font-family: 'Instrument Serif', Georgia, serif;
        font-size: 1.9rem;
        font-weight: 400;
        line-height: 1.1;
        color: var(--ink);
        letter-spacing: -0.015em;
        margin: 0 0 0.35rem 0;
    }
    .section-desc {
        color: var(--ink-2);
        font-size: 0.92rem;
        line-height: 1.5;
        margin: 0;
    }

    /* ============ TABS ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid var(--hairline);
        padding: 0;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 0;
        padding: 0.85rem 0 0.85rem 0;
        margin-right: 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.92rem;
        color: var(--ink-2);
        border-bottom: 2px solid transparent;
        transition: none;
    }
    .stTabs [aria-selected="true"] {
        color: var(--ink) !important;
        border-bottom: 2px solid var(--ink) !important;
        background: transparent !important;
    }
    .stTabs [data-baseweb="tab"]:hover { color: var(--ink); }
    .stTabs [data-baseweb="tab-highlight"] { display: none; }
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ============ SEÑALES (tabla editorial) ============ */
    .signal-table {
        background: var(--surface);
        border: 1px solid var(--hairline);
    }
    .signal-header {
        display: grid;
        grid-template-columns: 2.2fr 1.2fr 2fr 3fr 1.2fr;
        gap: 1.5rem;
        padding: 0.75rem 1.5rem;
        border-bottom: 1px solid var(--ink);
        background: var(--paper);
    }
    .signal-header span {
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--ink-3);
        font-weight: 600;
    }
    .signal-row {
        display: grid;
        grid-template-columns: 2.2fr 1.2fr 2fr 3fr 1.2fr;
        gap: 1.5rem;
        padding: 1.35rem 1.5rem;
        border-bottom: 1px solid var(--hairline);
        align-items: center;
    }
    .signal-row:last-child { border-bottom: none; }
    .signal-row:hover { background: #fcfcfa; }

    .sr-symbol {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: -0.005em;
        color: var(--ink);
    }
    .sr-company {
        font-size: 0.78rem;
        color: var(--ink-3);
        margin-top: 0.15rem;
    }

    .sr-price {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1rem;
        font-weight: 500;
        color: var(--ink);
        font-variant-numeric: tabular-nums;
    }
    .sr-price-label {
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--ink-3);
        margin-top: 0.15rem;
    }

    .sr-signal-main {
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        display: flex;
        align-items: baseline;
        gap: 0.5rem;
    }
    .sr-arrow { font-size: 0.85rem; }
    .sr-arrow.bull { color: var(--bull); }
    .sr-arrow.bear { color: var(--bear); }
    .sr-arrow.lat  { color: var(--neutral); }
    .sr-signal-sub {
        font-size: 0.72rem;
        color: var(--ink-3);
        margin-top: 0.15rem;
    }

    .sr-dist-bar {
        display: flex;
        height: 6px;
        border-radius: 0;
        overflow: hidden;
        background: var(--hairline);
    }
    .sr-dist-seg { height: 100%; }
    .sr-dist-seg.bull { background: var(--bull); }
    .sr-dist-seg.bear { background: var(--bear); }
    .sr-dist-seg.lat  { background: var(--neutral); }
    .sr-dist-labels {
        display: flex;
        justify-content: space-between;
        margin-top: 0.45rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--ink-3);
        font-variant-numeric: tabular-nums;
    }
    .sr-dist-labels span:nth-child(1) { color: var(--bull); }
    .sr-dist-labels span:nth-child(2) { color: var(--bear); }

    .sr-conf-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--ink);
        font-variant-numeric: tabular-nums;
        text-align: right;
    }
    .sr-conf-lbl {
        font-size: 0.68rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--ink-3);
        text-align: right;
        margin-top: 0.15rem;
    }

    /* ============ TERMÓMETRO ============ */
    .thermo {
        margin-top: 2rem;
        padding-top: 1.25rem;
        border-top: 1px solid var(--hairline);
    }
    .thermo-head {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.75rem;
    }
    .thermo-title {
        font-size: 0.7rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ink-3);
        font-weight: 600;
    }
    .thermo-axis-labels {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: var(--ink-3);
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .thermo-bar {
        display: flex;
        height: 26px;
        border: 1px solid var(--hairline-2);
    }
    .thermo-seg {
        flex: 1;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 500;
        color: #ffffff;
        font-variant-numeric: tabular-nums;
    }
    .thermo-ticks {
        display: flex;
        margin-top: 0.4rem;
    }
    .thermo-tick {
        flex: 1;
        text-align: center;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: var(--ink-3);
        font-variant-numeric: tabular-nums;
    }

    /* ============ CALLOUT ============ */
    .callout {
        border-left: 2px solid var(--ink);
        padding: 0.25rem 0 0.25rem 1.25rem;
        margin: 1.5rem 0 2rem 0;
        color: var(--ink-2);
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .callout strong { color: var(--ink); font-weight: 600; }

    /* ============ DATAFRAME ============ */
    .stDataFrame { border: 1px solid var(--hairline); }
    [data-testid="stDataFrame"] thead tr th {
        background: var(--paper) !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: var(--ink-3) !important;
        font-weight: 600 !important;
        border-bottom: 1px solid var(--ink) !important;
    }
    [data-testid="stDataFrame"] tbody tr td {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.82rem !important;
        font-variant-numeric: tabular-nums !important;
        border-bottom: 1px solid var(--hairline) !important;
    }

    /* ============ FOOTER ============ */
    .footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--hairline);
        display: flex;
        justify-content: space-between;
        font-size: 0.72rem;
        color: var(--ink-3);
        letter-spacing: 0.05em;
    }
    .footer-disclaimer {
        font-style: italic;
        max-width: 640px;
        line-height: 1.55;
        color: var(--ink-3);
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

        majority = y_te.value_counts(normalize=True).iloc[0]

        results_rows.append({
            "Ticker": t,
            "Baseline": majority,
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
# MASTHEAD
# =========================================================
forecast_day = next_business_day()
today = datetime.now()

st.markdown(f"""
<div class="masthead">
    <div class="masthead-brand">Stock Signal · Análisis de mercado</div>
    <div class="masthead-date">{today.strftime('%d · %m · %Y')} — Edición diaria</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# HERO
# =========================================================
st.markdown(f"""
<div class="hero">
    <div class="kicker">Informe de inteligencia de mercado</div>
    <h1 class="hero-title">Nueve acciones.<br><em>Una señal por acción.</em></h1>
    <p class="hero-lede">
        Análisis técnico asistido por inteligencia artificial sobre las principales
        compañías de semiconductores y software. Cada predicción se construye con
        cinco años de historia bursátil y dos modelos independientes.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# DATELINE
# =========================================================
st.markdown(f"""
<div class="dateline">
    <div class="dateline-label">Próximo día hábil</div>
    <div class="dateline-value">
        {format_spanish_date(forecast_day).capitalize()}
        <span class="muted">· hoy es {format_spanish_date(today)}</span>
    </div>
</div>
""", unsafe_allow_html=True)

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
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs(["Mercado", "Señales", "Modelo"])


# ---------------- TAB 1: MERCADO ----------------
with tab1:
    st.markdown("""
    <div class="section-head">
        <div class="section-kicker">Panorama</div>
        <h2 class="section-title">Cómo cerró el mercado hoy</h2>
        <p class="section-desc">
            Tamaño del recuadro proporcional al volumen negociado en dólares.
            Color según variación porcentual del cierre.
        </p>
    </div>
    """, unsafe_allow_html=True)

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

    # Termómetro editorial
    seg_labels = ["-3%", "-2%", "-1%", "0%", "+1%", "+2%", "+3%"]
    ticks = ["−3", "−2", "−1", "0", "+1", "+2", "+3"]

    segs_html = "".join(
        f'<div class="thermo-seg" style="background:{c};">{lbl}</div>'
        for c, lbl in zip(FINVIZ_COLORS, seg_labels)
    )
    ticks_html = "".join(f'<div class="thermo-tick">{t}</div>' for t in ticks)

    st.markdown(f"""
    <div class="thermo">
        <div class="thermo-head">
            <div class="thermo-title">Escala de variación diaria</div>
            <div class="thermo-axis-labels">
                <span>Negativo</span>
                <span>Positivo</span>
            </div>
        </div>
        <div class="thermo-bar">{segs_html}</div>
        <div class="thermo-ticks">{ticks_html}</div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- TAB 2: SEÑALES ----------------
with tab2:
    st.markdown("""
    <div class="section-head">
        <div class="section-kicker">Pronóstico</div>
        <h2 class="section-title">Señal del modelo para el próximo día hábil</h2>
        <p class="section-desc">
            Distribución de probabilidad asignada por el modelo a cada escenario.
            Ordenada de mayor a menor confianza.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>Nota.</strong> Una confianza del 45% no significa certeza.
        El modelo reparte el 100% de probabilidad entre tres escenarios posibles:
        subida mayor a 1%, bajada mayor a 1%, o movimiento lateral.
    </div>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="signal-table">
        <div class="signal-header">
            <span>Emisor</span>
            <span>Cierre</span>
            <span>Señal</span>
            <span>Distribución</span>
            <span style="text-align:right;">Confianza</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

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

        rows_html += f"""
        <div class="signal-row">
            <div>
                <div class="sr-symbol">{t}</div>
                <div class="sr-company">{company}</div>
            </div>
            <div>
                <div class="sr-price">${row['Precio']:.2f}</div>
                <div class="sr-price-label">USD</div>
            </div>
            <div>
                <div class="sr-signal-main">
                    <span class="sr-arrow {arrow_class}">{arrow}</span>
                    <span>{label}</span>
                </div>
                <div class="sr-signal-sub">Próximo día hábil</div>
            </div>
            <div>
                <div class="sr-dist-bar">
                    <div class="sr-dist-seg bull" style="width:{p_sub:.1f}%;"></div>
                    <div class="sr-dist-seg bear" style="width:{p_baj:.1f}%;"></div>
                    <div class="sr-dist-seg lat"  style="width:{p_lat:.1f}%;"></div>
                </div>
                <div class="sr-dist-labels">
                    <span>{p_sub:.0f}% ↑</span>
                    <span>{p_baj:.0f}% ↓</span>
                    <span>{p_lat:.0f}% →</span>
                </div>
            </div>
            <div>
                <div class="sr-conf-val">{conf:.0%}</div>
                <div class="sr-conf-lbl">{confidence_level(conf)}</div>
            </div>
        </div>
        """

    st.markdown(f'<div class="signal-table" style="border-top:none;">{rows_html}</div>',
                unsafe_allow_html=True)


# ---------------- TAB 3: MODELO ----------------
with tab3:
    st.markdown("""
    <div class="section-head">
        <div class="section-kicker">Metodología</div>
        <h2 class="section-title">Rendimiento de los modelos</h2>
        <p class="section-desc">
            Precisión sobre el 20% de los datos reservados para validación.
            Comparado contra el baseline trivial (predecir siempre la clase mayoritaria).
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="callout">
        <strong>KNN.</strong> Busca los 50 días históricos más parecidos al día actual y
        clasifica según lo que ocurrió después. &nbsp;
        <strong>XGBoost.</strong> Ensamblado de árboles de decisión con boosting que
        construye reglas no lineales sobre los cinco indicadores técnicos.
    </div>
    """, unsafe_allow_html=True)

    display_df = summary_df.copy()
    display_df = display_df.rename(columns={
        "Ticker": "Ticker",
        "Baseline": "Baseline",
        "KNN": "KNN",
        "XGBoost": "XGBoost",
        "Mejor": "Mejor",
        "MejorPrecision": "Máxima",
    })
    for c in ["Baseline", "KNN", "XGBoost", "Máxima"]:
        display_df[c] = display_df[c].apply(lambda x: f"{x*100:.1f}%")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)

    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(
        name="KNN", x=summary_df["Ticker"], y=summary_df["KNN"],
        marker_color="#0f2a4a",
    ))
    fig_cmp.add_trace(go.Bar(
        name="XGBoost", x=summary_df["Ticker"], y=summary_df["XGBoost"],
        marker_color="#8a6a2b",
    ))
    fig_cmp.add_trace(go.Scatter(
        name="Baseline", x=summary_df["Ticker"], y=summary_df["Baseline"],
        mode="lines+markers",
        line=dict(color="#a8a29e", dash="dot", width=1.5),
        marker=dict(size=6),
    ))
    fig_cmp.update_layout(
        barmode="group",
        height=400,
        paper_bgcolor="#faf9f6",
        plot_bgcolor="#faf9f6",
        font=dict(color="#14110f", family="Inter", size=12),
        yaxis=dict(
            tickformat=".0%", title=None,
            gridcolor="#e7e5e0", zerolinecolor="#e7e5e0",
        ),
        xaxis=dict(gridcolor="rgba(0,0,0,0)", title=None),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        bargap=0.35,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown(f"""
<div class="footer">
    <div class="footer-disclaimer">
        Documento con fines exclusivamente informativos y educativos.
        No constituye asesoría financiera ni recomendación de inversión.
        Los mercados son impredecibles.
    </div>
    <div style="text-align:right;">
        Stock Signal<br>
        {today.strftime('%d · %m · %Y')}
    </div>
</div>
""", unsafe_allow_html=True)

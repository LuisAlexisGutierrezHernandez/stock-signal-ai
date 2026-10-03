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
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(
    page_title="Stock Signal AI",
    page_icon="📈",
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
    "AVGO": "Broadcom", "GOOG": "Google", "META": "Meta",
    "MSFT": "Microsoft", "ORCL": "Oracle",
}

# Colores estilo Finviz (7 paradas de -3% a +3%)
FINVIZ_COLORS = [
    "#FF2C2C",  # -3%
    "#B03A3A",  # -2%
    "#5E2E2E",  # -1%
    "#4B5563",  #  0%
    "#235E34",  # +1%
    "#1A9C42",  # +2%
    "#1DD14F",  # +3%
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


def format_spanish_date(dt):
    days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    months = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    return f"{days[dt.weekday()]} {dt.day} de {months[dt.month - 1]} de {dt.year}"


def confidence_label(pct):
    if pct >= 0.60:
        return "Señal fuerte", "#059669", "💪"
    elif pct >= 0.45:
        return "Señal moderada", "#d97706", "⚡"
    else:
        return "Señal débil", "#dc2626", "⚠️"


# =========================================================
# CSS — TEMA CLARO
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }

    /* Texto base oscuro */
    h1, h2, h3, h4, h5, h6, p, span, div, label { color: #0f172a; }

    .hero-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #0891b2, #2563eb, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin: 0; letter-spacing: -1px;
    }
    .hero-subtitle {
        text-align: center; color: #64748b; font-size: 1rem;
        margin-top: 0.3rem; margin-bottom: 1.5rem;
    }

    .date-banner {
        background: linear-gradient(90deg, #e0f2fe, #ede9fe);
        border: 1px solid #bae6fd;
        border-radius: 16px; padding: 1.25rem 1.75rem;
        text-align: center; margin-bottom: 2rem;
        box-shadow: 0 4px 14px rgba(15,23,42,0.05);
    }
    .date-banner .label {
        color: #0369a1; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700;
    }
    .date-banner .value {
        color: #0f172a; font-size: 1.6rem; font-weight: 800; margin-top: 0.3rem;
    }
    .date-banner .sub { color: #475569; font-size: 0.9rem; margin-top: 0.4rem; }

    .signal-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 18px; padding: 1.5rem 1.75rem;
        margin-bottom: 1rem; position: relative; overflow: hidden;
        box-shadow: 0 2px 10px rgba(15,23,42,0.05);
    }
    .signal-card.bull { border-left: 5px solid #10b981; }
    .signal-card.bear { border-left: 5px solid #ef4444; }
    .signal-card.lat  { border-left: 5px solid #94a3b8; }

    .signal-ticker { font-size: 1.5rem; font-weight: 800; color: #0f172a; }
    .signal-company { font-size: 0.85rem; color: #64748b; font-weight: 500; }
    .signal-price { font-size: 1.1rem; color: #475569; font-weight: 600; }
    .signal-badge {
        display: inline-block; padding: 0.5rem 1.1rem; border-radius: 999px;
        font-weight: 700; font-size: 0.95rem;
    }
    .badge-bull { background: #d1fae5; color: #065f46; border: 1px solid #6ee7b7; }
    .badge-bear { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }
    .badge-lat  { background: #f1f5f9; color: #334155; border: 1px solid #cbd5e1; }

    .prob-row {
        display: flex; align-items: center; gap: 0.75rem; margin-top: 0.55rem;
        font-size: 0.9rem;
    }
    .prob-label { color: #334155; width: 155px; font-weight: 500; }
    .prob-bar-bg {
        flex: 1; height: 10px; background: #e2e8f0;
        border-radius: 999px; overflow: hidden;
    }
    .prob-bar-fill { height: 100%; border-radius: 999px; }
    .prob-pct { color: #64748b; width: 55px; text-align: right; font-weight: 700; font-size: 0.85rem; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: #f1f5f9;
        padding: 6px; border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 10px 18px;
        font-weight: 600; color: #64748b;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0891b2, #2563eb);
        color: white !important;
    }
    .stTabs [aria-selected="true"] p { color: white !important; }

    .section-title {
        color: #0f172a; font-size: 1.25rem; font-weight: 700;
        margin-top: 1.5rem; margin-bottom: 0.75rem;
        padding-left: 0.7rem; border-left: 4px solid #0891b2;
    }
    .info-box {
        background: #eff6ff; border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem; border-radius: 8px;
        color: #1e3a8a; font-size: 0.92rem; margin-top: 1rem;
    }
    .disclaimer {
        background: #fffbeb; border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem; border-radius: 8px;
        color: #92400e; font-size: 0.88rem; margin-top: 2rem;
    }

    /* Termómetro */
    .thermo-wrap {
        margin-top: 0.8rem;
        padding: 0.5rem 0.25rem 0.25rem 0.25rem;
    }
    .thermo-bar {
        display: flex; height: 34px; border-radius: 6px; overflow: hidden;
        border: 1px solid #cbd5e1;
        box-shadow: 0 1px 3px rgba(15,23,42,0.06);
    }
    .thermo-seg {
        flex: 1; display: flex; align-items: center; justify-content: center;
        color: white; font-weight: 700; font-size: 0.85rem;
    }
    .thermo-labels {
        display: flex; justify-content: space-between;
        margin-top: 0.4rem; color: #64748b; font-size: 0.78rem; font-weight: 500;
    }

    /* Footer limpio */
    .footer-note {
        color: #64748b; font-size: 0.8rem; text-align: center;
        padding: 1rem 0 2rem 0;
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
            "Clase Mayoritaria": majority,
            "KNN": acc_knn,
            "XGBoost": acc_xgb,
            "Mejor Modelo": "KNN" if acc_knn >= acc_xgb else "XGBoost",
            "Mejor Precisión": max(acc_knn, acc_xgb),
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
            "Precio actual": float(df["Close"].iloc[-1]),
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
# HERO + FECHA
# =========================================================
st.markdown('<h1 class="hero-title">📈 Stock Signal AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Inteligencia artificial que analiza acciones y predice hacia dónde irán</p>',
    unsafe_allow_html=True,
)

forecast_day = next_business_day()
today = datetime.now()

st.markdown(f"""
<div class="date-banner">
    <div class="label">🗓️ Pronóstico para el próximo día hábil bursátil</div>
    <div class="value">{format_spanish_date(forecast_day).capitalize()}</div>
    <div class="sub">Hoy es {format_spanish_date(today)}. Este análisis predice qué pasará
    en la próxima jornada en que abran los mercados.</div>
</div>
""", unsafe_allow_html=True)

# =========================================================
# CARGA
# =========================================================
with st.spinner("🔄 Analizando el mercado... esto toma ~30 segundos la primera vez."):
    try:
        processed, summary_df, signals_df, heat_df, importances = compute_everything()
    except Exception as e:
        st.error(f"⚠️ No se pudieron obtener los datos: {e}")
        st.stop()

if not processed:
    st.error("No se pudieron obtener datos. Intenta recargar la página.")
    st.stop()

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3 = st.tabs([
    "🔥 Mapa de calor del mercado",
    "🎯 Señales del día",
    "🧠 ¿Qué tan confiable es?",
])


# ---------------- TAB 1: MAPA DE CALOR ----------------
with tab1:
    st.markdown('<div class="section-title">Cómo se movieron hoy las 9 acciones analizadas</div>', unsafe_allow_html=True)
    st.caption(
        "Tamaño del recuadro = volumen negociado (liquidez).  "
        "Color = variación % del día: 🟢 verde subió, 🔴 rojo bajó."
    )

    heat_sorted = heat_df.sort_values("Change", ascending=False).reset_index(drop=True)

    labels = [
        f"<b>{r.Ticker}</b><br><span style='font-size:1.15em'>{r.Change:+.2f}%</span>"
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
                [0.00, FINVIZ_COLORS[0]],
                [0.17, FINVIZ_COLORS[1]],
                [0.33, FINVIZ_COLORS[2]],
                [0.50, FINVIZ_COLORS[3]],
                [0.67, FINVIZ_COLORS[4]],
                [0.83, FINVIZ_COLORS[5]],
                [1.00, FINVIZ_COLORS[6]],
            ],
            cmid=0,
            cmin=-3,
            cmax=3,
            line=dict(width=3, color="#ffffff"),
        ),
        textposition="middle center",
        textfont=dict(size=16, color="white", family="Inter"),
        hovertemplate=(
            "<b>%{label}</b><br>"
            "Empresa: %{customdata[0]}<br>"
            "Cambio diario: %{customdata[1]:+.2f}%<br>"
            "Precio: $%{customdata[2]:.2f}"
            "<extra></extra>"
        ),
        tiling=dict(pad=3),
        sort=True,
    ))

    fig_heat.update_layout(
        height=520,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    # ---- TERMÓMETRO ----
    st.markdown(f"""
    <div class="thermo-wrap">
        <div class="thermo-bar">
            <div class="thermo-seg" style="background:{FINVIZ_COLORS[0]};">-3%</div>
            <div class="thermo-seg" style="background:{FINVIZ_COLORS[1]};">-2%</div>
            <div class="thermo-seg" style="background:{FINVIZ_COLORS[2]};">-1%</div>
            <div class="thermo-seg" style="background:{FINVIZ_COLORS[3]};">0%</div>
            <div class="thermo-seg" style="background:{FINVIZ_COLORS[4]};">+1%</div>
            <div class="thermo-seg" style="background:{FINVIZ_COLORS[5]};">+2%</div>
            <div class="thermo-seg" style="background:{FINVIZ_COLORS[6]};">+3%</div>
        </div>
        <div class="thermo-labels">
            <span>Fuerte caída</span>
            <span>Sin cambio</span>
            <span>Fuerte subida</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------- TAB 2: SEÑALES ----------------
with tab2:
    st.markdown('<div class="section-title">Predicciones explicadas de forma simple</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 Para cada acción, mostramos <b>cómo reparte el modelo el 100% de probabilidad</b>
        entre los tres posibles resultados del próximo día hábil.
        Ordenamos desde la señal más segura hasta la menos segura.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for _, row in signals_df.sort_values("Confianza", ascending=False).iterrows():
        t = row["Ticker"]
        conf = row["Confianza"]
        conf_text, conf_color, conf_emoji = confidence_label(conf)

        if row["Señal"] == 2:
            badge_class, badge_text, card_class = "badge-bull", "🟢 SUBIDA esperada", "bull"
        elif row["Señal"] == 1:
            badge_class, badge_text, card_class = "badge-bear", "🔴 BAJADA esperada", "bear"
        else:
            badge_class, badge_text, card_class = "badge-lat", "⚪ SIN CAMBIO esperado", "lat"

        company = COMPANY_NAMES.get(t, t)
        p_sub = row["P_subida"] * 100
        p_baj = row["P_bajada"] * 100
        p_lat = row["P_sin_cambio"] * 100

        st.markdown(f"""
        <div class="signal-card {card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
                <div>
                    <div class="signal-ticker">{t}
                        <span style="font-weight:500; color:#64748b; font-size:1rem;">· {company}</span>
                    </div>
                    <div class="signal-price">Precio actual: ${row['Precio actual']:.2f}</div>
                </div>
                <div style="text-align:right;">
                    <span class="signal-badge {badge_class}">{badge_text}</span>
                </div>
            </div>

            <div style="margin-top:1rem;">
                <div style="color:#64748b; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.4rem;">
                    Reparto de probabilidad del modelo
                </div>

                <div class="prob-row">
                    <span class="prob-label">🟢 Subida &gt; 1%</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{p_sub:.0f}%; background:#10b981;"></div>
                    </div>
                    <span class="prob-pct">{p_sub:.0f}%</span>
                </div>

                <div class="prob-row">
                    <span class="prob-label">🔴 Bajada &gt; 1%</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{p_baj:.0f}%; background:#ef4444;"></div>
                    </div>
                    <span class="prob-pct">{p_baj:.0f}%</span>
                </div>

                <div class="prob-row">
                    <span class="prob-label">⚪ Sin cambio</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{p_lat:.0f}%; background:#94a3b8;"></div>
                    </div>
                    <span class="prob-pct">{p_lat:.0f}%</span>
                </div>
            </div>

            <div style="margin-top:0.85rem; color:{conf_color}; font-size:0.85rem; font-weight:700;">
                {conf_emoji} {conf_text} · el modelo asigna {conf:.0%} al resultado más probable
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- TAB 3: CONFIABILIDAD ----------------
with tab3:
    st.markdown('<div class="section-title">🧠 ¿Qué tan buenos son estos modelos?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>Para entender los números:</b> Imagina que el modelo mira 100 días del pasado y trata
        de adivinar qué pasó. Si acierta 45 veces, su precisión es 45%.<br><br>
        🔸 <b>Baseline (tonto):</b> Lo que acertarías si SIEMPRE dijeses "sube" sin pensar.<br>
        🔸 <b>KNN:</b> Busca días parecidos en el pasado y copia lo que pasó.<br>
        🔸 <b>XGBoost:</b> Construye reglas del tipo "si pasa A y B, entonces C".
    </div>
    """, unsafe_allow_html=True)

    display_df = summary_df.copy()
    display_df["Clase Mayoritaria"] = display_df["Clase Mayoritaria"].apply(lambda x: f"{x:.2%}")
    display_df["KNN"] = display_df["KNN"].apply(lambda x: f"{x:.2%}")
    display_df["XGBoost"] = display_df["XGBoost"].apply(lambda x: f"{x:.2%}")
    display_df["Mejor Precisión"] = display_df["Mejor Precisión"].apply(lambda x: f"{x:.2%}")
    display_df = display_df.rename(columns={
        "Clase Mayoritaria": "Baseline (tonto)",
        "Mejor Precisión": "Mejor del modelo",
    })
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-title">📊 Comparación visual</div>', unsafe_allow_html=True)
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="KNN", x=summary_df["Ticker"], y=summary_df["KNN"],
                             marker_color="#3b82f6"))
    fig_cmp.add_trace(go.Bar(name="XGBoost", x=summary_df["Ticker"], y=summary_df["XGBoost"],
                             marker_color="#a855f7"))
    fig_cmp.add_trace(go.Scatter(name="Baseline (tonto)",
                                 x=summary_df["Ticker"], y=summary_df["Clase Mayoritaria"],
                                 mode="lines+markers",
                                 line=dict(color="#f59e0b", dash="dash", width=2)))
    fig_cmp.update_layout(
        barmode="group",
        height=420,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#0f172a"),
        yaxis=dict(tickformat=".0%", title="Precisión", gridcolor="#e2e8f0"),
        xaxis=dict(gridcolor="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Aviso importante:</b> Esta herramienta es un proyecto educativo y de investigación.
        <b>NO es asesoría financiera.</b> Los mercados son impredecibles y los modelos de inteligencia
        artificial pueden equivocarse. Nunca tomes decisiones de inversión basándote únicamente en esta app.
        Consulta siempre a un profesional certificado.
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    f'<div class="footer-note">🚀 Stock Signal AI · Construido con Streamlit + scikit-learn + XGBoost · '
    f'Última actualización: {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>',
    unsafe_allow_html=True,
)

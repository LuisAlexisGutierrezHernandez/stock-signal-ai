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
# CONSTANTES FIJAS (idénticas al notebook original)
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

# =========================================================
# UTILIDADES DE FECHA
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
        return "Señal fuerte", "#34d399", "💪"
    elif pct >= 0.45:
        return "Señal moderada", "#fbbf24", "⚡"
    else:
        return "Señal débil", "#f87171", "⚠️"


# =========================================================
# CSS
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background: radial-gradient(circle at 20% 0%, #1e293b 0%, #0f172a 60%); }

    .hero-title {
        font-size: 3rem; font-weight: 800;
        background: linear-gradient(90deg, #22d3ee, #3b82f6, #a855f7);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin: 0; letter-spacing: -1px;
    }
    .hero-subtitle {
        text-align: center; color: #94a3b8; font-size: 1rem;
        margin-top: 0.3rem; margin-bottom: 1.5rem;
    }

    .date-banner {
        background: linear-gradient(90deg, rgba(34,211,238,0.12), rgba(168,85,247,0.12));
        border: 1px solid rgba(34,211,238,0.35);
        border-radius: 16px; padding: 1.25rem 1.75rem;
        text-align: center; margin-bottom: 2rem;
    }
    .date-banner .label {
        color: #94a3b8; font-size: 0.85rem;
        text-transform: uppercase; letter-spacing: 1.5px; font-weight: 600;
    }
    .date-banner .value {
        color: #22d3ee; font-size: 1.6rem; font-weight: 800; margin-top: 0.3rem;
    }
    .date-banner .sub { color: #cbd5e1; font-size: 0.9rem; margin-top: 0.4rem; }

    .signal-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px; padding: 1.5rem 1.75rem;
        margin-bottom: 1rem; position: relative; overflow: hidden;
    }
    .signal-card.bull { border-left: 5px solid #34d399; }
    .signal-card.bear { border-left: 5px solid #f87171; }
    .signal-card.lat  { border-left: 5px solid #cbd5e1; }

    .signal-ticker { font-size: 1.5rem; font-weight: 800; color: #f1f5f9; }
    .signal-company { font-size: 0.85rem; color: #94a3b8; font-weight: 500; }
    .signal-price { font-size: 1.1rem; color: #cbd5e1; font-weight: 600; }
    .signal-badge {
        display: inline-block; padding: 0.5rem 1.1rem; border-radius: 999px;
        font-weight: 700; font-size: 0.95rem;
    }
    .badge-bull { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.4); }
    .badge-bear { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.4); }
    .badge-lat  { background: rgba(148,163,184,0.15); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.4); }

    .prob-row {
        display: flex; align-items: center; gap: 0.75rem; margin-top: 0.55rem;
        font-size: 0.9rem;
    }
    .prob-label { color: #cbd5e1; width: 150px; font-weight: 500; }
    .prob-bar-bg {
        flex: 1; height: 10px; background: rgba(148,163,184,0.13);
        border-radius: 999px; overflow: hidden;
    }
    .prob-bar-fill { height: 100%; border-radius: 999px; transition: width 0.4s ease; }
    .prob-pct { color: #94a3b8; width: 55px; text-align: right; font-weight: 600; font-size: 0.85rem; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 6px; background: rgba(30,41,59,0.5);
        padding: 6px; border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 10px 18px;
        font-weight: 600; color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0891b2, #2563eb);
        color: white !important;
    }

    .section-title {
        color: #f1f5f9; font-size: 1.25rem; font-weight: 700;
        margin-top: 1.5rem; margin-bottom: 0.75rem;
        padding-left: 0.7rem; border-left: 4px solid #22d3ee;
    }
    .info-box {
        background: rgba(34,211,238,0.06); border-left: 4px solid #22d3ee;
        padding: 1rem 1.25rem; border-radius: 8px;
        color: #cbd5e1; font-size: 0.92rem; margin-top: 1rem;
    }
    .disclaimer {
        background: rgba(251,191,36,0.08); border-left: 4px solid #fbbf24;
        padding: 1rem 1.25rem; border-radius: 8px;
        color: #fcd34d; font-size: 0.88rem; margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# PIPELINE CACHEADO
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

    results_rows = []
    signals_rows = []
    heat_rows = []
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

        # Señal con TODOS los datos
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

        # Datos para el heatmap
        last_close = float(df["Close"].iloc[-1])
        prev_close = float(df["Close"].iloc[-2])
        change_pct = (last_close - prev_close) / prev_close * 100
        last_vol = float(df["Volume"].iloc[-1])
        dollar_volume = last_close * last_vol

        heat_rows.append({
            "Ticker": t,
            "Company": COMPANY_NAMES.get(t, t),
            "Change": change_pct,
            "Price": last_close,
            "Size": dollar_volume,
        })

        # Importancias
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
                [0.00, "#7f1d1d"],
                [0.30, "#b91c1c"],
                [0.45, "#4b0f0f"],
                [0.50, "#1f2937"],
                [0.55, "#0a3a1f"],
                [0.70, "#15803d"],
                [1.00, "#065f46"],
            ],
            cmid=0,
            cmin=-4,
            cmax=4,
            line=dict(width=3, color="#0f172a"),
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
        template="plotly_dark",
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
    )

    st.plotly_chart(fig_heat, use_container_width=True)

    # Leyenda inferior
    st.markdown("""
    <div style="display:flex; justify-content:center; gap:1.2rem; margin-top:0.5rem; color:#94a3b8; font-size:0.82rem;">
        <span>🔴 Bajó más de 2%</span>
        <span>🟥 Bajó</span>
        <span>⬛ Sin cambio</span>
        <span>🟩 Subió</span>
        <span>🟢 Subió más de 2%</span>
    </div>
    """, unsafe_allow_html=True)


# ---------------- TAB 2: SEÑALES DEL DÍA ----------------
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
                        <span style="font-weight:500; color:#94a3b8; font-size:1rem;">· {company}</span>
                    </div>
                    <div class="signal-price">Precio actual: ${row['Precio actual']:.2f}</div>
                </div>
                <div style="text-align:right;">
                    <span class="signal-badge {badge_class}">{badge_text}</span>
                </div>
            </div>

            <div style="margin-top:1rem;">
                <div style="color:#94a3b8; font-size:0.8rem; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:0.4rem;">
                    Reparto de probabilidad del modelo
                </div>

                <div class="prob-row">
                    <span class="prob-label">🟢 Subida &gt; 1%</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{p_sub:.0f}%; background:#34d399;"></div>
                    </div>
                    <span class="prob-pct">{p_sub:.0f}%</span>
                </div>

                <div class="prob-row">
                    <span class="prob-label">🔴 Bajada &gt; 1%</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{p_baj:.0f}%; background:#f87171;"></div>
                    </div>
                    <span class="prob-pct">{p_baj:.0f}%</span>
                </div>

                <div class="prob-row">
                    <span class="prob-label">⚪ Sin cambio</span>
                    <div class="prob-bar-bg">
                        <div class="prob-bar-fill" style="width:{p_lat:.0f}%; background:#cbd5e1;"></div>
                    </div>
                    <span class="prob-pct">{p_lat:.0f}%</span>
                </div>
            </div>

            <div style="margin-top:0.85rem; color:{conf_color}; font-size:0.85rem; font-weight:600;">
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
        🔸 <b>Baseline (tonto):</b> Lo que acertarías si SIEMPRE dijeses "sube" sin pensar.
        Hay que superarlo para que la IA sirva de algo.<br>
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
                                 mode="lines+markers", line=dict(color="#fbbf24", dash="dash", width=2)))
    fig_cmp.update_layout(
        barmode="group", template="plotly_dark", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", title="Precisión"),
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
st.caption(
    f"🚀 Stock Signal AI · Construido con Streamlit + scikit-learn + XGBoost · "
    f"Última actualización: {datetime.now().strftime('%d/%m/%Y %H:%M')}"
)

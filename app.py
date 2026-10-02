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
    "NVDA": "NVIDIA", "MU": "Micron Technology", "AMD": "Advanced Micro Devices",
    "INTC": "Intel", "AVGO": "Broadcom", "GOOG": "Google (Alphabet)",
    "META": "Meta (Facebook)", "MSFT": "Microsoft", "ORCL": "Oracle",
}

# =========================================================
# UTILIDADES DE FECHA
# =========================================================
def next_business_day(from_date=None):
    """Devuelve el siguiente día hábil (lunes a viernes) después de la fecha dada."""
    if from_date is None:
        from_date = datetime.now()
    d = from_date + timedelta(days=1)
    while d.weekday() >= 5:  # 5=sábado, 6=domingo
        d += timedelta(days=1)
    return d


def format_spanish_date(dt):
    days = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    months = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
              "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    return f"{days[dt.weekday()]} {dt.day} de {months[dt.month - 1]} de {dt.year}"


def confidence_label(pct):
    """Convierte la confianza en una etiqueta amigable."""
    if pct >= 0.60:
        return "Señal fuerte", "#34d399", "💪"
    elif pct >= 0.45:
        return "Señal moderada", "#fbbf24", "⚡"
    else:
        return "Señal débil", "#f87171", "⚠️"


# =========================================================
# CSS PERSONALIZADO
# =========================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #1e293b 0%, #0f172a 60%);
    }

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
    .date-banner .sub {
        color: #cbd5e1; font-size: 0.9rem; margin-top: 0.4rem;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.9), rgba(15,23,42,0.9));
        border: 1px solid rgba(148,163,184,0.15);
        border-radius: 16px; padding: 1.25rem 1.5rem;
        position: relative; overflow: hidden; height: 100%;
    }
    .metric-card::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #22d3ee, #3b82f6);
    }
    .metric-label {
        color: #94a3b8; font-size: 0.78rem; font-weight: 500;
        text-transform: uppercase; letter-spacing: 0.6px;
    }
    .metric-value {
        color: #f1f5f9; font-size: 1.85rem; font-weight: 700; margin-top: 0.3rem;
    }
    .metric-delta { font-size: 0.82rem; margin-top: 0.25rem; color: #94a3b8; }

    .signal-card {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid rgba(148,163,184,0.18);
        border-radius: 18px; padding: 1.5rem 1.75rem;
        margin-bottom: 1rem; position: relative; overflow: hidden;
    }
    .signal-card.bull { border-left: 5px solid #34d399; }
    .signal-card.bear { border-left: 5px solid #f87171; }
    .signal-card.lat  { border-left: 5px solid #cbd5e1; }

    .signal-ticker {
        font-size: 1.5rem; font-weight: 800; color: #f1f5f9;
    }
    .signal-company {
        font-size: 0.85rem; color: #94a3b8; font-weight: 500;
    }
    .signal-price {
        font-size: 1.1rem; color: #cbd5e1; font-weight: 600;
    }
    .signal-badge {
        display: inline-block; padding: 0.5rem 1.1rem; border-radius: 999px;
        font-weight: 700; font-size: 0.95rem;
    }
    .badge-bull { background: rgba(16,185,129,0.15); color: #34d399;
                  border: 1px solid rgba(16,185,129,0.4); }
    .badge-bear { background: rgba(239,68,68,0.15); color: #f87171;
                  border: 1px solid rgba(239,68,68,0.4); }
    .badge-lat  { background: rgba(148,163,184,0.15); color: #cbd5e1;
                  border: 1px solid rgba(148,163,184,0.4); }

    .signal-explain {
        color: #94a3b8; font-size: 0.92rem; margin-top: 0.75rem;
        padding-top: 0.75rem; border-top: 1px solid rgba(148,163,184,0.12);
    }

    .conf-bar-wrap {
        display: flex; align-items: center; gap: 0.75rem; margin-top: 0.6rem;
    }
    .conf-bar-bg {
        flex: 1; height: 8px; background: rgba(148,163,184,0.15);
        border-radius: 999px; overflow: hidden;
    }
    .conf-bar-fill { height: 100%; border-radius: 999px; }

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
        background: rgba(34,211,238,0.06);
        border-left: 4px solid #22d3ee;
        padding: 1rem 1.25rem; border-radius: 8px;
        color: #cbd5e1; font-size: 0.92rem; margin-top: 1rem;
    }
    .disclaimer {
        background: rgba(251,191,36,0.08);
        border-left: 4px solid #fbbf24;
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
    """Descarga, procesa, entrena y devuelve todo listo para la UI."""
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

        signals_rows.append({
            "Ticker": t,
            "Señal": pred,
            "Confianza": float(proba.max()),
            "Precio actual": float(df["Close"].iloc[-1]),
        })

        # Importancias
        xgb_full = XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH,
            learning_rate=LEARNING_RATE, random_state=RANDOM_STATE,
            verbosity=0, eval_metric="mlogloss"
        ).fit(X, y)
        importances[t] = pd.Series(xgb_full.feature_importances_, index=FEATURES).to_dict()

    return processed, pd.DataFrame(results_rows), pd.DataFrame(signals_rows), importances


# =========================================================
# HERO
# =========================================================
st.markdown('<h1 class="hero-title">📈 Stock Signal AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Inteligencia artificial que analiza acciones y predice hacia dónde irán</p>',
    unsafe_allow_html=True,
)

# =========================================================
# BANNER DE FECHA
# =========================================================
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
# CARGA DE DATOS
# =========================================================
with st.spinner("🔄 Analizando el mercado... esto toma ~30 segundos la primera vez."):
    try:
        processed, summary_df, signals_df, importances = compute_everything()
    except Exception as e:
        st.error(f"⚠️ No se pudieron obtener los datos: {e}")
        st.stop()

if not processed:
    st.error("No se pudieron obtener datos. Intenta recargar la página.")
    st.stop()

# =========================================================
# MÉTRICAS GLOBALES
# =========================================================
best_row = summary_df.sort_values("Mejor Precisión", ascending=False).iloc[0]
avg_acc = summary_df["Mejor Precisión"].mean()
bullish = int((signals_df["Señal"] == 2).sum())
high_conf = int((signals_df["Confianza"] >= 0.45).sum())

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Acciones analizadas</div>
        <div class="metric-value">{len(processed)}</div>
        <div class="metric-delta">Últimos 5 años de datos</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Señal más segura</div>
        <div class="metric-value">{high_conf} / {len(signals_df)}</div>
        <div class="metric-delta">Con confianza moderada o fuerte</div>
    </div>""", unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Predicen subida</div>
        <div class="metric-value">{bullish} / {len(signals_df)}</div>
        <div class="metric-delta">El resto predice bajada o sin cambio</div>
    </div>""", unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Precisión promedio</div>
        <div class="metric-value">{avg_acc:.1%}</div>
        <div class="metric-delta">Histórica · sobre datos de prueba</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Señales del día",
    "📊 Ver una acción en detalle",
    "🧠 ¿Qué tan confiable es?",
    "📖 Aprende los conceptos",
])


# ---------------- TAB 1: SEÑALES ----------------
with tab1:
    st.markdown('<div class="section-title">Predicciones de hoy explicadas de forma simple</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        💡 <b>¿Qué estás viendo?</b> Para cada acción, la inteligencia artificial analizó 5 años
        de datos históricos y predice qué podría pasar mañana. Ordenamos las señales
        desde la más segura hasta la menos segura.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    for _, row in signals_df.sort_values("Confianza", ascending=False).iterrows():
        t = row["Ticker"]
        conf = row["Confianza"]
        conf_text, conf_color, conf_emoji = confidence_label(conf)

        if row["Señal"] == 2:
            badge_class = "badge-bull"
            badge_text = "🟢 SUBIDA esperada"
            card_class = "bull"
            explain = "El sistema anticipa que el precio subirá más de 1%."
        elif row["Señal"] == 1:
            badge_class = "badge-bear"
            badge_text = "🔴 BAJADA esperada"
            card_class = "bear"
            explain = "El sistema anticipa que el precio bajará más de 1%."
        else:
            badge_class = "badge-lat"
            badge_text = "⚪ SIN CAMBIO esperado"
            card_class = "lat"
            explain = "El sistema anticipa que el precio se moverá poco (menos de 1%)."

        company = COMPANY_NAMES.get(t, t)

        st.markdown(f"""
        <div class="signal-card {card_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:1rem;">
                <div>
                    <div class="signal-ticker">{t} <span style="font-weight:500; color:#94a3b8; font-size:1rem;">· {company}</span></div>
                    <div class="signal-price">Precio actual: ${row['Precio actual']:.2f}</div>
                </div>
                <div style="text-align:right;">
                    <span class="signal-badge {badge_class}">{badge_text}</span>
                </div>
            </div>
            <div class="signal-explain">{explain}</div>
            <div class="conf-bar-wrap">
                <span style="color:{conf_color}; font-size:0.85rem; font-weight:600; white-space:nowrap;">
                    {conf_emoji} {conf_text}
                </span>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{conf*100:.0f}%; background:{conf_color};"></div>
                </div>
                <span style="color:#94a3b8; font-size:0.8rem; min-width:3rem; text-align:right;">{conf:.0%}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ---------------- TAB 2: DETALLE ----------------
with tab2:
    st.markdown('<div class="section-title">Elige una acción para ver su historia completa</div>', unsafe_allow_html=True)

    ticker_sel = st.selectbox(
        "Acción:",
        list(processed.keys()),
        format_func=lambda t: f"{t} — {COMPANY_NAMES.get(t, t)}",
    )
    df_sel = processed[ticker_sel]

    # Precio + Medias
    st.markdown(f'<div class="section-title">📈 Evolución del precio ({ticker_sel})</div>', unsafe_allow_html=True)
    st.caption("Cada línea de color es un promedio del precio, útil para ver la tendencia general.")

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["Close"], name="Precio real",
                                   line=dict(color="#22d3ee", width=2)))
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_40"], name="Tendencia corta (40 días)",
                                   line=dict(color="#facc15", width=1.3)))
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_80"], name="Tendencia media (80 días)",
                                   line=dict(color="#a855f7", width=1.3)))
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_160"], name="Tendencia larga (160 días)",
                                   line=dict(color="#f43f5e", width=1.3)))
    fig_price.update_layout(
        template="plotly_dark", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Precio (USD)",
    )
    st.plotly_chart(fig_price, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">🌡️ Termómetro de compra/venta (RSI)</div>', unsafe_allow_html=True)
        st.caption("Arriba de 70 = la acción está muy cara (posible bajada). Abajo de 30 = muy barata (posible subida).")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_sel.index, y=df_sel["RSI"],
                                     line=dict(color="#22d3ee", width=1.8), name="RSI"))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f87171",
                          annotation_text="Caro", annotation_position="right")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#34d399",
                          annotation_text="Barato", annotation_position="right")
        fig_rsi.update_layout(template="plotly_dark", height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=20, b=10),
                              yaxis=dict(range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">📊 Frecuencia histórica de cada resultado</div>', unsafe_allow_html=True)
        st.caption("De todas las veces que analizamos esta acción, ¿cuántas subió, bajó o se mantuvo?")
        dist = df_sel["Target"].value_counts(normalize=True).sort_index()
        label_map_short = {2: "🟢 Subió", 1: "🔴 Bajó", 0: "⚪ Sin cambio"}
        labels = [label_map_short[i] for i in dist.index]
        colors = ["#34d399" if i == 2 else "#f87171" if i == 1 else "#cbd5e1" for i in dist.index]
        fig_dist = go.Figure(go.Bar(x=labels, y=dist.values, marker_color=colors,
                                    text=[f"{v:.0%}" for v in dist.values],
                                    textposition="outside"))
        fig_dist.update_layout(template="plotly_dark", height=300,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=20, b=10),
                               yaxis=dict(tickformat=".0%", range=[0, max(dist.values) * 1.2]))
        st.plotly_chart(fig_dist, use_container_width=True)

    # Importancias
    st.markdown('<div class="section-title">🎯 ¿Qué indicadores pesaron más en la decisión?</div>', unsafe_allow_html=True)
    st.caption("Mientras más larga la barra, más importante fue ese indicador para el modelo.")
    imp = pd.Series(importances[ticker_sel]).sort_values()
    nice_names = {
        "RSI": "RSI (termómetro compra/venta)",
        "Relative_Range": "Volatilidad diaria",
        "Slope_SMA_40": "Tendencia corta",
        "Slope_SMA_80": "Tendencia media",
        "Slope_SMA_160": "Tendencia larga",
    }
    imp.index = [nice_names.get(i, i) for i in imp.index]
    fig_imp = px.bar(x=imp.values, y=imp.index, orientation="h",
                     labels={"x": "Importancia", "y": ""})
    fig_imp.update_traces(marker_color="#3b82f6")
    fig_imp.update_layout(template="plotly_dark", height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)


# ---------------- TAB 3: CONFIABILIDAD ----------------
with tab3:
    st.markdown('<div class="section-title">🧠 ¿Qué tan buenos son estos modelos?</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
        <b>Para entender los números:</b> Imagina que el modelo mira 100 días del pasado y trata
        de adivinar qué pasó. Si acierta 45 veces, su precisión es 45%.<br><br>
        🔸 <b>Clase Mayoritaria (baseline):</b> Es lo que acertarías si SIEMPRE dijeses "sube" sin pensar.
        Es el rival más tonto y hay que superarlo para que la IA sirva de algo.<br>
        🔸 <b>KNN:</b> Busca días parecidos en el pasado y copia lo que pasó.<br>
        🔸 <b>XGBoost:</b> Construye reglas tipo "si pasa A y B, entonces C".
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


# ---------------- TAB 4: APRENDE ----------------
with tab4:
    st.markdown('<div class="section-title">📖 Glosario sencillo</div>', unsafe_allow_html=True)
    st.markdown("""
    ### 🏢 ¿Qué es una acción?
    Es una pequeña parte de una empresa. Si compras una acción de NVIDIA, eres dueño de un
    pedacito de NVIDIA. Su precio sube cuando a la gente le va bien con la empresa y baja cuando
    hay malas noticias.

    ### 📈 ¿Qué significa "subida esperada"?
    El sistema predice que el precio de esa acción subirá **más de 1%** en el próximo día hábil.
    Por ejemplo: si cuesta $100, subiría al menos a $101.

    ### 📉 ¿Qué significa "bajada esperada"?
    El sistema predice que el precio bajará **más de 1%**. De $100 pasaría a $99 o menos.

    ### ⚪ ¿Qué significa "sin cambio"?
    El sistema cree que el precio se moverá **menos de 1%**, ni arriba ni abajo de forma clara.

    ### 🎯 ¿Qué es la "confianza"?
    Es qué tan seguro está el modelo. Una confianza de 40% significa que el modelo no está muy
    seguro. **Ojo: confianza alta no garantiza que acierte.** En bolsa nunca hay garantías.

    ### 🧠 ¿Qué es KNN?
    Imagina que tienes 5 años de historia del mercado. El modelo busca los 50 días más parecidos
    al día de hoy y mira qué pasó después. Si la mayoría subió, predice subida.

    ### 🌳 ¿Qué es XGBoost?
    Es un modelo que crea cientos de reglas tipo:
    *"Si el RSI es mayor a 70 Y la tendencia es bajista, entonces probablemente baje."*
    Combina muchas reglas simples para hacer una predicción más robusta.

    ### 🎚️ ¿Qué es el RSI?
    Es un número entre 0 y 100 que mide si una acción está "cara" o "barata":
    - **Arriba de 70:** Mucha gente la está comprando → puede bajar pronto.
    - **Abajo de 30:** Mucha gente la está vendiendo → puede subir pronto.

    ### 📏 ¿Qué son las medias móviles (SMA)?
    Es el precio promedio de los últimos N días. Sirven para ver la tendencia general:
    - Si el precio está **arriba** de la media → tendencia alcista.
    - Si está **abajo** → tendencia bajista.
    """)

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

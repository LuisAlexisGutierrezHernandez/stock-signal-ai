import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
st.set_page_config(
    page_title="Stock Signal AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

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
        margin-top: 0.3rem; margin-bottom: 2rem;
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

    .badge { display: inline-block; padding: 0.3rem 0.9rem; border-radius: 999px;
             font-weight: 600; font-size: 0.85rem; }
    .badge-bull { background: rgba(16,185,129,0.15); color: #34d399;
                  border: 1px solid rgba(16,185,129,0.35); }
    .badge-bear { background: rgba(239,68,68,0.15); color: #f87171;
                  border: 1px solid rgba(239,68,68,0.35); }
    .badge-lat  { background: rgba(148,163,184,0.15); color: #cbd5e1;
                  border: 1px solid rgba(148,163,184,0.35); }

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

    .disclaimer {
        background: rgba(251,191,36,0.08);
        border-left: 4px solid #fbbf24;
        padding: 1rem 1.25rem; border-radius: 8px;
        color: #fcd34d; font-size: 0.88rem; margin-top: 2rem;
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        border-right: 1px solid rgba(148,163,184,0.1);
    }
</style>
""", unsafe_allow_html=True)


# =========================================================
# FUNCIONES CACHEADAS
# =========================================================
@st.cache_data(ttl=3600, show_spinner=False)
def download_data(tickers_tuple, period):
    """Descarga datos históricos desde Yahoo Finance."""
    raw = yf.download(
        list(tickers_tuple), period=period, interval="1d",
        group_by="ticker", progress=False, auto_adjust=True
    )
    return raw


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
def process_all(tickers_tuple, period):
    """Descarga + limpia + features. Devuelve dict {ticker: df}."""
    raw = download_data(tickers_tuple, period)
    result = {}
    for t in tickers_tuple:
        try:
            sub = raw[t].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
            cleaned = clean_stock_data(sub)
            result[t] = create_features_and_target(cleaned)
        except Exception:
            continue
    return result


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### ⚙️ Configuración")

    DEFAULT_TICKERS = ["NVDA", "MU", "AMD", "INTC", "AVGO", "GOOG", "META", "MSFT", "ORCL"]

    selected = st.multiselect(
        "Acciones a analizar",
        options=DEFAULT_TICKERS + ["AAPL", "AMZN", "TSLA", "TSM", "QCOM", "PLTR"],
        default=DEFAULT_TICKERS,
        help="Selecciona una o varias acciones para analizar.",
    )

    period = st.selectbox(
        "Periodo histórico",
        options=["2y", "3y", "5y", "10y"],
        index=2,
        help="Cantidad de años de historia.",
    )

    st.markdown("---")
    st.markdown("### 🤖 Modelo")
    k_neighbors = st.slider("Vecinos KNN", 5, 100, 50, 5)
    n_estimators = st.slider("Árboles XGBoost", 50, 300, 100, 50)
    test_size = st.slider("% datos de prueba", 10, 40, 20, 5)

    st.markdown("---")
    st.caption("💡 *Los datos se cachean 1 hora para acelerar la app.*")


# =========================================================
# HERO
# =========================================================
st.markdown('<h1 class="hero-title">📈 Stock Signal AI</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Predicción de señales bursátiles con Machine Learning · '
    'KNN + XGBoost · Análisis técnico automatizado</p>',
    unsafe_allow_html=True,
)

if not selected:
    st.warning("⚠️ Selecciona al menos una acción en la barra lateral.")
    st.stop()

# =========================================================
# CARGA DE DATOS
# =========================================================
with st.spinner("🔄 Descargando y procesando datos de mercado..."):
    processed = process_all(tuple(selected), period)

if not processed:
    st.error("No se pudieron obtener datos. Intenta con otros tickers.")
    st.stop()

features = ["Slope_SMA_40", "Slope_SMA_80", "Slope_SMA_160", "RSI", "Relative_Range"]
label_map = {2: "ALCISTA", 1: "BAJISTA", 0: "LATERAL"}
badge_map = {2: "badge-bull", 1: "badge-bear", 0: "badge-lat"}
color_map = {2: "#34d399", 1: "#f87171", 0: "#cbd5e1"}

# =========================================================
# MÉTRICAS GLOBALES
# =========================================================
summary_rows = []
signals_rows = []

for t, df in processed.items():
    X = df[features]
    y = df["Target"]
    split = int(len(df) * (1 - test_size / 100))

    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    knn = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X_tr, y_tr)
    acc_knn = accuracy_score(y_te, knn.predict(X_te))

    xgb = XGBClassifier(
        n_estimators=n_estimators, max_depth=3, learning_rate=0.05,
        random_state=42, verbosity=0, use_label_encoder=False,
        eval_metric="mlogloss"
    ).fit(X_tr, y_tr)
    acc_xgb = accuracy_score(y_te, xgb.predict(X_te))

    majority = y_te.value_counts(normalize=True).iloc[0]
    best = "KNN" if acc_knn >= acc_xgb else "XGBoost"

    summary_rows.append({
        "Ticker": t,
        "Clase Mayoritaria": majority,
        "KNN": acc_knn,
        "XGBoost": acc_xgb,
        "Mejor Modelo": best,
        "Mejor Precisión": max(acc_knn, acc_xgb),
    })

    # Señal con TODOS los datos
    knn_all = KNeighborsClassifier(n_neighbors=k_neighbors).fit(X, y)
    latest = X.iloc[[-1]]
    pred = knn_all.predict(latest)[0]
    proba = knn_all.predict_proba(latest)[0]

    signals_rows.append({
        "Ticker": t,
        "Señal": label_map[pred],
        "Código": pred,
        "Confianza": proba.max(),
        "Precio actual": df["Close"].iloc[-1],
        "Retorno 3d real": df["Close"].pct_change(3).iloc[-1],
    })

summary_df = pd.DataFrame(summary_rows).sort_values("Mejor Precisión", ascending=False)
signals_df = pd.DataFrame(signals_rows)

# ---- Métricas superiores ----
c1, c2, c3, c4 = st.columns(4)
best_row = summary_df.iloc[0]

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Acciones analizadas</div>
        <div class="metric-value">{len(processed)}</div>
        <div class="metric-delta">Datos de {period}</div>
    </div>""", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Mejor precisión</div>
        <div class="metric-value">{best_row['Mejor Precisión']:.1%}</div>
        <div class="metric-delta">{best_row['Ticker']} · {best_row['Mejor Modelo']}</div>
    </div>""", unsafe_allow_html=True)

with c3:
    avg_acc = summary_df["Mejor Precisión"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Precisión promedio</div>
        <div class="metric-value">{avg_acc:.1%}</div>
        <div class="metric-delta">sobre {len(processed)} tickers</div>
    </div>""", unsafe_allow_html=True)

with c4:
    bullish = (signals_df["Código"] == 2).sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Señales alcistas hoy</div>
        <div class="metric-value">{bullish} / {len(signals_df)}</div>
        <div class="metric-delta">Modelo KNN entrenado con 100% de datos</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================================================
# TABS
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Señales del Día",
    "📊 Análisis Individual",
    "🏆 Comparación de Modelos",
    "ℹ️ Acerca de",
])


# ---------------- TAB 1: SEÑALES ----------------
with tab1:
    st.markdown('<div class="section-title">Predicciones más recientes (basadas en la última vela disponible)</div>', unsafe_allow_html=True)
    st.caption("El modelo KNN se entrena con el 100% del histórico y predice qué pasará en los próximos 3 días: subida >1%, caída >1%, o movimiento lateral.")

    for _, row in signals_df.sort_values("Confianza", ascending=False).iterrows():
        badge_class = badge_map[row["Código"]]
        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown(f"""
            <div style="background: rgba(30,41,59,0.55); border:1px solid rgba(148,163,184,0.15);
                        border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.6rem;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <span style="font-size:1.3rem; font-weight:700; color:#f1f5f9;">{row['Ticker']}</span>
                        <span style="color:#94a3b8; margin-left: 0.8rem; font-size:0.9rem;">
                            Precio: ${row['Precio actual']:.2f}
                        </span>
                    </div>
                    <span class="badge {badge_class}">{row['Señal']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_right:
            st.markdown(f"""
            <div style="background: rgba(30,41,59,0.55); border:1px solid rgba(148,163,184,0.15);
                        border-radius: 12px; padding: 1rem 1.25rem; margin-bottom: 0.6rem;
                        text-align:center;">
                <div style="color:#94a3b8; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.5px;">Confianza</div>
                <div style="color:#f1f5f9; font-size:1.4rem; font-weight:700;">{row['Confianza']:.1%}</div>
            </div>
            """, unsafe_allow_html=True)


# ---------------- TAB 2: ANÁLISIS INDIVIDUAL ----------------
with tab2:
    ticker_sel = st.selectbox("Selecciona una acción para análisis detallado", list(processed.keys()))
    df_sel = processed[ticker_sel]

    # Gráfico de precio + medias
    st.markdown(f'<div class="section-title">📈 {ticker_sel} — Precio y Medias Móviles</div>', unsafe_allow_html=True)

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["Close"], name="Cierre",
                                   line=dict(color="#22d3ee", width=2)))
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_40"], name="SMA 40",
                                   line=dict(color="#facc15", width=1.3)))
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_80"], name="SMA 80",
                                   line=dict(color="#a855f7", width=1.3)))
    fig_price.add_trace(go.Scatter(x=df_sel.index, y=df_sel["SMA_160"], name="SMA 160",
                                   line=dict(color="#f43f5e", width=1.3)))
    fig_price.update_layout(
        template="plotly_dark", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f'<div class="section-title">RSI (14)</div>', unsafe_allow_html=True)
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_sel.index, y=df_sel["RSI"], line=dict(color="#22d3ee")))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="#f87171")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="#34d399")
        fig_rsi.update_layout(template="plotly_dark", height=300,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig_rsi, use_container_width=True)

    with col_b:
        st.markdown(f'<div class="section-title">Distribución de Señales (Target)</div>', unsafe_allow_html=True)
        dist = df_sel["Target"].value_counts(normalize=True).sort_index()
        labels = [label_map[i] for i in dist.index]
        colors = [color_map[i] for i in dist.index]
        fig_dist = go.Figure(go.Bar(x=labels, y=dist.values, marker_color=colors,
                                    text=[f"{v:.1%}" for v in dist.values],
                                    textposition="outside"))
        fig_dist.update_layout(template="plotly_dark", height=300,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=20, b=10),
                               yaxis=dict(tickformat=".0%"))
        st.plotly_chart(fig_dist, use_container_width=True)

    # Importancia de features
    st.markdown(f'<div class="section-title">🎯 Importancia de Indicadores (XGBoost)</div>', unsafe_allow_html=True)
    X_s = df_sel[features]; y_s = df_sel["Target"]
    xgb_s = XGBClassifier(n_estimators=n_estimators, max_depth=3, learning_rate=0.05,
                          random_state=42, verbosity=0, use_label_encoder=False,
                          eval_metric="mlogloss").fit(X_s, y_s)
    imp = pd.Series(xgb_s.feature_importances_, index=features).sort_values()
    fig_imp = px.bar(x=imp.values, y=imp.index, orientation="h",
                     labels={"x": "Importancia", "y": "Indicador"})
    fig_imp.update_traces(marker_color="#3b82f6")
    fig_imp.update_layout(template="plotly_dark", height=320,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig_imp, use_container_width=True)


# ---------------- TAB 3: COMPARACIÓN ----------------
with tab3:
    st.markdown('<div class="section-title">🏆 Comparación de rendimiento por acción</div>', unsafe_allow_html=True)

    display_df = summary_df.copy()
    display_df["Clase Mayoritaria"] = display_df["Clase Mayoritaria"].apply(lambda x: f"{x:.2%}")
    display_df["KNN"] = display_df["KNN"].apply(lambda x: f"{x:.2%}")
    display_df["XGBoost"] = display_df["XGBoost"].apply(lambda x: f"{x:.2%}")
    display_df["Mejor Precisión"] = display_df["Mejor Precisión"].apply(lambda x: f"{x:.2%}")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Gráfico comparativo
    fig_cmp = go.Figure()
    fig_cmp.add_trace(go.Bar(name="KNN", x=summary_df["Ticker"], y=summary_df["KNN"],
                             marker_color="#3b82f6"))
    fig_cmp.add_trace(go.Bar(name="XGBoost", x=summary_df["Ticker"], y=summary_df["XGBoost"],
                             marker_color="#a855f7"))
    fig_cmp.add_trace(go.Scatter(name="Baseline (clase mayoritaria)",
                                 x=summary_df["Ticker"], y=summary_df["Clase Mayoritaria"],
                                 mode="lines+markers", line=dict(color="#fbbf24", dash="dash")))
    fig_cmp.update_layout(
        barmode="group", template="plotly_dark", height=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(tickformat=".0%", title="Precisión"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=20, b=10),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)


# ---------------- TAB 4: ACERCA DE ----------------
with tab4:
    st.markdown('<div class="section-title">ℹ️ ¿Cómo funciona esta herramienta?</div>', unsafe_allow_html=True)
    st.markdown("""
    Esta app entrena **dos modelos de Machine Learning** para predecir si una acción
    subirá, bajará o se mantendrá lateral en los próximos **3 días**.

    ### 🧠 Indicadores utilizados como features
    - **Slope SMA 40, 80, 160:** pendientes de medias móviles (tendencia).
    - **RSI (14):** índice de fuerza relativa (sobrecompra/sobreventa).
    - **Relative Range:** rango relativo diario (volatilidad).

    ### 🎯 Etiquetas del Target
    - 🟢 **ALCISTA:** retorno a 3 días > +1%
    - 🔴 **BAJISTA:** retorno a 3 días < −1%
    - ⚪ **LATERAL:** cualquier otro caso

    ### 🤖 Modelos
    - **KNN (K-Nearest Neighbors):** clasifica según los vecinos más cercanos en el espacio de features.
    - **XGBoost:** árboles de decisión con boosting, muy potente en datos tabulares.

    ### 📏 Referencias de comparación
    - **Clase Mayoritaria:** siempre predecir la clase más frecuente (baseline tonto).
    - Si el modelo **no supera claramente** al baseline, la señal tiene poco valor predictivo.
    """)

    st.markdown("""
    <div class="disclaimer">
        ⚠️ <b>Aviso importante:</b> Esta herramienta es un proyecto educativo y de investigación.
        <b>NO es asesoría financiera.</b> Los mercados son impredecibles y los modelos de ML
        pueden fallar. Nunca tomes decisiones de inversión basándote únicamente en esta app.
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

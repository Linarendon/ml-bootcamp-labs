
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Monitoreo del Modelo",
    page_icon="🎯",
    layout="wide"
)

# Título principal
st.title("🎯 Dashboard de Monitoreo del Modelo de Producción Petrolera")
st.markdown("---")

# Funciones auxiliares
@st.cache_data
def load_and_prepare_data():
    """Cargar y preparar los datos"""
    df = pd.read_csv('Production_data.csv')
    X = df.drop(['Well_ID', 'Production_Rate'], axis=1)
    y = df['Production_Rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X.columns

@st.cache_resource
def train_model():
    """Entrenar el modelo"""
    X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_prepare_data()
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model

def calculate_metrics(y_true, y_pred):
    """Calcular métricas del modelo"""
    return {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R²': r2_score(y_true, y_pred),
        'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    }

# Cargar datos y modelo
X_train_scaled, X_test_scaled, y_train, y_test, feature_names = load_and_prepare_data()
model = train_model()
y_pred = model.predict(X_test_scaled)

# Sidebar con controles
st.sidebar.header("⚙️ Configuración")
show_predictions = st.sidebar.checkbox("Mostrar Predicciones vs Reales", True)
show_residuals = st.sidebar.checkbox("Mostrar Análisis de Residuos", True)
show_metrics = st.sidebar.checkbox("Mostrar Métricas Detalladas", True)
show_distribution = st.sidebar.checkbox("Mostrar Distribuciones", True)

# Métricas principales en la parte superior
metrics = calculate_metrics(y_test, y_pred)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("RMSE", f"{metrics['RMSE']:.3f}")
with col2:
    st.metric("MAE", f"{metrics['MAE']:.3f}")
with col3:
    st.metric("R²", f"{metrics['R²']:.3f}")
with col4:
    st.metric("MAPE", f"{metrics['MAPE']:.2f}%")

st.markdown("---")

# Gráficos principales
if show_predictions:
    st.subheader("📊 Predicciones vs. Valores Reales")
    fig_pred = px.scatter(
        x=y_test, y=y_pred, 
        labels={'x': 'Valores Reales', 'y': 'Predicciones'},
        title="Predicciones vs. Valores Reales"
    )
    fig_pred.add_shape(
        type="line",
        x0=y_test.min(), y0=y_test.min(),
        x1=y_test.max(), y1=y_test.max(),
        line=dict(color="red", width=2, dash="dash")
    )
    st.plotly_chart(fig_pred, use_container_width=True)

if show_residuals:
    st.subheader("📈 Análisis de Residuos")
    residuals = y_test - y_pred

    col1, col2 = st.columns(2)

    with col1:
        fig_residuals = px.scatter(
            x=y_pred, y=residuals,
            labels={'x': 'Predicciones', 'y': 'Residuos'},
            title="Residuos vs. Predicciones"
        )
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_residuals, use_container_width=True)

    with col2:
        fig_hist_residuals = px.histogram(
            x=residuals,
            title="Distribución de Residuos",
            labels={'x': 'Residuos'}
        )
        st.plotly_chart(fig_hist_residuals, use_container_width=True)

if show_distribution:
    st.subheader("📊 Distribuciones")

    col1, col2 = st.columns(2)

    with col1:
        fig_pred_dist = px.histogram(
            x=y_pred,
            title="Distribución de Predicciones",
            labels={'x': 'Predicciones'}
        )
        st.plotly_chart(fig_pred_dist, use_container_width=True)

    with col2:
        errors = np.abs(y_test - y_pred)
        fig_error_dist = px.histogram(
            x=errors,
            title="Distribución de Errores Absolutos",
            labels={'x': 'Error Absoluto'}
        )
        st.plotly_chart(fig_error_dist, use_container_width=True)

if show_metrics:
    st.subheader("📋 Métricas Detalladas")

    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Métrica', 'Valor'])
    st.dataframe(metrics_df, use_container_width=True)

    # Información adicional
    st.info(f"""
    **Información del Modelo:**
    - Tipo: Regresión Lineal
    - Características: {len(feature_names)}
    - Muestras de prueba: {len(y_test)}
    - Sesgo promedio: {np.mean(y_pred - y_test):.3f}
    - Desviación estándar de residuos: {np.std(y_test - y_pred):.3f}
    """)

# Footer
st.markdown("---")
st.markdown("**Dashboard generado automáticamente** | Última actualización: " + 
           pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
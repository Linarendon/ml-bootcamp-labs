# Radar_V2.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import unicodedata

st.title("Comparación de correlaciones por pozo e intervalo")

# --- 1. Cargar archivo ---
st.sidebar.header("Cargar archivo CSV")
uploaded_file = st.sidebar.file_uploader("Selecciona el archivo con resultados", type=["csv"])

def normalizar_columna(col):
    """Normaliza nombres de columnas: quita tildes, espacios, mayúsculas."""
    col = str(col)
    col = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('utf-8')
    col = col.replace(" ", "_").replace("-", "_").upper().strip()
    return col

if uploaded_file is not None:
    try:
        # --- Leer CSV ---
        df = pd.read_csv(uploaded_file, sep=None, engine='python')
        df.columns = [normalizar_columna(c) for c in df.columns]

        st.success("Archivo cargado correctamente ✅")
        st.write("Vista previa de los datos:")
        st.dataframe(df.head())

        # --- Validar columnas necesarias ---
        columnas_necesarias = {"POZO", "INTERVALO", "CORRELACION"}
        if not columnas_necesarias.issubset(df.columns):
            st.error(f"El archivo cargado no contiene las columnas requeridas: {columnas_necesarias}")
            st.write("Columnas detectadas en el archivo:", list(df.columns))
            st.stop()

        # --- Agrupar por pozo e intervalo ---
        df_summary = df.groupby(["POZO", "INTERVALO"])["CORRELACION"].mean().reset_index()

        pozos = df_summary["POZO"].unique()
        intervalos = df_summary["INTERVALO"].unique()

        # --- Crear gráfico radar ---
        fig = go.Figure()

        for pozo in pozos:
            valores = []
            for intervalo in intervalos:
                valor = df_summary.loc[
                    (df_summary["POZO"] == pozo) & (df_summary["INTERVALO"] == intervalo),
                    "CORRELACION"
                ]
                valores.append(valor.values[0] if not valor.empty else 0)

            # Cerrar el polígono
            valores += [valores[0]]
            intervalos_cerrado = list(intervalos) + [intervalos[0]]

            fig.add_trace(go.Scatterpolar(
                r=valores,
                theta=intervalos_cerrado,
                fill='toself',
                name=pozo
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 0.8])),
            showlegend=True,
            title="Radar de correlaciones promedio por pozo e intervalo"
        )

        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error al procesar el archivo: {e}")

else:
    st.info("Por favor carga un archivo CSV con las columnas: POZO, INTERVALO, CORRELACION.")

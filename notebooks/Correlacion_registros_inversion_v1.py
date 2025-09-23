# Correlacion_heatmap_consistencia_app_v2.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from pathlib import Path

st.set_page_config(page_title="Heatmap Correlaciones (Con Consistencia)", layout="wide")

# -------------------------
# Helpers
# -------------------------
def str_to_bool_consistent(x):
    """Convierte valores diversos a boolean (True = consistente)."""
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    if s in ("si", "sí", "yes", "y", "1", "true", "t", "ok"):
        return True
    if s in ("no", "n", "0", "false", "f"):
        return False
    # si no está claro, devolver NaN
    return np.nan

def expected_sign(reg, inv):
    """Reglas esperadas (devuelve 'Positiva' / 'Negativa' / 'N/A')."""
    r = str(reg).strip().upper()
    i = str(inv).strip().upper()
    # reglas según tu especificación
    if r == "GR" and i in ("IP", "VPVS", "VP/V S", "VP/V S".replace(" ", "")):
        return "Positiva"
    if r == "VSH" and i in ("IP", "VPVS"):
        return "Positiva"
    if r in ("PERM", "PERMEABILIDAD") and i in ("IP", "VPVS"):
        return "Negativa"
    if r in ("POR", "POROSIDAD", "POROSITY") and i in ("IP", "VPVS"):
        return "Negativa"
    # fallback: si no coincide con reglas explícitas, intentar reglas por tokens
    if r.startswith("GR"):
        return "Positiva"
    if r.startswith("VSH"):
        return "Positiva"
    if r.startswith("PER") or r.startswith("K"):
        return "Negativa"
    if r.startswith("POR"):
        return "Negativa"
    return "N/A"

def build_annot_string(corr_val, cons_flag, exp_sign, style="num_sym"):
    """Construye la cadena de anotación según estilo:
       - 'num' -> solo número
       - 'num_sym' -> número + símbolo
       - 'num_sym_rule' -> número + símbolo + (regla)
    """
    if pd.isna(corr_val):
        return ""
    num = f"{corr_val:.2f}"
    symbol = "✅" if cons_flag else "❌"
    if style == "num":
        return num
    if style == "num_sym":
        return f"{num}\n{symbol}"
    if style == "num_sym_rule":
        rule = exp_sign if exp_sign is not None else ""
        return f"{num}\n{symbol}\n({rule})"
    return num

# -------------------------
# App UI - carga de datos
# -------------------------
st.title("🔥 Heatmap de Correlaciones (Registros vs Inversión) con Consistencia y Regla Esperada")

ruta_default = "/workspaces/ml-bootcamp-labs/data/raw/Correlacion_Registros_Inversion_cvs.csv"
uploaded = st.sidebar.file_uploader("Sube CSV (opcional) — separador ';'", type=["csv"])

if uploaded is not None:
    df = pd.read_csv(uploaded, sep=";")
else:
    df = pd.read_csv(ruta_default, sep=";")

# Normalizar nombres de columnas por si hay espacios invisibles
df.columns = [c.strip() for c in df.columns]

st.sidebar.markdown("**Vista rápida de columnas leídas**")
st.sidebar.write(list(df.columns))

# columnas esperadas
expected_cols = ["POZO", "DELTA", "DELTA ABSOLUTO", "CURVA_REGISTRO", "CURVA_INVERSION",
                 "CORRELACION", "INTERVALO_FT", "SIGNO_ESPERADO", "CONSISTENCIA", "LISTA"]

# mostrar preview
st.subheader("Vista previa de datos")
st.dataframe(df.head(8))

# -------------------------
# Preparaciones
# -------------------------
# Normalizar campos de texto importantes
for col in ["POZO", "CURVA_REGISTRO", "CURVA_INVERSION", "SIGNO_ESPERADO", "CONSISTENCIA"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip()

# añadir columna booleana de consistencia (media si multiples filas)
if "CONSISTENCIA" in df.columns:
    df["CONSISTENCIA_BOOL"] = df["CONSISTENCIA"].apply(str_to_bool_consistent)
else:
    df["CONSISTENCIA_BOOL"] = np.nan

# crear par de curvas
df["PAR_CURVAS"] = df["CURVA_REGISTRO"].str.upper() + "_vs_" + df["CURVA_INVERSION"].str.upper()

# -------------------------
# Sidebar filtros
# -------------------------
st.sidebar.header("Filtros")
pozos = sorted(df["POZO"].unique().tolist())
pozo_sel = st.sidebar.multiselect("Pozos (filtrar)", pozos, default=pozos)

curvas_reg = sorted(df["CURVA_REGISTRO"].unique().tolist())
curvas_inv = sorted(df["CURVA_INVERSION"].unique().tolist())
cur_reg_sel = st.sidebar.multiselect("Curvas Registro (filtrar)", curvas_reg, default=curvas_reg)
cur_inv_sel = st.sidebar.multiselect("Curvas Inversión (filtrar)", curvas_inv, default=curvas_inv)

mode = st.sidebar.radio("Modo de visualización", options=["Solo correlación", "Correlación + Consistencia y Regla"])
annot_style = st.sidebar.selectbox("Estilo de anotación", options=[
    ("Número", "num"),
    ("Número + símbolo (✅/❌)", "num_sym"),
    ("Número + símbolo + regla esperada", "num_sym_rule")
], index=1)
annot_style = annot_style[1]

# -------------------------
# Filtrar DF
# -------------------------
df_f = df[
    (df["POZO"].isin(pozo_sel)) &
    (df["CURVA_REGISTRO"].isin(cur_reg_sel)) &
    (df["CURVA_INVERSION"].isin(cur_inv_sel))
].copy()

if df_f.empty:
    st.warning("⚠️ No hay datos para los filtros seleccionados. Ajusta filtros.")
    st.stop()

# -------------------------
# Pivot tablas (correlación y consistencia)
# -------------------------
# correlación: media si existen varias filas
pivot_corr = df_f.pivot_table(index="POZO", columns=["CURVA_REGISTRO", "CURVA_INVERSION"],
                              values="CORRELACION", aggfunc="mean")
# consistencia: tomamos la media de booleanos -> proporción consistente (0..1)
pivot_cons = df_f.pivot_table(index="POZO", columns=["CURVA_REGISTRO", "CURVA_INVERSION"],
                              values="CONSISTENCIA_BOOL", aggfunc="mean")

# convert multiindex columns to ordered list for consistent plotting
cols = pivot_corr.columns
n_rows, n_cols = pivot_corr.shape

# -------------------------
# Construir matriz de anotaciones (strings)
# -------------------------
annot = pd.DataFrame(index=pivot_corr.index, columns=pivot_corr.columns, dtype=object)
for r in pivot_corr.index:
    for c in pivot_corr.columns:
        corr_val = pivot_corr.loc[r, c]
        cons_val = pivot_cons.loc[r, c] if (c in pivot_cons.columns and r in pivot_cons.index) else np.nan
        # definir consistente si proporción >= 0.5
        cons_flag = bool(cons_val >= 0.5) if not pd.isna(cons_val) else False
        reg_expected = expected_sign(c[0], c[1])
        if mode == "Solo correlación":
            # solo número
            s = build_annot_string(corr_val, cons_flag, reg_expected, style="num")
        else:
            s = build_annot_string(corr_val, cons_flag, reg_expected, style=annot_style)
        annot.loc[r, c] = s

# -------------------------
# Dibujar heatmap con seaborn
# -------------------------
# figure size adaptativa
fig_w = max(12, 0.6 * max(1, n_cols))
fig_h = max(6, 0.3 * max(1, n_rows))
fig, ax = plt.subplots(figsize=(fig_w, fig_h))

# seaborn heatmap: pasar los valores numéricos en pivot_corr (NaN aceptadas)
sns.set(font_scale=0.9)
sns.heatmap(pivot_corr, annot=annot.values, fmt="", cmap="coolwarm", center=0,
            cbar_kws={'label': 'Correlación'}, linewidths=0.5, linecolor='gray', ax=ax)

ax.set_title("Heatmap: Correlaciones Pozo vs Pares (Curva Registro vs Curva Inversión)")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

st.pyplot(fig)

# -------------------------
# Leyenda / Reglas esperadas (sidebar y debajo del mapa)
# -------------------------
st.sidebar.markdown("## 📖 Reglas de consistencia esperadas")
st.sidebar.markdown("""
- **GR – IP → Positiva**  
- **GR – VPVS → Positiva**  
- **VSH – IP → Positiva**  
- **VSH – VPVS → Positiva**  
- **PERM – IP → Negativa**  
- **PERM – VPVS → Negativa**  
- **POR – IP → Negativa**  
- **POR – VPVS → Negativa**
""")

st.markdown("### 📖 Reglas de consistencia esperadas (referencia)")
st.markdown("""
- **GR – IP → Positiva**  
- **GR – VPVS → Positiva**  
- **VSH – IP → Positiva**  
- **VSH – VPVS → Positiva**  
- **PERM – IP → Negativa**  
- **PERM – VPVS → Negativa**  
- **POR – IP → Negativa**  
- **POR – VPVS → Negativa**
""")

# -------------------------
# Exportar imagen JPEG
# -------------------------
buf = io.BytesIO()
fig.savefig(buf, format="jpeg", dpi=200, bbox_inches='tight')
buf.seek(0)

st.download_button("📥 Descargar heatmap (JPEG)", data=buf, file_name="heatmap_correlaciones_consistencia.jpg", mime="image/jpeg")

# También mostrar una pequeña tabla resumen con conteos de consistencia
st.subheader("Resumen de consistencia por par de curvas")
cons_summary = df_f.groupby(["CURVA_REGISTRO", "CURVA_INVERSION"]).agg(
    total_entries=("CONSISTENCIA_BOOL", "count"),
    consistency_mean=("CONSISTENCIA_BOOL", "mean")
).reset_index()
# convertir consistency_mean a porcentaje
cons_summary["consistency_pct"] = (cons_summary["consistency_mean"] * 100).round(1)
st.dataframe(cons_summary.sort_values(by="consistency_pct", ascending=False).reset_index(drop=True))

st.success("Heatmap generado. Usa los filtros para explorar diferentes pozos/pares de curvas.")

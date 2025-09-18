# CURVAS_CLASIFICADAS_APP_EXTENDED.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score
)

st.set_page_config(layout="wide", page_title="Validación Curvas Clasificadas - Extended")

# ---------- Config / paths ----------
CSV_PATH = "/workspaces/ml-bootcamp-labs/data/raw/CURVAS_CLASIFICADAS_CVS.csv"
RESULTS_DIR = "/workspaces/ml-bootcamp-labs/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------- Load data ----------
@st.cache_data
def load_data(path=CSV_PATH):
    df = pd.read_csv(path, sep=";")
    # normalize Tipo text
    if "Tipo" in df.columns:
        df["Tipo"] = df["Tipo"].astype(str).str.strip().str.upper()
    # numeric profundidad
    df["Profundidad"] = pd.to_numeric(df["Profundidad"], errors="coerce")
    # ensure Valor numeric (classes may be int-like)
    df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    return df

df = load_data()

st.title("📊 Validación de Curvas Clasificadas — Extended")
st.markdown("Añadido: métricas avanzadas, gráficos por clase, matriz de errores y exportación de gráficos.")

# quick diagnostics
with st.expander("📌 Datos (diagnóstico)"):
    st.write("Archivo:", CSV_PATH)
    st.write("Columnas:", df.columns.tolist())
    st.write("Tipos únicos en 'Tipo' (muestra):", df["Tipo"].unique()[:50])
    st.write("Pozos disponibles:", df["Pozo"].unique()[:20])
    st.dataframe(df.head(10))

# ---------- Pivot to wide ----------
# Make a pivot table where each Tipo becomes a column
df_pivot = df.pivot_table(index=["Profundidad", "Pozo"], columns="Tipo", values="Valor", aggfunc="first").reset_index()
df_pivot.columns.name = None

# Sidebar controls
st.sidebar.header("Filtros y opciones")

# select pozo
pozos = df_pivot["Pozo"].dropna().unique().tolist()
pozo_sel = st.sidebar.selectbox("Selecciona Pozo", pozos)

# depth inputs
min_prof = int(df_pivot["Profundidad"].min())
max_prof = int(df_pivot["Profundidad"].max())
prof_min = st.sidebar.number_input("Profundidad mínima", min_value=min_prof, max_value=max_prof, value=min_prof, step=1)
prof_max = st.sidebar.number_input("Profundidad máxima", min_value=min_prof, max_value=max_prof, value=max_prof, step=1)

# available curves from pivot (exclude Profundidad, Pozo)
available_curves = [c for c in df_pivot.columns.tolist() if c not in ("Profundidad", "Pozo")]
available_curves_sorted = sorted(available_curves)

# choose curves
st.sidebar.subheader("Curvas a comparar")
col1, col2 = st.sidebar.columns(2)
with col1:
    curve_gt = col1.selectbox("Ground truth (real)", available_curves_sorted)
with col2:
    curve_pred = col2.selectbox("Predicción (modelo)", available_curves_sorted, index=min(1, len(available_curves_sorted)-1))

# alignment method
st.sidebar.subheader("Alineación")
align_method = st.sidebar.radio("Método de alineación", options=["Exact match (dropna)", "Tolerance (merge_asof)"])
tolerance = 33
if align_method == "Tolerance (merge_asof)":
    tolerance = st.sidebar.number_input("Tolerancia (ft)", min_value=0, max_value=200, value=33, step=1)

# Buttons for saving
st.sidebar.subheader("Export")
save_plots = st.sidebar.checkbox("Habilitar guardado automático de gráficas (.jpg)", value=False)

# Filter by pozo and depth
df_filt = df_pivot[
    (df_pivot["Pozo"] == pozo_sel) &
    (df_pivot["Profundidad"] >= prof_min) &
    (df_pivot["Profundidad"] <= prof_max)
].copy()

if df_filt.empty:
    st.error("No hay datos para ese pozo y rango de profundidad. Revisa selección.")
    st.stop()

# Check selected curves exist
if curve_gt not in df_filt.columns or curve_pred not in df_filt.columns:
    st.error(f"Las curvas seleccionadas no están presentes en el DataFrame filtrado: {curve_gt} o {curve_pred}")
    st.stop()

# Prepare aligned dataset
if align_method == "Exact match (dropna)":
    df_align = df_filt[["Profundidad", curve_gt, curve_pred]].dropna().copy()
    # keep Profundidad for plotting
    df_align = df_align.rename(columns={curve_gt: "GT", curve_pred: "PRED"})
else:
    # merge_asof requires sorted by Profundidad and both sides present as separate DF
    df_real = df_filt[["Profundidad", curve_gt]].dropna().sort_values("Profundidad").rename(columns={curve_gt: "GT"})
    df_pred = df_filt[["Profundidad", curve_pred]].dropna().sort_values("Profundidad").rename(columns={curve_pred: "PRED"})
    # do asof merge (nearest within tolerance)
    df_align = pd.merge_asof(df_real, df_pred, on="Profundidad", direction="nearest", tolerance=tolerance).dropna()

# Cast to int classes if possible
try:
    df_align["GT"] = df_align["GT"].astype(int)
    df_align["PRED"] = df_align["PRED"].astype(int)
except Exception:
    # if cast fails, leave as is
    pass

# If no aligned rows, warn
if df_align.empty:
    st.warning("❌ No hay puntos alineados entre GT y Pred en el rango/tolerancia seleccionados.")
    st.stop()

# Compute confusion matrix and metrics
y_true = df_align["GT"].values
y_pred = df_align["PRED"].values

labels = sorted(list(set(np.unique(y_true)) | set(np.unique(y_pred))))
# ensure labels are ints if possible
labels = [int(l) for l in labels]

cm = confusion_matrix(y_true, y_pred, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)

# Absolute confusion matrix plot
fig1, ax1 = plt.subplots(figsize=(5,4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_xlabel("Predicho")
ax1.set_ylabel("Real")
ax1.set_title("Matriz de Confusión (conteos)")

# Normalized per-row (recall per class)
with np.errstate(all='ignore'):
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
cm_norm_df = pd.DataFrame(np.nan_to_num(cm_norm), index=labels, columns=labels)

fig2, ax2 = plt.subplots(figsize=(5,4))
sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Greens", ax=ax2)
ax2.set_xlabel("Predicho")
ax2.set_ylabel("Real")
ax2.set_title("Matriz Normalizada (recall por clase)")

# Misclassification matrix (zero diagonal)
cm_errors = cm.copy()
np.fill_diagonal(cm_errors, 0)
cm_err_df = pd.DataFrame(cm_errors, index=labels, columns=labels)

fig3, ax3 = plt.subplots(figsize=(5,4))
sns.heatmap(cm_err_df, annot=True, fmt="d", cmap="Reds", ax=ax3)
ax3.set_xlabel("Predicho")
ax3.set_ylabel("Real")
ax3.set_title("Matriz de Errores (sin diagonal)")

# Metrics
accuracy = np.trace(cm) / cm.sum() if cm.sum() > 0 else 0.0
balanced_acc = balanced_accuracy_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
prec_per_class = precision_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
rec_per_class = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
report_dict = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report_dict).transpose()

# Precision bar chart per class
fig4, ax4 = plt.subplots()
sns.barplot(x=[str(l) for l in labels], y=prec_per_class, ax=ax4)
ax4.set_ylim(0,1)
ax4.set_ylabel("Precisión")
ax4.set_xlabel("Clase")
ax4.set_title("Precisión por clase (precision)")

# Plot GT vs Pred along depth: scatter +/- jitter to see overlaps
fig5, ax5 = plt.subplots(figsize=(4,8))
ax5.scatter(df_align["GT"].values, df_align["Profundidad"].values, 
            label="Ground Truth (GT)", alpha=0.6, marker='o', color="blue")
ax5.scatter(df_align["PRED"].values + 0.08, df_align["Profundidad"].values, 
            label="Predicción (PRED)", alpha=0.6, marker='x', color="orange")

ax5.invert_yaxis()
ax5.set_xlabel("Clase")
ax5.set_ylabel("Profundidad (ft)")
ax5.set_title("Comparación en profundidad (GT vs Pred)")
ax5.set_xlim(min(labels)-0.5, max(labels)+0.5)
ax5.legend()   # <-- activa la leyenda con convención


# Display widgets + metrics in the app
left_col, right_col = st.columns([1,1])

with left_col:
    st.subheader("Matriz de confusión")
    st.pyplot(fig1)
    st.subheader("Matriz Normalizada")
    st.pyplot(fig2)

with right_col:
    st.subheader("Matriz de errores")
    st.pyplot(fig3)
    st.subheader("Precisión por clase")
    st.pyplot(fig4)

st.subheader("📈 Métricas globales")
st.write(f"Total puntos evaluados: **{len(df_align)}**")
st.write(f"Accuracy (global): **{accuracy:.3f}**")
st.write(f"Balanced accuracy: **{balanced_acc:.3f}**")
st.write(f"Cohen's Kappa: **{kappa:.3f}**")
st.dataframe(report_df)

st.subheader("📉 Comparación en profundidad")
st.pyplot(fig5)

# Save figures as JPEG option
if save_plots:
    saved_files = []
    figs = {
        "confusion_counts": fig1,
        "confusion_norm": fig2,
        "confusion_errors": fig3,
        "precision_per_class": fig4,
        "depth_comparison": fig5
    }
    for name, fig in figs.items():
        out_path = os.path.join(RESULTS_DIR, f"{pozo_sel}_{curve_gt}_vs_{curve_pred}_{name}.jpg")
        # save
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        saved_files.append(out_path)
    st.success(f"Gráficas guardadas en {RESULTS_DIR}")
    # allow download of a sample file (first)
    if saved_files:
        with open(saved_files[0], "rb") as f:
            st.download_button("Descargar primer JPG guardado", f.read(), file_name=os.path.basename(saved_files[0]), mime="image/jpeg")

# Export aligned table and report
export_df = df_align.copy()
export_df = export_df.rename(columns={"GT": f"{curve_gt}_GT", "PRED": f"{curve_pred}_PRED"})
csv_bytes = export_df.to_csv(index=False).encode("utf-8")
st.download_button("Descargar tabla alineada (CSV)", csv_bytes, file_name=f"{pozo_sel}_{curve_gt}_vs_{curve_pred}_aligned.csv", mime="text/csv")

# also export report_df
report_csv = report_df.to_csv().encode("utf-8")
st.download_button("Descargar reporte (CSV)", report_csv, file_name=f"{pozo_sel}_{curve_gt}_vs_{curve_pred}_report.csv", mime="text/csv")



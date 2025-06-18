**SESIÓN 08: LABORATORIO - INTERPRETACIÓN Y VISUALIZACIÓN DE RESULTADOS**

**Objetivo:**

*   Aplicar técnicas para interpretar modelos de Machine Learning.
*   Identificar la importancia de las características (feature importance).
*   Crear visualizaciones efectivas para comunicar resultados a stakeholders técnicos y no técnicos.
*   Diseñar dashboards para el monitoreo de predicciones.

**Duración:** 2 horas

**Contenidos:**

1.  **Preparación del Entorno (15 minutos)**
    *   Importar las librerías necesarias (Pandas, NumPy, Scikit-learn, Matplotlib/Seaborn, SHAP).
    *   Cargar un dataset y un modelo entrenado (se proporcionará un ejemplo, o se usará el resultado de sesiones anteriores).

2.  **Cálculo de la Importancia de las Características (30 minutos)**
    *   Utilizar métodos inherentes al modelo (ej: `feature_importances_` en Árboles de Decisión).
    *   Implementar permutation importance (Scikit-learn).
    *   Introducción a SHAP (SHapley Additive exPlanations) para una interpretación más completa (opcional, si el tiempo lo permite).

3.  **Visualización de Resultados Predictivos (45 minutos)**
    *   Crear gráficos de predicciones vs. valores reales (scatter plots).
    *   Generar gráficos de residuos (residual plots) para evaluar la calidad del ajuste.
    *   Implementar visualizaciones interactivas (ej: con Plotly o Bokeh) para explorar predicciones individuales.

4.  **Diseño de Dashboards para el Monitoreo (30 minutos)**
    *   Identificar las métricas clave para el monitoreo del modelo (ej: rendimiento, distribución de predicciones).
    *   Diseñar un dashboard simple (puede ser en Jupyter Notebook o utilizando una herramienta como Streamlit) que muestre estas métricas y visualizaciones.

**Recursos:**

*   Dataset de ejemplo (CSV, si es necesario).
*   Modelo entrenado de una sesión anterior (si es necesario).
*   Notebook de Jupyter con código base (funciones básicas, sugerencias).
*   Documentación de las librerías (Scikit-learn, Matplotlib, Seaborn, SHAP, Plotly/Bokeh, Streamlit).

**Entregables:**

*   Notebook de Jupyter con el código implementado y las visualizaciones generadas.
*   Diseño de un dashboard simple para el monitoreo del modelo (puede ser un prototipo).
*   Presentación corta (5 minutos) resumiendo los hallazgos principales y el diseño del dashboard.

**Criterios de Evaluación:**

*   Corrección en el cálculo de la importancia de las características.
*   Efectividad de las visualizaciones para comunicar resultados.
*   Claridad en la interpretación de los resultados.
*   Coherencia del diseño del dashboard con los objetivos de monitoreo.
*   Calidad de la presentación y documentación.

---

**Laboratorio Detallado usando un Modelo de Regresión de Producción Petrolera:**

**Dataset:** Utilizaremos el dataset 'Production\_data.csv' (descrito en laboratorios anteriores).

**Modelo:** Asumimos que ya tienes un modelo de Regresión Lineal (o Polinómica) entrenado del laboratorio de la Sesión 05.

**Paso 1: Preparación del Entorno (15 minutos):**

1.  Importa las bibliotecas necesarias:

    ```python
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.inspection import permutation_importance
    import shap
    import plotly.express as px
    import streamlit as st # Para construir el Dashboard
    import warnings
    warnings.filterwarnings('ignore')
    ```

2.  Carga el dataset y el modelo (recuerda que este laboratorio asume que tienes un modelo entrenado previamente):

    ```python
    df = pd.read_csv('Production_data.csv')

    X = df.drop(['Well_ID', 'Production_Rate'], axis=1)
    y = df['Production_Rate']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)  # Modelo ya entrenado
    ```

**Paso 2: Cálculo de la Importancia de las Características (30 minutos):**

1.  Importancia inherente al modelo (para modelos basados en árboles):

    ```python
    # Este paso es aplicable si usas un modelo basado en árboles (ej: DecisionTreeRegressor)
    # feature_importances = model.feature_importances_
    # for feature, importance in zip(X.columns, feature_importances):
    #     print(f"{feature}: {importance:.3f}")
    ```

2.  Permutation Importance:

    ```python
    result = permutation_importance(model, X_test_scaled, y_test, n_repeats=10, random_state=42)
    importances = result.importances_mean
    for feature, importance in zip(X.columns, importances):
        print(f"{feature}: {importance:.3f}")
    ```

3.  (Opcional) SHAP values:

    ```python
    explainer = shap.LinearExplainer(model, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    ```

**Paso 3: Visualización de Resultados Predictivos (45 minutos):**

1.  Predicciones vs. Valores Reales:

    ```python
    y_pred = model.predict(X_test_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    plt.xlabel("Valores Reales")
    plt.ylabel("Predicciones")
    plt.title("Predicciones vs. Valores Reales")
    plt.show()
    ```

2.  Gráfico de Residuos:

    ```python
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='--')
    plt.xlabel("Predicciones")
    plt.ylabel("Residuos")
    plt.title("Gráfico de Residuos")
    plt.show()
    ```

3.  (Opcional) Visualizaciones Interactivas con Plotly:

    ```python
    fig = px.scatter(x=y_test, y=y_pred, labels={'x': 'Valores Reales', 'y': 'Predicciones'}, title="Predicciones vs. Valores Reales (Interactivo)")
    fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), x1=y_test.max(), y1=y_test.max(), line=dict(color="black", width=1, dash="dash"))
    fig.show()
    ```

**Paso 4: Diseño de Dashboards para el Monitoreo (30 minutos):**

1.  Identifica métricas clave:

    *   RMSE
    *   R²
    *   Distribución de predicciones

2.  Implementa un dashboard simple con Streamlit (ejemplo básico):

    ```python
    st.title("Dashboard de Monitoreo del Modelo de Producción")

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"R²: {r2:.2f}")

    st.subheader("Predicciones vs. Valores Reales")
    fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': 'Valores Reales', 'y': 'Predicciones'})
    st.plotly_chart(fig_pred)

    st.subheader("Distribución de Predicciones")
    fig_dist = px.histogram(x=y_pred, labels={'x': 'Predicciones'})
    st.plotly_chart(fig_dist)
    ```

    Para ejecutar este dashboard, guarda el código como `app.py` y ejecuta `streamlit run app.py` en la terminal.


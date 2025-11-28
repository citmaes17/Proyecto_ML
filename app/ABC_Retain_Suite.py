
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import joblib

# ---------------------------------------------------------
# Configuración básica de la página
# ---------------------------------------------------------
st.set_page_config(
    page_title="ABC Retain Suite – ChurnRadar Valioso",
    layout="wide"
)


# ---------------------------------------------------------
# Carga del modelo de churn (pipeline entrenado)
# ---------------------------------------------------------
@st.cache_resource
def load_churn_model():
    """
    Carga el pipeline entrenado de churn_valioso.
    Busca en ./models y en ../models.
    """
    posibles_rutas = [
        Path("models/churn_pipeline.pkl"),
        Path("../models/churn_pipeline.pkl")
    ]

    for ruta in posibles_rutas:
        if ruta.exists():
            try:
                modelo = joblib.load(ruta)
                return modelo, None
            except Exception as e:
                return None, f"Error cargando el modelo desde {ruta}: {e}"

    return None, "No se encontró el archivo 'churn_pipeline.pkl' en ./models ni en ../models."


# ---------------------------------------------------------
# Segmentación de clientes (K-Means "on the fly")
# ---------------------------------------------------------
def describir_clusters(perfil_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Genera una descripción de negocio para cada cluster,
    comparando contra la mediana de cada variable.
    """
    desc_list = []

    medianas = perfil_clusters.median()

    for cluster_id, row in perfil_clusters.iterrows():
        partes = []

        # Gasto total
        if row["MntTotal"] >= medianas["MntTotal"]:
            partes.append("gasto alto")
        else:
            partes.append("gasto bajo")

        # Frecuencia de compra
        if row["TotalPurchases"] >= medianas["TotalPurchases"]:
            partes.append("frecuencia alta")
        else:
            partes.append("frecuencia baja")

        # Ingresos
        if row["Income"] >= medianas["Income"]:
            partes.append("ingresos altos")
        else:
            partes.append("ingresos más bajos")

        # Canal preferido
        canales = ["Perc_WebPurchases", "Perc_CatalogPurchases", "Perc_StorePurchases"]
        canal_max = row[canales].idxmax()
        mapa_canal = {
            "Perc_WebPurchases": "prefiere canal web",
            "Perc_CatalogPurchases": "prefiere catálogo",
            "Perc_StorePurchases": "prefiere tienda física",
        }
        partes.append(mapa_canal.get(canal_max, "mix de canales"))

        # Actividad web (visitas)
        if row["NumWebVisitsMonth"] >= medianas["NumWebVisitsMonth"]:
            partes.append("mucha actividad web")
        else:
            partes.append("poca actividad web")

        desc = ", ".join(partes)
        desc_list.append({"Cluster": int(cluster_id), "Descripción_segmento": desc})

    return pd.DataFrame(desc_list).sort_values("Cluster")


def segmentar_clientes(df: pd.DataFrame):
    """
    Crea segmentos de clientes usando K-Means sobre variables de comportamiento.
    Devuelve:
        - df con columna 'Cluster'
        - perfil medio por cluster
        - tabla descriptiva de cada segmento
    """
    # Variables que usamos en el notebook de segmentación
    seg_cols = [
        "Recency",
        "CustomerTenure",
        "MntTotal",
        "TotalPurchases",
        "Income",
        "Perc_WebPurchases",
        "Perc_CatalogPurchases",
        "Perc_StorePurchases",
        "NumWebVisitsMonth",
        "Kidhome",
        "Teenhome",
    ]

    faltantes = [c for c in seg_cols if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias para la segmentación: "
            + ", ".join(faltantes)
        )

    seg_df = df.copy()

    # Imputación sencilla para Income y algún posible NaN
    seg_df[seg_cols] = seg_df[seg_cols].copy()
    for col in seg_cols:
        if seg_df[col].isna().any():
            if seg_df[col].dtype.kind in "biufc":
                seg_df[col] = seg_df[col].fillna(seg_df[col].median())
            else:
                seg_df[col] = seg_df[col].fillna(seg_df[col].mode()[0])

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(seg_df[seg_cols])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    seg_df["Cluster"] = clusters

    perfil_clusters = seg_df.groupby("Cluster")[seg_cols].mean().round(2)
    desc_clusters = describir_clusters(perfil_clusters)

    return seg_df, perfil_clusters, desc_clusters


# ---------------------------------------------------------
# Scoring de churn valioso con el modelo entrenado
# ---------------------------------------------------------
def scorear_churn_valioso(df: pd.DataFrame, modelo):
    """
    Aplica el pipeline entrenado para obtener probabilidad y predicción de churn valioso.
    Añade columnas:
        - Churn_Valioso_Prob
        - Churn_Valioso_Pred
        - Riesgo_Churn (Bajo / Medio / Alto)
    """
    feature_cols = [
        "Recency",
        "MntTotal",
        "TotalPurchases",
        "Income",
        "Perc_CatalogPurchases",
        "NumWebVisitsMonth",
        "Kidhome",
        "Teenhome",
        "Education",
        "Marital_Status",
    ]

    faltantes = [c for c in feature_cols if c not in df.columns]
    if faltantes:
        raise ValueError(
            "Faltan columnas necesarias para el modelo de churn valioso: "
            + ", ".join(faltantes)
        )

    X = df[feature_cols].copy()

    # El pipeline ya se encarga del preprocesamiento (imputación, escalado, OHE, etc.)
    probs = modelo.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    df_scored = df.copy()
    df_scored["Churn_Valioso_Prob"] = probs
    df_scored["Churn_Valioso_Pred"] = preds

    # Nivel de riesgo para hacerlo más "de negocio"
    condiciones = [
        df_scored["Churn_Valioso_Prob"] >= 0.7,
        df_scored["Churn_Valioso_Prob"] >= 0.4,
    ]
    elecciones = ["Alto", "Medio"]
    df_scored["Riesgo_Churn"] = np.select(condiciones, elecciones, default="Bajo")

    return df_scored


# ---------------------------------------------------------
# Carga de datos demo (si existe)
# ---------------------------------------------------------
@st.cache_data
def load_demo_data():
    """
    Intenta cargar una base de ejemplo del proyecto.
    Puedes ajustar esta ruta a tu dataset master o test.
    """
    posibles_rutas = [
        Path("data/superstore_master_test.csv"),
        Path("data/superstore_master.csv"),
        Path("../data/superstore_master_test.csv"),
        Path("../data/superstore_master.csv"),
        Path("data/superstore_procesado.csv"),
        Path("../data/superstore_procesado.csv"),
    ]

    for ruta in posibles_rutas:
        if ruta.exists():
            try:
                df = pd.read_csv(ruta)
                return df, None
            except Exception as e:
                return None, f"Error cargando datos demo desde {ruta}: {e}"

    return None, "No se encontró un dataset demo en las rutas esperadas."


# ---------------------------------------------------------
# App principal
# ---------------------------------------------------------
def main():
    st.title("ABC Retain Suite – ChurnRadar Valioso")
    st.markdown(
        """
Este módulo de **ABC Retain Suite** está pensado como un prototipo listo para negocio:

- Usa el histórico de clientes de un negocio tipo *Marketing Campaign / Superstore*.
- Segmenta la base en **4 perfiles de clientes** (K-Means).
- Aplica un modelo entrenado de **Churn Valioso** para detectar clientes de alto valor en riesgo.
- Permite **filtrar y descargar** la lista de clientes priorizados para campañas de retención.
"""
    )

    # Carga del modelo
    modelo_churn, error_modelo = load_churn_model()

    # ---------------------------------------------
    # SIDEBAR: carga de datos
    # ---------------------------------------------
    with st.sidebar:
        st.header("1. Cargar datos")

        usar_demo = st.checkbox("Usar datos demo del proyecto (si existen)", value=True)

        archivo_subido = st.file_uploader(
            "O sube un CSV con la **misma estructura** que el dataset del proyecto",
            type=["csv"],
        )

    df = None
    mensaje_origen = ""

    if usar_demo:
        df_demo, error_demo = load_demo_data()
        if df_demo is not None:
            df = df_demo.copy()
            mensaje_origen = "Datos demo del proyecto"
        else:
            st.warning(error_demo)

    if (df is None) and (archivo_subido is not None):
        try:
            df = pd.read_csv(archivo_subido)
            mensaje_origen = f"Archivo subido: {archivo_subido.name}"
        except Exception as e:
            st.error(f"Error leyendo el archivo subido: {e}")
            return

    if df is None:
        st.info("Selecciona 'Usar datos demo' o sube un CSV para empezar.")
        return

    st.success(f"✅ Datos cargados desde: **{mensaje_origen}**")

    # Intentamos segmentar y scorear solo una vez
    df_seg = df.copy()
    perfil_clusters = None
    desc_clusters = None

    try:
        df_seg, perfil_clusters, desc_clusters = segmentar_clientes(df)
    except ValueError as e:
        st.warning("Segmentación no disponible: " + str(e))

    df_scored = None
    if modelo_churn is not None:
        try:
            df_scored = scorear_churn_valioso(df_seg, modelo_churn)
        except ValueError as e:
            st.warning("Scoring de churn no disponible: " + str(e))

    # Creamos las pestañas principales
    tab_panel, tab_seg, tab_churn, tab_export = st.tabs(
        ["📊 Panel ejecutivo", "🧩 Segmentación", "🔥 Churn Valioso", "📤 Exportar campañas"]
    )

    # -------------------------------------------------
    # 📊 Panel ejecutivo
    # -------------------------------------------------
    with tab_panel:
        st.subheader("Resumen ejecutivo")

        col1, col2, col3, col4 = st.columns(4)

        n_clientes = df.shape[0]
        with col1:
            st.metric("Nº de clientes", n_clientes)

        # Churn real (si existe) o predicho
        churn_rate = None
        churn_source = ""

        if "Churn_Valioso_KMeans" in df.columns:
            churn_rate = df["Churn_Valioso_KMeans"].mean() * 100
            churn_source = "Etiqueta histórica (Churn_Valioso_KMeans)"
        elif df_scored is not None:
            churn_rate = df_scored["Churn_Valioso_Pred"].mean() * 100
            churn_source = "Predicción del modelo (Churn_Valioso_Pred)"

        if churn_rate is not None:
            with col2:
                st.metric("Churn valioso (%)", f"{churn_rate:.2f}")
        else:
            with col2:
                st.metric("Churn valioso (%)", "N/D")

        # Ticket / CLV simple aproximado (si existen columnas)
        if "MntTotal" in df.columns:
            total_mnt = df["MntTotal"].sum()
            with col3:
                st.metric("Gasto total histórico", f"{total_mnt:,.0f}")
        else:
            with col3:
                st.metric("Gasto total histórico", "N/D")

        if "CLV_log" in df.columns:
            clv_medio = df["CLV_log"].mean()
            with col4:
                st.metric("CLV_log promedio", f"{clv_medio:.2f}")
        else:
            with col4:
                st.metric("CLV_log promedio", "N/D")

        st.caption(f"Fuente de churn valioso: {churn_source or 'No disponible en este dataset.'}")

        st.markdown("---")

        # Gráficos de alto nivel: tamaño de clusters y churn por cluster
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Distribución de clientes por cluster**")
            if ("Cluster" in df_seg.columns) and (df_seg["Cluster"].notna().any()):
                cluster_sizes = df_seg["Cluster"].value_counts().sort_index()
                st.bar_chart(cluster_sizes)
            else:
                st.info("No se pudo calcular la segmentación para mostrar los clusters.")

        with col_b:
            st.markdown("**Churn valioso por cluster (%)**")
            if (df_scored is not None) and ("Cluster" in df_scored.columns):
                # Usamos etiqueta real si existe, si no usamos la predicción
                if "Churn_Valioso_KMeans" in df_scored.columns:
                    col_churn = "Churn_Valioso_KMeans"
                else:
                    col_churn = "Churn_Valioso_Pred"

                churn_cluster = (
                    df_scored.groupby("Cluster")[col_churn]
                    .mean()
                    .mul(100)
                    .round(2)
                )
                st.bar_chart(churn_cluster)
            else:
                st.info("No se pudo calcular churn por cluster (falta modelo o cluster).")

        st.markdown(
            """
**Cómo leer este panel:**

- A la izquierda ves qué tan equilibrada está la base entre los distintos segmentos.
- A la derecha ves qué segmentos concentran mayor % de churn valioso (histórico o predicho).
- Esta vista sirve para responder preguntas tipo:
  - *“¿Dónde se me están yendo los clientes valiosos?”*
  - *“Si solo puedo atacar dos segmentos este mes, cuáles son los más críticos?”*
"""
        )

    # -------------------------------------------------
    # 🧩 Segmentación
    # -------------------------------------------------
    with tab_seg:
        st.subheader("Segmentación de clientes (K-Means)")

        if ("Cluster" not in df_seg.columns) or (perfil_clusters is None):
            st.error("No se pudo realizar la segmentación con los datos disponibles.")
        else:
            st.markdown("**Tamaño de cada cluster:**")
            st.dataframe(df_seg["Cluster"].value_counts().rename("n_clientes"))

            st.markdown("**Perfil medio por cluster (variables de comportamiento):**")
            st.dataframe(perfil_clusters)

            st.markdown("**Descripción de negocio de cada segmento:**")
            st.dataframe(desc_clusters)

            st.markdown(
                """
**Cómo interpretar los segmentos:**

- *Gasto alto / bajo* → media de **MntTotal** frente al resto.
- *Frecuencia alta / baja* → basada en **TotalPurchases**.
- *Ingresos altos / más bajos* → comparando **Income** con otros clusters.
- *Canal preferido* → máximo entre `Perc_WebPurchases`, `Perc_CatalogPurchases`, `Perc_StorePurchases`.
- *Mucha / poca actividad web* → según `NumWebVisitsMonth` frente a la mediana.
"""
            )

            st.markdown("---")
            st.subheader("Explorar clientes de un segmento")

            clusters_unicos = sorted(df_seg["Cluster"].unique().tolist())
            cluster_sel = st.selectbox("Selecciona un cluster", clusters_unicos)

            df_cluster = df_seg[df_seg["Cluster"] == cluster_sel].copy()
            st.write(f"Número de clientes en el cluster {cluster_sel}: {df_cluster.shape[0]}")

            cols_ver = [
                c for c in [
                    "Id", "Recency", "MntTotal", "TotalPurchases",
                    "Income", "Perc_WebPurchases", "Perc_CatalogPurchases",
                    "Perc_StorePurchases", "NumWebVisitsMonth", "Kidhome",
                    "Teenhome", "Cluster"
                ] if c in df_cluster.columns
            ]

            st.dataframe(df_cluster[cols_ver].head(50))

    # -------------------------------------------------
    # 🔥 Churn Valioso
    # -------------------------------------------------
    with tab_churn:
        st.subheader("Churn Valioso – Riesgo y priorización")

        if df_scored is None:
            st.error("No se pudo aplicar el modelo de churn valioso (falta modelo o columnas).")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Distribución de churn valioso (predicho):**")
                dist = (
                    df_scored["Churn_Valioso_Pred"]
                    .value_counts(normalize=True)
                    .rename(index={0: "No churn valioso", 1: "Churn valioso"})
                    .mul(100)
                    .round(2)
                    .rename("Proporción (%)")
                )
                st.dataframe(dist)

            with col2:
                st.markdown("**Distribución por nivel de riesgo:**")
                riesgo_counts = (
                    df_scored["Riesgo_Churn"]
                    .value_counts()
                    .rename("n_clientes")
                )
                st.dataframe(riesgo_counts)
                st.bar_chart(riesgo_counts)

            st.markdown(
                """
**Lectura rápida:**

- *Churn valioso* indica qué % de la base está en riesgo relevante para negocio.
- *Riesgo Alto / Medio / Bajo* permite priorizar campañas según la capacidad de inversión del mes.
"""
            )

            # Si existe etiqueta real, comparamos modelo vs realidad
            if "Churn_Valioso_KMeans" in df_scored.columns:
                st.markdown("---")
                st.subheader("Comparación contra etiqueta histórica (si existe)")

                from sklearn.metrics import confusion_matrix, classification_report

                y_true = df_scored["Churn_Valioso_KMeans"]
                y_pred = df_scored["Churn_Valioso_Pred"]

                cm = confusion_matrix(y_true, y_pred)
                st.write("Matriz de confusión (0 = no churn valioso, 1 = churn valioso):")
                st.dataframe(pd.DataFrame(
                    cm,
                    index=["Real 0", "Real 1"],
                    columns=["Pred 0", "Pred 1"]
                ))

                st.text("Reporte de clasificación:")
                st.text(classification_report(y_true, y_pred, digits=4))

                st.markdown(
                    """
Esto sirve para explicar, en un lenguaje sencillo, qué tan bien el modelo
reproduce el patrón histórico de churn valioso que definimos en el CDA.
"""
                )

    # -------------------------------------------------
    # 📤 Exportar campañas
    # -------------------------------------------------
    with tab_export:
        st.subheader("Filtrar y exportar clientes para campañas de retención")

        if df_scored is None:
            st.error("No se pudo aplicar el modelo de churn valioso, no hay datos para exportar.")
        else:
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)

            with col_f1:
                if "Cluster" in df_scored.columns:
                    opciones_cluster = ["Todos"] + sorted(df_scored["Cluster"].unique().tolist())
                    cluster_sel = st.selectbox("Cluster", opciones_cluster, index=0)
                else:
                    cluster_sel = "Todos"

            with col_f2:
                niveles_riesgo = ["Alto", "Medio", "Bajo"]
                riesgo_sel = st.multiselect(
                    "Nivel de riesgo",
                    niveles_riesgo,
                    default=["Alto", "Medio"]
                )

            with col_f3:
                min_prob = st.slider(
                    "Probabilidad mínima de churn valioso",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                )

            with col_f4:
                n_top = st.number_input(
                    "Máx. clientes a mostrar (top-N por probabilidad)",
                    min_value=10,
                    max_value=5000,
                    value=200,
                    step=10,
                )

            mask = df_scored["Churn_Valioso_Prob"] >= min_prob

            if "Riesgo_Churn" in df_scored.columns:
                mask &= df_scored["Riesgo_Churn"].isin(riesgo_sel)

            if "Cluster" in df_scored.columns and cluster_sel != "Todos":
                mask &= df_scored["Cluster"] == cluster_sel

            df_filtrado = df_scored[mask].copy()

            # Ordenamos por probabilidad de churn (descendente)
            df_filtrado = df_filtrado.sort_values(
                by="Churn_Valioso_Prob",
                ascending=False
            ).head(n_top)

            st.write(f"Número de clientes que cumplen los filtros: **{df_filtrado.shape[0]}**")

            cols_ver = [
                c for c in [
                    "Id", "Cluster", "Riesgo_Churn", "Churn_Valioso_Prob",
                    "Recency", "MntTotal", "TotalPurchases", "Income"
                ] if c in df_filtrado.columns
            ]
            st.dataframe(df_filtrado[cols_ver])

            # Descarga de la lista filtrada
            csv_bytes = df_filtrado.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Descargar lista filtrada (CSV)",
                data=csv_bytes,
                file_name="clientes_churn_valioso_priorizados.csv",
                mime="text/csv",
            )

            st.markdown(
                """
Esta salida se puede conectar directamente con:

- Plataformas de email marketing.
- Equipos de call center / ventas.
- Cargas en CRM para crear listas dinámicas de “clientes en riesgo”.

La idea es que el usuario de negocio no tenga que saber nada de modelos;
solo filtra, exporta y activa campañas.
"""
            )


if __name__ == "__main__":
    main()

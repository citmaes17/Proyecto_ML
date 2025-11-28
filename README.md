
# ABC Retain Suite – ChurnRadar Valioso  
Proyecto de Segmentación de Clientes y Churn Valioso (Superstore)

---

## 1. Resumen ejecutivo

Este repositorio contiene un proyecto completo de **Machine Learning aplicado a marketing relacional**, usando un dataset tipo *Superstore / Marketing Campaign*.

El objetivo principal es doble:

1. **Entender el comportamiento de la base de clientes** mediante:
   - EDA,
   - CDA (análisis estadístico),
   - segmentación con K-Means.

2. **Clasificar clientes en riesgo de “churn valioso”**  
   (clientes de alto valor que han dejado de comprar) y exponerlo en una app:

> **ABC Retain Suite – Módulo 1: ChurnRadar Valioso**  
> App + servicio de actualización para ayudar a priorizar campañas de retención.

---

## 2. Objetivos del proyecto

### Objetivo de negocio

- Identificar **qué clientes valiosos** están dejando de comprar.
- Entregar **segmentos** de clientes con significado de negocio.
- Generar una **lista priorizada para campañas de retención**:
  - a quién llamar,
  - a quién enviar email,
  - en qué segmento enfocarse primero.

### Objetivo de Data Science

- Construir un pipeline completo:

  1. **0 – Split maestro**: separar train/test antes de todo.
  2. **1 – EDA**: conocer la base, crear variables.
  3. **2 – CDA**: validar con estadística lo visto en el EDA.
  4. **3 – Segmentación K-Means**: 4 clusters de comportamiento.
  5. **4 – Modelo supervisado de churn valioso**.
  6. **5 – Evaluación final sobre test_master**.

- Integrar el modelo en una app de **Streamlit** que haga scoring y permita exportar campañas.

---

## 3. Stack tecnológico

- **Python 3.x**
- **Pandas**, **NumPy**
- **Matplotlib**, **Seaborn**
- **Scikit-learn**
- **Joblib**
- **Streamlit**

---

## 4. Estructura del repositorio

```text
Proyecto_ML/
├── data/
│   ├── superstore_data.csv           # Dataset original (crudo)
│   ├── superstore_modelado.csv       # Dataset modelado
│   ├── superstore_para_retencion     # Data set para retencion
│   ├── superstore_master.csv         # Base limpia + features (previa al split)
│   ├── train_master.csv              # Split de entrenamiento
│   ├── test_master.csv               # Split de test final (no tocado hasta el final)
│ │
├── notebooks/
│   ├── 0_Split_Master_Superstore.ipynb   # Split purista train/test
│   ├── 1_EDA_Superstore.ipynb            # EDA + creación de variables
│   ├── 2_CDA_Superstore.ipynb            # CDA + definición de churn valioso
│   ├── 3_Segmentacion_Clientes.ipynb     # K-Means + interpretación de clusters
│   ├── 4_Modelo_Churn_Valioso.ipynb      # Modelo supervisado + validación
│   └── 5_Evaluacion_TestMaster.ipynb     # Evaluación final en test_master
│
├── utils/
│   └── data_overview.py              # Clase/funciones para resumen rápido de datos
│
├── models/
│   └── churn_pipeline.pkl            # Pipeline de preprocesado + modelo entrenado
│
├── app/
│   └── ABC_Retain_Suite.py   # App de Streamlit (módulo 1 de la suite)
│
├── reports/
│   └── ABC_Retai_Suite_Tecnico.pdf # Presentación técnica del proyecto
│   └── ABC_Retai_Suite.pdf         # Presentación del negocio
│
└── README.md                         # Este documento
```

*(Los nombres de algunos ficheros pueden variar ligeramente, pero la idea de estructura es esta.)*

---

## 5. Datos y variables clave

### 5.1 Dataset

- Base de clientes con información de:
  - Fecha de alta,
  - Compras por distintos canales,
  - Importe total gastado,
  - Visitas web,
  - Variables demográficas básicas (Income, Kidhome, Teenhome, Education, Marital_Status).

### 5.2 Feature engineering principal

En el EDA se construyen variables clave para comportamiento de cliente:

- **Recency**: días desde la última compra.
- **CustomerTenure**: antigüedad del cliente (días desde alta).
- **MntTotal**: gasto total histórico.
- **TotalPurchases**: número total de compras.
- **Perc_WebPurchases**: % de compras por web.
- **Perc_CatalogPurchases**: % de compras por catálogo.
- **Perc_StorePurchases**: % de compras en tienda física.
- **NumWebVisitsMonth**: visitas web mensuales.
- **CLV_simple** = `MntTotal * TotalPurchases`.
- **CLV_log** = `log1p(MntTotal) * log1p(TotalPurchases)`  
  (más estable, usado para definir valor del cliente y en el CDA,  
  **no se usa como feature en el modelo supervisado**).

### 5.3 Definición de Churn Valioso K-Means

1. Se aplica **K-Means sobre Recency** para encontrar el cluster más inactivo.
2. El cluster más inactivo tiene media ≈ **83 días sin comprar**.
3. Se definen las condiciones:
   - **Inactivo**: `Recency ≥ 83`.
   - **Valioso**: `CLV_log ≥ mediana`.
4. Etiquetas:
   - `Churn_Valioso_KMeans = 1` si (Inactivo & Valioso), si no 0.
   - `Churn_KMeans = 1` si `Recency ≥ 83` (churn simple, solo inactividad).

Resultados globales (en la base original):

- **Churn Valioso K-Means** ≈ **8.1 %**.
- **Churn simple (K-Means)** ≈ **16.5 %**.

Se demuestra en el CDA que **Churn Valioso** es mucho más informativo que el churn simple.

---

## 6. Flujo de trabajo y notebooks

### 6.1 `0_Split_Master_Superstore.ipynb`

**Objetivo:**  
Hacer el split **purista** train/test antes de cualquier análisis, para reservar un **test_master** completamente virgen para la evaluación final del modelo.

**Pasos principales:**

1. Carga `superstore_master.csv` (base ya limpia y con features básicas).
2. Hace un split estratificado sobre la etiqueta de churn valioso (`Churn_Valioso_KMeans`).
3. Guarda:
   - `train_master.csv`
   - `test_master.csv`  
     en la carpeta `data/`.

---

### 6.2 `1_EDA_Superstore.ipynb`

**Objetivo:**  
Exploración inicial de la data de entrenamiento y creación de variables.

**Pasos:**

1. Carga `train_master.csv`.
2. Revisión rápida:
   - tamaños,
   - tipos de variables,
   - nulos,
   - estadísticas descriptivas.
3. Creación de variables de comportamiento (si no venían ya creadas).
4. Visualizaciones:
   - distribuciones de Recency, MntTotal, TotalPurchases, Income, etc.
   - mix de canales,
   - visitas web.
5. Exploración inicial de churn simple y churn valioso (sin hacer aún CDA formal).

---

### 6.3 `2_CDA_Superstore.ipynb`

**Objetivo:**  
Validar estadísticamente lo que se vio en el EDA y **definir formalmente el churn valioso**.

**Pasos:**

1. Confirmación de la definición de `Churn_Valioso_KMeans`:
   - K-Means sobre Recency → umbral de inactividad ≈ 83 días.
   - Combinación con CLV_log ≥ mediana.
2. Cálculo de tasas:
   - Churn valioso ≈ 8.1 %,
   - Churn simple ≈ 16.5 %.
3. **CDA numérica (Mann-Whitney)**:
   - Variables analizadas: Recency, MntTotal, TotalPurchases, Income, Perc_CatalogPurchases, NumWebVisitsMonth, Perc_StorePurchases, etc.
   - Se reportan p-values y **rank-biserial**.
   - Se concluye que:
     - Recency tiene efecto enorme (≈ 0.91),
     - MntTotal, TotalPurchases, Income, Perc_CatalogPurchases son relevantes,
     - NumWebVisitsMonth está inversamente asociada.
4. **CDA categórica (Chi-cuadrado)**:
   - Kidhome, Teenhome con asociación significativa con churn valioso.
   - Education, Marital_Status tienen efecto más suave.
5. Conclusiones:
   - Churn valioso **no es aleatorio**.
   - Hay señal suficiente para construir un modelo supervisado.

---

### 6.4 `3_Segmentacion_Clientes.ipynb`

**Objetivo:**  
Segmentar clientes en **4 clusters** con K-Means y describirlos en lenguaje de negocio. Cruzar segmentación con churn valioso.

**Features para clustering:**

- Recency, CustomerTenure, MntTotal, TotalPurchases, Income,
- Perc_WebPurchases, Perc_CatalogPurchases, Perc_StorePurchases,
- NumWebVisitsMonth,
- Kidhome, Teenhome.

**Pasos:**

1. Imputación simple y **escalado estándar** de variables numéricas.
2. K-Means con `k = 4` (trade-off entre simplicidad e interpretabilidad).
3. Cálculo de medias por cluster → tabla de perfil.
4. Heatmaps y visualizaciones comparando clusters.
5. **Cruce con Churn_Valioso_KMeans**:
   - Se calculan porcentajes de churn valioso dentro de cada cluster.
   - Se identifican segmentos de alto riesgo vs bajo riesgo.
6. Interpretación de negocio:
   - Segmentos con alto valor y alto riesgo,
   - Segmentos de bajo valor casi sin churn valioso,
   - Diferencias por canal y perfil demográfico.

---

### 6.5 `4_Modelo_Churn_Valioso.ipynb`

**Objetivo:**  
Entrenar un modelo supervisado para clasificar **churn valioso** usando solo `train_master.csv` (sin tocar test_master).

**Target:**

- `Churn_Valioso_KMeans`.

**Features usadas:**

- Numéricas:
  - Recency, MntTotal, TotalPurchases, Income,
  - Perc_CatalogPurchases, NumWebVisitsMonth,
  - Kidhome, Teenhome.
- Categóricas:
  - Education, Marital_Status.

> ⚠️ **CLV_log NO se usa como feature**, aunque se usó para definir el churn valioso.  
> Esto evita fuga de información.

**Preprocesamiento y modelo:**

- `ColumnTransformer`:
  - Numéricas → `SimpleImputer(median)` + `StandardScaler`.
  - Categóricas → `SimpleImputer(most_frequent)` + `OneHotEncoder`.
- `Pipeline` con:
  - `preprocess` + `model`.

**Modelos y tuning:**

- LogisticRegression:
  - Penalty: L1 / L2,
  - C: [0.1, 1.0, 10.0],
  - `class_weight='balanced'`.
- RandomForestClassifier:
  - `n_estimators`: [100, 300],
  - `max_depth`: [None, 5, 10],
  - `min_samples_split`: [2, 5],
  - `class_weight='balanced'`.
- `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- `GridSearchCV` con métrica **ROC-AUC**.

**Resultados en validación (hold-out):**

- Mejor modelo: **RandomForestClassifier**  
  (`n_estimators=100`, `max_depth=5`, `min_samples_split=2`, `class_weight='balanced'`).
- Métricas:
  - ROC-AUC ≈ **0.9933**.
  - Accuracy ≈ **0.9955**.
  - Precision (clase 1 – churn valioso) ≈ **0.9722**.
  - Recall (clase 1) ≈ **0.9722**.
- Importancias:
  - Recency domina,
  - luego MntTotal, TotalPurchases, Income, Perc_CatalogPurchases,
  - y en menor medida Kidhome, NumWebVisitsMonth, Teenhome y algunas categorías.

**Salida:**

- Se guarda el pipeline completo (preprocesado + modelo) en:
  - `models/churn_pipeline.pkl`.

---

### 6.6 `5_Evaluacion_TestMaster.ipynb`

**Objetivo:**  
Hacer la **evaluación final** del modelo usando `test_master.csv` (datos nunca vistos).

**Pasos:**

1. Carga `test_master.csv`.
2. Carga el `churn_pipeline.pkl`.
3. Obtiene:
   - Probabilidades de churn valioso.
   - Predicciones finales.
4. Calcula métricas sobre test:
   - ROC-AUC,
   - matriz de confusión,
   - precision, recall, F1 de la clase positiva.
5. Compara resultados de validación vs test para comprobar que no hay sobreajuste grave.
6. Deja listas las columnas necesarias (`Churn_Valioso_Pred`, `Churn_Valioso_Prob`) para que la app las use.

---

## 7. Modelo de churn valioso: resumen

- **Tipo de problema**: clasificación binaria (`Churn_Valioso_KMeans` = 1 / 0).
- **Distribución** (en train):
  - ≈ 8% clase positiva (churn valioso),
  - ≈ 92% clase negativa.
- **Enfoque**:
  - Pipeline con preprocesado separando numéricas y categóricas.
  - Modelos comparados: Logistic Regression vs RandomForest.
  - Selección vía GridSearchCV y ROC-AUC.
- **Mejor modelo**:
  - RandomForest con profundidad moderada.
- **Uso en producción / demo**:
  - El modelo se serializa con Joblib,
  - La app de Streamlit lo usa para hacer scoring.

---

## 8. Segmentación K-Means: resumen

- **Número de clusters**: k = 4.

- **Variables**:
  - Recency, CustomerTenure, MntTotal, TotalPurchases, Income,
  - Perc_WebPurchases, Perc_CatalogPurchases, Perc_StorePurchases,
  - NumWebVisitsMonth, Kidhome, Teenhome.

- **Preprocesamiento**:
  - Imputación + StandardScaler.

- **Interpretación**:
  - Se obtienen perfiles como:
    - Segmentos de **alto gasto y alta frecuencia** (clientes premium).
    - Segmentos de **bajo gasto y baja frecuencia**.
    - Segmentos más digitales vs más de tienda física.
  - Al cruzar con churn valioso:
    - Algunos clusters concentran un % mucho mayor de churn valioso.
    - Otros apenas aportan churn valioso → baja prioridad de inversión.

---

## 9. App: ABC Retain Suite – ChurnRadar Valioso

La app está en:

```text
app/app_ABC_Retain_Suite_ChurnRadar.py
```

### 9.1 Ejecución

Desde la carpeta raíz del proyecto:

```bash
cd Proyecto_ML
streamlit run app/app_ABC_Retain_Suite_ChurnRadar.py
```

Requisitos:

- Tener instalado `streamlit` y las dependencias del proyecto.
- Tener el modelo entrenado en `models/churn_pipeline.pkl`.
- Tener disponible una base con las mismas columnas que `train_master.csv` / `test_master.csv`.

### 9.2 Pestañas (según diseño actual)

La app está pensada como el **Módulo 1 de ABC Retain Suite**.

1. **📊 Panel ejecutivo**
   - Número de clientes.
   - % de churn valioso (histórico o predicho).
   - Gasto total y CLV medio.
   - Gráficos:
     - Distribución por cluster.
     - Churn valioso por cluster (%).

2. **🧩 Segmentación**
   - Tamaño de cada cluster.
   - Tabla con medias por cluster.
   - Descripción de cada segmento en lenguaje de negocio.
   - Filtro para explorar clientes de un cluster concreto.

3. **🔥 Churn Valioso**
   - Distribución de churn valioso (0/1).
   - Distribución por nivel de riesgo (Alto / Medio / Bajo).
   - Si la base tiene etiqueta histórica, se muestra:
     - matriz de confusión,
     - classification_report.

4. **📤 Exportar campañas**
   - Filtros:
     - cluster,
     - nivel de riesgo,
     - probabilidad mínima de churn,
     - top-N clientes.
   - Tabla con clientes priorizados.
   - Botón para descargar CSV listo para activación en campañas.

---

## 10. Cómo reproducir el proyecto

### 10.1 Requisitos

1. Clonar este repositorio.
2. Crear un entorno virtual (opcional pero recomendado):

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scriptsctivate      # Windows
```

3. Instalar dependencias:

```bash
pip install -r requirements.txt
```

*(Si no hay `requirements.txt`, instalar: pandas, numpy, scikit-learn, matplotlib, seaborn, joblib, streamlit.)*

### 10.2 Orden recomendado de ejecución

1. Ejecutar `0_Split_Master_Superstore.ipynb`  
   → genera `train_master.csv` y `test_master.csv`.

2. Ejecutar `1_EDA_Superstore.ipynb`  
   → análisis exploratorio sobre `train_master`.

3. Ejecutar `2_CDA_Superstore.ipynb`  
   → definición y validación de `Churn_Valioso_KMeans`.

4. Ejecutar `3_Segmentacion_Clientes.ipynb`  
   → K-Means y perfiles de segmentos.

5. Ejecutar `4_Modelo_Churn_Valioso.ipynb`  
   → entrenamiento del modelo y guardado de `churn_pipeline.pkl`.

6. Ejecutar `5_Evaluacion_TestMaster.ipynb`  
   → evaluación final en `test_master`.

7. Lanzar la app de Streamlit:

```bash
streamlit run app/app_ABC_Retain_Suite_ChurnRadar.py
```

---

## 11. Limitaciones y trabajo futuro

- La definición de **churn valioso** se basa en:
  - Recency ≥ 83 días + CLV_log ≥ mediana.  
  El modelo reproduce muy bien esa definición, pero:
  - está fuertemente anclado a Recency,
  - funciona como excelente **clasificador del estado actual**,
  - no es un modelo de *early warning* puro (a varios meses vista).
- El dataset corresponde a un solo contexto de negocio:
  - la generalización a otros sectores requiere reentrenar el modelo con sus datos.
- Trabajo futuro:
  - redefinir churn con ventanas temporales (30–60–90 días),
  - incorporar más señales temporales y digitales,
  - construir un módulo específico de **recomendación (cross-sell / up-sell)**,
  - añadir interpretabilidad avanzada (SHAP, etc.) si el contexto lo requiere.

---

## 12. Contacto / créditos

Este proyecto forma parte de un **proceso formativo y de portfolio en Data Science y Machine Learning aplicado a marketing**, y se integra como el primer módulo de:

> **ABC Retain Suite** – herramientas para cuidar el valor de tus clientes.

```markdown
Autor: [Tu nombre aquí]  
Rol: Data Scientist / Analista de Marketing Data-Driven
```

Puedes adaptar este README para GitHub, para portfolio profesional, para entregas académicas o para presentar el proyecto a equipos de negocio y Data Science.

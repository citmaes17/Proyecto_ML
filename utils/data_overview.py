import pandas as pd

class DataOverview:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def resumen(self):
        print("📊 Dimensiones:", self.df.shape)
        print("\n🧩 Tipos de datos:")
        print(self.df.dtypes.value_counts())
        print("\n🚫 Nulos por columna (top 10):")
        print(self.df.isnull().sum().sort_values(ascending=False).head(10))
        print(f"\n📎 Duplicados: {self.df.duplicated().sum()} filas duplicadas")

    def categorias_unicas(self, n: int = 5):
        cat_cols = self.df.select_dtypes(include='object').columns
        print("🔠 Variables categóricas (primeros valores únicos):")
        for col in cat_cols:
            vals = self.df[col].astype(str).unique()[:n]
            print(f"- {col}: {vals}")

    def resumen_numericas(self):
        from IPython.display import display
        display(self.df.describe().T.round(2))

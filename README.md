# 1a-EXPERIENC-IA-DATASCIENCE

Proyecto de análisis, modelado predictivo y aplicación Streamlit para el
**forecasting de ventas diarias** de productos deportivos. A partir del
histórico 2021–2024 se entrena un modelo que predice las unidades vendidas, y
una app interactiva permite simular distintos escenarios de descuento y de
precios de la competencia para noviembre de 2025.

---

## Estructura del proyecto

```
1a-EXPERIENC-IA-DATASCIENCE/
├── app/
│   └── app.py                  # Aplicación Streamlit (simulador de ventas)
├── data/
│   ├── raw/                    # Datos originales (no modificar)
│   │   ├── entrenamiento/
│   │   │   ├── ventas.csv          # Ventas históricas 2021-2024
│   │   │   └── competencia.csv     # Precios de competencia 2021-2024
│   │   └── inferencia/
│   │       └── ventas_2025_inferencia.csv   # Datos a predecir (nov 2025)
│   └── processed/              # Datos derivados (se regeneran con los notebooks)
│       ├── df_processed.csv                 # Dataset de entrenamiento con features
│       └── inferencia_df_transformado.csv   # Datos de inferencia con features
├── models/
│   └── modelo_final.joblib     # Modelo entrenado (HistGradientBoostingRegressor)
├── notebooks/
│   ├── entrenamiento.ipynb     # EDA + feature engineering + entrenamiento
│   └── forecasting.ipynb       # Transforma los datos de inferencia 2025
├── reports/
│   ├── bf2024_real_vs_pred.csv # Real vs predicho en Black Friday 2024
│   └── figs/                   # Gráficos generados
├── requirements.txt
└── README.md
```

---

## Requisitos

- **Python 3.9 o superior**
- Las dependencias están en `requirements.txt`. La versión de `scikit-learn`
  está fijada (`==1.6.1`) porque debe coincidir con la usada para entrenar
  `modelo_final.joblib`; si la cambias, el modelo podría no cargar.

---

## Instalación

```bash
# 1. Clonar el repositorio
git clone https://github.com/JorgeAlcaide7/1a-EXPERIENC-IA-DATASCIENCE.git
cd 1a-EXPERIENC-IA-DATASCIENCE

# 2. Crear y activar un entorno virtual
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## Uso

### Lanzar la aplicación

```bash
streamlit run app/app.py
```

> Ejecútalo **desde la raíz del proyecto** (las rutas al modelo y a los datos
> son relativas a esa carpeta). Se abrirá una pestaña en el navegador.

En la app puedes:
- Elegir un producto.
- Ajustar el **descuento** aplicado sobre el precio base.
- Simular tres **escenarios de competencia** (precios actuales, −5 % o +5 %).
- Ver la predicción diaria de unidades e ingresos para noviembre 2025, con el
  Black Friday (día 28) destacado.

### Regenerar datos y modelo (opcional)

Solo es necesario si cambias los datos o el feature engineering. Ejecuta los
notebooks **en orden** desde la carpeta `notebooks/`:

1. `entrenamiento.ipynb` → genera `data/processed/df_processed.csv` y
   `models/modelo_final.joblib`.
2. `forecasting.ipynb` → genera `data/processed/inferencia_df_transformado.csv`
   (depende del paso anterior).

---

## El modelo

- **Algoritmo:** `HistGradientBoostingRegressor` (scikit-learn).
- **Objetivo (target):** `unidades_vendidas` por producto y día.
- **Features principales:** variables de calendario (día, mes, festivos, Black
  Friday, etc.), precio propio y de la competencia, descuento, ratio de precio,
  *lags* de ventas (1 a 7 días) y media móvil de 7 días, todo calculado **por
  producto**.

### Validación y métricas

La evaluación se hace con un **holdout temporal**: se entrena con 2021–2023 y se
valida con **2024** (datos que el modelo no ha visto).

| Métrica (validación 2024) | Valor |
|---|---|
| MAE  | ~0.88 |
| R²   | ~0.93 |
| MAPE | ~18 % |

El modelo que se despliega (`modelo_final.joblib`) se reentrena con **todos**
los datos 2021–2024 para aprovechar el máximo de información antes de producción.
El error medido sobre esos mismos datos (*in-sample*) **no** debe usarse como
indicador de rendimiento: la referencia es siempre el holdout 2024.

---

## Flujo de datos (pipeline)

```
ventas.csv + competencia.csv
        │  (entrenamiento.ipynb: merge + feature engineering)
        ▼
df_processed.csv  ──►  modelo_final.joblib
        │
        │  (forecasting.ipynb: mismo pipeline sobre datos 2025)
        ▼
inferencia_df_transformado.csv  ──►  app.py (simulación recursiva)
```

La app realiza un **forecasting recursivo**: predice día a día y realimenta cada
predicción como *lag* del día siguiente, de modo que un cambio de precio se
propaga a lo largo de todo el horizonte simulado.

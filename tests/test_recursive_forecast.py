"""Test del forecasting recursivo de la app (protege contra el Bug 2).

El Bug 2 hacía que predecir_recursivo() buscara columnas con un nombre
inexistente ('unidades_vendidas_lag_1' en vez de '..._lag1'), por lo que la
realimentación se saltaba en silencio y los lags quedaban congelados.

Usamos un modelo falso cuya predicción depende SOLO de lag1 (devuelve lag1 + 1).
Si la realimentación funciona, cada predicción se convierte en el lag1 del día
siguiente y la serie crece de uno en uno. Si estuviera rota, lag1 se quedaría
congelado y la serie no crecería.
"""
import numpy as np
import pandas as pd


class _FakeModel:
    def __init__(self, feats):
        self.feature_names_in_ = np.array(feats)

    def predict(self, X):
        return np.array([X["unidades_vendidas_lag1"].iloc[0] + 1.0])


def test_la_prediccion_se_realimenta_como_lag(app_function):
    predecir_recursivo = app_function("predecir_recursivo")

    n = 4
    feats = [f"unidades_vendidas_lag{i}" for i in range(1, 8)] + ["unidades_vendidas_ma7"]
    df = pd.DataFrame({
        "fecha": pd.date_range("2025-11-01", periods=n),
        **{c: [0.0] * n for c in feats},
    })
    # Solo el primer día tiene una semilla en lag1; los demás deben rellenarse
    # con las predicciones si la realimentación funciona.
    df["unidades_vendidas_lag1"] = [10.0, 0.0, 0.0, 0.0]

    preds = predecir_recursivo(_FakeModel(feats), df, feats)

    # día0 = 10+1 = 11, y luego 12, 13, 14 si lag1 se realimenta correctamente
    assert preds == [11.0, 12.0, 13.0, 14.0]


def test_no_devuelve_predicciones_negativas(app_function):
    predecir_recursivo = app_function("predecir_recursivo")

    class _NegModel:
        feature_names_in_ = np.array(["unidades_vendidas_lag1"])

        def predict(self, X):
            return np.array([-5.0])

    df = pd.DataFrame({
        "fecha": pd.date_range("2025-11-01", periods=3),
        "unidades_vendidas_lag1": [1.0, 1.0, 1.0],
    })
    preds = predecir_recursivo(_NegModel(), df, ["unidades_vendidas_lag1"])
    assert all(p >= 0 for p in preds)

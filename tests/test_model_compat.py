"""Tests de compatibilidad entre el modelo y los datos de inferencia.

Protegen contra desajustes entre las features que el modelo espera y las
columnas disponibles en el CSV de inferencia que consume la app.
"""
import joblib
import pandas as pd


def test_modelo_carga(root):
    modelo = joblib.load(root / "models" / "modelo_final.joblib")
    assert hasattr(modelo, "feature_names_in_")
    assert len(modelo.feature_names_in_) > 0


def test_features_presentes_en_inferencia(root):
    modelo = joblib.load(root / "models" / "modelo_final.joblib")
    df = pd.read_csv(root / "data" / "processed" / "inferencia_df_transformado.csv")
    faltan = [f for f in modelo.feature_names_in_ if f not in df.columns]
    assert not faltan, f"Faltan estas features en el CSV de inferencia: {faltan}"


def test_el_modelo_predice_sobre_inferencia(root):
    modelo = joblib.load(root / "models" / "modelo_final.joblib")
    df = pd.read_csv(root / "data" / "processed" / "inferencia_df_transformado.csv")
    feats = list(modelo.feature_names_in_)
    preds = modelo.predict(df[feats].head(10))
    assert len(preds) == 10

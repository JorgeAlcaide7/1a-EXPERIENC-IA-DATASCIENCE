"""Tests del cálculo de lags y media móvil (protegen contra el Bug 1).

El Bug 1 consistía en calcular los lags agrupando solo por año, lo que mezclaba
ventas de productos distintos, y en una media móvil que incluía el día actual
(fuga del target). Estos tests comparan el dataset procesado contra un cálculo
de referencia independiente hecho desde los datos crudos.
"""
import pandas as pd


def _cargar(root):
    raw = pd.read_csv(
        root / "data" / "raw" / "entrenamiento" / "ventas.csv",
        parse_dates=["fecha"],
    ).sort_values(["producto_id", "fecha"])
    proc = pd.read_csv(
        root / "data" / "processed" / "df_processed.csv",
        parse_dates=["fecha"],
    )
    return raw, proc


def test_lag1_es_del_mismo_producto(root):
    """lag1 debe ser la venta del día anterior del MISMO producto."""
    raw, proc = _cargar(root)
    raw["lag1_ref"] = raw.groupby("producto_id")["unidades_vendidas"].shift(1)

    m = proc.merge(
        raw[["fecha", "producto_id", "lag1_ref"]],
        on=["fecha", "producto_id"], how="left",
    )
    comparable = m["lag1_ref"].notna()
    assert comparable.any(), "No hay filas comparables; revisa los datos"
    assert (m.loc[comparable, "unidades_vendidas_lag1"]
            == m.loc[comparable, "lag1_ref"]).all()


def test_todos_los_lags_correctos(root):
    """Comprueba lag1..lag7 contra la referencia por producto."""
    raw, proc = _cargar(root)
    for lag in range(1, 8):
        raw[f"ref_lag{lag}"] = raw.groupby("producto_id")["unidades_vendidas"].shift(lag)

    cols = [f"ref_lag{lag}" for lag in range(1, 8)]
    m = proc.merge(
        raw[["fecha", "producto_id"] + cols],
        on=["fecha", "producto_id"], how="left",
    )
    for lag in range(1, 8):
        comparable = m[f"ref_lag{lag}"].notna()
        assert (m.loc[comparable, f"unidades_vendidas_lag{lag}"]
                == m.loc[comparable, f"ref_lag{lag}"]).all(), f"lag{lag} incorrecto"


def test_ma7_no_incluye_el_dia_actual(root):
    """La media móvil de 7 días NO debe incluir el día actual (evita fuga)."""
    raw, proc = _cargar(root)
    # Referencia correcta: media de los <=7 días anteriores (shift(1) excluye hoy)
    raw["ma7_ref"] = raw.groupby("producto_id")["unidades_vendidas"].transform(
        lambda x: x.rolling(7, min_periods=1).mean().shift(1)
    )
    m = proc.merge(
        raw[["fecha", "producto_id", "ma7_ref"]],
        on=["fecha", "producto_id"], how="left",
    )
    comparable = m["ma7_ref"].notna()
    assert comparable.any()
    assert ((m.loc[comparable, "unidades_vendidas_ma7"]
             - m.loc[comparable, "ma7_ref"]).abs() < 1e-6).all()

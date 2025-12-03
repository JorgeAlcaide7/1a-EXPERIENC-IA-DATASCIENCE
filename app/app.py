import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="Simulador de Ventas Nov 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        color: white;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelo y datos
@st.cache_resource
def load_model():
    try:
        model_path = Path("models/modelo_final.joblib")
        if not model_path.exists():
            st.error(f"❌ El archivo del modelo no existe en: {model_path.absolute()}")
            st.info("ℹ️ Ejecuta el notebook 'entrenamiento.ipynb' hasta el final para generar el modelo")
            st.stop()
        
        model = joblib.load(model_path)
        st.success(f"✅ Modelo cargado correctamente ({len(model.feature_names_in_)} features)")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        st.info(f"📁 Ruta absoluta esperada: {Path('models/modelo_final.joblib').absolute()}")
        st.stop()

@st.cache_data
def load_data():
    try:
        data_path = Path("data/processed/inferencia_df_transformado.csv")
        if not data_path.exists():
            st.error(f"❌ El archivo de datos no existe en: {data_path.absolute()}")
            st.info("ℹ️ Asegúrate de que el archivo 'inferencia_df_transformado.csv' existe en data/processed/")
            st.stop()
        
        df = pd.read_csv(data_path)
        df['fecha'] = pd.to_datetime(df['fecha'])
        st.success(f"✅ Datos cargados correctamente ({len(df)} registros)")
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        st.info(f"📁 Ruta absoluta esperada: {Path('data/processed/inferencia_df_transformado.csv').absolute()}")
        st.stop()

# Función para actualizar lags recursivamente
def actualizar_lags(df_producto, predicciones):
    """Actualiza los lags día por día con las predicciones"""
    df_actualizado = df_producto.copy()
    
    for i in range(len(df_actualizado)):
        if i == 0:
            # Día 1: usar lags originales del archivo
            continue
        else:
            # Días 2-30: actualizar lags con predicciones previas
            # Desplazar lags hacia la derecha
            for lag in range(7, 1, -1):
                col_actual = f'unidades_vendidas_lag_{lag}'
                col_anterior = f'unidades_vendidas_lag_{lag-1}'
                if col_actual in df_actualizado.columns and col_anterior in df_actualizado.columns:
                    df_actualizado.loc[df_actualizado.index[i], col_actual] = \
                        df_actualizado.loc[df_actualizado.index[i-1], col_anterior]
            
            # lag_1 = predicción del día anterior
            if 'unidades_vendidas_lag_1' in df_actualizado.columns:
                df_actualizado.loc[df_actualizado.index[i], 'unidades_vendidas_lag_1'] = predicciones[i-1]
            
            # Actualizar media móvil con las últimas 7 predicciones
            if 'unidades_vendidas_ma7' in df_actualizado.columns:
                inicio = max(0, i - 7)
                valores_ma = predicciones[inicio:i]
                if len(valores_ma) > 0:
                    df_actualizado.loc[df_actualizado.index[i], 'unidades_vendidas_ma7'] = np.mean(valores_ma)
    
    return df_actualizado

# Función de predicción recursiva
def predecir_recursivo(modelo, df_producto, feature_names):
    """Realiza predicciones día por día actualizando lags"""
    predicciones = []
    df_trabajo = df_producto.copy().sort_values('fecha').reset_index(drop=True)
    
    for i in range(len(df_trabajo)):
        # Preparar features para este día
        X_dia = df_trabajo.loc[[i], feature_names]
        
        # Predecir
        pred = modelo.predict(X_dia)[0]
        pred = max(0, pred)  # No permitir predicciones negativas
        predicciones.append(pred)
        
        # Actualizar lags para el siguiente día (si no es el último)
        if i < len(df_trabajo) - 1:
            # Desplazar lags
            for lag in range(7, 1, -1):
                col_actual = f'unidades_vendidas_lag_{lag}'
                col_anterior = f'unidades_vendidas_lag_{lag-1}'
                if col_actual in df_trabajo.columns and col_anterior in df_trabajo.columns:
                    df_trabajo.loc[i+1, col_actual] = df_trabajo.loc[i, col_anterior]
            
            # lag_1 = predicción actual
            if 'unidades_vendidas_lag_1' in df_trabajo.columns:
                df_trabajo.loc[i+1, 'unidades_vendidas_lag_1'] = pred
            
            # Actualizar MA7
            if 'unidades_vendidas_ma7' in df_trabajo.columns:
                inicio = max(0, i - 6)
                valores_ma = predicciones[inicio:i+1]
                df_trabajo.loc[i+1, 'unidades_vendidas_ma7'] = np.mean(valores_ma)
    
    return predicciones

# Función para simular ventas
def simular_ventas(df, modelo, producto, descuento_pct, escenario_comp):
    """Simula ventas con los parámetros seleccionados"""
    
    # Filtrar producto
    df_producto = df[df['nombre'] == producto].copy().sort_values('fecha').reset_index(drop=True)
    
    if len(df_producto) == 0:
        st.error(f"No se encontraron datos para el producto: {producto}")
        return None
    
    # Ajustar precio_venta según descuento
    descuento_multiplicador = 1 + (descuento_pct / 100)
    df_producto['precio_venta'] = df_producto['precio_base'] * descuento_multiplicador
    
    # Ajustar precios de competencia
    comp_ajuste = {'Actual (0%)': 1.0, 'Competencia -5%': 0.95, 'Competencia +5%': 1.05}
    mult_comp = comp_ajuste[escenario_comp]
    
    for col in ['Amazon', 'Decathlon', 'Deporvillage']:
        if col in df_producto.columns:
            df_producto[col] = df_producto[col] * mult_comp
    
    # Recalcular precio_competencia y ratio_precio
    if all(col in df_producto.columns for col in ['Amazon', 'Decathlon', 'Deporvillage']):
        df_producto['precio_competencia'] = df_producto[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    
    if 'precio_competencia' in df_producto.columns:
        df_producto['ratio_precio'] = df_producto['precio_venta'] / df_producto['precio_competencia'].replace(0, 1)
    
    if 'precio_base' in df_producto.columns:
        df_producto['descuento_porcentaje'] = ((df_producto['precio_venta'] - df_producto['precio_base']) / 
                                                df_producto['precio_base']) * 100
    
    # Obtener features del modelo
    feature_names = modelo.feature_names_in_
    
    # Verificar que todas las features existen
    missing_features = [f for f in feature_names if f not in df_producto.columns]
    if missing_features:
        st.error(f"⚠️ Faltan columnas requeridas: {missing_features[:5]}")
        return None
    
    # Predecir recursivamente
    predicciones = predecir_recursivo(modelo, df_producto, feature_names)
    
    # Agregar predicciones y calcular ingresos
    df_producto['unidades_predichas'] = predicciones
    df_producto['ingresos_proyectados'] = df_producto['unidades_predichas'] * df_producto['precio_venta']
    
    return df_producto

# Cargar modelo y datos
modelo = load_model()
df_base = load_data()

# SIDEBAR
with st.sidebar:
    st.markdown("## 🎯 Controles de Simulación")
    st.markdown("---")
    
    # Selector de producto
    productos = sorted(df_base['nombre'].unique())
    producto_seleccionado = st.selectbox(
        "📦 Seleccionar Producto",
        productos,
        index=0
    )
    
    st.markdown("---")
    
    # Slider de descuento
    descuento = st.slider(
        "💰 Ajuste de Descuento (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Ajusta el descuento aplicado sobre el precio base"
    )
    
    st.markdown("---")
    
    # Escenario de competencia
    st.markdown("🏪 **Escenario de Competencia**")
    escenario = st.radio(
        "",
        ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        index=0,
        help="Simula cambios en los precios de la competencia"
    )
    
    st.markdown("---")
    
    # Botón de simulación
    simular = st.button("🚀 Simular Ventas", use_container_width=True, type="primary")

# ZONA PRINCIPAL
st.markdown(f"# 📊 Dashboard de Predicción - Noviembre 2025")
st.markdown(f"### Producto: **{producto_seleccionado}**")
st.markdown("---")

if simular:
    with st.spinner("🔄 Ejecutando simulación recursiva..."):
        resultados = simular_ventas(df_base, modelo, producto_seleccionado, descuento, escenario)
    
    if resultados is not None:
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            unidades_totales = resultados['unidades_predichas'].sum()
            st.metric("📦 Unidades Totales", f"{int(unidades_totales):,}")
        
        with col2:
            ingresos_totales = resultados['ingresos_proyectados'].sum()
            st.metric("💶 Ingresos Proyectados", f"€{ingresos_totales:,.2f}")
        
        with col3:
            precio_promedio = resultados['precio_venta'].mean()
            st.metric("💵 Precio Promedio", f"€{precio_promedio:.2f}")
        
        with col4:
            descuento_promedio = resultados['descuento_porcentaje'].mean()
            st.metric("🏷️ Descuento Promedio", f"{descuento_promedio:.1f}%")
        
        st.markdown("---")
        
        # Gráfico de predicción diaria
        st.markdown("## 📈 Predicción Diaria de Ventas")
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.set_style("whitegrid")
        
        dias = resultados['fecha'].dt.day.values
        ventas = resultados['unidades_predichas'].values
        
        # Línea de predicción
        ax.plot(dias, ventas, marker='o', linewidth=2.5, markersize=6, 
                color='#667eea', label='Ventas Predichas')
        
        # Marcar Black Friday (día 28)
        bf_idx = np.where(dias == 28)[0]
        if len(bf_idx) > 0:
            bf_idx = bf_idx[0]
            ax.axvline(x=28, color='red', linestyle='--', linewidth=2, alpha=0.7)
            ax.plot(28, ventas[bf_idx], 'ro', markersize=12, label='Black Friday')
            ax.annotate('🛍️ Black Friday', xy=(28, ventas[bf_idx]), 
                       xytext=(28, ventas[bf_idx] + max(ventas)*0.1),
                       fontsize=12, fontweight='bold', color='red',
                       ha='center', arrowprops=dict(arrowstyle='->', color='red', lw=2))
        
        ax.set_xlabel('Día de Noviembre', fontsize=12, fontweight='bold')
        ax.set_ylabel('Unidades Vendidas', fontsize=12, fontweight='bold')
        ax.set_title('Predicción de Ventas - Noviembre 2025', fontsize=14, fontweight='bold')
        ax.set_xticks(range(1, 31, 2))
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("---")
        
        # Tabla detallada
        st.markdown("## 📋 Detalle Diario")
        
        tabla = resultados[['fecha', 'precio_venta', 'precio_competencia', 'descuento_porcentaje', 
                           'unidades_predichas', 'ingresos_proyectados']].copy()
        
        tabla['dia_semana'] = tabla['fecha'].dt.day_name()
        tabla['fecha_str'] = tabla['fecha'].dt.strftime('%d/%m/%Y')
        tabla['es_bf'] = tabla['fecha'].dt.day == 28
        
        # Formatear columnas
        tabla_display = tabla[['fecha_str', 'dia_semana', 'precio_venta', 'precio_competencia', 
                               'descuento_porcentaje', 'unidades_predichas', 'ingresos_proyectados']].copy()
        
        tabla_display.columns = ['Fecha', 'Día Semana', 'Precio Venta (€)', 'Precio Competencia (€)', 
                                 'Descuento (%)', 'Unidades', 'Ingresos (€)']
        
        tabla_display['Precio Venta (€)'] = tabla_display['Precio Venta (€)'].apply(lambda x: f"€{x:.2f}")
        tabla_display['Precio Competencia (€)'] = tabla_display['Precio Competencia (€)'].apply(lambda x: f"€{x:.2f}")
        tabla_display['Descuento (%)'] = tabla_display['Descuento (%)'].apply(lambda x: f"{x:.1f}%")
        tabla_display['Unidades'] = tabla_display['Unidades'].apply(lambda x: f"{int(x):,}")
        tabla_display['Ingresos (€)'] = tabla_display['Ingresos (€)'].apply(lambda x: f"€{x:,.2f}")
        
        # Destacar Black Friday
        def highlight_bf(row):
            if '28/11/2025' in row['Fecha']:
                return ['background-color: #ffcccc; font-weight: bold'] * len(row)
            return [''] * len(row)
        
        st.dataframe(tabla_display.style.apply(highlight_bf, axis=1), use_container_width=True, height=600)
        
        st.markdown("---")
        
        # Comparativa de escenarios
        st.markdown("## 🔄 Comparativa de Escenarios de Competencia")
        
        escenarios = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        resultados_escenarios = {}
        
        with st.spinner("Calculando escenarios..."):
            for esc in escenarios:
                res_esc = simular_ventas(df_base, modelo, producto_seleccionado, descuento, esc)
                if res_esc is not None:
                    resultados_escenarios[esc] = {
                        'unidades': res_esc['unidades_predichas'].sum(),
                        'ingresos': res_esc['ingresos_proyectados'].sum()
                    }
        
        col1, col2, col3 = st.columns(3)
        
        for i, (esc, col) in enumerate(zip(escenarios, [col1, col2, col3])):
            with col:
                if esc in resultados_escenarios:
                    st.markdown(f"### {esc}")
                    st.metric("Unidades", f"{int(resultados_escenarios[esc]['unidades']):,}")
                    st.metric("Ingresos", f"€{resultados_escenarios[esc]['ingresos']:,.2f}")
        
        st.success("✅ Simulación completada exitosamente")
        
else:
    st.info("👆 Configura los parámetros en el panel lateral y pulsa '🚀 Simular Ventas' para comenzar")
    
    # Mostrar información del dataset
    st.markdown("### 📊 Información del Dataset")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Productos", len(df_base['nombre'].unique()))
    with col2:
        st.metric("Días en Noviembre", len(df_base['fecha'].unique()))
    with col3:
        st.metric("Features del Modelo", len(modelo.feature_names_in_))
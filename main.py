
import joblib
import streamlit as st
import pandas as pd

# --- Configuración de la Página ---
# Esto debe ser lo primero que se ejecute en el script.
st.set_page_config(
    page_title="Predictor de porcentaje de sílica",
    page_icon="🧪",
    layout="wide"
)

# --- Carga del Modelo ---
# Usamos @st.cache_resource para que el modelo se cargue solo una vez y se mantenga en memoria,
# lo que hace que la aplicación sea mucho más rápida.
@st.cache_resource
def load_model(model_path):
    """Carga el modelo entrenado desde un archivo .joblib."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(f"Error: No se encontró el archivo del modelo en {model_path}. Asegúrate de que el archivo del modelo esté en el directorio correcto.")
        return None

# Cargamos nuestro modelo campeón. Streamlit buscará en la ruta 'model.joblib'.
# Asegúrate de que el archivo del modelo (final_model.joblib) esté subido y renombrado a model.joblib
model = load_model('model.joblib')

# --- Barra Lateral para las Entradas del Usuario ---
with st.sidebar:
    st.header("⚙️ Parámetros de Entrada")
    st.markdown("""
    Ajusta los deslizadores para que coincidan con los parámetros operativos de lixiviación.
    """)

    # Slider para el % iron concentrate
    # Usamos un nombre de variable interno más limpio
    iron_concentrate_input = st.slider(
        label='% de concentración de hierro',
        min_value=60.00, # Aseguramos que sea float
        max_value=70.00, # Aseguramos que sea float
        value=66.00, # Valor inicial (aseguramos que sea float)
        step=0.05 # Cambiado el paso a 0.1
    )
    st.caption("Representa el porcentaje de concentración de hierro.")

    # Slider para el flujo de amina
    # Usamos un nombre de variable interno más limpio
    amina_flow_input = st.slider(
        label='Flujo de amina',
        min_value=240,
        max_value=740,
        value=540,
        step=1
    )
    st.caption("Flujo de amina")

    # Slider para la Flotation Column Air Flow
    # Usamos un nombre de variable interno más limpio
    flotation_column_airflow_input = st.slider(
        label='Flujo de aire en la columa de flotación',
        min_value=-175,
        max_value=305,
        value=250,
        step=1
    )
    st.caption("Flujo de aire en la columna de flotación")

# --- Contenido de la Página Principal ---
st.title("🧪 Predictor de procentage de sílica")
st.markdown("""
¡Bienvenido! Esta aplicación utiliza un modelo de machine learning para predecir el porcentaje de concentración de sílica en el proceso de lixiviación basándose en parámetros operativos clave.

**Esta herramienta puede ayudar a los ingenieros de procesos y operadores a:**
- **Optimizar** las condiciones de operación para obtener el porcentage de sílica final.
- **Predecir** el impacto de los cambios en el proceso antes de implementarlos.
- **Solucionar** problemas potenciales simulando diferentes escenarios.
""")

# --- Lógica de Predicción ---
# Solo intentamos predecir si el modelo se ha cargado correctamente.
if model is not None:
    # El botón principal que el usuario presionará para obtener un resultado.
    if st.button('🚀 Predecir el porcentaje de silica', type="primary"):
        # Creamos un DataFrame de pandas con las entradas del usuario.
        # ¡Es crucial que los nombres de las columnas coincidan exactamente con los que el modelo espera!
        # Asegúrate de que estas claves coincidan con los nombres de las características usadas para entrenar el modelo.
        # Corregimos el orden de las columnas para que coincida con el error message: ['Amina Flow', 'Flotation Column 03 Air Flow', '% Iron Concentrate']
        df_input = pd.DataFrame({
            'Amina Flow': [amina_flow_input],
            'Flotation Column 03 Air Flow': [flotation_column_airflow_input],
            '% Iron Concentrate': [iron_concentrate_input]
        })

        # Hacemos la predicción
        try:
            prediction_value = model.predict(df_input)
            st.subheader("📈 Resultado de la Predicción")
            # Mostramos el resultado en un cuadro de éxito, formateado a dos decimales.
            st.success(f"**Porcentaje Predicho:** `{prediction_value[0]:.2f}%`")
            st.info("Este valor representa el porcentaje de sílica presente en la operación.")
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
else:
    st.warning("El modelo no pudo ser cargado. Por favor, verifica la ruta del archivo del modelo.")

st.divider()

# --- Sección de Explicación ---
with st.expander("ℹ️ Sobre la Aplicación"):
    st.markdown("""
    **¿Cómo funciona?**

    1.  **Datos de Entrada:** Proporcionas los parámetros operativos clave usando los deslizadores en la barra lateral.
    2.  **Predicción:** El modelo de machine learning pre-entrenado recibe estas entradas y las analiza basándose en los patrones que aprendió de datos históricos.
    3.  **Resultado:** La aplicación muestra el porcentaje final predicho.

    **Detalles del Modelo:**

    * **Tipo de Modelo:** `Regression Model` (XGBoost Optimizado)
    * **Propósito:** Predecir el valor continuo del rendimiento de la destilación.
    * **Características Usadas:** Porcentaje de concentración de acero, Flujo de amina y Flujo de aire en la columna de flotación.
    """)

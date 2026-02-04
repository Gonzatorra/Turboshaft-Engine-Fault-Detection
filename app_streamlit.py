import streamlit as st
import pandas as pd
from paginas import tab_contexto, tab_eda, tab_modelos

#cONFIGURACIÓN BÁSICA
st.set_page_config(
    page_title="Detección de Fallos en Motores Turboshaft",
    layout="wide"
)
st.title("🚁 Detección de Fallos en Motores Turboshaft")
st.markdown("---")





#CARGA DE DATOS
file_name = "data/Helicopter_Turboshaft_Fault_Detection.csv"

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(file_name)
        
        # Convertimos explícitamente a datetime nada más cargar.
        # 'coerce' convertirá errores en NaT (Not a Time) para no romper el código.
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
        return df
    except FileNotFoundError:
        st.error(f"No se ha encontrado el archivo {file_name}. Verifica la ruta.")
        return None
    except Exception as e:
        st.error(f"Error cargando el archivo: {e}")
        return None

# Solo cargar datos si no están en sesion
if 'df' not in st.session_state:
    df = load_data()
    if df is not None:
        st.session_state['df'] = df




#MENU LATERAL
st.sidebar.header("Pestañas")
seccion = st.sidebar.radio(
    "Selecciona una sección:",
    ["Contexto", "EDA", "Modelado Machine Learning"]
)




#SECCION SELECCIONADA
if 'df' in st.session_state and st.session_state['df'] is not None:
    if seccion == "Contexto":
        tab_contexto.run()
    elif seccion == "EDA":
        tab_eda.run()
    elif seccion == "Modelado Machine Learning":
        tab_modelos.run()
else:
    st.warning("No se han cargado los datos. Verifica que el CSV exista en la carpeta `data/`.")

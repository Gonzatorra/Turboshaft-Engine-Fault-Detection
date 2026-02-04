import pandas as pd
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings('ignore')


def run():
    # ----TAB 1: INFORMACION BASICA----#
    st.header("🌐 Contexto del Proyecto")
    container1 = st.container(width=6000)
    container1.subheader("Evolución de la Industria 4.0")
    container1.markdown("La industria ha ido evolucionando a lo largo del tiempo, desde el siglo XVIII, cuando surgieron las primeras máquinas que reemplazaron el trabajo manual de los trabajadores, hasta hoy, cuando siguen surgiendo nuevas tecnologías para facilitar, mejorar y optimizar el rendimiento en la fabricación y el mantenimiento de sistemas industriales.")
    container1.markdown("Tradicionalmente, estos avances se han enfocado en procesos repetitivos o de producción masiva, como la fabricación de tornillos, puertas o componentes mecánicos. Sin embargo, en las últimas décadas, la digitalización y la inteligencia artificial han abierto la puerta a aplicar los mismos principios de optimización en sectores más complejos y críticos, como la industria aeronáutica.")
    container1.subheader("Importancia en la Industria Aeronáutica")
    container1.markdown("En este ámbito, los helicópteros son un caso especialmente interesante, pues su funcionamiento depende de sistemas muy sensibles, los cuales están sometidos a grandes esfuerzos mecánicos y donde cualquier fallo puede repercutir en consecuencias graves. Es por esto que es de vital importancia ser capaz de **detectar y predecir fallos en uno de los componentes más importantes del helicóptero, como son los motores** (corazón de helicóptero), permitiendo así mejorar la seguridad y la eficiencia en su construcción.")
    st.markdown("---")
    container1.subheader("Carga dataset")
    container1.markdown("En esta primera fase cargaremos y analizaremos la información básica de los datos del dataset escogido de kaggle: "
                        "https://www.kaggle.com/datasets/ziya07/helicopter-turboshaft-detection-dataset")


    #----INFORMACION BASICA----#
    if 'df' in st.session_state:
        st.subheader("📊 Información básica del Dataset")

        df = st.session_state['df']
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total de Registros", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total de Columnas", df.shape[1])
        with col3:
            st.metric("Columnas Numéricas",
                    f"{df.select_dtypes(include=[np.number]).shape[1]}")


        #----PRIMEROS REGISTROS----#
        with st.expander("👁️ Ver primeros registros"):
            st.dataframe(df.head())



        #----ESTRUCTURA DEL DATASET----#
        with st.expander("📋 Ver estructura del dataset"):
            buffer = []
            for col in df.columns:
                buffer.append({
                    "Columna": col,
                    "Tipo": str(df[col].dtype),
                    "No Nulos": df[col].notnull().sum(),
                    "Nulos": df[col].isnull().sum(),
                    "Únicos": df[col].nunique()
                })
            st.dataframe(pd.DataFrame(buffer))



        #----VALORES NULOS----#
        with st.expander("Ø Ver valores nulos"):
            missing_data = df.isnull().sum()
            missing_data = missing_data.rename("Valores nulos")
            st.dataframe(missing_data)



        #----DATOS TEMPORALES----#
        with st.expander("📆 Datos temporales"):
            st.text(f"Primer registro: {df['Timestamp'].min()}")
            st.text(f"Ultimo registro: {df['Timestamp'].max()}")
            st.text(
                f"Duracion total: {df['Timestamp'].max() - df['Timestamp'].min()}")
            st.markdown("---")
            diferencias_tiempo = df['Timestamp'].diff().dropna()
            frecuencia_esperada = pd.Timedelta(
                hours=1)  # Frecuencia esperada = 1h
            consistencia_frecuencia = (
                diferencias_tiempo == frecuencia_esperada).mean()
            st.text(
                f"Consistencia de frecuencia (1h): {consistencia_frecuencia:.2%}")

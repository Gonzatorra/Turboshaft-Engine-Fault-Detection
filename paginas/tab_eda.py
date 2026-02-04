import streamlit  as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')


from utils import *
from plots import * 


#----TAB 2: ANALISIS EXPLORATORIO----#
def run():
    st.header("🔍 Análisis Exploratorio (EDA)")
    #Verificar si los datos estan cargados
    if 'df' not in st.session_state:
        st.warning("Los datos no se han cargado. Ve a la pestaña 'Contexto' y verifica la carga.")
    else:
        df = st.session_state['df']
        
        # Filtro de fechas en sidebar (solo para este tab)
        if 'Timestamp' in df.columns:
            st.sidebar.subheader("Filtro Temporal")
            
            # Convertir a datetime si no lo está
            if not pd.api.types.is_datetime64_any_dtype(df['Timestamp']):
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            
            # Obtener fechas mínimas y máximas
            min_date = df['Timestamp'].min()
            max_date = df['Timestamp'].max()
            
            # Widget de selección de rango de fechas
            selected_dates = st.sidebar.date_input(
                "Selecciona el intervalo de fechas:",
                value=(min_date.date(), max_date.date()),
                min_value=min_date.date(),
                max_value=max_date.date()
            )
            
            # Aplicar filtro si se seleccionaron dos fechas
            if len(selected_dates) == 2:
                start_date, end_date = selected_dates
                start_datetime = pd.Timestamp(start_date)
                end_datetime = pd.Timestamp(end_date)
                
                # Filtrar el dataframe
                df_filtered = df[(df['Timestamp'] >= start_datetime) & 
                                (df['Timestamp'] <= end_datetime)]
                st.sidebar.success(f"Filtrado: {len(df_filtered)} registros de {start_date} a {end_date}")
            else:
                df_filtered = df
                st.sidebar.info("Selecciona un intervalo de fechas válido (inicio y fin)")
        else:
            df_filtered = df
            st.sidebar.info("No hay columna de fechas disponible para filtrar")
        
        st.success(f"Analizando dataset con {df_filtered.shape[0]} registros")
            
        if 'Fault_Label_Encoded' in df_filtered.columns:
                df_filtered = df_filtered.drop("Fault_Label_Encoded", axis=1)
        #Opciones
        opcion_analisis = st.selectbox(
            "Selecciona tipo de análisis:",
            ["📊 Estadísticas Descriptivas", 
             "📈 Distribuciones",
             "🔗 Matriz de Correlación",
             "❗ Fallos",
             "⚙️ Sensores",
             "⌚️ Tendencia temporal",
             "🧭 PCA"]
        )
        


        #----Estadisticas Descriptivas----#
        if opcion_analisis == "📊 Estadísticas Descriptivas":
            st.subheader("Estadísticas Descriptivas")
            df_numerico = df_filtered.select_dtypes(include=[np.number])
            st.dataframe(df_numerico.describe())
            
        #----Distribuciones----#
        elif opcion_analisis == "📈 Distribuciones":
            st.subheader("Análisis de Distribuciones y Outliers")
            
            st.write("Selecciona variables para visualizar su distribución:")
            #Seleccionar columnas numericas
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            
            if numeric_cols:
                selected_cols = st.multiselect(
                    "Selecciona columnas para visualizar:",
                    numeric_cols,
                )

            if len(selected_cols) == 0:
                resultados_df = generar_df_estadisticas_outliers(df_filtered[numeric_cols])
                st.info("No has seleccionado ninguna variable. Se mostrarán todas las distribuciones.")
            else:
                resultados_df = generar_df_estadisticas_outliers(df_filtered[selected_cols])
  
            fig = plot_distribuciones_outliers(df_filtered, resultados_df)
            #Mostrar en app
            plt.tight_layout()
            st.pyplot(fig)


        #----Correlaciones----#
        elif opcion_analisis == "🔗 Matriz de Correlación":
            st.subheader("🔗 Matriz de Correlación")

            #Seleccionar columnas numericas
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

            st.write("Selecciona columnas para incluir en la matriz:")
            if numeric_cols:
                selected_corr_cols = st.multiselect(
                    "Selecciona columnas para visualizar:",
                    numeric_cols
                )

            if len(selected_corr_cols) == 0:
                corr_matrix = calculo_matriz_correlacion(df_filtered, numeric_cols)
                # corr_df = df_filtered[numeric_cols].corr()
                st.info("No has seleccionado ninguna columna. Se mostrará la matriz completa.")
            else:
                corr_matrix = df_filtered[selected_corr_cols].corr()
  
            fig = plot_correlacion_heatmap(corr_matrix)
            #Mostrar en app
            plt.tight_layout()
            st.pyplot(fig)


        #----Analisis de fallos----#
        elif opcion_analisis == "❗ Fallos":
            st.subheader("❗ Fallos")
            tab4, tab5, tab6 = st.tabs(["Análisis básico",
                            "Distribución Normal y Fallos",
                            "Distribución de fallos"])

             
            with tab4:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    df_filtered["fallo"] = np.where(df_filtered["Fault_Label"] == "Normal", "Normal", "Fallos")
                    fig = pie_chart(df_filtered, "fallo", paleta = ["#3f7afa", "#ff5e5e"], titulo="Distribucion: Normal vs Fallos")            
                    st.pyplot(fig)

            with tab5: 
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:   
                    paleta = definicion_paleta(df_filtered, "Fault_Label")
                    fig = bar_chart(df_filtered, "Fault_Label", paleta, "Distribucion de Tipos de Fallo", "Cantidad", "Tipo de Fallo")
                    st.pyplot(fig)

            with tab6:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    df_fallos = df_filtered[df_filtered['Fault_Label'] != 'Normal']
                    fig = pie_chart(df_fallos, "Fault_Label", titulo="Distribucion de Tipos de Fallos")
                    st.pyplot(fig)





        #----Analisis de sensores----#
        elif opcion_analisis == "⚙️ Sensores":
            st.subheader("⚙️ Sensores")
            st.write("Selecciona sensores y fallos a analizar:")

            # Sensores disponibles
            sensores_disponibles = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            if len(sensores_disponibles) == 0:
                st.warning("No hay datos de sensores para analizar.")

            # Fallos disponibles sin Normal
            fallos_disponibles = df_filtered['Fault_Label'].unique().tolist()
            if len(fallos_disponibles) == 0:
                st.warning("No hay fallos para analizar.")

            # Multiselect principal
            fallos_seleccionados = st.multiselect(
                "Selecciona tipos de fallo a comparar:",
                fallos_disponibles,
            )
            sensores_seleccionados = st.multiselect(
                "Selecciona sensores a comparar:",
                sensores_disponibles,
                key="sensores_tab_principal" #Para diferenciar del temporal
            )

         
            if len(fallos_seleccionados) == 0 and len(sensores_seleccionados) == 0:
                st.info("No has seleccionado ningún sensor ni fallo. Se mostrarán todos ellos.")
                fallos_seleccionados = fallos_disponibles
                sensores_seleccionados = sensores_disponibles


            # Tabs
            tab7, tab8, tab9= st.tabs([
                "Heatmap medianas",
                "Boxplots sensores por tipo de fallo",
                "Sensores activados por fallo",
            ])

   
            #--- Tab 7:Heatmap medianas---#
            with tab7:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    median_matrix = calcular_medianas_por_fallo(df_filtered, fallos_seleccionados, sensores_seleccionados)
                    fig = plot_heatmap_medianas(median_matrix)
                    plt.tight_layout()
                    st.pyplot(fig)

            #---Tab 8: Boxplots---#
            with tab8:
                col1, col2, col3 = st.columns([1, 2, 1])

                if len(fallos_seleccionados) > 0 and len(sensores_seleccionados) == 0:
                    #Solo fallos seleccionados --> todos los sensores
                    df_filtrado = df_filtered[df_filtered['Fault_Label'].isin(fallos_seleccionados)]
                    sensores_a_graficar = [col for col in df_filtrado.select_dtypes(include='number').columns if col != 'Fault_Label']
                    fallos_a_graficar = fallos_seleccionados

                elif len(fallos_seleccionados) == 0 and len(sensores_seleccionados) > 0:
                    #Solo sensores seleccionados --> todos los fallos
                    sensores_existentes = [s for s in sensores_seleccionados if s in df_filtered.columns]
                    df_filtrado = df_filtered[sensores_existentes + ['Fault_Label']]
                    sensores_a_graficar = sensores_existentes
                    fallos_a_graficar = df_filtered['Fault_Label'].unique().tolist()
    
                else:
                    #Ambos seleccionados o ninguno
                    sensores_existentes = [s for s in sensores_seleccionados if s in df_filtered.columns]
                    df_filtrado = df_filtered[df_filtered['Fault_Label'].isin(fallos_seleccionados)][sensores_existentes + ['Fault_Label']]
                    sensores_a_graficar = sensores_existentes
                    fallos_a_graficar = fallos_seleccionados

                #Calcular medianas y pintar
                with col2:
                    median_matrix = calcular_medianas_por_fallo(df_filtrado, fallos_a_graficar, sensores_a_graficar)
                    fig = boxplots_por_sensor(df_filtrado, sensores_a_graficar, fallos_a_graficar)
                    plt.tight_layout()
                    st.pyplot(fig)


                #---Tab 9: Radarplots---#
                with tab9:
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col2:
                        # Sensores a graficar
                        sensores_a_graficar = sensores_seleccionados if len(sensores_seleccionados) > 0 else [col for col in df_filtered.select_dtypes(include='number').columns if col != 'Fault_Label']

                        # Filtrar df
                        df_filtrado = df_filtered[df_filtered['Fault_Label'].isin(fallos_a_graficar)]

                        # Calcular medianas
                        median_matrix = calcular_medianas_por_fallo(df_filtrado, fallos_a_graficar, sensores_a_graficar)

                        # Pintar radar
                        fig = radar_plot(fallos_a_graficar, median_matrix, sensores_a_graficar)
                        plt.tight_layout()
                        st.pyplot(fig)



            st.markdown("---")

            st.subheader("Timeseries")
            # Dividir la pantalla en dos columnas: izquierda para selección, derecha para plots
            col1, col2 = st.columns([1, 3])

            with col1:
                fallos_disponibles = []
                for f in df_filtered['Fault_Label'].unique():
                    if f != "Normal":
                        fallos_disponibles.append(f)

                sensores_seleccionados = st.multiselect(
                    "Selecciona sensores a comparar:",
                    sensores_disponibles,
                    key="sensores_timeline"
                )



                if len(sensores_seleccionados) == 0:
                    st.info("No has seleccionado ningún sensor. Se mostrarán todos ellos.")
                    sensores_seleccionados = sensores_disponibles

            with col2:
                #Tabs en la columna derecha
                tab10, tab11 = st.tabs([
                    "Timeline sensores",
                    "Cambio de sensores"
                ])

                

                #---Tab 10: Time Series fallos---#
                with tab10:
                    #Controles en la columna izquierda
                    fallo_timeline = st.radio(
                        "Selecciona un tipo de fallo a comparar:",
                        fallos_disponibles,
                        key="fallo_timeline"
                    )
                
                    n_ejemplos = st.slider(
                        "Número de ejemplos a mostrar",
                        min_value=1,
                        max_value=10,
                        )
                    fig, ejemplos_turbine = timeline_por_tipo_fallo(
                        df_filtered,
                        sensores_seleccionados,
                        fallo_timeline,
                        n_ejemplos=n_ejemplos
                    )
                    plt.tight_layout()
                    st.pyplot(fig)



                #---Tab 11: Cambio sensores---#
                with tab11:
                    plt.tight_layout()
                    fig = plot_cambio_5h(df_filtered, sensores_seleccionados)
                    st.pyplot(fig)







        elif opcion_analisis == "⌚️ Tendencia temporal":
            st.subheader("⌚️ Tendencias temporales (Time-Series)")

            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()

            col = st.selectbox("Selecciona una variable para analizar su evolución temporal:", numeric_cols)

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=df_filtered.index,
                y=df_filtered[col],
                mode="lines",
                line=dict(color="royalblue", width=2),
                name=col
            ))

            fig.update_layout(
                title=f"Evolución Temporal de {col}",
                xaxis_title="Tiempo (índice de registro)",
                yaxis_title=col,
                height=400,
                template="simple_white"
            )

            st.plotly_chart(fig, use_container_width=True)


        elif opcion_analisis == "🧭 PCA":
            st.subheader("🧭 PCA de los 3 principales componentes")

            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
            clean_numeric = [c for c in numeric_cols if c not in ["Fault_Code"]]

            X = df_filtered[clean_numeric].dropna()
            y = df_filtered["Fault_Label"]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            pca = PCA(n_components=3)
            pcs = pca.fit_transform(X_scaled)
            loadings = pca.components_.T

            pca_df = pd.DataFrame({
                "PC1": pcs[:, 0],
                "PC2": pcs[:, 1],
                "PC3": pcs[:, 2],
                "Fault_Label": y
            })

            fig = go.Figure()

            fig.add_trace(go.Scatter3d(
                x=pca_df["PC1"],
                y=pca_df["PC2"],
                z=pca_df["PC3"],
                mode='markers',
                marker=dict(
                    size=4,
                    opacity=0.75,
                    color=pca_df["Fault_Label"].astype('category').cat.codes,
                    colorscale="Viridis"
                ),
                text=y,
                hovertemplate="<b>Clase:</b> %{text}",
                name="Muestras"
            ))

            arrow_scale = 3

            for i, feature in enumerate(clean_numeric):
                fig.add_trace(go.Scatter3d(
                    x=[0, loadings[i, 0] * arrow_scale],
                    y=[0, loadings[i, 1] * arrow_scale],
                    z=[0, loadings[i, 2] * arrow_scale],
                    mode='lines+text',
                    line=dict(color="black", width=6),
                    text=[None, feature],
                    textposition="top center",
                    name=feature,
                    showlegend=False
                ))

            var_exp = pca.explained_variance_ratio_ * 100
            fig.update_layout(
                title=(
                    f"PCA 3D Biplot<br>"
                    f"Varianza explicada: "
                    f"PC1={var_exp[0]:.1f}% • "
                    f"PC2={var_exp[1]:.1f}% • "
                    f"PC3={var_exp[2]:.1f}%"
                ),
                scene=dict(
                    xaxis_title=f"PC1 ({var_exp[0]:.1f}%)",
                    yaxis_title=f"PC2 ({var_exp[1]:.1f}%)",
                    zaxis_title=f"PC3 ({var_exp[2]:.1f}%)",
                ),
                height=700,
                template="simple_white",
                margin=dict(l=0, r=0, t=80, b=0)
            )

            st.plotly_chart(fig, use_container_width=True)
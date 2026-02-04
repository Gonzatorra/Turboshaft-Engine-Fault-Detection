import streamlit  as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from scipy.stats import gaussian_kde
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score
import torch
import bentoml # NUEVO

from utils import preprocesar_datos, split_data, evaluar_modelo_scikit
from utils_modelos import entrenar_evaluar_pytorch
from utils_modelos import crear_secuencias
from modelos.mlp import MLP
from modelos.mlp_mejorado import MLP_mejorado
from modelos.lstm import LSTM_basico
from plots import plot_confusion_matrix
# Importamos la clase para que el pickle funcione al cargar
from modelos.lstm_ventana_variable import LSTM_vv 

from utils import *
from modelos.mlp import MLP

#funcion para evaluar desde bento
def evaluar_modelo_bento(model, X_test, y_test, nombre_modelo, le, es_secuencial=False, es_pytorch=False, ventana=50, es_variable=False):
    """
    Evalúa un modelo cargado de BentoML sin re-entrenar.
    Replica el diccionario de resultados que espera 'mostrar_resultados_modelo'.
    """
    #Predecir
    if es_pytorch:
        model.eval()
        with torch.no_grad():
            if es_secuencial:
                # Crear secuencias con la ventana especificada
                X_seq, y_seq = crear_secuencias(X_test, y_test, ventana)
                
                # Validación de seguridad
                if len(X_seq) == 0:
                    st.error("El test set es muy pequeño para esta ventana.")
                    return None
                
                X_tensor = torch.tensor(X_seq, dtype=torch.float32)
                y_true = y_seq 
                
                # Manejo específico del modelo de Ventana Variable
                if es_variable:
                    # Como estamos evaluando con ventanas fijas creadas por 'crear_secuencias', todas tienen la misma longitud 'ventana'. Creamos el tensor lengths.
                    lengths = torch.full((X_tensor.size(0),), ventana, dtype=torch.long)
                    logits = model(X_tensor, lengths)
                else:
                    logits = model(X_tensor)
                    
                y_pred = torch.argmax(logits, dim=1).numpy()
            else:
                # MLP tabular
                X_tensor = torch.tensor(X_test, dtype=torch.float32)
                y_true = y_test
                logits = model(X_tensor)
                y_pred = torch.argmax(logits, dim=1).numpy()
    else:
        y_true = y_test
        y_pred = model.predict(X_test)
    
    #alcular Métricas
    return {
        'modelo': model,
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_global': f1_score(y_true, y_pred, average='weighted'),
        'f1_por_clase': f1_score(y_true, y_pred, average=None, zero_division=0),
        'recall_por_clase': recall_score(y_true, y_pred, average=None, zero_division=0),
        'precision_por_clase': precision_score(y_true, y_pred, average=None, zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
        'y_true': y_true,       
        'y_pred': y_pred,
        'confusion_matrix': confusion_matrix(y_true, y_pred),
        'classes': le.classes_,
        'class_distribution': {clase: np.sum(y_true == i) for i, clase in enumerate(le.classes_)}
    }

def run():
    st.header("🤖 Modelado de Machine Learning (Evaluación BentoML)")

    #Cargar datos desde sesión
    df = st.session_state['df']
    
    #Filtro de fechas en sidebar
    if 'Timestamp' in df.columns:
        st.sidebar.subheader("Filtro Temporal")
        
        #Convertir a datetime
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
            
            # Calcular diferencia en días
            dias_diferencia = (end_datetime - start_datetime).days + 1  # +1 para incluir ambos extremos
            
            # Validar intervalo mínimo de 7 días
            if dias_diferencia < 7:
                st.sidebar.error(f"Intervalo insuficiente: {dias_diferencia} días. Se requiere al menos 7 días de datos para entrenar modelos.")
                st.sidebar.info(f"Selecciona un intervalo de al menos 7 días.")
                
                # Mostrar dataset completo como fallback
                df_filtered = df
                st.sidebar.warning("Usando dataset completo por intervalo insuficiente")
            else:
                # Filtrar el dataframe
                df_filtered = df[(df['Timestamp'] >= start_datetime) & 
                                (df['Timestamp'] <= end_datetime)]
                st.sidebar.success(f"Intervalo válido: {dias_diferencia} días")
                st.sidebar.success(f"Filtrado: {len(df_filtered)} registros de {start_date} a {end_date}")
                
        else:
            df_filtered = df
            st.sidebar.info("Selecciona un intervalo de fechas válido (inicio y fin)")
    else:
        df_filtered = df
        st.sidebar.info("No hay columna de fechas disponible para filtrar")
    
    #Convertimos a datetime para evitar error
    if 'Timestamp' in df_filtered.columns:
        df_filtered['Timestamp'] = pd.to_datetime(df_filtered['Timestamp'])

    # Parámetros interactivos
    st.sidebar.subheader("Parámetros de Evaluación")
    test_size = st.sidebar.slider("Tamaño del test (%)", 10, 30, 20)
    
    #Para que las predicciones del modelo (0,1,2) coincidan con el texto real
    try:
        le = bentoml.sklearn.load_model("turboshaft_le:latest")
        st.session_state['le'] = le
    except:
        st.error("Error: No está el LabelEncoder en BentoML. Ejecuta el notebook para guardar los modelos.")
        return

    #Preprocesamiento manual para usar el Encoder cargado
    df_proc = df_filtered.copy()
    if 'Fault_Label' in df_proc.columns:
        df_proc['Fault_Label_Encoded'] = le.transform(df_proc['Fault_Label'])
        y = df_proc['Fault_Label_Encoded']
        
        # Eliminamos columnas no numéricas para X
        cols_drop = ['Fault_Label', 'Fault_Label_Encoded', 'Timestamp']
        X = df_proc.drop(columns=[c for c in cols_drop if c in df_proc.columns])
        X = X.select_dtypes(include=[np.number])
    else:
        st.error("Columna Fault_Label no encontrada.")
        st.stop()

    #Split
    #solo usaremos X_test para evaluar
    X_train, X_test, y_train, y_test = split_data(X, y, test_size/100)

    #cargar features de bento
    try:
        # Intentamos cargar las features que el modelo espera
        loaded_features = bentoml.picklable_model.load_model("turboshaft_features:latest")
        
        # Sobrescribimos la selección para asegurar que no falle por falta de columnas
        available_features = loaded_features
        selected_features = loaded_features
        
        with st.expander("Ver variables del modelo"):
             st.write(selected_features)
             
    except:
        st.warning("No se encontró lista de features en BentoML. Usando todas las disponibles.")
        selected_features = X.columns.tolist()

    # inicializamos el dataframe de importancia
    # filtrado global de datos
    # Rellenamos con 0 si falta alguna columna en el CSV nuevo
    for col in selected_features:
        if col not in X_train.columns:
            X_train[col] = 0
            X_test[col] = 0
            
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]
    
    st.info(f"Evaluando sobre {len(X_test)} muestras de prueba.")
    st.markdown("---")

    # Crear pestañas para los modelos
    tab1, tab2, tab3, tab8 = st.tabs([
        "Modelos Shallow Learning",
        "MLP Básico",
        "MLP Avanzado",
        "LSTM"
    ])

    #TAB 1: Shallow Learning
    with tab1:
        tab4, tab5, tab6, tab7 = st.tabs([
            "Random Forest",
            "Gradient Boosting",
            "XGBoost",
            "Support Vector Machine (SVM)"
        ])
        
        with tab4:
            st.subheader("Random Forest (BentoML)")
            st.write("Parámetros: Pre-entrenados.")
            if st.button("Evaluar Random Forest"):
                with st.spinner("Cargando y evaluando..."):
                    model = bentoml.sklearn.load_model("modelo_randomforest:latest")
                    res = evaluar_modelo_bento(model, X_test, y_test, "Random Forest", le)
                    mostrar_resultados_modelo(res, le)
                
        with tab5:
            st.subheader("Gradient Boosting (BentoML)")
            st.write("Parámetros: Pre-entrenados.")
            if st.button("Evaluar Gradient Boosting"):
                with st.spinner("Cargando y evaluando..."):
                    model = bentoml.sklearn.load_model("modelo_gbm:latest")
                    res = evaluar_modelo_bento(model, X_test, y_test, "GBM", le)
                    mostrar_resultados_modelo(res, le)
                
        with tab6:
            st.subheader("XGBoost (BentoML)")
            st.write("Parámetros: Pre-entrenados.")
            if st.button("Evaluar XGBoost"):
                with st.spinner("Cargando y evaluando..."):
                    model = bentoml.sklearn.load_model("modelo_xgboost:latest")
                    res = evaluar_modelo_bento(model, X_test, y_test, "XGBoost", le)
                    mostrar_resultados_modelo(res, le)
                
        with tab7:
            st.subheader("Support Vector Machine (BentoML)")
            st.write("Parámetros: Pre-entrenados.")
            if st.button("Evaluar SVM"):
                with st.spinner("Cargando y evaluando..."):
                    model = bentoml.sklearn.load_model("modelo_svm:latest")
                    res = evaluar_modelo_bento(model, X_test, y_test, "SVM", le)                
                    mostrar_resultados_modelo(res, le)


    #TAB 2: MLP Básico
    with tab2:
        st.subheader("MLP Básico (BentoML)")
        st.write("Parámetros: Pre-entrenados.")
        # Los sliders originales están ahí visualmente pero no afectan
        if st.button("Evaluar MLP Básico"):
            with st.spinner("Cargando y evaluando..."):
                model = bentoml.pytorch.load_model("modelo_mlp:latest")
                model.to("cpu")
                res = evaluar_modelo_bento(model, X_test.values, y_test.values, "MLP", le, es_pytorch=True)
                mostrar_resultados_modelo(res, le)

    #TAB 3: MLP Avanzado
    with tab3:
        st.subheader("MLP Avanzado (BentoML)")
        
        
        if st.button("Evaluar MLP Avanzado"):
            with st.spinner("Cargando y evaluando..."):
                # Carga el modelo_mlp_mejorado
                model = bentoml.pytorch.load_model("modelo_mlp_mejorado:latest")
                model.to("cpu")
                res = evaluar_modelo_bento(model, X_test.values, y_test.values, "MLP Mejorado", le, es_pytorch=True)
                mostrar_resultados_modelo(res, le)

    #TAB 8: LSTM
    with tab8:
        st.subheader("LSTM (BentoML)")
        
        # Selector para los 3 tipos de lstm
        tipo_lstm = st.selectbox(
            "Selecciona la variante de LSTM a evaluar:",
            ["LSTM Base", "LSTM Class Weights", "LSTM Ventana Variable"]
        )
        
        #Slider de ventana
        #Importante: Debe coincidir con el entrenamiento. Ponemos 50 por defecto.
        ventana = st.slider("Tamaño de la ventana (para generar secuencias de test)", 10, 200, 50, step=5)
        
        
        if st.button(f"Evaluar {tipo_lstm}"):
            col1, col2 = st.columns(2)
            
            # Lógica de carga según selección
            nombre_bento = ""
            es_variable = False
            
            if tipo_lstm == "LSTM Base":
                nombre_bento = "modelo_lstm:latest"
            elif tipo_lstm == "LSTM Class Weights":
                nombre_bento = "modelo_lstm_classweights:latest"
            elif tipo_lstm == "LSTM Ventana Variable":
                nombre_bento = "modelo_lstm_ventanavariable:latest"
                es_variable = True
            
            with st.spinner(f"Cargando {nombre_bento} y evaluando..."):
                try:
                    model = bentoml.pytorch.load_model(nombre_bento)
                    model.to("cpu")
                    
                    res = evaluar_modelo_bento(
                        model,
                        X_test.values,
                        y_test.values,
                        tipo_lstm,
                        le,
                        es_secuencial=True,
                        es_pytorch=True,
                        ventana=ventana,
                        es_variable=es_variable
                    )
                    
                    if res:
                        mostrar_resultados_modelo(res, le)
                        
                except Exception as e:
                    st.error(f"Error cargando {nombre_bento}: {str(e)}")
                    st.info("¿Has ejecutado el bloque de guardado masivo en el notebook?")
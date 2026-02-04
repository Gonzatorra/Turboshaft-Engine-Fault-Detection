# FUNCIONALIDADES PRINCIPALES:
# 1. Deteccion y analisis de outliers (IQR)
# 2. Calculo de correlaciones y patrones por fallo
# 3. Preprocesamiento y split de datos
# 4. Evaluacion de modelos (scikit-learn y PyTorch)
# 5. Utilidades para modelos LSTM con ventanas fijas/variables
# 6. Visualización de resultados




#----------------------------------------#
#------------- DEPENDENCIAS -------------#
#----------------------------------------#
from IPython.display import display
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Evaluacion de modelos
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report, recall_score, precision_score, cohen_kappa_score, balanced_accuracy_score, matthews_corrcoef, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Importaciones propias
from plots import plot_confusion_matrix  

import bentoml
import json
import streamlit as st

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, cohen_kappa_score, balanced_accuracy_score, confusion_matrix
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modelos.ventana_lstm import SequenceDataset, collate_fn





#------------------------------------#
#------------- OUTLIERS -------------#
#------------------------------------#
def calcular_outliers_iqr(df):
    """
    Calcula informacion de outliers para cada columna numerica de un DataFrame usando el criterio del IQR.

    Parametros:
    -----------
    df (pd.DataFrame): DataFrame con los datos numericos a analizar.

    Retorna:
    --------
    dict: Diccionario donde cada clave es una columna numerica y cada valor es otro diccionario con:
        - 'count': numero de outliers
        - 'percent': porcentaje de outliers
        - 'lower_bound': limite inferior para detectar outliers
        - 'upper_bound': limite superior para detectar outliers
    """
    
    # Seleccionar columnas numericas
    numeric_cols = df.select_dtypes(include=[np.number]).columns
        
    outliers_info = {}
        
    # Calcular outliers para cada columna
    for columna in numeric_cols:
        # Calculo de cuartiles (25% y 75%)
        Q1 = df[columna].quantile(0.25)
        Q3 = df[columna].quantile(0.75)
        
        # Rango intercuartil (IQR)
        IQR = Q3 - Q1
            
        # Limites para detectar outliers
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
            
        # Deteccion de outliers
        outliers = df[(df[columna] < limite_inferior) | (df[columna] > limite_superior)]
        porcentaje_outliers = (len(outliers) / len(df)) * 100
        
        # Almacenar informacion
        outliers_info[columna] = {
            'count': len(outliers),
            'percent': porcentaje_outliers,
            'lower_bound': limite_inferior,
            'upper_bound': limite_superior
        }

    return outliers_info





def generar_df_estadisticas_outliers(df):
    """
    Genera un DataFrame con estadisticas de outliers para cada columna numerica, para usar en visualizaciones.

    Parametros:
    -----------
    - df (pd.DataFrame): DataFrame con los datos numericos a analizar.

    Retorna:
    --------
    pd.DataFrame: DataFrame con las columnas:
        - count: número de outliers
        - percent: porcentaje de outliers
        - lower_bound: límite inferior IQR
        - upper_bound: límite superior IQR
        Ordenado de menor a mayor porcentaje de outliers.
    """
    
    # Deteccion de outliers
    outliers_info = calcular_outliers_iqr(df)

    #Anadir a DataFrame y ordenar por porcentaje
    resultados_df = pd.DataFrame(outliers_info).T
    resultados_df = resultados_df.sort_values('percent', ascending=True)
    
    return resultados_df










#---------------------------------------#
#------------- CORRELACION -------------#
#---------------------------------------#
def calculo_matriz_correlacion(df, numeric_cols = None):
    """
    Calcula la matriz de correlacion de un DataFrame.
    
    Parametros:
    -----------
        - df (pd.DataFrame): DataFrame con los datos.
        - numeric_cols (list, opcional): Lista de columnas numericas a usar para la correlacion.
                                        Si es None, se usan todas las columnas numericas.

    Retorna:
    --------
        - pd.DataFrame: Matriz de correlacion.
    """

    # Seleccionar columnas numericas si no se especifican
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Calcular matriz de correlacion
    corr_matrix = df[numeric_cols].corr()
    
    return corr_matrix










#----------------------------------#
#------------- FALLOS -------------#
#----------------------------------#
def calcular_medianas_por_fallo(df, fallos, sensores_principales):
    """
    Calcula la mediana de cada sensor para cada tipo de fallo.

    Parametros:
    - df (pd.DataFrame): DataFrame con los datos.
    - fallos (list): lista de tipos de fallo.
    - sensores_principales (list): lista de nombres de columnas de sensores.

    Retorna:
    - pd.DataFrame: matriz de medianas (fallos x sensores).
    """
    
    # Crear dataframe con fallos como filas y sensores como columnas
    median_matrix = pd.DataFrame(index=fallos, columns=sensores_principales)
    
    # Calcular medianas para cada combinacion de fallo y sensor
    for fallo in fallos:
        for sensor in sensores_principales:
            # Filtrar por dupla fallo-sensor y calcular mediana
            median_matrix.loc[fallo, sensor] = df[df['Fault_Label'] == fallo][sensor].median()
    
    return median_matrix.astype(float)










#-----------------------------------------#
#------------- MODELOS BASIC -------------#
#-----------------------------------#
def preprocesar_datos(df, target_col="Fault_Label"):
    """
    Preprocesa el dataset para modelos de machine learning, codificando la variable objetivo y 
    seleccionando unicamente las variables numericas como caracteristicas.

    Parametros:
    - df (pd.DataFrame): DataFrame con los datos originales.
    - target_col (str, opcional): Nombre de la columna objetivo (por defecto 'Fault_Label').

    Retorna:
    - X (pd.DataFrame): DataFrame con las variables predictoras numericas.
    - y (pd.Series): Serie con la variable objetivo codificada numericamente.
    - le (LabelEncoder): Objeto LabelEncoder ajustado para decodificar las clases.
    """
    
    # Codificar variable objetivo, pasar cada clase a un numero
    le = LabelEncoder()
    
    # Agregar columna codificada
    df[f"{target_col}_Encoded"] = le.fit_transform(df[target_col])
    
    # Seleccionar solo columnas numericas para el modelo
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Eliminar la columna objetivo codificada de las features
    if f"{target_col}_Encoded" in numeric_cols:
        numeric_cols.remove(f"{target_col}_Encoded")
    
    # Dividir en X e y
    X = df[numeric_cols]
    y = df[f"{target_col}_Encoded"]
    
    return X, y, le





def split_data(X, y, test_size=0.3, random_state=111):
    """
    Divide los datos en conjuntos de entrenamiento y prueba manteniendo la distribucion de clases.

    Parametros:
    - X (pd.DataFrame): Variables predictoras.
    - y (pd.Series): Variable objetivo codificada.
    - test_size (float, opcional): Proporción del conjunto de prueba (por defecto 0.3).
    - random_state (int, opcional): Semilla aleatoria para reproducibilidad (por defecto 111).

    Retorna:
    - tuple: (X_train, X_test, y_train, y_test) en el mismo formato de entrada.
    """
    # Dividir en train y test con estratificacion para mantener distribucion de clases
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)





def evaluar_modelo_scikit(modelo, X_train, X_test, y_train, y_test, nombre_modelo, le):
    """
    Entrena y evalúa un modelo de clasificación de scikit-learn, devolviendo metricas
    de rendimiento.

    Parametros:
    - modelo (sklearn.base.BaseEstimator): Modelo de clasificacion compatible con scikit-learn.
    - X_train (pd.DataFrame): Conjunto de entrenamiento (features).
    - X_test (pd.DataFrame): Conjunto de prueba (features).
    - y_train (pd.Series): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series): Etiquetas del conjunto de prueba.
    - nombre_modelo (str): Nombre descriptivo del modelo evaluado.
    - le (LabelEncoder): Objeto LabelEncoder ajustado para mapear clases originales.

    Retorna:
    - dict: Diccionario con los siguientes elementos:
        - 'modelo': Modelo entrenado.
        - 'accuracy': Exactitud global del modelo.
        - 'balanced_accuracy': Exactitud balanceada (media del recall por clase).
        - 'f1_global': F1 ponderado global.
        - 'f1_por_clase': Vector de F1-score por clase.
        - 'recall_por_clase': Vector de recall por clase.
        - 'precision_por_clase': Vector de precision por clase.
        - 'kappa': Estadístico de Cohen’s Kappa (robusto frente a desbalanceo).
        - 'y_true': Etiquetas verdaderas del conjunto de prueba.
        - 'y_pred': Predicciones del modelo.
        - 'y_pred_proba': Probabilidades de prediccion (si el modelo las soporta).
        - 'confusion_matrix': Matriz de confusion.
        - 'classes': Nombres de las clases originales.
        - 'class_distribution': Distribución real de clases en el conjunto de prueba.
    """
    
    # Entrenar modelo
    modelo.fit(X_train, y_train)
    
    # Predecir
    y_pred = modelo.predict(X_test)
    y_pred_proba = modelo.predict_proba(X_test) if hasattr(modelo, "predict_proba") else None
    
    # Calculo de metricas globales
    acc = accuracy_score(y_test, y_pred)
    f1_global = f1_score(y_test, y_pred, average='weighted')
    
    # Calculo de metricas por clase
    f1_por_clase = f1_score(y_test, y_pred, average=None)
    recall_por_clase = recall_score(y_test, y_pred, average=None)
    precision_por_clase = precision_score(y_test, y_pred, average=None)
    
    # Calculo de matriz de confusion
    cm = confusion_matrix(y_test, y_pred)
    
    # Metricas extra para desbalanceo
    # Cohen's Kappa (mejor que accuracy para desbalanceo)
    kappa = cohen_kappa_score(y_test, y_pred)
    
    # Balanced Accuracy (promedio de recall por clase)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    
    return {
        'modelo': modelo,
        'accuracy': acc,
        'balanced_accuracy': balanced_acc,
        'f1_global': f1_global,
        'f1_por_clase': f1_por_clase,
        'recall_por_clase': recall_por_clase,
        'precision_por_clase': precision_por_clase,
        'kappa': kappa,
        'y_true': y_test,       
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'confusion_matrix': cm,
        'classes': le.classes_,
        'class_distribution': {clase: np.sum(y_test == i) for i, clase in enumerate(le.classes_)}
    }
    
    

    
    
    
    
    
    
    
#----------------------------------------------#
#------------- MOSTRAR RESULTADOS -------------#
#----------------------------------------------#
def mostrar_resultados_notebook(resultados, le):
    """
    Muestra los resultados de un modelo de clasificacion en un entorno de notebook (Jupyter),
    incluyendo metricas globales, metricas por clase y la matriz de confusion.

    Parametros:
    - resultados (dict): Diccionario de metricas generado por la funcion de evaluacion del modelo.
      Debe contener, al menos, las claves:
        ['accuracy', 'balanced_accuracy', 'f1_global', 'kappa',
         'y_true', 'y_pred', 'precision_por_clase', 'recall_por_clase',
         'f1_por_clase', 'classes', 'class_distribution'].
    - le (LabelEncoder): Objeto LabelEncoder utilizado para mapear las etiquetas codificadas
      a sus nombres originales de clase.

    Retorna:
    - La funcion imprime las metricas y muestra la matriz de confusion en el notebook.
    """
        
    print("Resultados del Modelo")
    print("---------------------")
    
    # Metricas globales
    print(f"Accuracy: {resultados['accuracy']:.3f}")
    print(f"Balanced Accuracy: {resultados['balanced_accuracy']:.3f}")
    print(f"F1 Global: {resultados['f1_global']:.3f}")
    print(f"Kappa: {resultados['kappa']:.3f}")
    print(f"Muestras en test: {len(resultados['y_true'])}\n")
    
    # Metricas por clase
    print("Métricas por Clase:")
    metrics_data = []
    for i, clase in enumerate(resultados['classes']):
        metrics_data.append({
            'Clase': clase,
            'Precision': resultados['precision_por_clase'][i],
            'Recall': resultados['recall_por_clase'][i],
            'F1': resultados['f1_por_clase'][i],
        })
    
    df_class_metrics = pd.DataFrame(metrics_data)
    display(df_class_metrics.style.format({
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1': '{:.3f}'
    }))
    
    # Distribucion de clases
    print("Distribución de clases en el conjunto de test:")
    for clase, count in resultados['class_distribution'].items():
        porcentaje = (count / len(resultados['y_true'])) * 100
        print(f"  {clase}: {count} muestras ({porcentaje:.2f}%)")
    
    # Matriz de confusion
    print("\nMatriz de Confusión:")
    fig = plot_confusion_matrix(
        y_true=resultados['y_true'],
        y_pred=resultados['y_pred'],
        class_names=le.classes_
    )
    plt.show()

    
    


def mostrar_resultados_modelo(resultados, le):
    """
    Muestra los resultados de un modelo de clasificacion en una interfaz de Streamlit, 
    incluyendo metricas globales, metricas por clase, distribución de clases y la matriz de confusion.

    Parametros:
    - resultados (dict): Diccionario con los resultados del modelo, generado por la funcion
      de evaluación. Debe contener las metricas globales y por clase, así como las predicciones.
    - le (LabelEncoder): Objeto LabelEncoder utilizado para recuperar los nombres de las clases originales.

    Retorna:
    - tuple: (col1, col2), objetos de columnas de Streamlit con el layout visual del reporte.
    """
    
    # Crear dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Resultados del Modelo")
        
        # Pestañas para organizar
        tab_global, tab_clases, tab_detalle = st.tabs(["Global", "Por Clase", "Detalle"])
        
        with tab_global:
            # Metricas globales
            g1, g2, g3, g4 = st.columns(4)
            with g1:
                st.metric("Accuracy", f"{resultados['accuracy']:.3f}")
            with g2:
                st.metric("Balanced Acc", f"{resultados['balanced_accuracy']:.3f}")
            with g3:
                st.metric("F1 Global", f"{resultados['f1_global']:.3f}")
            with g4:
                st.metric("Kappa", f"{resultados['kappa']:.3f}")
            
            # Distribucion
            st.caption(f"Muestras en test: {len(resultados['y_true'])}")
        
        with tab_clases:
            # Metricas por clase
            metrics_data = []
            for i, clase in enumerate(resultados['classes']):
                metrics_data.append({
                    'Clase': clase,
                    'Precision': resultados['precision_por_clase'][i],
                    'Recall': resultados['recall_por_clase'][i],
                    'F1': resultados['f1_por_clase'][i],
                })
            
            df_class_metrics = pd.DataFrame(metrics_data)
            st.dataframe(df_class_metrics.style.format({
                'Precision': '{:.3f}',
                'Recall': '{:.3f}',
                'F1': '{:.3f}'
            }))
        
        with tab_detalle:
            # Informacion adicional sobre la distribucion (desbalanceo)
            st.write("**Distribución de clases:**")
            for clase, count in resultados['class_distribution'].items():
                st.progress(count / len(resultados['y_true']), 
                          text=f"{clase}: {count} muestras")
    
    with col2:
        st.subheader("Matríz de Confusión")
        fig = plot_confusion_matrix(
            y_true=resultados['y_true'],
            y_pred=resultados['y_pred'],
            class_names=le.classes_
        )
        st.pyplot(fig)
    
    return col1, col2





def mostrar_resultados_notebook_variable(resultados, le):
    """
    Muestra los resultados de un modelo en notebook para el caso de modelos con ventana variable.
    Incluye el mismo conjunto de metricas que la versión estandar, pero ajusta la visualizacion
    de la matriz de confusión para manejar posibles diferencias en el numero de clases o etiquetas.

    Parametros:
    - resultados (dict): Diccionario con las metricas del modelo, incluyendo predicciones y valores reales.
    - le (LabelEncoder): Objeto LabelEncoder con las clases originales del modelo.

    Retorna:
    - None. La función imprime las metricas globales y por clase, y muestra la matriz de confusion.
    """
    
    print("Resultados del Modelo (Variable)")
    print("--------------------------------")

    # Metricas globales
    acc = resultados.get('accuracy', 0)
    bal_acc = resultados.get('balanced_accuracy', 0)
    f1 = resultados.get('f1_global', 0)
    kappa = resultados.get('kappa', 0)
    
    print(f"Accuracy: {acc:.3f}")
    print(f"Balanced Accuracy: {bal_acc:.3f}")
    print(f"F1 Global: {f1:.3f}")
    print(f"Kappa: {kappa:.3f}")
    print(f"Muestras en test: {len(resultados['y_true'])}\n")
    
    # Metricas por clase
    print("Métricas por Clase:")
    metrics_data = []
    clases = le.classes_
    
    for i, clase in enumerate(clases):
        prec = resultados['precision_por_clase'][i] if i < len(resultados['precision_por_clase']) else 0
        rec = resultados['recall_por_clase'][i] if i < len(resultados['recall_por_clase']) else 0
        f1_cls = resultados['f1_por_clase'][i] if i < len(resultados['f1_por_clase']) else 0
        
        metrics_data.append({
            'Clase': clase,
            'Precision': prec,
            'Recall': rec,
            'F1': f1_cls,
        })
        
        
    
    df_class_metrics = pd.DataFrame(metrics_data)
    display(df_class_metrics.style.format({
        'Precision': '{:.3f}',
        'Recall': '{:.3f}',
        'F1': '{:.3f}'
    }))
    
    # Distribucion de clases
    print("Distribución de clases en el conjunto de test:")
    if 'class_distribution' in resultados:
        for clase, count in resultados['class_distribution'].items():
            porcentaje = (count / len(resultados['y_true'])) * 100
            print(f"  {clase}: {count} muestras ({porcentaje:.2f}%)")
    
    print("\nMatriz de Confusión:")
    
    y_true = resultados['y_true']
    y_pred = resultados['y_pred']
    
    todos_los_indices = np.arange(len(clases))
    
    fig = plot_confusion_matrix(
        y_true, 
        y_pred, 
        class_names=clases, 
        labels=todos_los_indices
    )
    plt.show()










#----OTROS----#
def definicion_paleta(df, columna):
    """
    Genera un diccionario de colores que asigna un color unico a cada valor distinto
    de una columna categorica del DataFrame, utilizando una paleta predefinida de Seaborn.

    Parametros:
    - df (pd.DataFrame): DataFrame que contiene la columna categorica.
    - columna (str): Nombre de la columna para la cual se generará la paleta de colores.

    Retorna:
    - dict: Diccionario con pares {valor_categoria: color_RGB}, donde cada valor
      unico de la columna esta asociado a un color diferente.
    """
    # Obtener valores unicos
    variables = df[columna].unique().tolist()
    
    # Generar paleta de colores
    paleta = sns.color_palette("tab20", len(variables))
    
    # Mapear valores a colores
    color_map = {valor: paleta[i] for i, valor in enumerate(variables)}
    
    return color_map





def analisis_patrones_por_fallo(fallos, median_matrix):
    """
    Analiza los patrones de comportamiento de los sensores para cada tipo de fallo,
    identificando los sensores con valores significativamente altos o bajos y el
    sensor mas característico de cada tipo de fallo.

    Parametros:
    - fallos (list): Lista con los nombres o etiquetas de los fallos a analizar.
    - median_matrix (pd.DataFrame): Matriz de medianas con los fallos en filas y
      los sensores en columnas (valores estandarizados o centrados en torno al promedio).

    Retorna:
    - None. Imprime en consola los sensores con valores altos, bajos y el sensor
      mas característico para cada tipo de fallo analizado.
    """

    print("Análisis de patrones por fallo:")
    print("--------------------------------")

    # Analizar cada fallo
    for fallo in fallos:
        print(f"\n{fallo}:")
        # Calculo de medianas para el fallo
        medians = median_matrix.loc[fallo]
        
        #Sensores mas altos que el promedio
        altos = medians[medians > 0.5].sort_values(ascending=False)

        # Si hay, printearlo
        if len(altos) > 0:
            print(f"  Sensores ALTOS: {', '.join([f'{s}:{v:.2f}' for s,v in altos.items()])}")
        
        
        # Sensores mas bajos que el promedio
        bajos = medians[medians < -0.5].sort_values()

        # Si hay, printearlo
        if len(bajos) > 0:
            print(f"  Sensores BAJOS: {', '.join([f'{s}:{v:.2f}' for s,v in bajos.items()])}")
        
        #Sensor mas caracteristico
        caracteristico = medians.abs().idxmax()
        
        print(f"  Sensor más CARACTERÍSTICO: {caracteristico} ({medians[caracteristico]:.2f})")

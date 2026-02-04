
# FUNCIONALIDADES PRINCIPALES:
# 3. Preprocesamiento y division de datos
# 4. Evaluacion de modelos (scikit-learn y PyTorch)
# 5. Utilidades para modelos LSTM con ventanas fijas o variables


#----------------------------------------#
#------------- DEPENDENCIAS -------------#
#----------------------------------------#

# --- Librerias estandar ---
import json
from collections import Counter
import numpy as np
import pandas as pd


# --- Scikit-learn ---
from sklearn.utils import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    cohen_kappa_score, balanced_accuracy_score, confusion_matrix
)

# --- Machine Learning: XGBoost e Imbalanced Learn ---
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# --- PyTorch ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Herramientas internas del proyecto ---
from utils import (
    preprocesar_datos,
    split_data,
    evaluar_modelo_scikit,
    mostrar_resultados_notebook
)

# --- Servir modelos ---
import bentoml





#------------------------------------------------#
#------------- PREPARACION DE DATOS -------------#
#------------------------------------------------#
def preparar_datos(df):
    """
    Realiza el preprocesamiento basico del dataset, incluyendo la codificacion de la variable
    objetivo y la division del conjunto de datos en entrenamiento y prueba, manteniendo la
    distribucion de clases.

    Parametros:
    - df (pd.DataFrame): DataFrame original con los datos, incluyendo la variable objetivo.

    Retorna:
    - X_train (pd.DataFrame): Conjunto de entrenamiento (caracteristicas).
    - X_test (pd.DataFrame): Conjunto de prueba (caracteristicas).
    - y_train (pd.Series): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series): Etiquetas del conjunto de prueba.
    - le (LabelEncoder): Objeto LabelEncoder ajustado a las clases originales.
    """
    
    # Preprocesamiento, codificación y division (X, y, le)
    X, y, le = preprocesar_datos(df)
    
    # Division en train y test
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Distribución de clases en entrenamiento: {np.bincount(y_train)}")
    print(f"Distribución de clases en prueba: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test, le






def aplicar_smote(X_train, y_train):
    """
    Aplica la tecnica SMOTE para balancear las clases del conjunto de entrenamiento,
    generando muestras sinteticas de las clases minoritarias.

    Parametros:
    - X_train (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de entrenamiento.
    - y_train (pd.Series o np.ndarray): Etiquetas del conjunto de entrenamiento.

    Retorna:
    - X_res (np.ndarray): Conjunto de entrenamiento balanceado (caracteristicas).
    - y_res (np.ndarray): Etiquetas correspondientes al conjunto balanceado.
    """
    
    # Aplicar SMOTE para balancear clases minoritarias en el conjunto de entrenamiento
    smote = SMOTE(random_state=111)
    
    # Ajustamos los datos
    X_res, y_res = smote.fit_resample(X_train, y_train)

    print(f"Distribucion original: {Counter(y_train)}")
    print(f"Distribución de clases después de SMOTE: {np.bincount(y_res)}")

    return X_res, y_res





def medias_moviles(df, ventana=5):
    """
    Calcula medias moviles y desviaciones estandar moviles para cada sensor del dataset,
    con el objetivo de suavizar el ruido de los sensores y capturar tendencias temporales.

    Parametros:
    - df (pd.DataFrame): DataFrame con los datos originales de sensores.
    - ventana (int, opcional): Tamano de la ventana temporal para el calculo de medias
      y desviaciones estandar (por defecto 5).

    Retorna:
    - df_ingenieria (pd.DataFrame): DataFrame extendido con las nuevas variables de ingenieria
      de caracteristicas (medias y desviaciones moviles por sensor).
    """
    #creamos "medias moviles" (el promedio de los ultimos 5 valores del sensor) para suavizar el ruido de los sensores
    #esto ayuda al modelo a ver la tendencia general en lugar de picos sueltos.

    #identificamos las columnas de sensores (excluyendo la label y timestamp)
    cols_sensores = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Fault_Label' in cols_sensores: cols_sensores.remove('Fault_Label')

    #creamos nuevas variables con una ventana de tiempo
    #(el promedio de los ultimos 5 datos)
    ventana = ventana
    
    df_ingenieria = df.copy()

    for col in cols_sensores:
        #Tendencia
        df_ingenieria[f'{col}_mean_{ventana}'] = df_ingenieria[col].rolling(window=ventana).mean()
        
        #std
        df_ingenieria[f'{col}_std_{ventana}'] = df_ingenieria[col].rolling(window=ventana).std()

    #para la limpieza (al crear ventanas, las primeras filas quedan vacias osea que las rellenamos).
    df_ingenieria = df_ingenieria.fillna(method='bfill').fillna(method='ffill')

    print(f"Dimensiones nuevas del dataset: {df_ingenieria.shape}")

    return df_ingenieria


    

def seleccion_caracteristicas(X, y):
    """
    Selecciona las caracteristicas mas relevantes para el modelo utilizando la importancia
    de caracteristicas del algoritmo Random Forest.

    Parametros:
    - X (pd.DataFrame): Variables predictoras (features).
    - y (pd.Series): Variable objetivo codificada.
    
    Retorna:
    - X_filtrado (pd.DataFrame): Subconjunto de X con las caracteristicas mas importantes.
    """
    
    print(f"Características originales ({X.shape[1]}): {X.columns.tolist()}")

    #Usamos Random Forest para ver que sensores importan de verdad
    sel_model = RandomForestClassifier(n_estimators=50, random_state=111, n_jobs=-1)
    sel_model.fit(X, y)

    #Obtenemos la importancia y seleccionamos las mejores
    #cogemos los 9 mejores
    importancias = pd.Series(sel_model.feature_importances_, index=X.columns)
    mejores_features = importancias.nlargest(9).index.tolist()

    print(f"Características seleccionadas ({len(mejores_features)}): {mejores_features}")

    return X[mejores_features]




def preparar_datos_lstm(df, features):
    """
    Prepara los datos para su uso en un modelo LSTM, dividiendo el conjunto original en
    entrenamiento y prueba, y extrayendo unicamente las caracteristicas especificadas.

    Parametros:
    - df (pd.DataFrame): DataFrame con los datos originales, incluyendo la columna objetivo.
    - features (list): Lista de nombres de las columnas que se utilizaran como caracteristicas.

    Retorna:
    - X_train_raw (np.ndarray): Conjunto de entrenamiento con las caracteristicas seleccionadas.
    - X_test_raw (np.ndarray): Conjunto de prueba con las caracteristicas seleccionadas.
    - y_train_raw (np.ndarray): Etiquetas del conjunto de entrenamiento.
    - y_test_raw (np.ndarray): Etiquetas del conjunto de prueba.
    """
    # Extraemos las caracteristicas y la variable objetivo
    X_raw = df[features].values
    y_raw = df['target'].values

    # Dividimos en train y test (80%-20%)
    split = int(len(X_raw) * 0.8)
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]

    return X_train_raw, X_test_raw, y_train_raw, y_test_raw







def crear_secuencias_variables(df, features_cols, target_col, sequence_id_col=None):
    """
    Crea secuencias de longitud variable a partir de un DataFrame, para entrenamiento
    de modelos secuenciales como LSTM.

    Parametros:
    - df (pd.DataFrame): DataFrame con los datos.
    - features_cols (list): Lista de columnas que seran utilizadas como features.
    - target_col (str): Nombre de la columna objetivo (label).
    - sequence_id_col (str, opcional): Columna que identifica secuencias independientes.
      Si se proporciona, cada grupo se convierte en una secuencia. Si no, se crean
      secuencias cuando target_col != 0.

    Retorna:
    - X_seqs (list de np.ndarray): Lista de secuencias de features (cada secuencia es un array 2D).
    - y_seqs (list): Lista de etiquetas correspondientes a cada secuencia.
    """
    
    X_seqs = []
    y_seqs = []
    
    # Si se proporciona una columna de ID de secuencia, agrupar por ella
    if sequence_id_col and sequence_id_col in df.columns:
        for _, group in df.groupby(sequence_id_col):
            X_seqs.append(group[features_cols].values)
            y_seqs.append(group[target_col].iloc[-1])
            
    else:
        current_seq = []
        
        # Crear secuencias basadas en la columna target_col
        for i, row in df.iterrows():
            current_seq.append(row[features_cols].values)
            
            # Crear una nueva secuencia cuando target_col != 0
            if row[target_col] != 0: 
                X_seqs.append(np.array(current_seq))
                y_seqs.append(row[target_col])
                current_seq = []
        if len(current_seq) > 0:
            pass 

    return X_seqs, y_seqs






#----EXTRAS LSTM----#
def crear_secuencias(X, y, ventana):
    """
    Crea secuencias de longitud fija a partir de arrays de features y etiquetas,
    para entrenamiento de modelos secuenciales como LSTM.

    Parametros:
    - X (np.ndarray o pd.DataFrame): Array de features.
    - y (np.ndarray o pd.Series): Array de etiquetas correspondientes.
    - ventana (int): Longitud de la secuencia (numero de pasos de tiempo).

    Retorna:
    - X_seq (np.ndarray): Array de secuencias de features, con forma (num_secuencias, ventana, num_features).
    - y_seq (np.ndarray): Array de etiquetas correspondientes al ultimo paso de cada secuencia.
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - ventana):
        # Para cada posicion, tomar una secuencia de longitud de la ventana
        X_seq.append(X[i:i+ventana])
        y_seq.append(y[i+ventana-1])  # Etiqueta del ultimo paso
    return np.array(X_seq), np.array(y_seq)










#-------------------------------------------#
#------------- MODELOS SHALLOW -------------#
#-------------------------------------------#
def random_forest(X_train, X_test, y_train, y_test, le):
    """
    Entrena un modelo Random Forest, evalua sus resultados y guarda el modelo junto
    con el label encoder y las caracteristicas en BentoML.

    Parametros:
    - X_train (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de entrenamiento.
    - X_test (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de prueba.
    - y_train (pd.Series o np.ndarray): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series o np.ndarray): Etiquetas del conjunto de prueba.
    - le (LabelEncoder): Objeto LabelEncoder ajustado para decodificar clases.

    Retorna:
    - resultados_rf (dict): Diccionario con las metricas y resultados de evaluacion
      del modelo Random Forest.
    """
    # Entrenamos el modelo
    rf_model = RandomForestClassifier(n_estimators=10, random_state=111, class_weight='balanced')
    resultados_rf = evaluar_modelo_scikit(rf_model, X_train, X_test, y_train, y_test, "Random Forest", le)

    # Guardamos el modelo en BentoML
    bentoml.sklearn.save_model(
        "modelo_RandomForest", 
        rf_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    # Guardamos las caracteristicas si X_train es un DataFrame
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_rf





def gbm(X_train, X_test, y_train, y_test, le):
    """
    Entrena un modelo Gradient Boosting, evalua sus resultados y guarda el modelo junto
    con el label encoder y las caracteristicas en BentoML.

    Parametros:
    - X_train (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de entrenamiento.
    - X_test (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de prueba.
    - y_train (pd.Series o np.ndarray): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series o np.ndarray): Etiquetas del conjunto de prueba.
    - le (LabelEncoder): Objeto LabelEncoder ajustado para decodificar clases.

    Retorna:
    - resultados_gbm (dict): Diccionario con las metricas y resultados de evaluacion
      del modelo Gradient Boosting.
    """
    # Entrenamos el modelo
    gbm_model = GradientBoostingClassifier(n_estimators=50, random_state=111)
    resultados_gbm = evaluar_modelo_scikit(gbm_model, X_train, X_test, y_train, y_test, "Gradient Boosting", le)

    # Guardamos el modelo en BentoML
    bentoml.sklearn.save_model(
        "modelo_GBM",
        gbm_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    # Guardamos las caracteristicas si X_train es un DataFrame
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_gbm  





def xgboost(X_train, X_test, y_train, y_test, le):
    """
    Entrena un modelo XGBoost, ajustando el parametro scale_pos_weight para balancear clases
    si es necesario, evalua sus resultados y guarda el modelo junto con el label encoder
    y las caracteristicas en BentoML.

    Parametros:
    - X_train (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de entrenamiento.
    - X_test (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de prueba.
    - y_train (pd.Series o np.ndarray): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series o np.ndarray): Etiquetas del conjunto de prueba.
    - le (LabelEncoder): Objeto LabelEncoder ajustado para decodificar clases.

    Retorna:
    - resultados_xgb (dict): Diccionario con las metricas y resultados de evaluacion
      del modelo XGBoost.
    """
    counts = np.bincount(y_train)
    ratio = counts[0] / counts[1] if len(counts) > 1 else 1
    
    # Entrenamos el modelo
    xgb_model = XGBClassifier(
        n_estimators=100, 
        random_state=111, 
        use_label_encoder=False, 
        eval_metric='mlogloss', 
        scale_pos_weight=ratio
    )

    resultados_xgb = evaluar_modelo_scikit(xgb_model, X_train, X_test, y_train, y_test, "XGBoost", le)

    # Guardamos el modelo en BentoML
    bentoml.sklearn.save_model(
        "modelo_XGBoost", 
        xgb_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    # Guardamos las caracteristicas si X_train es un DataFrame
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_xgb





def svm(X_train, X_test, y_train, y_test, le):
    """
    Entrena un modelo SVM con kernel RBF, evalua sus resultados y guarda el modelo junto
    con el label encoder y las caracteristicas en BentoML.

    Parametros:
    - X_train (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de entrenamiento.
    - X_test (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de prueba.
    - y_train (pd.Series o np.ndarray): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series o np.ndarray): Etiquetas del conjunto de prueba.
    - le (LabelEncoder): Objeto LabelEncoder ajustado para decodificar clases.

    Retorna:
    - resultados_svm (dict): Diccionario con las metricas y resultados de evaluacion
      del modelo SVM.
    """
    # Entrenamos el modelo
    svm_model = SVC(kernel='rbf', probability=True, random_state=111, class_weight='balanced')
    resultados_svm = evaluar_modelo_scikit(svm_model, X_train, X_test, y_train, y_test, "SVM", le)
    
    # Guardamos el modelo en BentoML
    bentoml.sklearn.save_model(
        "modelo_SVM", 
        svm_model,
        signatures={"predict": {"batchable": True}}
    )
    
    bentoml.sklearn.save_model("turboshaft_le", le)
    
    # Guardamos las caracteristicas si X_train es un DataFrame
    if hasattr(X_train, 'columns'):
        bentoml.picklable_model.save_model(
            "turboshaft_features", 
            X_train.columns.tolist()
        )
        
    print(f"Modelo guardado en BentoML.")
    
    return resultados_svm










#----------------------------------------#
#------------- MODELOS DEEP -------------#
#----------------------------------------#

def entrenar_evaluar_pytorch(model, X_train, X_test, y_train, y_test, le,
                             epochs=50, batch_size=32, lr=0.001, device='cpu',
                             class_weights=None, 
                             nombre_bento=None):
    
    """
    Entrena y evalua un modelo de PyTorch sobre datos tabulares, calculando metricas
    globales y por clase, y guardando el modelo en BentoML si se especifica.

    Parametros:
    - model (torch.nn.Module): Modelo de PyTorch a entrenar.
    - X_train (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de entrenamiento.
    - X_test (pd.DataFrame o np.ndarray): Caracteristicas del conjunto de prueba.
    - y_train (pd.Series o np.ndarray): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series o np.ndarray): Etiquetas del conjunto de prueba.
    - le (LabelEncoder): Objeto LabelEncoder ajustado para decodificar clases.
    - epochs (int, opcional): Numero de epocas de entrenamiento (por defecto 50).
    - batch_size (int, opcional): Tamano del batch para el DataLoader (por defecto 32).
    - lr (float, opcional): Learning rate para el optimizador Adam (por defecto 0.001).
    - device (str o torch.device, opcional): Dispositivo para entrenar el modelo (CPU/GPU).
    - class_weights (np.ndarray, opcional): Pesos de clase para la funcion de perdida.
    - nombre_bento (str, opcional): Nombre para guardar el modelo en BentoML.

    Retorna:
    - resultados (dict): Diccionario con metricas globales, metricas por clase, 
      matriz de confusion, distribucion de clases, predicciones, etiquetas reales
      y el modelo entrenado.
    """
    
    #Si nos pasan un DataFrame, guardamos sus columnas y extraemos los valores
    feature_names = None
    if isinstance(X_train, pd.DataFrame):
        feature_names = X_train.columns.tolist()
        X_train_data = X_train.values
    else:
        X_train_data = X_train

    if isinstance(X_test, pd.DataFrame):
        X_test_data = X_test.values
    else:
        X_test_data = X_test

    if isinstance(y_train, pd.Series): y_train = y_train.values
    if isinstance(y_test, pd.Series): y_test = y_test.values

    #Convertir a tensores
    X_train_tensor = torch.tensor(X_train_data, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_test_tensor = torch.tensor(X_test_data, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    model = model.to(device)
    
    # Definir el criterio
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Entrenamiento
    model.train()
    for epoch in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    
    # Evaluacion
    model.eval()
    all_preds = []
    all_true = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())
    
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    
    
    # Metricas
    acc = accuracy_score(all_true, all_preds)
    f1_global = f1_score(all_true, all_preds, average='weighted')
    
    # Si se especifica, guardamos el modelo en BentoML
    if nombre_bento:
        bentoml.pytorch.save_model(
            nombre_bento, 
            model,
            signatures={"__call__": {"batchable": True, "batch_dim": 0}}
        )
        
        bentoml.sklearn.save_model("turboshaft_le", le)
        
        if feature_names:
            bentoml.picklable_model.save_model("turboshaft_features", feature_names)
        
        print(f"Modelo {nombre_bento} guardado correctamente.")

    # Devolvemos los resultados
    resultados = {
        'accuracy': acc,
        'f1_global': f1_global,
        'f1_por_clase': f1_score(all_true, all_preds, average=None),
        'recall_por_clase': recall_score(all_true, all_preds, average=None), 
        'precision_por_clase': precision_score(all_true, all_preds, average=None),  
        'kappa': cohen_kappa_score(all_true, all_preds),  
        'balanced_accuracy': balanced_accuracy_score(all_true, all_preds),  
        'y_pred': all_preds,
        'y_true': all_true,
        'model': model,
        'classes': le.classes_,  
        'class_distribution': {clase: np.sum(all_true == i) for i, clase in enumerate(le.classes_)},  
        'confusion_matrix': confusion_matrix(all_true, all_preds)  
    }
    
    return resultados






def entrenar_evaluar_lstm(model, 
                          X_train=None, X_test=None, 
                          y_train=None, y_test=None, 
                          le=None, 
                          train_dataset=None, test_dataset=None,
                          epochs=50, batch_size=32, lr=0.001, 
                          device='cpu', class_weights=None, 
                          ventana_variable=False, collate_fn=None,
                          ventana=50,
                          nombre_bento=None):
    """
    Entrena y evalua un modelo LSTM, soportando tanto secuencias fijas como variables,
    calculando metricas globales y por clase, y guardando el modelo en BentoML si se especifica.

    Parametros:
    - model (torch.nn.Module): Modelo LSTM a entrenar.
    - X_train (pd.DataFrame o np.ndarray, opcional): Caracteristicas del conjunto de entrenamiento.
    - X_test (pd.DataFrame o np.ndarray, opcional): Caracteristicas del conjunto de prueba.
    - y_train (pd.Series o np.ndarray, opcional): Etiquetas del conjunto de entrenamiento.
    - y_test (pd.Series o np.ndarray, opcional): Etiquetas del conjunto de prueba.
    - le (LabelEncoder, opcional): Objeto LabelEncoder ajustado para decodificar clases.
    - train_dataset (torch.utils.data.Dataset, opcional): Dataset de entrenamiento (para secuencias variables).
    - test_dataset (torch.utils.data.Dataset, opcional): Dataset de prueba (para secuencias variables).
    - epochs (int, opcional): Numero de epocas de entrenamiento (por defecto 50).
    - batch_size (int, opcional): Tamano del batch (por defecto 32).
    - lr (float, opcional): Learning rate para el optimizador Adam (por defecto 0.001).
    - device (str o torch.device, opcional): Dispositivo para entrenar el modelo (CPU/GPU).
    - class_weights (np.ndarray, opcional): Pesos de clase para la funcion de perdida.
    - ventana_variable (bool, opcional): Si True, usa secuencias de longitud variable con pack_padded_sequence.
    - collate_fn (callable, opcional): Funcion de collation para DataLoader si ventana_variable=True.
    - ventana (int, opcional): Tamano de la ventana para secuencias fijas (por defecto 50).
    - nombre_bento (str, opcional): Nombre para guardar el modelo en BentoML.

    Retorna:
    - resultados (dict): Diccionario con metricas globales y por clase, matriz de confusion,
      distribucion de clases, predicciones, etiquetas reales, modelo entrenado, ventana usada
      y si la ventana fue variable.
    """
    #Preparacion de datos para BentoML
    feature_names = None
    if X_train is not None and hasattr(X_train, 'columns'):
        feature_names = X_train.columns.tolist()

    #Crear DataLoaders
    if ventana_variable:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    else:
        #Crear secuencias fijas
        X_train_seq, y_train_seq = crear_secuencias(X_train, y_train, ventana)
        X_test_seq, y_test_seq = crear_secuencias(X_test, y_test, ventana)

        train_dataset = TensorDataset(torch.tensor(X_train_seq, dtype=torch.float32),
                                      torch.tensor(y_train_seq, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test_seq, dtype=torch.float32),
                                     torch.tensor(y_test_seq, dtype=torch.long))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

    #Preparar modelo y criterio
    model = model.to(device)
    
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #Entrenamiento
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            optimizer.zero_grad()
            
            if ventana_variable:
                xb, yb, lengths = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb, lengths)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)

            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()

    #Evaluacion
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            if ventana_variable:
                xb, yb, lengths = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb, lengths)
            else:
                xb, yb = batch
                xb, yb = xb.to(device), yb.to(device)
                outputs = model(xb)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(yb.cpu().numpy())

    #Metricas
    all_true = np.array(all_true)
    all_preds = np.array(all_preds)
    all_labels = np.arange(len(le.classes_))

    # Guardar el modelo en BentoML si se especifica
    if nombre_bento:
        bentoml.pytorch.save_model(
            nombre_bento, 
            model,
            signatures={
                "__call__": {
                    "batchable": True, 
                    "batch_dim": 0
                }
            }
        )
        
        # Guardar el label encoder
        if le is not None:
            bentoml.sklearn.save_model("turboshaft_le", le)
        
        # Guardar las caracteristicas si estan disponibles
        if feature_names:
            bentoml.picklable_model.save_model("turboshaft_features", feature_names)
        
        print(f"Modelo {nombre_bento} guardado.")


    # Devolvemos los resultados
    resultados = {
        'accuracy': accuracy_score(all_true, all_preds),
        'balanced_accuracy': balanced_accuracy_score(all_true, all_preds),
        'f1_global': f1_score(all_true, all_preds, average='weighted'),
        'f1_por_clase': f1_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'recall_por_clase': recall_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'precision_por_clase': precision_score(all_true, all_preds, average=None, labels=all_labels, zero_division=0),
        'kappa': cohen_kappa_score(all_true, all_preds),
        'y_true': all_true,
        'y_pred': all_preds,
        'model': model,
        'classes': le.classes_,
        'class_distribution': {clase: np.sum(all_true == i) for i, clase in enumerate(le.classes_)},
        'confusion_matrix': confusion_matrix(all_true, all_preds, labels=all_labels),
        'ventana_usada': ventana,
        'ventana_variable': ventana_variable
    }

    return resultados








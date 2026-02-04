import torch  #MANTENER SIEMPRE PRIMERO
import bentoml
import numpy as np

# DEFINICIÓN DEL SERVICIO
@bentoml.service(name="turboshaft_service_master")
class TurboshaftService:
    
    def __init__(self):
        """
        Carga de todos los modelos (Tabulares, Fijos y Variables).
        """
        #Modelos Tabulares
        self.rf = bentoml.sklearn.load_model("modelo_randomforest:latest")
        self.gbm = bentoml.sklearn.load_model("modelo_gbm:latest")
        self.xgb = bentoml.sklearn.load_model("modelo_xgboost:latest")
        self.svm = bentoml.sklearn.load_model("modelo_svm:latest")
        
        #Modelos Deep Learning (PyTorch)
        self.mlp = bentoml.pytorch.load_model("modelo_mlp:latest")
        self.mlp_imp = bentoml.pytorch.load_model("modelo_mlp_mejorado:latest")
        
        #LSTM Fijos
        self.lstm = bentoml.pytorch.load_model("modelo_lstm:latest")
        self.lstm_cw = bentoml.pytorch.load_model("modelo_lstm_classweights:latest")
        
        #LSTM Variable
        self.lstm_vv = bentoml.pytorch.load_model("modelo_lstm_ventanavariable:latest") 
        
        #Configurar PyTorch en CPU y Eval
        for model in [self.mlp, self.mlp_imp, self.lstm, self.lstm_cw, self.lstm_vv]:
            model.to("cpu")
            model.eval()

        #utilidades
        self.le = bentoml.sklearn.load_model("turboshaft_le:latest")

    def _decodificar(self, prediccion_numerica):
        """Traduce de número (0,1,2) a texto ('Normal', 'Fallo X')."""
        try:
            if hasattr(prediccion_numerica, 'cpu'):
                prediccion_numerica = prediccion_numerica.detach().cpu().numpy()
            
            prediccion_numerica = np.ravel(prediccion_numerica)
            nombres = self.le.inverse_transform(prediccion_numerica)
            
            if hasattr(nombres, 'tolist'):
                return nombres.tolist()
            else:
                return list(nombres)
        except Exception as e:
            return [f"Error traduciendo: {str(e)}"]

    #API 1: TABULAR
    @bentoml.api
    def predecir_tabular(self, input_json: dict) -> dict:
        datos = np.array(input_json.get("datos"), dtype=np.float32)
        modelo_nombre = input_json.get("modelo")
        
        resultado_num = None
        
        #Lógica de selección
        if modelo_nombre == "Random Forest":
            resultado_num = self.rf.predict(datos)
        elif modelo_nombre == "GBM":
            resultado_num = self.gbm.predict(datos)
        elif modelo_nombre == "XGBoost":
            resultado_num = self.xgb.predict(datos)
        elif modelo_nombre == "SVM":
            resultado_num = self.svm.predict(datos)
        elif modelo_nombre == "MLP":
            with torch.no_grad():
                logits = self.mlp(torch.tensor(datos))
                resultado_num = logits.argmax(dim=1)
        elif modelo_nombre == "MLP Mejorado":
            with torch.no_grad():
                logits = self.mlp_imp(torch.tensor(datos))
                resultado_num = logits.argmax(dim=1)
        else:
            return {"error": f"Modelo tabular '{modelo_nombre}' no encontrado."}

        clases = self._decodificar(resultado_num)
        return {"prediccion": clases, "modelo_usado": modelo_nombre}

    #API 2: SECUENCIAL
    @bentoml.api
    def predecir_secuencial(self, input_json: dict) -> dict:
        """
        Espera un JSON con:
        - "datos": Array 3D [batch, seq_len, features]
        - "modelo": Nombre del modelo
        - "lengths": (SOLO para LSTM Variable) Lista con la longitud real de cada secuencia
        """
        datos = np.array(input_json.get("datos"), dtype=np.float32)
        modelo_nombre = input_json.get("modelo")
        
        tensor_datos = torch.tensor(datos)
        resultado_num = None
        
        with torch.no_grad():
            if modelo_nombre == "LSTM Base":
                logits = self.lstm(tensor_datos)
                resultado_num = logits.argmax(dim=1)
                
            elif modelo_nombre == "LSTM Class Weights":
                logits = self.lstm_cw(tensor_datos)
                resultado_num = logits.argmax(dim=1)
                
            elif modelo_nombre == "LSTM Ventana Variable":
                #Necesitamos el argumento extra 'lengths'
                lengths = input_json.get("lengths")
                
                if lengths is None:
                    return {"error": "El modelo de ventana variable requiere el campo 'lengths' en el JSON."}
                
                #Convertir lengths a Tensor Long (Int64)
                #Deben estar en CPU para pack_padded_sequence si el tensor de datos está en CPU
                tensor_lengths = torch.tensor(lengths, dtype=torch.long)
                
                #Llamada especial con dos argumentos
                logits = self.lstm_vv(tensor_datos, tensor_lengths)
                resultado_num = logits.argmax(dim=1)
                
            else:
                return {"error": f"Modelo secuencial '{modelo_nombre}' no encontrado."}

        clases = self._decodificar(resultado_num)
        return {"prediccion": clases, "modelo_usado": modelo_nombre}
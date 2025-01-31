from datetime import datetime
import json
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from dateutil.relativedelta import relativedelta
from sklearn.preprocessing import StandardScaler

class LoanService:
    def __init__(self):
        # Cargar el modelo y el escalador
        self.model = tf.keras.models.load_model('app/services/model.keras')
        self.scaler = joblib.load('app/services/scaler.pkl')
        
        # Pesos para cada clase
        self.weights = {
            "Incumplidos": -2,
            "Activos": 2,
            "Pagados": 3,
            "Morosos": -1,
            "Emitidos": 1
        }
        
        # Parámetros del scorecard
        self.OFFSET = 600  # Ajusta la escala base del puntaje
        self.FACTOR = 10  # Ajusta el peso del impacto del logaritmo
        
        # Definir clases del modelo
        self.class_names = ["Incumplidos", "Activos", "Pagados", "Morosos", "Emitidos"]
    
    def validate_and_convert_data(self, data):
        """
        Verifica y convierte los tipos de datos en el diccionario de entrada.
        """
        df = pd.DataFrame([data])
        
        # Definir los tipos correctos de datos
        correct_types = {
            'out_prncp': 'float', 'out_prncp_inv': 'float', 'last_pymnt_amnt': 'float',
            'total_rec_prncp': 'float', 'total_pymnt': 'float', 'total_pymnt_inv': 'float',
            'recoveries': 'float', 'collection_recovery_fee': 'float', 'funded_amnt': 'float',
            'total_rec_int': 'float', 'funded_amnt_inv': 'float', 'loan_amnt': 'float',
            'installment': 'float', 'total_rev_hi_lim': 'float', 'tot_cur_bal': 'float',
            'initial_list_status_w': 'int', 'int_rate': 'float', 'dti': 'float',
            'revol_bal': 'float', 'revol_util': 'float'
        }
        
        for column, correct_type in correct_types.items():
            if column in df.columns:
                df[column] = df[column].astype(correct_type)
        
        return df
    
    def calculate_probabilities(self, data):
        """
        Calcula las probabilidades predichas por el modelo para cada clase.
        """
        probabilities = self.model.predict(data)
        return {class_name: prob for class_name, prob in zip(self.class_names, probabilities[0])}
    
    def calculate_score(self, probabilities):
        """
        Calcula un puntaje general a partir de las probabilidades y los pesos.
        """
        total_prob = sum(probabilities.values())
        normalized_probs = {k: v / total_prob for k, v in probabilities.items()}
        
        score = self.OFFSET
        for class_name, prob in normalized_probs.items():
            weight = self.weights.get(class_name, 0)
            score += weight * self.FACTOR * np.log(prob + 1e-6)
        
        return round(score, 2)
    
    def normalize_score(self, score, min_score=300, max_score=850):
        """
        Normaliza el puntaje al rango deseado (300-850).
        """
        current_min, current_max = 200, 1000
        normalized_score = min_score + (score - current_min) * (max_score - min_score) / (current_max - current_min)
        return round(max(min(normalized_score, max_score), min_score), 2)
    
    def get_score(self, json_data):
        """
        Preprocesa los datos, calcula las probabilidades y obtiene el puntaje final.
        """
        df = self.validate_and_convert_data(json_data)
        data = self.scaler.transform(df)
        probabilities = self.calculate_probabilities(data)
        raw_score = self.calculate_score(probabilities)
        final_score = self.normalize_score(raw_score)
        
        result = {"score": float(final_score), **{k: round(float(v), 3) for k, v in probabilities.items()}}
        
        return json.dumps(result)

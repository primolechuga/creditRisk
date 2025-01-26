from datetime import datetime
import json
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from dateutil.relativedelta import relativedelta
from tensorflow.keras.models import load_model

class LoanService:

    def __init__(self):
        
        # Parámetros del modelo
        self.model = load_model('app/services/modelo.h5')
        self.grade_classes = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}
        self.verification_status_classes = {'Not Verified': 0, 'Source Verified': 1, 'Verified': 2}
        self.home_ownership_classes = {'ANY': 0, 'MORTAGE': 1, 'NONE': 2, 'OTHER': 3, 'OWN': 4, 'RENT': 5}
        self.purpose_classes = {'car': 0, 'credit_card': 1, 'debt_consolidation': 2, 'educational': 3, 'home_improvement': 4, 'house': 5, 'major_purchase': 6, 'medical': 7, 'moving': 8, 'other': 9, 'renewable_energy': 10, 'small_business': 11, 'vacation': 12, 'wedding': 13}
        self.scaler = StandardScaler()
        self.weights = {"Pagado": 3, "En proceso": 2, "Incumplido": -3}  # Pesos por clase
    

    def preprocess_data(self, data):
        
        # Convertir variables categóricas a numéricas
        data["grade"] = self.grade_classes[data["grade"]]
        data["verification_status"] = self.verification_status_classes[data["verification_status"]]
        data["home_ownership"] = self.home_ownership_classes[data["home_ownership"]]
        data["purpose"] = self.purpose_classes[data["purpose"]]

        ref_date = datetime(2025, 1, 21)

        def preprocess_date(fecha_inicio, fecha_fin):
            diferencia = relativedelta(fecha_fin, fecha_inicio)
            return diferencia.years * 12 + diferencia.months

        # Convertir las fechas en el diccionario de datos
        data["mths_since_issue_d"] = preprocess_date(datetime.strptime(data["mths_since_issue_d"], "%Y-%m-%dT%H:%M:%S.%fZ"), ref_date)
        data["mths_since_last_pymnt_d"] = preprocess_date(datetime.strptime(data["mths_since_last_pymnt_d"], "%Y-%m-%dT%H:%M:%S.%fZ"), ref_date)
        data["mths_since_last_credit_pull_d"] = preprocess_date(datetime.strptime(data["mths_since_last_credit_pull_d"], "%Y-%m-%dT%H:%M:%S.%fZ"), ref_date)
        

        #Crear un arreglo de numpy con los datos
        data = np.array(list(data.values())).reshape(1, -1)

        #Escalado de los datos
        feature_array = self.scaler.fit_transform(data)

        return feature_array
    

    def calculate_probabilities(self, data):
        # Calcular probabilidades de predicción
        probabilities = self.model.predict(data)
        
        # Definir nombres de clases
        class_names = ["Pagado", "En proceso", "Incumplido"]
        
        # Extraer probabilidades por clase
        class_probabilities = {class_name: prob for class_name, prob in zip(class_names, probabilities[0])}

        return class_probabilities
    
    
    # Función para calcular el puntaje basado en probabilidades
    
    def calculate_score(self, probabilities, offset=600, factor=50):
        
        # Asegurar que las probabilidades sumen 1
        total_prob = sum(probabilities.values())
        normalized_probs = {k: v / total_prob for k, v in probabilities.items()}
        
        # Score general combinando las clases con sus pesos
        score = offset
        for class_name, prob in normalized_probs.items():
            weight = self.weights.get(class_name, 0)  # Default weight is 0 if not defined
            score += weight * factor * np.log(prob + 1e-6)  # Evitar log(0) con un epsilon
        
        return round(score, 0)
    
    
    def get_score(self, json_data):
        
        # Preprocesar los datos
        data = self.preprocess_data(json_data)
        
        # Calcular probabilidades de predicción
        probabilities = self.calculate_probabilities(data)
        
        # Calcular el puntaje
        score = self.calculate_score(probabilities)
        

        result = {
            "pagado": round(float(probabilities["Pagado"]), 1),
            "en_proceso": round(float(probabilities["En proceso"]), 1),
            "incumplido": round(float(probabilities["Incumplido"]), 1),
            "score": round(float(score), 1)
        }

        return json.dumps(result)
    






    
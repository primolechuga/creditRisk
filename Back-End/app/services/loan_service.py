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
        try:
            self.model = tf.keras.models.load_model('app/services/model.keras')
            self.scaler = joblib.load('app/services/scaler.pkl')
            
            # Obtener los nombres de las características del escalador
            # Esto es crucial para mantener el mismo orden de características
            self.feature_names = self.scaler.feature_names_in_
            
            print(f"Características del escalador: {self.feature_names}")
        except Exception as e:
            print(f"Error al cargar modelo o escalador: {str(e)}")
            # Si no podemos obtener las características del escalador, usamos las importantes
            self.feature_names = [
                'total_rev_hi_lim', 'tot_cur_bal', 'dti', 'revol_bal', 
                'months_until_2025', 'annual_inc', 'total_acc', 'open_acc',
                'mths_since_last_delinq', 'emp_length', 'mths_since_last_major_derog',
                'inq_last_6mths', 'mths_since_last_record', 'delinq_2yrs', 'pub_rec',
                'verification_status', 'home_ownership_MORTGAGE', 'home_ownership_RENT',
                'purpose_credit_card', 'purpose_debt_consolidation'
            ]
        
        # Definir clases del modelo (10 clases)
        self.class_names = [
            "Current", 
            "Fully Paid", 
            "Charged Off", 
            "Late (31-120)", 
            "In Grace Period", 
            "Late (16-30)", 
            "Issued", 
            "Default", 
            "Does not meet the credit policy. Status: Fully Paid", 
            "Does not meet the credit policy. Status: Charged Off"
        ]
        
        # Pesos para cada clase (ajustados para 10 clases)
        self.weights = {
            "Current": 2,
            "Fully Paid": 3,
            "Charged Off": -3,
            "Late (31-120)": -2,
            "In Grace Period": 0,
            "Late (16-30)": -1,
            "Issued": 1,
            "Default": -4,
            "Does not meet the credit policy. Status: Fully Paid": 0,
            "Does not meet the credit policy. Status: Charged Off": -2
        }
        
        # Parámetros del scorecard
        self.OFFSET = 600
        self.FACTOR = 12
    
    def preprocess_data(self, data):
        """
        Realiza el preprocesamiento necesario en los datos de entrada.
        """
        # Convertir a DataFrame si es un diccionario
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = pd.DataFrame(data)
        
        # Debugging
        print(f"Columnas en los datos de entrada: {df.columns.tolist()}")
            
        # Calcular meses hasta 2025 si no está presente
        if 'months_until_2025' not in df.columns:
            current_date = datetime.now()
            target_date = datetime(2025, 12, 31)
            df['months_until_2025'] = (target_date.year - current_date.year) * 12 + target_date.month - current_date.month
        
        # Rellenar valores nulos
        # Para variables de tiempo desde eventos negativos, usar 999 (nunca ocurrido)
        time_columns = ['mths_since_last_delinq', 'mths_since_last_record', 'mths_since_last_major_derog']
        for col in time_columns:
            if col in df.columns:
                df[col] = df[col].fillna(999)
        
        # Para variables numéricas estándar, rellenar con 0
        numeric_columns = [
            'total_rev_hi_lim', 'tot_cur_bal', 'dti', 'revol_bal', 'annual_inc', 
            'total_acc', 'open_acc', 'emp_length', 'inq_last_6mths', 
            'delinq_2yrs', 'pub_rec'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Verificar si verification_status es categórico y convertirlo a numérico
        if 'verification_status' in df.columns and df['verification_status'].dtype == 'object':
            verification_mapping = {
                'Not Verified': 0,
                'Source Verified': 1,
                'Verified': 2
            }
            df['verification_status'] = df['verification_status'].map(verification_mapping)
        
        # Procesar home_ownership si no está en formato one-hot
        if 'home_ownership' in df.columns:
            # Crear variables dummy
            mortgage_mask = df['home_ownership'] == 'MORTGAGE'
            rent_mask = df['home_ownership'] == 'RENT'
            
            df['home_ownership_MORTGAGE'] = mortgage_mask.astype(int)
            df['home_ownership_RENT'] = rent_mask.astype(int)
        
        # Procesar purpose si no está en formato one-hot
        if 'purpose' in df.columns:
            # Crear variables dummy
            credit_card_mask = df['purpose'] == 'credit_card'
            debt_cons_mask = df['purpose'] == 'debt_consolidation'
            
            df['purpose_credit_card'] = credit_card_mask.astype(int)
            df['purpose_debt_consolidation'] = debt_cons_mask.astype(int)
        
        # Asegurar que todas las características del escalador estén presentes
        for feature in self.feature_names:
            if feature not in df.columns:
                print(f"Característica faltante: {feature} - añadiendo con valor 0")
                df[feature] = 0
        
        # Seleccionar solo las características en el mismo orden que el escalador
        # Esto es crítico para evitar el error "Feature names must be in the same order as they were in fit"
        df_selected = df[self.feature_names]
        
        print(f"Columnas después del preprocesamiento: {df_selected.columns.tolist()}")
        return df_selected
    
    def calculate_probabilities(self, data):
        """
        Calcula las probabilidades predichas por el modelo para cada clase.
        """
        try:
            # Escalar los datos
            scaled_data = self.scaler.transform(data)
            print(f"Datos escalados: {scaled_data}")
            
            # Predecir probabilidades
            probabilities = self.model.predict(scaled_data)
            
            # Mapear a diccionario de clases
            return {class_name: float(prob) for class_name, prob in zip(self.class_names, probabilities[0])}
        except Exception as e:
            print(f"Error en calculate_probabilities: {str(e)}")
            # En caso de error, devolver probabilidades uniformes
            return {class_name: 1.0/len(self.class_names) for class_name in self.class_names}
    
    def calculate_score(self, probabilities):
        """
        Calcula un puntaje general a partir de las probabilidades y los pesos.
        """
        # Normalizar probabilidades
        total_prob = sum(probabilities.values())
        normalized_probs = {k: v / total_prob for k, v in probabilities.items()}
        
        # Calcular score base
        score = self.OFFSET
        
        # Añadir contribución ponderada de cada clase
        for class_name, prob in normalized_probs.items():
            weight = self.weights.get(class_name, 0)
            score += weight * self.FACTOR * (prob ** 1.5)  # Potencia para amplificar diferencias pequeñas  
        
        # Ajustar para evitar valores extremos
        return round(min(max(score, 300), 850), 2)
    
    def get_risk_category(self, score):
        """
        Determina la categoría de riesgo basada en el puntaje crediticio.
        """
        if score >= 750:
            return "Riesgo Muy Bajo"
        elif score >= 700:
            return "Riesgo Bajo"
        elif score >= 650:
            return "Riesgo Moderado"
        elif score >= 600:
            return "Riesgo Alto"
        else:
            return "Riesgo Muy Alto"
    
    def predict(self, json_data):
        """
        Método principal que procesa los datos, predice la clase y calcula el puntaje.
        """
        try:
            # Preprocesar datos
            if isinstance(json_data, str):
                data = json.loads(json_data)
            else:
                data = json_data
            
            print(f"Datos recibidos: {data}")
                
            # Preprocesar
            preprocessed_data = self.preprocess_data(data)
            
            # Calcular probabilidades
            probabilities = self.calculate_probabilities(preprocessed_data)
            
            # Calcular puntaje
            score = self.calculate_score(probabilities)
            
            # Determinar categoría de riesgo
            risk_category = self.get_risk_category(score)
            
            # Encontrar la clase más probable
            most_likely_class = max(probabilities.items(), key=lambda x: x[1])[0]
            
            # Preparar resultado
            result = {
                "score": float(score),
                "risk_category": risk_category,
                "most_likely_class": most_likely_class,
                "probabilities": {k: round(float(v), 4) for k, v in probabilities.items()}
            }
            
            return json.dumps(result)
            
        except Exception as e:
            error_message = f"Error en la predicción: {str(e)}"
            print(f"Error detallado: {error_message}")
            import traceback
            traceback.print_exc()
            return json.dumps({"error": error_message})

# # Ejemplo de uso
# if __name__ == "__main__":
#     # Datos de prueba basados en las características importantes
#     test_data = {
#         "total_rev_hi_lim": 120000,
#         "tot_cur_bal": 80000,
#         "dti": 18.5,
#         "revol_bal": 15000,
#         "months_until_2025": 3,
#         "annual_inc": 95000,
#         "total_acc": 12,
#         "open_acc": 5,
#         "mths_since_last_delinq": 48,
#         "emp_length": 10,
#         "mths_since_last_major_derog": 62,
#         "inq_last_6mths": 1,
#         "mths_since_last_record": 999,
#         "delinq_2yrs": 0,
#         "pub_rec": 0,
#         "verification_status": 2,  # Verified
#         "home_ownership_MORTGAGE": 1,
#         "home_ownership_RENT": 0,
#         "purpose_credit_card": 0,
#         "purpose_debt_consolidation": 1
#     }
    
#     # Instanciar servicio y hacer predicción
#     loan_service = LoanService()
#     result = loan_service.predict(test_data)
#     print(result)
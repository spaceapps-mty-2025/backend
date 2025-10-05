# app.py (Versión Final, Sincronizada y Robusta)

from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from functools import wraps

# --- 1. Configuración Inicial y Carga de Artefactos ---
app = Flask(__name__)
API_KEY = os.environ.get('API_KEY', 'WXviSp$hK8') # Clave por defecto

try:
    model = joblib.load('exoplanet_model.joblib')
    scaler = joblib.load('data_scaler.joblib')
    feature_means = joblib.load('feature_means.joblib')
    
    # ¡MEJORA CLAVE! El orden de las columnas se define automáticamente desde el artefacto.
    # Esto garantiza una sincronización perfecta con el modelo entrenado.
    FINAL_COLUMN_ORDER = list(feature_means.keys())
    
    print("✅ Modelo, escalador y promedios cargados correctamente.")
    print(f"✅ El modelo espera {len(FINAL_COLUMN_ORDER)} características.")

except FileNotFoundError:
    print("❌ Error: No se encontraron los archivos .joblib. Asegúrate de que estén en la misma carpeta.")
    model, scaler, feature_means, FINAL_COLUMN_ORDER = None, None, None, []

# --- Decorador para la API Key ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-Key') and request.headers.get('X-API-Key') == API_KEY:
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "No autorizado. Proporciona una API Key válida en la cabecera 'X-API-Key'."}), 401
    return decorated_function

# --- 2. Definición del "Contrato" de la API ---
MANDATORY_FEATURES = [
    'period', 'duration', 'transit_depth', 'planet_radius', 'eq_temp', 'insol_flux',
    'stellar_eff_temp', 'stellar_logg', 'stellar_radius'
]

# --- 3. API Endpoint para Predicciones ---
@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    if not all([model, scaler, feature_means]):
        return jsonify({"error": "El modelo no está disponible. Revisa los logs del servidor."}), 500

    data = request.get_json()
    if not data:
        return jsonify({"error": "Cuerpo de la solicitud vacío o no es JSON válido."}), 400

    missing_features = [f for f in MANDATORY_FEATURES if f not in data]
    if missing_features:
        return jsonify({"error": f"Faltan campos obligatorios: {', '.join(missing_features)}"}), 400

    try:
        df = pd.DataFrame([data])
        
        # Rellenar campos opcionales faltantes con los promedios del entrenamiento
        for col in FINAL_COLUMN_ORDER:
            if col not in df.columns:
                df[col] = feature_means.get(col, 0)

        # Re-crear las características de ingeniería
        df['snr_per_srad'] = df['koi_model_snr'] / df['stellar_radius']
        df['depth_per_srad_sq'] = df['transit_depth'] / (df['stellar_radius'] ** 2)
        
        # Asegurar el orden final de las columnas
        df = df[FINAL_COLUMN_ORDER]
        
        data_scaled = scaler.transform(df)
        prediction_code = model.predict(data_scaled)[0]
        probabilities = model.predict_proba(data_scaled)[0]
        
        label_map = {0: 'Confirmado', 1: 'Candidato'}
        predicted_label = label_map.get(prediction_code, "Desconocido")
        confidence = probabilities[prediction_code]

        response = {
            "predicted_label": predicted_label,
            "confidence": f"{confidence:.2%}",
            "probabilities": {
                "Confirmado": f"{probabilities[0]:.2%}",
                "Candidato": f"{probabilities[1]:.2%}"
            }
        }
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": f"Error durante la predicción: {str(e)}"}), 500

# --- 4. Iniciar el Servidor ---
if __name__ == '__main__':
    print("🚀 Iniciando servidor de predicción en http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
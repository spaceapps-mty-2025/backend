from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
from functools import wraps

# --- 1. Configuración Inicial y Carga de Artefactos ---
app = Flask(__name__)

# Lee la API Key desde una variable de entorno para seguridad.
# Le pondremos un valor por defecto solo para pruebas locales.
API_KEY = os.environ.get('API_KEY', 'WXviSp$hK8')

try:
    model = joblib.load('exoplanet_model.joblib')
    scaler = joblib.load('data_scaler.joblib')
    feature_means = joblib.load('feature_means.joblib')
    print("✅ Modelo, escalador y promedios de características cargados correctamente.")
except FileNotFoundError:
    print("❌ Error: Asegúrate de que los 3 archivos .joblib estén en la misma carpeta.")
    model, scaler, feature_means = None, None, None

# --- Decorador para requerir la API Key ---
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Buscamos la clave en la cabecera 'X-API-Key'
        if request.headers.get('X-API-Key') and request.headers.get('X-API-Key') == API_KEY:
            return f(*args, **kwargs)
        else:
            return jsonify({"error": "No autorizado. Proporciona una API Key válida en la cabecera 'X-API-Key'."}), 401
    return decorated_function

# --- 2. Definición del "Contrato" de la API ---
MANDATORY_FEATURES = [
    'period', 'duration', 'transit_depth', 'planet_radius', 'eq_temp', 'insol_flux',
    'stellar_eff_temp', 'stellar_logg', 'stellar_radius', 'ra', 'dec'
]
FINAL_COLUMN_ORDER = [
    'koi_fpflag_nt', 'koi_fpflag_ss', 'koi_fpflag_co', 'koi_fpflag_ec', 'period',
    'koi_time0bk', 'koi_impact', 'duration', 'transit_depth', 'planet_radius',
    'eq_temp', 'insol_flux', 'koi_model_snr', 'koi_tce_plnt_num', 'stellar_eff_temp',
    'stellar_logg', 'stellar_radius', 'ra', 'dec', 'koi_kepmag',
    'st_pmra', 'st_pmdec', 'transit_midpoint', 'tess_mag', 'stellar_dist',
    'snr_per_srad', 'depth_per_srad_sq'
]

# --- 3. API Endpoint para Predicciones (ahora protegido) ---
@app.route('/predict', methods=['POST'])
@require_api_key  # <-- ¡Aquí aplicamos la seguridad!
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
        for col in FINAL_COLUMN_ORDER:
            if col not in df.columns:
                df[col] = feature_means.get(col, 0)

        df['snr_per_srad'] = df['koi_model_snr'] / df['stellar_radius']
        df['depth_per_srad_sq'] = df['transit_depth'] / (df['stellar_radius'] ** 2)
        
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

# --- 4. Iniciar el Servidor (Solo para pruebas locales) ---
# El proveedor cloud usará un servidor de producción (Gunicorn), no este.
if __name__ == '__main__':
    print("🚀 Iniciando servidor de predicción en http://127.0.0.1:5000")
    app.run(port=5000, debug=True)

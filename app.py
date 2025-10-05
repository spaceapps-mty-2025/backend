# app.py (Versión Final, Sincronizada y Robusta)

from flask import Flask, request, jsonify, url_for
import pandas as pd
import joblib
import os
from functools import wraps
import lightkurve as lk
from lightkurve import search_targetpixelfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend sin interfaz gráfica
import matplotlib.pyplot as plt
from datetime import datetime
import json

# --- 1. Configuración Inicial y Carga de Artefactos ---
app = Flask(__name__)
API_KEY = os.environ.get('API_KEY', 'WXviSp$hK8') # Clave por defecto

# Crear carpeta para imágenes si no existe
IMAGES_FOLDER = 'static/images'
os.makedirs(IMAGES_FOLDER, exist_ok=True)

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

# --- 3.2. API Endpoint para Análisis de Estrellas y Generación de Imágenes ---
@app.route('/analyze-star/<star_id>', methods=['GET'])
@require_api_key
def analyze_star(star_id):
    """
    Analiza una estrella por su ID (ej: 6922244 para KIC 6922244)
    y genera un gráfico de la curva de luz plegada.
    Si ya existe una imagen para este ID, la devuelve sin regenerar.
    """

    if not star_id or not star_id.strip():
        return jsonify({"image_url": None, "message": "ID de estrella vacío"}), 400

    # Verificar si ya existe una imagen para este ID
    existing_filename = f"kic_{star_id}.png"
    existing_filepath = os.path.join(IMAGES_FOLDER, existing_filename)
    metadata_filepath = os.path.join(IMAGES_FOLDER, f"kic_{star_id}_metadata.json")

    if os.path.exists(existing_filepath) and os.path.exists(metadata_filepath):
        print(f"♻️  Imagen ya existe para KIC {star_id}, reutilizando...")
        image_url = url_for('static', filename=f'images/{existing_filename}', _external=True)

        # Cargar parámetros guardados
        with open(metadata_filepath, 'r') as f:
            metadata = json.load(f)

        return jsonify({
            "image_url": image_url,
            "message": "Imagen existente recuperada (no se regeneró)",
            "star_id": f"KIC {star_id}",
            "parameters": metadata.get("parameters", {}),
            "cached": True
        }), 200

    try:
        # 1. Buscar el archivo de píxeles
        print(f"🔍 Buscando datos para KIC {star_id}...")
        search_result = search_targetpixelfile(
            f'KIC {star_id}',
            author="Kepler",
            cadence="long",
            quarter=4
        )

        if len(search_result) == 0:
            return jsonify({
                "image_url": None,
                "message": f"No se encontraron datos para KIC {star_id}"
            }), 404

        # 2. Descargar el archivo de píxeles
        print(f"📥 Descargando datos...")
        pixelFile = search_result.download()

        if pixelFile is None:
            return jsonify({
                "image_url": None,
                "message": "Error al descargar datos de la estrella"
            }), 500

        # 3. Convertir a curva de luz
        print(f"📊 Generando curva de luz...")
        lc = pixelFile.to_lightcurve(aperture_mask=pixelFile.pipeline_mask)

        # 4. Aplanar la curva
        flat_lc = lc.flatten()

        # 5. Generar periodograma BLS
        print(f"🔬 Analizando periodicidad...")
        period = np.linspace(1, 5, 20)
        bls = lc.to_periodogram(method='bls', period=period, frequency_factor=500)

        # 6. Extraer parámetros del planeta
        planet_x_period = bls.period_at_max_power
        planet_x_t0 = bls.transit_time_at_max_power
        planet_x_dur = bls.duration_at_max_power

        # 7. Generar la imagen final (curva de luz plegada)
        print(f"🎨 Generando imagen...")
        fig, ax = plt.subplots(figsize=(10, 6))
        lc.fold(period=planet_x_period, epoch_time=planet_x_t0).scatter(ax=ax)
        ax.set_xlim(-2, 2)
        ax.set_title(f'Curva de Luz Plegada - KIC {star_id}')
        ax.set_xlabel('Tiempo desde tránsito [días]')
        ax.set_ylabel('Flujo normalizado')

        # 8. Guardar la imagen (sin timestamp para evitar duplicados)
        plt.tight_layout()
        plt.savefig(existing_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # 9. Guardar metadatos en JSON
        parameters = {
            "period": float(planet_x_period.value),
            "transit_time": float(planet_x_t0.value),
            "duration": float(planet_x_dur.value)
        }

        metadata = {
            "star_id": f"KIC {star_id}",
            "parameters": parameters,
            "generated_at": datetime.now().isoformat()
        }

        with open(metadata_filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        # 10. Generar URL de la imagen
        image_url = url_for('static', filename=f'images/{existing_filename}', _external=True)

        print(f"✅ Imagen generada exitosamente: {existing_filename}")

        return jsonify({
            "image_url": image_url,
            "message": "Análisis completado exitosamente",
            "star_id": f"KIC {star_id}",
            "parameters": parameters,
            "cached": False
        }), 200

    except ValueError as ve:
        print(f"❌ Error de validación: {str(ve)}")
        return jsonify({
            "image_url": None,
            "message": f"ID de estrella inválido: {str(ve)}"
        }), 400

    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")
        return jsonify({
            "image_url": None,
            "message": "Error al procesar la estrella. Verifica que el ID sea válido."
        }), 500

# --- 4. Iniciar el Servidor ---
if __name__ == '__main__':
    print("🚀 Iniciando servidor de predicción en http://127.0.0.1:5000")
    app.run(port=5000, debug=False)
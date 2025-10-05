# SCRIPT FINAL-FINAL PARA KAGGLE (ENTRENAR Y GUARDAR)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import numpy as np

# --- PASO 1: Carga de Ambos Datasets ---
koi_df = pd.read_csv('/kaggle/input/koi-dataset/cumulative_2025.10.04_15.09.17.csv', skiprows=31, engine='python')
tess_df = pd.read_csv('/kaggle/input/tess-dataset/TOI_2025.10.04_20.11.03.csv', skiprows=27)

# --- PASO 2: Estandarización y Unión ---
# ... (la sección de mapeo de columnas y target es idéntica a la anterior)
koi_map = {
    'koi_period': 'period', 'koi_duration': 'duration', 'koi_depth': 'transit_depth',
    'koi_prad': 'planet_radius', 'koi_teq': 'eq_temp', 'koi_insol': 'insol_flux',
    'koi_steff': 'stellar_eff_temp', 'koi_slogg': 'stellar_logg', 'koi_srad': 'stellar_radius'
}
koi_df.rename(columns=koi_map, inplace=True)
koi_df = koi_df[koi_df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])].copy()
koi_df['disposition'] = koi_df['koi_disposition'].map({'CONFIRMED': 0, 'CANDIDATE': 1})

tess_map = {
    'pl_orbper': 'period', 'pl_trandurh': 'duration', 'pl_trandep': 'transit_depth',
    'pl_rade': 'planet_radius', 'pl_eqt': 'eq_temp', 'pl_insol': 'insol_flux',
    'st_teff': 'stellar_eff_temp', 'st_logg': 'stellar_logg', 'st_rad': 'stellar_radius',
    'pl_tranmid': 'transit_midpoint', 'st_dist': 'stellar_dist', 'st_tmag': 'tess_mag'
}
tess_df.rename(columns=tess_map, inplace=True)
tess_df = tess_df[tess_df['tfopwg_disp'].isin(['CP', 'PC'])].copy()
tess_df['disposition'] = tess_df['tfopwg_disp'].map({'CP': 0, 'PC': 1})

combined_df = pd.concat([koi_df, tess_df], ignore_index=True, sort=False)

# --- ¡CAMBIO IMPORTANTE AQUÍ! ---
# Se eliminan TODAS las columnas no deseadas del DataFrame combinado
cols_to_drop_final = [
    'kepid', 'koi_score', 'kepoi_name', 'kepler_name', 'koi_disposition',
    'koi_pdisposition', 'tfopwg_disp', 'rowid', 'koi_teq_err1', 'koi_teq_err2',
    'tid', 'toi' # IDs de TESS que tampoco son características
]
# Eliminamos solo las que existen para evitar errores
existing_cols_to_drop = [col for col in cols_to_drop_final if col in combined_df.columns]
combined_df.drop(columns=existing_cols_to_drop, inplace=True)

# --- El resto del script continúa ---
combined_df['snr_per_srad'] = combined_df['koi_model_snr'] / combined_df['stellar_radius']
combined_df['depth_per_srad_sq'] = combined_df['transit_depth'] / (combined_df['stellar_radius'] ** 2)

X = combined_df.select_dtypes(include=np.number).drop(columns=['disposition'], errors='ignore')
y = combined_df['disposition'].dropna()
X = X.loc[y.index]

feature_means = X.mean().to_dict()
X.fillna(feature_means, inplace=True)

# --- PASO 3: Entrenamiento (con 100% de los datos) ---
print(f"Entrenando modelo final con {X.shape[1]} características.")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scale_pos_weight_value = (y == 0).sum() / (y == 1).sum()
model = xgb.XGBClassifier(
    objective='binary:logistic', eval_metric='logloss', n_estimators=500,
    learning_rate=0.05, max_depth=5, scale_pos_weight=scale_pos_weight_value,
    random_state=42, use_label_encoder=False
)
model.fit(X_scaled, y)
print("✅ ¡Modelo final entrenado correctamente!")

# --- PASO 4: Guardar los Artefactos Finales ---
joblib.dump(model, 'exoplanet_model.joblib')
joblib.dump(scaler, 'data_scaler.joblib')
joblib.dump(feature_means, 'feature_means.joblib')
print("✅ Artefactos finales de producción guardados.")
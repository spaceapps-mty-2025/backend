# SCRIPT DE ENTRENAMIENTO FINAL (AJUSTADO)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import numpy as np

# --- PASO 1: Carga de Ambos Datasets ---
# Asegúrate de que las rutas a tus archivos son correctas
koi_df = pd.read_csv('/kaggle/input/kepler/cumulative_2025.10.05_08.30.28.csv', skiprows=53, engine='python')
archive_df = pd.read_csv('/kaggle/input/kepler/k2pandc_2025.10.05_08.30.33.csv', skiprows=98, engine='python')

# --- PASO 2: Estandarización y Unión ---
# Mapeo para KOI
koi_map = {
    'koi_period': 'period', 'koi_prad': 'planet_radius', 'koi_insol': 'insol_flux', 'koi_teq': 'eq_temp',
    'koi_steff': 'stellar_eff_temp', 'koi_slogg': 'stellar_logg', 'koi_srad': 'stellar_radius',
    'koi_kepmag': 'kepmag', 'koi_impact': 'impact', 'koi_duration': 'duration', 'koi_depth': 'transit_depth'
}
koi_df.rename(columns=koi_map, inplace=True)
koi_df = koi_df[koi_df['koi_disposition'].isin(['CONFIRMED', 'CANDIDATE'])].copy()
koi_df['disposition'] = koi_df['koi_disposition'].map({'CONFIRMED': 0, 'CANDIDATE': 1})

# Mapeo para el segundo dataset de Kepler/Archive
archive_map = {
    'pl_orbper': 'period', 'pl_rade': 'planet_radius', 'pl_insol': 'insol_flux', 'pl_eqt': 'eq_temp',
    'st_teff': 'stellar_eff_temp', 'st_logg': 'stellar_logg', 'st_rad': 'stellar_radius', 'sy_kepmag': 'kepmag'
}
archive_df.rename(columns=archive_map, inplace=True)
archive_df = archive_df[archive_df['disposition'].isin(['CONFIRMED', 'CANDIDATE'])].copy()
archive_df['disposition'] = archive_df['disposition'].map({'CONFIRMED': 0, 'CANDIDATE': 1})

# Combinar
combined_df = pd.concat([koi_df, archive_df], ignore_index=True, sort=False)

# --- AJUSTE: Lista de columnas a eliminar actualizada ---
cols_to_drop = [
    # IDs y Nombres
    'kepid', 'kepoi_name', 'kepler_name', 'pl_name', 'hostname',
    # Columnas de target originales y texto
    'koi_disposition', 'default_flag', 'disp_refname', 'discoverymethod', 'soltype', 'pl_controv_flag',
    'pl_refname', 'pl_bmassprov', 'ttv_flag', 'st_refname', 'st_spectype',
    'sy_refname', 'rastr', 'decstr', 'rowupdate', 'pl_pubdate', 'releasedate',
    # Columnas eliminadas por decisión del usuario
    'koi_tce_plnt_num', 'st_pmra', 'st_pmdec',
    # Columnas de error y límites para prevenir data leakage
    'pl_orbpererr1', 'pl_orbpererr2', 'pl_orbperlim', 'pl_orbsmaxerr1', 'pl_orbsmaxerr2', 'pl_orbsmaxlim',
    'pl_radeerr1', 'pl_radeerr2', 'pl_radelim', 'pl_radjerr1', 'pl_radjerr2', 'pl_radjlim',
    'pl_bmasseerr1', 'pl_bmasseerr2', 'pl_bmasselim', 'pl_bmassjerr1', 'pl_bmassjerr2', 'pl_bmassjlim',
    'pl_orbeccenerr1', 'pl_orbeccenerr2', 'pl_orbeccenlim', 'pl_insolerr1', 'pl_insolerr2', 'pl_insollim',
    'pl_eqterr1', 'pl_eqterr2', 'pl_eqtlim', 'st_tefferr1', 'st_tefferr2', 'st_tefflim', 'st_raderr1',
    'st_raderr2', 'st_radlim', 'st_masserr1', 'st_masserr2', 'st_masslim', 'st_meterr1', 'st_meterr2',
    'st_metlim', 'st_loggerr1', 'st_loggerr2', 'st_logglim', 'sy_disterr1', 'sy_disterr2', 'sy_vmagerr1',
    'sy_vmagerr2', 'sy_kmagerr1', 'sy_kmagerr2', 'sy_gaiamagerr1', 'sy_gaiamagerr2'
]

existing_cols_to_drop = [col for col in cols_to_drop if col in combined_df.columns]
combined_df.drop(columns=existing_cols_to_drop, inplace=True)

# --- Ingeniería de características ---
combined_df['snr_per_srad'] = combined_df['koi_model_snr'] / combined_df['stellar_radius']
combined_df['depth_per_srad_sq'] = combined_df['transit_depth'] / (combined_df['stellar_radius'] ** 2)

# --- Preprocesamiento Final ---
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

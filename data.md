# Documentación de Parámetros del Modelo de Detección de Exoplanetas

Este documento define el contrato de datos completo para la API de predicción. Describe todos los parámetros que el modelo fue entrenado para recibir, especificando cuáles son obligatorios y cuáles son opcionales.

---

##  obligatorioCampos Obligatorios

Los siguientes **9 campos son requeridos** en cada solicitud a la API. Corresponden a las características fundamentales que ambos datasets (KOI y TESS) tienen en común.

| Parámetro Unificado (API) | Nombre Original (KOI) | Nombre Original (TESS) | Descripción Breve |
| :---------------------- | :-------------------- | :--------------------- | :---------------- |
| `period` | `koi_period` | `pl_orbper` | Período Orbital del planeta (en días). |
| `duration` | `koi_duration` | `pl_trandurh` | Duración del tránsito (en horas). |
| `transit_depth` | `koi_depth` | `pl_trandep` | Profundidad del tránsito (en partes por millón, ppm). |
| `planet_radius` | `koi_prad` | `pl_rade` | Radio del planeta (en radios terrestres). |
| `eq_temp` | `koi_teq` | `pl_eqt` | Temperatura de Equilibrio del planeta (en Kelvin). |
| `insol_flux` | `koi_insol` | `pl_insol` | Flujo de insolación que recibe el planeta. |
| `stellar_eff_temp` | `koi_steff` | `st_teff` | Temperatura de la superficie de la estrella (en Kelvin). |
| `stellar_logg` | `koi_slogg` | `st_logg` | Gravedad en la superficie de la estrella (log10 cm/s²). |
| `stellar_radius` | `koi_srad` | `st_rad` | Radio de la estrella (en radios solares). |

---

## opcionalCampos Opcionales

Estos campos proporcionan información adicional que puede mejorar la precisión. Si no se envían, la API los rellenará con un valor promedio y la predicción seguirá funcionando.

### Coordenadas (Compartidas)

| Parámetro Unificado (API) | Nombre Original (KOI) | Nombre Original (TESS) | Descripción Breve |
| :---------------------- | :-------------------- | :--------------------- | :---------------- |
| `ra` | `ra` | `ra` | Coordenada celestial de Ascensión Recta (en grados). |
| `dec` | `dec` | `dec` | Coordenada celestial de Declinación (en grados). |

### Específicos del Dataset KOI

| Parámetro Unificado (API) | Nombre Original (KOI) | Nombre Original (TESS) | Descripción Breve |
| :---------------------- | :-------------------- | :--------------------- | :---------------- |
| `koi_fpflag_nt` | `koi_fpflag_nt` | N/A | Flag: Señal no similar a tránsito (0=No, 1=Sí). |
| `koi_fpflag_ss` | `koi_fpflag_ss` | N/A | Flag: Indica posible eclipse estelar (0=No, 1=Sí). |
| `koi_fpflag_co` | `koi_fpflag_co` | N/A | Flag: Indica centroide de luz desplazado (0=No, 1=Sí). |
| `koi_fpflag_ec` | `koi_fpflag_ec` | N/A | Flag: Indica contaminación por efemérides (0=No, 1=Sí). |
| `koi_time0bk` | `koi_time0bk` | N/A | Época del tránsito (en BJD). |
| `koi_impact` | `koi_impact` | N/A | Parámetro de impacto del tránsito (0 = centrado). |
| `koi_model_snr` | `koi_model_snr` | N/A | Relación señal a ruido (SNR) de la detección. |
| `koi_kepmag` | `koi_kepmag` | N/A | Brillo de la estrella en la banda del telescopio Kepler. |

### Específicos del Dataset TESS

| Parámetro Unificado (API) | Nombre Original (KOI) | Nombre Original (TESS) | Descripción Breve |
| :---------------------- | :-------------------- | :--------------------- | :---------------- |
| `transit_midpoint` | N/A | `pl_tranmid` | Punto medio del tránsito (en BJD). |
| `tess_mag` | N/A | `st_tmag` | Brillo de la estrella en la banda del telescopio TESS. |
| `stellar_dist` | N/A | `st_dist` | Distancia al sistema estelar (en parsecs). |
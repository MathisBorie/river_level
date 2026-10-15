import time, json, datetime as dt, requests, numpy as np
URL = "https://archive-api.open-meteo.com/v1/archive"
H = ["temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature", "precipitation", "rain",
     "snowfall", "snow_depth", "weather_code", "pressure_msl", "surface_pressure", "cloud_cover", "cloud_cover_low",
     "cloud_cover_mid", "cloud_cover_high", "et0_fao_evapotranspiration", "vapour_pressure_deficit", "wind_speed_10m",
     "wind_speed_100m", "wind_direction_10m", "wind_gusts_10m", "shortwave_radiation", "soil_temperature_0_to_7cm",
     "soil_moisture_0_to_7cm"]
dernier = [0.0]
def req(n_pts, debut="2023-06-01", fin="2023-06-01", horaires=(), journalieres=()):
    lat = np.round(np.linspace(44.0, 45.0, n_pts), 4) if n_pts > 1 else [44.5]
    p = {"latitude": list(lat), "longitude": [6.5] * n_pts, "start_date": debut, "end_date": fin}
    if horaires: p["hourly"] = list(horaires)
    if journalieres: p["daily"] = list(journalieres)
    r = requests.post(URL, data=p, timeout=120); dernier[0] = time.time()
    return r.status_code, ("" if r.status_code == 200 else (r.text[:120] or "réponse vide"))
def attendre_minute():
    while time.time() - dernier[0] < 63: time.sleep(1)
    while dt.datetime.now().second > 2: time.sleep(0.5)
def peser(nom, r, m, sonde):
    attendre_minute()
    t0 = time.time()
    s, why = req(600 - m)
    if s != 200: return {"test": nom, "erreur_remplissage": s, "raison": why}
    statuts = [req(**sonde) for _ in range(r)]
    if any(st != 200 for st, _ in statuts):
        return {"test": nom, "erreur_sonde": [st for st, _ in statuts], "raison": [w for _, w in statuts if w][:1]}
    c = 0
    while c < m + 10:
        s, why = req(1)
        if s != 200: break
        c += 1
    return {"test": nom, "r": r, "m": m, "places_restantes": c, "poids_par_requete": round((m - c) / r, 3),
            "raison_refus": why, "duree_s": round(time.time() - t0, 1), "heure": dt.datetime.now().strftime("%H:%M:%S")}
TESTS = [
    ("1 point, 1 an, 1 variable horaire (x10, précision 0,1)", 10, 40, dict(n_pts=1, debut="2022-01-01", fin="2022-12-31", horaires=H[:1])),
    ("5 points, 1 an, 1 variable horaire", 1, 30, dict(n_pts=5, debut="2022-01-01", fin="2022-12-31", horaires=H[:1])),
    ("1 point, 1 an, 9 horaires + 1 journalière", 1, 40, dict(n_pts=1, debut="2022-01-01", fin="2022-12-31", horaires=H[:9], journalieres=["temperature_2m_max"])),
]
for nom, r, m, sonde in TESTS:
    print(json.dumps(peser(nom, r, m, sonde), ensure_ascii=False), flush=True)

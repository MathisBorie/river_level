"""
river_web.py — cœur de River Lab en version 100% NAVIGATEUR (Pyodide).

Différences avec river.py (version serveur) :
  - HTTP asynchrone via pyodide.http.pyfetch (pas de `requests`) ;
  - scikit-learn uniquement (pas de TensorFlow) : Ridge, Linéaire, Gradient Boosting ;
  - pas de shapely (point-dans-polygone maison), pas de folium/matplotlib.

Le JS appelle les fonctions `async` d'ici ; l'état entraîné reste en mémoire
Python (STORE) le temps de la session.
"""
import time
import os
import math
import asyncio
import warnings
warnings.filterwarnings("ignore")  # silence les notices pandas (fragmentation, to_datetime)
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from pyodide.http import pyfetch

STORE = {}      # code -> entraînement {"test","feats","targets","modeles","scores","meta"}
ZONES = {}      # code -> bassin versant {"geojson_bassins","points_par_zone","noms_zones","surface_bv"}
SELECTION = {}  # code -> sélection de points {"points_preselectionnes","altitudes_preselection","coords_finales"}
VARS_METEO = ["rain_sum", "temperature_2m_mean", "snowfall_sum"]   # journalières natives

# ---------------------------------------------------------------- persistance
# Le worker monte un système de fichiers IndexedDB sur /persist (le "disque" du
# navigateur). On y sauvegarde modèles + points par station (joblib) pour qu'ils
# survivent d'une visite à l'autre. Après une écriture, on lève un drapeau : le
# worker appelle FS.syncfs() pour graver vraiment dans IndexedDB.
_PERSIST = "/persist"
_A_SYNCHRONISER = [False]


def besoin_sync():
    """Appelé par le worker après chaque requête : True s'il faut graver le FS."""
    v = _A_SYNCHRONISER[0]; _A_SYNCHRONISER[0] = False
    return v


def _sauver(code):
    """Écrit modèles + zones + sélection d'une station sur le disque du navigateur."""
    try:
        os.makedirs(_PERSIST, exist_ok=True)
        paquet = {"store": STORE.get(code), "zones": ZONES.get(code), "selection": SELECTION.get(code)}
        joblib.dump(paquet, f"{_PERSIST}/{code}.joblib", compress=3)
        _A_SYNCHRONISER[0] = True
    except Exception as e:
        print("[persist] échec sauvegarde", code, ":", e)


def _charger_tout():
    """Recharge en mémoire toutes les stations sauvegardées (au démarrage)."""
    if not os.path.isdir(_PERSIST):
        return
    for f in os.listdir(_PERSIST):
        if not f.endswith(".joblib"):
            continue
        code = f[:-len(".joblib")]
        try:
            p = joblib.load(f"{_PERSIST}/{f}")
            if p.get("store"): STORE[code] = p["store"]
            if p.get("zones"): ZONES[code] = p["zones"]
            if p.get("selection"): SELECTION[code] = p["selection"]
        except Exception as e:
            print("[persist] échec chargement", code, ":", e)


def _inventaire():
    """Liste des stations stockées sur cet appareil (pour le gestionnaire)."""
    total, stations = 0, []
    for code in sorted(set(list(STORE.keys()) + list(ZONES.keys()))):
        chemin = f"{_PERSIST}/{code}.joblib"
        octets = os.path.getsize(chemin) if os.path.exists(chemin) else 0
        total += octets
        st = STORE.get(code)
        modeles = []
        if st:
            noms = list(st.get("scores", {}).keys())
            par = octets // max(1, len(noms)) if noms else 0
            for nom in noms:
                modeles.append({"nom": nom, "octets": par, "score": st["scores"][nom]["r2"]})
        nom_station = (st["meta"]["nom"] if st else None) or (ZONES.get(code, {}) and code) or code
        stations.append({"code": code, "nom": nom_station, "octets_total": octets,
                         "octets_test": 0, "octets_travail": 0, "modeles": modeles})
    return {"octets_total": total, "stations": stations}


def _supprimer(code, cible):
    chemin = f"{_PERSIST}/{code}.joblib"
    octets = os.path.getsize(chemin) if os.path.exists(chemin) else 0
    if cible == "station":
        STORE.pop(code, None); ZONES.pop(code, None); SELECTION.pop(code, None)
        if os.path.exists(chemin):
            os.remove(chemin)
        _A_SYNCHRONISER[0] = True
        return {"libelle": f"station {code}", "octets_liberes": octets}
    if cible.startswith("modele:"):
        nom = cible.split(":", 1)[1]
        st = STORE.get(code)
        libere = octets // max(1, len(st.get("scores", {}))) if st else 0
        if st and nom in st.get("modeles", {}):
            st["modeles"].pop(nom, None); st["scores"].pop(nom, None)
            if st["modeles"]:
                _sauver(code)
            else:  # plus aucun modèle : on retire la station
                return _supprimer(code, "station")
        return {"libelle": f"modèle {nom}", "octets_liberes": libere}
    return {"libelle": cible, "octets_liberes": 0}


async def _fetch_json(url, essais=4):
    """GET JSON tolérant : retente sur 429 (quota par minute Open-Meteo) avec
    back-off, et transforme un corps d'erreur Open-Meteo en exception claire."""
    for k in range(essais):
        r = await pyfetch(url)
        statut = getattr(r, "status", 200)
        if statut == 429:
            if k == essais - 1:
                raise RuntimeError("Quota Open-Meteo par minute atteint — réessaie dans une minute.")
            await asyncio.sleep(min(60, 15 * (k + 1)))
            continue
        data = await r.json()
        if isinstance(data, dict) and data.get("error") and data.get("reason"):
            raise RuntimeError("Open-Meteo : " + str(data["reason"]))
        return data


def _parse_dates(serie):
    s = pd.to_datetime(serie)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s.dt.normalize()


# ----------------------------------------------------------------- données brutes
async def infos_station(code):
    d = await _fetch_json(f"https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations?code_station={code}&format=json&size=1")
    s = d["data"][0]
    return {"nom": s.get("libelle_station"), "lat": s.get("latitude_station"), "lon": s.get("longitude_station")}


async def debit(code, start, end):
    url = (f"https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab?code_entite={code}"
           f"&grandeur_hydro_elab=QmnJ&date_debut_obs_elab={start}&date_fin_obs_elab={end}&size=20000&sort=asc")
    d = (await _fetch_json(url))["data"]
    df = pd.DataFrame(d)[["date_obs_elab", "resultat_obs_elab"]]
    df.columns = ["date", "debit_L_s"]
    df["date"] = _parse_dates(df["date"])
    return df.dropna().reset_index(drop=True)


def _cap_archive(end, forecast):
    """L'archive Open-Meteo (ERA5) s'arrête à HIER : on plafonne la date de fin
    pour ne pas dépasser la plage autorisée (le forecast, lui, va dans le futur)."""
    if forecast:
        return end
    hier = (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return min(end, hier)   # comparaison lexicographique OK sur dates ISO


async def meteo_moyenne(coords, start, end, forecast=False):
    """Météo journalière moyennée sur les points `coords` (1 requête batchée)."""
    end = _cap_archive(end, forecast)
    base = "https://api.open-meteo.com/v1/forecast" if forecast else "https://archive-api.open-meteo.com/v1/archive"
    lat = ",".join(str(c[0]) for c in coords)
    lon = ",".join(str(c[1]) for c in coords)
    daily = ",".join(VARS_METEO)
    url = f"{base}?latitude={lat}&longitude={lon}&daily={daily}&start_date={start}&end_date={end}"
    data = await _fetch_json(url)
    points = data if isinstance(data, list) else [data]
    dfs = []
    for p in points:
        if "daily" not in p:
            continue
        dp = pd.DataFrame(p["daily"])
        dp["date"] = pd.to_datetime(dp["time"]).dt.normalize()
        dfs.append(dp.drop(columns=["time"]))
    if not dfs:
        return pd.DataFrame(columns=["date"] + VARS_METEO)
    return pd.concat(dfs).groupby("date", as_index=False).mean()


# ----------------------------------------------------------------- géométrie
# Équivalents « maison » de shapely (indisponible dans Pyodide) : lecture de
# géométries GeoJSON (Polygon / MultiPolygon), test point-dans-polygone par
# lancer de rayon, surface par la formule du lacet.
def _rings(geom):
    """Liste de polygones, chacun = [anneau_ext, trou1, ...], anneau = [[lon,lat],...]."""
    t = geom.get("type"); c = geom.get("coordinates")
    if t == "Polygon":
        return [c]
    if t == "MultiPolygon":
        return list(c)
    return []


def _bounds(geom):
    xs, ys = [], []
    for poly in _rings(geom):
        for lon, lat in poly[0]:
            xs.append(lon); ys.append(lat)
    return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)


def _dans_anneau(x, y, ring):
    inside = False; n = len(ring); j = n - 1
    for i in range(n):
        xi, yi = ring[i][0], ring[i][1]
        xj, yj = ring[j][0], ring[j][1]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def _dans_geom(x, y, geom):
    for poly in _rings(geom):
        if _dans_anneau(x, y, poly[0]) and not any(_dans_anneau(x, y, h) for h in poly[1:]):
            return True
    return False


def _aire_anneau(ring):
    s = 0.0; n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]; x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def _aire_km2(geom):
    aire_deg = 0.0; lat_ref = 0.0
    for poly in _rings(geom):
        a = _aire_anneau(poly[0]) - sum(_aire_anneau(h) for h in poly[1:])
        aire_deg += a
        lat_ref = poly[0][0][1]
    return aire_deg * (111.32 ** 2) * math.cos(math.radians(lat_ref))


# ----------------------------------------------------------------- bassin versant
async def _surface_bv(code):
    """Surface (km²) du bassin versant amont, fiche SITE Hub'Eau (code[:8])."""
    try:
        d = await _fetch_json("https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/sites"
                              f"?code_site={code[:8]}&format=json&size=1")
        data = d.get("data", [])
        if data and data[0].get("surface_bv"):
            return float(data[0]["surface_bv"])
    except Exception:
        pass
    return None


def _zones_amont(zones, surface_bv, code, log):
    """Ne garde que les zones réellement EN AMONT via la codification hydro
    française (région > secteur > sous-secteur > zone, numérotées source→aval),
    en s'arrêtant quand la surface cumulée atteint la surface officielle."""
    code_zone = code[:4]
    region, secteur, sous_secteur = code_zone[:1], code_zone[:2], code_zone[:3]
    cd = lambda f: (f["properties"].get("CdZoneHydro") or "")
    tier1 = sorted([f for f in zones if cd(f)[:3] == sous_secteur and cd(f) <= code_zone], key=cd, reverse=True)
    tier2 = sorted([f for f in zones if cd(f)[:2] == secteur and cd(f)[:3] < sous_secteur], key=cd, reverse=True)
    tier3 = sorted([f for f in zones if cd(f)[:1] == region and cd(f)[:2] < secteur], key=cd, reverse=True)
    if not tier1:
        log("Aucune zone amont identifiable via le code hydro — toutes les zones proches sont conservées.")
        return zones, []
    candidats = tier1 + (tier2 + tier3 if surface_bv else [])
    retenues, surface_cumulee = [], 0.0
    for f in candidats:
        if retenues and surface_bv and surface_cumulee >= 0.95 * surface_bv:
            break
        retenues.append(f); surface_cumulee += _aire_km2(f["geometry"])
    exclues = [f for f in zones if f not in retenues]
    if surface_bv:
        log(f"Bassin amont reconstitué : {len(retenues)} zone(s), {surface_cumulee:.0f} km² "
            f"(surface officielle : {surface_bv:.0f} km²).")
    else:
        log(f"Bassin amont : {len(retenues)} zone(s) du sous-secteur {sous_secteur} (surface officielle inconnue).")
    return retenues, exclues


async def determiner_zones(code, log, prog, arret, n_grille=15, max_par_zone=25):
    """Récupère les bassins versants Sandre amont + une grille de points candidats
    strictement à l'intérieur. Remplit ZONES[code]."""
    info = await _infos(code)
    lat0, lon0 = info["lat"], info["lon"]
    if lat0 is None or lon0 is None:
        raise ValueError("Position de la station inconnue.")
    arret()
    prog("Surface du bassin versant", None); log("Lecture de la surface du bassin (Hub'Eau)…")
    surface = await _surface_bv(code)
    rayon_km = min(300.0, max(25.0, 2.5 * math.sqrt(surface))) if surface else 60.0
    dlat = rayon_km / 111.0
    dlon = rayon_km / (111.0 * math.cos(math.radians(lat0)))
    bbox = f"{lon0-dlon},{lat0-dlat},{lon0+dlon},{lat0+dlat}"

    arret()
    prog("Interrogation de l'API Sandre", None); log("🌍 Récupération des bassins versants (Sandre)…")
    url = ("https://services.sandre.eaufrance.fr/geo/sandre?SERVICE=WFS&VERSION=1.0.0&REQUEST=GetFeature"
           f"&typeName=ZoneHydro_FXX&outputFormat=geojson&bbox={bbox}")
    data = await _fetch_json(url)
    zones = data.get("features", [])
    log(f"{len(zones)} zones hydrographiques trouvées dans le rayon de recherche.")
    await asyncio.sleep(0)

    retenues, _exclues = _zones_amont(zones, surface, code, log)
    geojson = {"type": "FeatureCollection", "features": retenues}

    prog("Génération des points candidats", None); log("🎯 Génération des points à l'intérieur des zones…")
    points_par_zone = {}
    for i, f in enumerate(retenues):
        arret()
        nom = f["properties"].get("LbZoneHydro") or f"Zone_{i}"
        geom = f["geometry"]
        min_lon, min_lat, max_lon, max_lat = _bounds(geom)
        valides = []
        for la in np.linspace(min_lat, max_lat, n_grille):
            for lo in np.linspace(min_lon, max_lon, n_grille):
                if _dans_geom(float(lo), float(la), geom):
                    valides.append((round(float(la), 4), round(float(lo), 4)))
        if len(valides) > max_par_zone:
            idx = np.linspace(0, len(valides) - 1, max_par_zone, dtype=int)
            valides = [valides[k] for k in idx]
        points_par_zone[nom] = valides
        log(f"   → {nom} : {len(valides)} points.")
        await asyncio.sleep(0)

    ZONES[code] = {"geojson_bassins": geojson, "points_par_zone": points_par_zone,
                   "noms_zones": list(points_par_zone.keys()), "surface_bv": surface}
    SELECTION.pop(code, None)   # une nouvelle définition de zones invalide l'ancienne sélection
    nb = sum(len(v) for v in points_par_zone.values())
    prog("Bassin versant prêt", 100); log(f"✅ {len(retenues)} zone(s), {nb} points candidats.")
    return {"nb_zones": len(retenues), "nb_points": nb}


# ------------------------------------------------- sélection des points (2 temps)
async def _fetch_elevations(coords, lot=100):
    """Altitude (m) de chaque point via l'endpoint élévation d'Open-Meteo (gratuit,
    distinct du quota météo). Renvoie une liste alignée sur coords (None si échec)."""
    out = []
    for i in range(0, len(coords), lot):
        sub = coords[i:i + lot]
        la = ",".join(str(c[0]) for c in sub); lo = ",".join(str(c[1]) for c in sub)
        try:
            d = await _fetch_json(f"https://api.open-meteo.com/v1/elevation?latitude={la}&longitude={lo}")
            out.extend(d.get("elevation") or [None] * len(sub))
        except Exception:
            out.extend([None] * len(sub))
    return out


async def _meteo_par_point(points, start, end, daily_vars, hourly_vars, forecast=False):
    """Météo PAR POINT (1 requête batchée) : renvoie [(df_horaire, df_journalier), ...]
    dans l'ordre de `points`."""
    end = _cap_archive(end, forecast)
    base = "https://api.open-meteo.com/v1/forecast" if forecast else "https://archive-api.open-meteo.com/v1/archive"
    la = ",".join(str(c[0]) for c in points); lo = ",".join(str(c[1]) for c in points)
    url = f"{base}?latitude={la}&longitude={lo}&start_date={start}&end_date={end}"
    if daily_vars: url += "&daily=" + ",".join(daily_vars)
    if hourly_vars: url += "&hourly=" + ",".join(hourly_vars)
    data = await _fetch_json(url)
    if isinstance(data, dict): data = [data]
    out = []
    for pt in data:
        df_d = pd.DataFrame()
        if "daily" in pt:
            df_d = pd.DataFrame(pt["daily"]); df_d["date"] = pd.to_datetime(df_d["time"]); df_d = df_d.drop(columns=["time"])
        df_h = pd.DataFrame()
        if "hourly" in pt:
            df_h = pd.DataFrame(pt["hourly"]); df_h["date"] = pd.to_datetime(df_h["time"]); df_h = df_h.drop(columns=["time"])
        out.append((df_h, df_d))
    return out


def _hourly_to_daily_mean(df_h):
    if df_h.empty:
        return pd.DataFrame(columns=["date"])
    df = df_h.copy(); df["date"] = df["date"].dt.floor("D")
    return df.groupby("date", as_index=False).mean()


def _coords_en_km(coords):
    """(lat, lon) -> km locaux (plan tangent, origine = barycentre)."""
    lat = np.array([c[0] for c in coords], dtype=float); lon = np.array([c[1] for c in coords], dtype=float)
    x = (lon - lon.mean()) * 111.32 * math.cos(math.radians(lat.mean()))
    y = (lat - lat.mean()) * 111.32
    return np.column_stack([x, y])


def _rangs_normalises(v):
    """Chaque valeur -> son rang dans [0,1] (insensible aux unités)."""
    v = np.asarray(v, dtype=float); n = len(v)
    if n <= 1:
        return np.zeros(n)
    return np.argsort(np.argsort(v)) / (n - 1)


async def _preselection(code, n_points, poids_altitude, poids_zones, log, prog, arret,
                        poids_alt_dist=1.0, sigma_km=None):
    """ÉTAPE 1 (sans quota météo) : glouton SOUS-MODULAIRE sur la grille de
    candidats — couverture spatiale (x,y,altitude) + altitude haute + équilibre
    entre zones. Garantie (1-1/e) du théorème de Nemhauser-Wolsey-Fisher."""
    zo = ZONES[code]
    candidats, zone_de = [], []
    for iz, pts in enumerate(zo["points_par_zone"].values()):
        for pt in pts:
            candidats.append((round(float(pt[0]), 4), round(float(pt[1]), 4))); zone_de.append(iz)
    n = len(candidats)
    if n == 0:
        raise ValueError("Aucun point candidat dans le bassin.")
    prog("Altitude des candidats", None); log(f"⛰️ Altitude de {n} points candidats (gratuit)…")
    alt_brut = await _fetch_elevations(candidats)
    if n_points >= n:
        return candidats, [float(a) if a is not None else 0.0 for a in alt_brut]
    zone_de = np.array(zone_de); n_zones = len(zo["points_par_zone"])
    altitudes = np.array([a if a is not None else np.nan for a in alt_brut], dtype=float)
    if np.isnan(altitudes).all():
        altitudes = np.zeros(n)
    else:
        altitudes = np.where(np.isnan(altitudes), np.nanmean(altitudes), altitudes)
    h_min, h_max = altitudes.min(), altitudes.max()
    alt_norm = (altitudes - h_min) / (h_max - h_min) if h_max > h_min else np.zeros(n)

    xy = _coords_en_km(candidats)
    etendue = max(1.0, np.sqrt(((xy - xy.mean(0)) ** 2).sum(1)).mean())
    pts3d = np.column_stack([xy, alt_norm * etendue * poids_alt_dist])
    if sigma_km is None:
        sigma_km = 0.5 * etendue
    d2 = ((pts3d[:, None, :] - pts3d[None, :, :]) ** 2).sum(-1)
    similarite = np.exp(-d2 / (sigma_km ** 2))

    prog("Présélection (altitude + couverture + zones)", None)
    log(f"🧮 Présélection gloutonne de {n_points} points parmi {n}…")
    meilleure_sim = np.zeros(n); compte_zone = np.zeros(n_zones); choisis = []; dispo = set(range(n))
    for _ in range(n_points):
        arret()
        best_gain, best_i = -np.inf, None
        for i in dispo:
            gain = (np.maximum(similarite[i] - meilleure_sim, 0).sum()
                    + poids_altitude * alt_norm[i]
                    + poids_zones * (math.sqrt(compte_zone[zone_de[i]] + 1) - math.sqrt(compte_zone[zone_de[i]])))
            if gain > best_gain:
                best_gain, best_i = gain, i
        choisis.append(best_i); dispo.discard(best_i)
        meilleure_sim = np.maximum(meilleure_sim, similarite[best_i])
        compte_zone[zone_de[best_i]] += 1
        await asyncio.sleep(0)   # laisse passer le suivi entre deux points
    pre = [candidats[i] for i in choisis]
    alts = [float(altitudes[i]) for i in choisis]
    log(f"✅ {len(choisis)} points présélectionnés | altitude {min(alts):.0f}–{max(alts):.0f} m.")
    return pre, alts


async def _selection_finale(code, points, n_final, fenetre_annees, poids_pluie, poids_neige,
                            log, prog, arret, lag_max=20, sigma_redondance_km=None):
    """ÉTAPE 2 (quota léger) : ne télécharge QUE les points présélectionnés sur une
    fenêtre courte. score_pluie = somme des corrélations croisées positives
    pluie(t)→débit(t+k) ; score_neige = stock de neige moyen. Normalisés par rang,
    combinés, puis glouton pertinence × (1 - redondance de proximité)."""
    if n_final >= len(points):
        return {"coords_finales": list(points), "temps_reponse_jours": 0}
    fin = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)   # l'archive s'arrête à hier
    debut = fin - pd.Timedelta(days=int(365.25 * fenetre_annees))
    s, e = debut.strftime("%Y-%m-%d"), fin.strftime("%Y-%m-%d")
    prog("Étape 2 : téléchargement léger (pluie + neige)", None)
    log(f"📥 Étape 2 : {len(points)} points sur {s}→{e} (pluie quotidienne + neige)…")
    df_eau = await debit(code, s, e)
    serie = df_eau.dropna(subset=["debit_L_s"]).set_index("date")["debit_L_s"].sort_index()
    res = await _meteo_par_point(points, s, e, daily_vars=["rain_sum"], hourly_vars=["snow_depth"])
    await asyncio.sleep(0)

    score_pluie, score_neige, lags = [], [], []
    for (df_h, df_d) in res:
        arret()
        pluie = df_d.set_index("date")["rain_sum"].sort_index() if (not df_d.empty and "rain_sum" in df_d) else pd.Series(dtype=float)
        if not df_h.empty and "snow_depth" in df_h:
            neige_j = _hourly_to_daily_mean(df_h).set_index("date")["snow_depth"]
            score_neige.append(float(neige_j.mean()))
        else:
            score_neige.append(0.0)
        somme, meilleur_r, meilleur_k = 0.0, -1.0, 0
        if not pluie.empty:
            for k in range(lag_max + 1):
                r = pluie.corr(serie.shift(-k))
                if pd.notna(r):
                    somme += max(0.0, r)
                    if r > meilleur_r:
                        meilleur_r, meilleur_k = r, k
        score_pluie.append(somme); lags.append(meilleur_k)

    score_pluie = np.array(score_pluie); score_neige = np.array(score_neige)
    pertinence = poids_pluie * _rangs_normalises(score_pluie) + poids_neige * _rangs_normalises(score_neige)

    xy = _coords_en_km(points)
    if sigma_redondance_km is None:
        etendue = max(1.0, np.sqrt(((xy - xy.mean(0)) ** 2).sum(1)).mean())
        sigma_redondance_km = 0.5 * etendue
    d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
    proximite = np.exp(-d2 / (sigma_redondance_km ** 2))

    choisis, redondance = [], np.zeros(len(points))
    for _ in range(n_final):
        arret()
        eff = pertinence * (1 - redondance)
        for c in choisis:
            eff[c] = -np.inf
        i = int(np.argmax(eff)); choisis.append(i)
        redondance = np.maximum(redondance, proximite[i])
    coords = [points[i] for i in choisis]
    temps_reponse = int(np.median([lags[i] for i in choisis]))
    log(f"✅ Étape 2 : {len(coords)} points retenus. Temps de réponse médian du bassin ~{temps_reponse} j.")
    return {"coords_finales": coords, "temps_reponse_jours": temps_reponse}


# ----------------------------------------------------------------- features
def construire(df_eau, df_meteo, past, horizon):
    df = pd.merge(df_eau, df_meteo, on="date", how="inner").sort_values("date").reset_index(drop=True)
    feats, targets = [], []
    for i in range(1, past + 1):
        c = f"debit_J-{i}"; df[c] = df["debit_L_s"].shift(i); feats.append(c)
    for i in range(1, past + 1):
        for v in VARS_METEO:
            c = f"{v}_J-{i}"; df[c] = df[v].shift(i); feats.append(c)
    for i in range(0, horizon + 1):
        for v in VARS_METEO:
            c = f"{v}_J+{i}"; df[c] = df[v].shift(-i); feats.append(c)
    doy = df["date"].dt.dayofyear
    df["sin"] = np.sin(2 * np.pi * doy / 365.25); df["cos"] = np.cos(2 * np.pi * doy / 365.25)
    feats += ["sin", "cos"]
    for h in range(0, horizon + 1):
        c = f"cible_J+{h}"; df[c] = df["debit_L_s"].shift(-h); targets.append(c)
    df = df.dropna().reset_index(drop=True)
    return df, feats, targets


def _gb():
    base = HistGradientBoostingRegressor(max_depth=8, max_leaf_nodes=31, learning_rate=0.05,
                                         max_iter=200, l2_regularization=10, min_samples_leaf=7,
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=15,
                                         random_state=42)
    return base


class MultiGB:
    """Un HistGradientBoosting par horizon (équivalent léger du serveur)."""
    def __init__(self, estimateurs): self.estimateurs = estimateurs
    def predict(self, X): return np.column_stack([e.predict(X) for e in self.estimateurs])


# ----------------------------------------------------------------- entraînement
NOMS_MODELE = {"ridge": "Ridge", "lineaire": "Régression linéaire",
               "gradient_boosting": "Gradient Boosting", "lineaire_pca": "Linéaire (PCA)"}
MODELES_DISPO = ("gradient_boosting", "ridge", "lineaire", "lineaire_pca")


class ModelePCA:
    """Régression linéaire sur les features compressées par PCA (sklearn, sans
    TensorFlow). Le proxy .predict transforme X avant de prédire."""
    def __init__(self, pca, lin): self.pca = pca; self.lin = lin
    def predict(self, X): return self.lin.predict(self.pca.transform(X))


async def _fit_modele(nom, Xtr, Ytr, Xte, Yte, targets, prog=None, arret=None, idx=0, nb=1, seuil_pca=99):
    """Entraîne UN modèle et renvoie (objet, {r2, detail}). Le GB (1/horizon)
    rend la main régulièrement pour que le suivi reste fluide."""
    prog = prog or (lambda *a, **k: None); arret = arret or (lambda: None)
    if nom == "ridge":
        m = Ridge(alpha=10000.0).fit(Xtr, Ytr)
    elif nom == "lineaire":
        m = LinearRegression().fit(Xtr, Ytr)
    elif nom == "lineaire_pca":
        energie = min(0.999, max(0.5, seuil_pca / 100.0))
        pca = PCA(n_components=energie, svd_solver="full").fit(Xtr)
        lin = LinearRegression().fit(pca.transform(Xtr), Ytr)
        m = ModelePCA(pca, lin)
    elif nom == "gradient_boosting":
        ests = []; Y = Ytr.values; nh = len(targets)
        for h in range(nh):
            arret()
            est = _gb(); est.fit(Xtr, Y[:, h]); ests.append(est)
            prog("Entraînement : Gradient Boosting (1 modèle/horizon)",
                 int((idx + (h + 1) / nh) / nb * 100), idx + 1, nb)
            if h % 2 == 0:
                await asyncio.sleep(0)
        m = MultiGB(ests)
    else:
        return None, None
    pred = m.predict(Xte)
    r2 = r2_score(Yte, pred, multioutput="raw_values")
    return m, {"r2": float(r2_score(Yte, pred)), "detail": [float(x) for x in r2]}


async def entrainer(code, coords=None, past=15, horizon=15, annees=10,
                    modeles=("ridge", "lineaire", "gradient_boosting"),
                    log=None, prog=None, arret=None, debut_str=None, fin_str=None, seuil_pca=99):
    log = log or (lambda m: None)
    prog = prog or (lambda *a, **k: None)
    arret = arret or (lambda: None)
    t0 = time.time()
    infos = await infos_station(code)
    lat, lon = infos["lat"], infos["lon"]
    if not coords:
        # grille grossière autour de la station (la sélection fine viendra ensuite)
        d = 0.08
        coords = [(round(lat + dy, 4), round(lon + dx, 4))
                  for dy in (-d, 0, d) for dx in (-d, 0, d)]
    hier = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)   # l'archive météo s'arrête à hier
    fin = min(pd.Timestamp(fin_str).normalize() if fin_str else hier, hier)
    debut = pd.Timestamp(debut_str).normalize() if debut_str else (fin - pd.Timedelta(days=int(365.25 * annees)))
    s, e = debut.strftime("%Y-%m-%d"), fin.strftime("%Y-%m-%d")

    arret()
    prog("Téléchargement du débit", None); log(f"Téléchargement du débit ({s} → {e})…")
    df_eau = await debit(code, s, e)
    arret()
    prog("Téléchargement de la météo", None); log(f"Téléchargement de la météo sur {len(coords)} points…")
    df_meteo = await meteo_moyenne(coords, s, e)
    t_dl = time.time() - t0
    await asyncio.sleep(0)

    prog("Construction des variables", None); log("Construction des variables explicatives…")
    df, feats, targets = construire(df_eau, df_meteo, past, horizon)
    n = len(df); split = int(n * 0.8)
    if n < 50:
        raise ValueError(f"Trop peu de données exploitables ({n} lignes) sur cette période.")
    Xtr, Xte = df[feats].iloc[:split], df[feats].iloc[split:]
    Ytr, Yte = df[targets].iloc[:split], df[targets].iloc[split:]
    await asyncio.sleep(0)

    resultats, objets = {}, {}
    tf = time.time()
    nb = len(modeles)
    for idx, nom in enumerate(modeles):
        arret()
        prog(f"Entraînement : {NOMS_MODELE.get(nom, nom)}", int(idx / nb * 100), idx + 1, nb)
        log(f"Entraînement : {NOMS_MODELE.get(nom, nom)}…")
        await asyncio.sleep(0)
        m, sc = await _fit_modele(nom, Xtr, Ytr, Xte, Yte, targets, prog, arret, idx, nb, seuil_pca)
        if m is None:
            continue
        objets[nom] = m
        resultats[nom] = sc
    t_fit = time.time() - tf
    prog("Entraînement terminé", 100)

    test_df = df.iloc[split:].reset_index(drop=True)
    STORE[code] = {"test": test_df, "feats": feats, "targets": targets, "modeles": objets,
                   # données gardées en mémoire pour ré-entraîner un autre modèle sans re-télécharger
                   "data": {"Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte},
                   "scores": resultats,
                   "meta": {"nom": infos["nom"], "coords": coords, "past": past, "horizon": horizon}}
    return {"nom_station": infos["nom"], "lignes": int(n), "n_points": len(coords),
            "resultats": resultats, "t_dl": t_dl, "t_fit": t_fit, "total": time.time() - t0,
            "dates_test": [d.strftime("%Y-%m-%d") for d in test_df["date"]]}


# ----------------------------------------------------------------- jobs (fond)
# On imite le système de job du serveur : le pipeline tourne comme tâche asyncio
# concurrente ; entre deux téléchargements (et via asyncio.sleep(0)) le worker
# reste libre de répondre aux requêtes de suivi -> la barre de progression avance.
JOBS = {}
_JOB_SEQ = [0]


class _Arret(Exception):
    pass


def _job_nouveau(type_job, code):
    _JOB_SEQ[0] += 1
    jid = _JOB_SEQ[0]
    JOBS[jid] = {"id": jid, "type": type_job, "code": code, "statut": "en_cours",
                 "log": [], "progression": None, "resultat": None, "erreur": None, "arret": False}
    return jid


def _job_dict(job, depuis=0):
    return {"log": job["log"][depuis:], "nb_lignes_log": len(job["log"]),
            "progression": job["progression"], "statut": job["statut"],
            "resultat": job["resultat"], "erreur": job["erreur"]}


def _cbs(job):
    """Callbacks log / progression / arrêt liés à un job."""
    def log(m): job["log"].append(str(m))
    def prog(phase, pct=None, courant=None, total=None):
        job["progression"] = {"phase": phase, "pct": pct, "courant": courant, "total": total}
    def arret():
        if job["arret"]:
            raise _Arret()
    return log, prog, arret


async def _run_zones(jid, code, body):
    job = JOBS[jid]
    log, prog, arret = _cbs(job)
    try:
        res = await determiner_zones(code, log, prog, arret,
                                     n_grille=int(body.get("n_points_grille", 15)),
                                     max_par_zone=int(body.get("max_points_par_zone", 25)))
        _sauver(code)
        job["resultat"] = res
        job["statut"] = "termine"
    except _Arret:
        log("⏹️ Arrêté à la demande."); job["statut"] = "arrete"
    except Exception as e:
        import traceback
        job["erreur"] = f"{type(e).__name__}: {e}"
        log("❌ " + traceback.format_exc()[-700:]); job["statut"] = "erreur"


async def _run_points(jid, code, body):
    """Points choisis À LA MAIN sur la carte : entraîne directement dessus."""
    job = JOBS[jid]
    log, prog, arret = _cbs(job)
    try:
        pts = [(float(p[0]), float(p[1])) for p in (body.get("points") or [])]
        if not pts:
            raise ValueError("Aucun point fourni.")
        SELECTION[code] = {"points_preselectionnes": None, "altitudes_preselection": None,
                           "coords_finales": [list(p) for p in pts]}
        log(f"📍 {len(pts)} point(s) choisi(s) à la main.")
        modele = body.get("modele") if body.get("modele") in MODELES_DISPO else "gradient_boosting"
        info = await entrainer(code, coords=pts,
                               past=int(body.get("past_day", 20)), horizon=int(body.get("predict_day", 15)),
                               modeles=(modele,),
                               log=log, prog=prog, arret=arret,
                               debut_str=body.get("start_train"), fin_str=body.get("end_train"),
                               seuil_pca=float(body.get("seuil_energie") or 99))
        sc = info["resultats"].get(modele, {})
        _sauver(code)
        log(f"✅ Modèle {NOMS_MODELE.get(modele, modele)} prêt — fiabilité {sc.get('r2', 0) * 100:.0f} % (sauvegardé)")
        job["resultat"] = {"score_gradient_boosting": sc.get("r2"), "coords_finales": [list(p) for p in pts]}
        job["statut"] = "termine"
    except _Arret:
        log("⏹️ Arrêté à la demande."); job["statut"] = "arrete"
    except Exception as e:
        import traceback
        job["erreur"] = f"{type(e).__name__}: {e}"
        log("❌ " + traceback.format_exc()[-700:]); job["statut"] = "erreur"


async def ajouter_modeles(code, noms, log, prog, arret, seuil_pca=99):
    """Entraîne un/des modèle(s) SUPPLÉMENTAIRE(S) sur les données déjà en mémoire
    (aucun re-téléchargement) et les ajoute à STORE[code]."""
    st = STORE.get(code)
    if not st or "data" not in st:
        raise ValueError("Analyse d'abord la rivière (données d'entraînement absentes de la mémoire).")
    d = st["data"]; targets = st["targets"]; nb = len(noms)
    for idx, nom in enumerate(noms):
        arret()
        prog(f"Entraînement : {NOMS_MODELE.get(nom, nom)}", int(idx / nb * 100), idx + 1, nb)
        log(f"Entraînement : {NOMS_MODELE.get(nom, nom)}…")
        await asyncio.sleep(0)
        m, sc = await _fit_modele(nom, d["Xtr"], d["Ytr"], d["Xte"], d["Yte"], targets, prog, arret, idx, nb, seuil_pca)
        if m is None:
            continue
        st["modeles"][nom] = m; st["scores"][nom] = sc
        log(f"   → {NOMS_MODELE.get(nom, nom)} : fiabilité {sc['r2'] * 100:.0f} %")
    prog("Terminé", 100)
    return {"modeles": list(st["scores"].keys())}


async def _run_ajouter(jid, code, noms, seuil_pca=99):
    job = JOBS[jid]
    log, prog, arret = _cbs(job)
    try:
        noms = [n for n in noms if n in MODELES_DISPO]
        if not noms:
            raise ValueError("Aucun modèle valide à entraîner.")
        job["resultat"] = await ajouter_modeles(code, noms, log, prog, arret, seuil_pca=seuil_pca)
        _sauver(code)
        log("✅ Terminé (sauvegardé sur cet appareil).")
        job["statut"] = "termine"
    except _Arret:
        log("⏹️ Arrêté à la demande."); job["statut"] = "arrete"
    except Exception as e:
        import traceback
        job["erreur"] = f"{type(e).__name__}: {e}"
        log("❌ " + traceback.format_exc()[-700:]); job["statut"] = "erreur"


async def _run_pipeline(jid, code, body):
    """Pipeline « 2 temps » complet (comme le serveur) : bassin versant (grille
    dense) → présélection altitude+couverture → sélection corrélation+neige →
    entraînement sur les points RETENUS. Les points apparaissent sur la carte au
    fil de l'analyse (SELECTION[code] mis à jour à chaque étape)."""
    job = JOBS[jid]
    log, prog, arret = _cbs(job)
    try:
        past = int(body.get("past_day", 20))
        horizon = int(body.get("predict_day", 15))
        sel = body.get("selection") or {}
        n_pre = int(sel.get("n_preselection", 30))
        n_fin = int(sel.get("n_final", 5))

        # 0. bassin versant (grille dense)
        await determiner_zones(code, log, prog, arret,
                               n_grille=int(sel.get("densite_grille", 25)), max_par_zone=60)
        await asyncio.sleep(0)

        # 1. présélection altitude + couverture + zones (sans quota météo)
        pre, alts = await _preselection(code, n_pre,
                                        poids_altitude=float(sel.get("poids_altitude", 1.0)),
                                        poids_zones=float(sel.get("poids_zones", 1.0)),
                                        log=log, prog=prog, arret=arret)
        SELECTION[code] = {"points_preselectionnes": [list(p) for p in pre],
                           "altitudes_preselection": alts, "coords_finales": None}
        await asyncio.sleep(0)

        # 2. sélection finale corrélation pluie + neige (quota léger)
        diag = await _selection_finale(code, pre, n_fin,
                                       fenetre_annees=float(sel.get("fenetre_annees", 2)),
                                       poids_pluie=float(sel.get("poids_pluie", 1.0)),
                                       poids_neige=float(sel.get("poids_neige", 1.0)),
                                       log=log, prog=prog, arret=arret)
        coords = diag["coords_finales"]
        SELECTION[code]["coords_finales"] = [list(c) for c in coords]
        await asyncio.sleep(0)

        # 3. entraînement sur les points RETENUS (un seul modèle, GB par défaut)
        modele = body.get("modele") if body.get("modele") in MODELES_DISPO else "gradient_boosting"
        info = await entrainer(code, coords=coords, past=past, horizon=horizon,
                               modeles=(modele,),
                               log=log, prog=prog, arret=arret,
                               debut_str=body.get("start_train"), fin_str=body.get("end_train"),
                               seuil_pca=float(body.get("seuil_energie") or 99))
        sc = info["resultats"].get(modele, {})
        _sauver(code)
        log(f"✅ Modèle {NOMS_MODELE.get(modele, modele)} prêt — fiabilité {sc.get('r2', 0) * 100:.0f} % (sauvegardé sur cet appareil)")
        job["resultat"] = {"score_gradient_boosting": sc.get("r2"),
                           "coords_finales": [list(c) for c in coords],
                           "temps_reponse_jours": diag["temps_reponse_jours"]}
        job["statut"] = "termine"
    except _Arret:
        log("⏹️ Arrêté à la demande.")
        job["statut"] = "arrete"
    except Exception as e:
        import traceback
        job["erreur"] = f"{type(e).__name__}: {e}"
        log("❌ " + traceback.format_exc()[-700:])
        job["statut"] = "erreur"


def backtest(code, modele, date_str, nb_jours=None):
    """Données du fan chart (prévision vs réalité) pour une date du jeu de test."""
    st = STORE.get(code)
    if not st or modele not in st["modeles"]:
        return None
    test, feats, targets, meta = st["test"], st["feats"], st["targets"], st["meta"]
    horizon = meta["horizon"]
    h_max = horizon if nb_jours is None else max(1, min(int(nb_jours), horizon))
    date = pd.to_datetime(date_str)
    ligne = test[test["date"] == date]
    if ligne.empty:
        return None
    X = ligne[feats]
    pred = st["modeles"][modele].predict(X)[0][:h_max + 1]
    vrai = ligne[[f"cible_J+{h}" for h in range(h_max + 1)]].values[0]
    debit0 = ligne["debit_L_s"].values[0]
    contexte = test[(test["date"] >= date - pd.Timedelta(days=15)) & (test["date"] <= date)]
    D = 1000.0
    dates = [date] + list(pd.date_range(date + pd.Timedelta(days=1), periods=h_max + 1))
    pts = [{"date": d.strftime("%Y-%m-%d"),
            "prev": float(v) / D, "reel": float(r) / D}
           for d, v, r in zip(dates, [debit0] + list(pred), [debit0] + list(vrai))]
    return {"pivot": date.strftime("%Y-%m-%d"), "hybride": False,
            "observe": [{"date": d.strftime("%Y-%m-%d"), "debit": float(x) / D}
                        for d, x in zip(contexte["date"], contexte["debit_L_s"])],
            "points": pts}


async def prevision(code, modele, nb_jours=None):
    """Prévision réelle à partir d'AUJOURD'HUI : débit récent (Hub'Eau) + météo
    PRÉVISIONNELLE (Open-Meteo forecast) sur les points retenus, puis prédiction
    des prochains jours. Renvoie un fan chart (pivot = aujourd'hui)."""
    st = STORE.get(code)
    if not st or modele not in st["modeles"]:
        return {"erreur": "modèle non entraîné"}
    feats, meta = st["feats"], st["meta"]
    coords, past, horizon = meta["coords"], meta["past"], meta["horizon"]
    today = pd.Timestamp.now().normalize()
    start = (today - pd.Timedelta(days=past + 7)).strftime("%Y-%m-%d")

    df_eau = await debit(code, start, today.strftime("%Y-%m-%d"))
    df_eau = df_eau.dropna(subset=["debit_L_s"]).sort_values("date").reset_index(drop=True)
    if df_eau.empty:
        return {"erreur": "Aucun débit récent exploitable sur Hub'Eau."}
    serie = df_eau.set_index("date")["debit_L_s"].asfreq("D").interpolate(limit_direction="both")
    date_J0 = df_eau["date"].iloc[-1]
    debit0 = float(df_eau["debit_L_s"].iloc[-1])
    decalage = int((today - date_J0).days)
    if decalage > horizon:
        return {"erreur": f"Le dernier débit publié date de {decalage} j — au-delà de l'horizon du modèle ({horizon} j)."}
    nb_max = horizon - decalage + 1
    nb_jours = min(15, nb_max) if nb_jours is None else min(int(nb_jours), nb_max)
    horizons = list(range(decalage, decalage + nb_jours))

    # Météo prévisionnelle : l'horizon forecast varie (~+14 à +16 j) -> on réduit si refus.
    df_meteo = None
    for cap in (15, 13, 11, 9, 7):
        end = (today + pd.Timedelta(days=cap)).strftime("%Y-%m-%d")
        try:
            df_meteo = await meteo_moyenne(coords, start, end, forecast=True)
            if df_meteo is not None and not df_meteo.empty:
                break
        except Exception:
            df_meteo = None
    if df_meteo is None or df_meteo.empty:
        return {"erreur": "Météo prévisionnelle indisponible (Open-Meteo forecast)."}
    idx = pd.date_range(date_J0 - pd.Timedelta(days=past), date_J0 + pd.Timedelta(days=horizon))
    dm = df_meteo.set_index("date").reindex(idx)
    dm = dm.fillna(dm.mean())   # jours au-delà du forecast -> moyenne (saisonnière) de la fenêtre

    f = {}
    for i in range(1, past + 1):
        di = date_J0 - pd.Timedelta(days=i)
        f[f"debit_J-{i}"] = float(serie.loc[di]) if di in serie.index and pd.notna(serie.loc[di]) else debit0
    for i in range(1, past + 1):
        for v in VARS_METEO:
            f[f"{v}_J-{i}"] = float(dm.loc[date_J0 - pd.Timedelta(days=i), v])
    for i in range(0, horizon + 1):
        for v in VARS_METEO:
            f[f"{v}_J+{i}"] = float(dm.loc[date_J0 + pd.Timedelta(days=i), v])
    doy = date_J0.dayofyear
    f["sin"] = math.sin(2 * math.pi * doy / 365.25); f["cos"] = math.cos(2 * math.pi * doy / 365.25)
    X = pd.DataFrame([f])[feats]
    pred = st["modeles"][modele].predict(X)[0]

    D = 1000.0
    contexte = df_eau[df_eau["date"] >= date_J0 - pd.Timedelta(days=15)]
    observe = [{"date": d.strftime("%Y-%m-%d"), "debit": float(x) / D}
               for d, x in zip(contexte["date"], contexte["debit_L_s"])]
    points = [{"date": date_J0.strftime("%Y-%m-%d"), "prev": debit0 / D}]
    for h in horizons:
        d = date_J0 + pd.Timedelta(days=h)
        points.append({"date": d.strftime("%Y-%m-%d"), "prev": float(pred[h]) / D})
    sc = st["scores"].get(modele, {})
    return {"pivot": today.strftime("%Y-%m-%d"), "hybride": False,
            "observe": observe, "points": points, "modele": modele,
            "score": sc.get("r2"), "scores_detail": sc.get("detail"), "unite": "m³/s"}


def pca_analyse(code, seuil):
    """PCA seule (sans entraîner) : variance expliquée par composante + variance
    cumulée, et nombre de composantes atteignant l'énergie demandée. Sert au
    graphe PCA et au choix de l'énergie conservée."""
    st = STORE.get(code)
    if not st or "data" not in st:
        return {"erreur": "Analyse d'abord la rivière (aucune donnée en mémoire)."}
    Xtr = st["data"]["Xtr"]
    p = PCA(svd_solver="full").fit(Xtr)
    ratio = p.explained_variance_ratio_ * 100.0
    cum = np.cumsum(ratio)
    energie = float(seuil)
    n = int(np.searchsorted(cum, energie) + 1)
    n = max(1, min(n, len(cum)))
    return {"n_composantes": n, "n_total": int(len(cum)), "energie": energie,
            "n_features": int(Xtr.shape[1]),
            "variance_cumulee": [float(x) for x in cum],
            "variance_par_comp": [float(x) for x in ratio]}


# ==========================================================================
# BACKEND CLIENT : un routeur qui imite l'API Flask, appelé par le worker
# Pyodide. Tout tourne 100% navigateur : carte, historique, bassin versant,
# sélection 2 temps, entraînement (barre de progression), backtest, prévision.
# ==========================================================================
import json as _json
from urllib.parse import urlparse, parse_qs

INFOS = {}   # code -> {"nom","lat","lon"}


async def _infos(code):
    if code not in INFOS:
        INFOS[code] = await infos_station(code)
    return INFOS[code]


async def stations():
    url = ("https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations?format=json&size=10000"
           "&fields=code_station,libelle_station,latitude_station,longitude_station,en_service,libelle_cours_eau,code_departement")
    out = []
    d = await _fetch_json(url)
    for s in d.get("data", []):
        if s.get("latitude_station") is None or s.get("longitude_station") is None:
            continue
        out.append({"code": s["code_station"], "nom": s.get("libelle_station") or "Sans nom",
                    "lat": s["latitude_station"], "lon": s["longitude_station"],
                    "en_service": bool(s.get("en_service")), "cours_eau": s.get("libelle_cours_eau"),
                    "departement": s.get("code_departement")})
    return out


async def etat(code):
    info = await _infos(code)
    st = STORE.get(code)
    zo = ZONES.get(code)
    sel = SELECTION.get(code)
    modeles = []
    if st:
        for nom, r in st.get("scores", {}).items():
            modeles.append({"nom": nom, "score": r["r2"], "espace": "brut", "hybride": False})
    # points finaux : la sélection auto/manuelle si elle existe, sinon ceux de l'entraînement
    coords_finales = None
    if sel and sel.get("coords_finales"):
        coords_finales = sel["coords_finales"]
    elif st:
        coords_finales = st["meta"]["coords"]
    return {"code_station": code, "nom_station": info.get("nom"),
            "lat_station": info.get("lat"), "lon_station": info.get("lon"),
            "zones_definies": bool(zo),
            "geojson_bassins": zo["geojson_bassins"] if zo else None,
            "points_par_zone": zo["points_par_zone"] if zo else None,
            "points_preselectionnes": sel["points_preselectionnes"] if sel else None,
            "altitudes_preselection": sel["altitudes_preselection"] if sel else None,
            "coords_finales": coords_finales,
            "donnees_pretes": bool(st), "donnees_train_presentes": bool(st),
            "pca_prete": False, "n_composantes_pca": None,
            "predict_day": st["meta"]["horizon"] if st else None,
            "past_day": st["meta"]["past"] if st else None,
            "modeles": modeles, "sauvegarde_existante": bool(st),
            "gros_fichiers": [], "cartes_folium": [], "params": {"agregations": {}}}


async def periode(code):
    # Première date (asc) + dernière (fenêtre récente large).
    base = f"https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab?code_entite={code}&grandeur_hydro_elab=QmnJ&sort=asc"
    d = await _fetch_json(base + "&size=1")
    if not d.get("data"):
        return {"erreur": "Aucune donnée QmnJ pour cette station."}
    premiere = pd.to_datetime(d["data"][0]["date_obs_elab"]).tz_localize(None) if pd.to_datetime(d["data"][0]["date_obs_elab"]).tzinfo else pd.to_datetime(d["data"][0]["date_obs_elab"])
    fin = pd.Timestamp.now().normalize()
    debut_rec = (fin - pd.Timedelta(days=1500)).strftime("%Y-%m-%d")
    d2 = await _fetch_json(base + f"&size=20000&date_debut_obs_elab={debut_rec}")
    derniere = pd.to_datetime(d2["data"][-1]["date_obs_elab"]).tz_localize(None) if d2.get("data") else fin
    debut_expl = max(premiere, pd.Timestamp("1940-01-01"))
    nb = round((derniere - debut_expl).days / 365.25, 1)
    # On garde seulement les avertissements utiles (période trop courte) ; la note
    # ERA5-depuis-1940 est purement informative -> retirée.
    avert = None
    if nb < 3:
        avert = f"Attention : seulement ~{nb} an(s) de données exploitables — fiabilité limitée."
    return {"debut": debut_expl.strftime("%Y-%m-%d"), "fin": derniere.strftime("%Y-%m-%d"),
            "debut_debit": premiere.strftime("%Y-%m-%d"), "nb_annees": nb, "avertissement": avert}


def dates_test(code):
    st = STORE.get(code)
    if not st:
        return {"erreur": "Pas de jeu de test (entraîne d'abord)."}
    dts = [d.strftime("%Y-%m-%d") for d in st["test"]["date"]]
    return {"min": dts[0], "max": dts[-1], "dates": dts}


def quota_zero():
    lib = {"minute": "par minute", "heure": "par heure", "jour": "par jour", "mois": "par mois"}
    lim = {"minute": 600, "heure": 5000, "jour": 10000, "mois": 300000}
    return {"fenetres": [{"fenetre": k, "libelle": v, "utilise": 0, "limite": lim[k],
                          "restant": lim[k], "pct": 0.0, "reset_dans_s": 0, "reset_texte": "—"} for k, v in lib.items()]}


async def historique(code, debut, fin):
    df = await debit(code, debut, fin)
    if df.empty:
        return {"erreur": "Aucune donnée sur cette période."}
    fr = lambda d: d.strftime("%d/%m/%Y")
    dm = df["debit_L_s"] / 1000.0
    serie = [{"date": d.strftime("%Y-%m-%d"), "debit": float(v)} for d, v in zip(df["date"], dm)]
    imin, imax = int(dm.idxmin()), int(dm.idxmax())
    return {"unite": "m³/s", "serie": serie, "nb_jours": int(len(df)),
            "debit_min": float(dm.min()), "debit_max": float(dm.max()), "debit_moyen": float(dm.mean()),
            "date_min": fr(df["date"].iloc[imin]), "date_max": fr(df["date"].iloc[imax])}


async def _traiter(method, path, body):
    """Routeur : (méthode, chemin, corps JSON) -> chaîne JSON, comme Flask."""
    body = _json.loads(body) if body else {}
    u = urlparse(path); p = u.path; q = parse_qs(u.query)
    g1 = lambda k, d=None: q.get(k, [d])[0]

    if p == "/api/config":
        return _json.dumps({"public": False, "analytics": None})
    if p == "/api/stations":
        return _json.dumps(await stations())
    if p == "/api/quota":
        return _json.dumps(quota_zero())
    if p in ("/api/quota/limites", "/api/quota/reinitialiser") and method == "POST":
        return _json.dumps({"ok": True})   # quota géré par le navigateur : réglages sans effet
    if p == "/api/stockage":
        return _json.dumps(_inventaire())
    if p == "/api/stockage/supprimer" and method == "POST":
        return _json.dumps(_supprimer(body.get("code"), body.get("cible", "station")))

    import re
    m = re.match(r"^/api/riviere/([^/]+)$", p)
    if m:
        return _json.dumps(await etat(m.group(1)))
    m = re.match(r"^/api/riviere/([^/]+)/periode$", p)
    if m:
        return _json.dumps(await periode(m.group(1)))
    m = re.match(r"^/api/riviere/([^/]+)/dates-test$", p)
    if m:
        return _json.dumps(dates_test(m.group(1)))
    m = re.match(r"^/api/riviere/([^/]+)/historique$", p)
    if m:
        return _json.dumps(await historique(m.group(1), g1("debut"), g1("fin")))
    m = re.match(r"^/api/riviere/([^/]+)/prevision$", p)
    if m:
        return _json.dumps(await prevision(m.group(1), g1("modele", "gradient_boosting"),
                                           int(g1("nb_jours")) if g1("nb_jours") else None))
    m = re.match(r"^/api/riviere/([^/]+)/pca$", p)
    if m:
        return _json.dumps(pca_analyse(m.group(1), float(g1("seuil") or 99)))
    m = re.match(r"^/api/riviere/([^/]+)/backtest$", p)
    if m:
        r = backtest(m.group(1), g1("modele", "gradient_boosting"), g1("date"), int(g1("nb_jours") or 15))
        st = STORE.get(m.group(1))
        if r and st:
            sc = st["scores"].get(g1("modele", "gradient_boosting"), {})
            r["modele"] = g1("modele", "gradient_boosting"); r["score"] = sc.get("r2"); r["scores_detail"] = sc.get("detail"); r["unite"] = "m³/s"
        return _json.dumps(r) if r else _json.dumps({"erreur": "date indisponible ou modèle non entraîné"})

    # ---- jobs de fond ----
    m = re.match(r"^/api/riviere/([^/]+)/zones$", p)
    if m and method == "POST":
        jid = _job_nouveau("zones", m.group(1))
        asyncio.ensure_future(_run_zones(jid, m.group(1), body))
        return _json.dumps({"job_id": jid})
    m = re.match(r"^/api/riviere/([^/]+)/points$", p)
    if m and method == "POST":
        jid = _job_nouveau("points_manuels", m.group(1))
        asyncio.ensure_future(_run_points(jid, m.group(1), body))
        return _json.dumps({"job_id": jid})
    m = re.match(r"^/api/riviere/([^/]+)/pipeline$", p)
    if m and method == "POST":
        jid = _job_nouveau("pipeline", m.group(1))
        asyncio.ensure_future(_run_pipeline(jid, m.group(1), body))
        return _json.dumps({"job_id": jid})
    m = re.match(r"^/api/riviere/([^/]+)/entrainer-modele$", p)
    if m and method == "POST":
        jid = _job_nouveau("entrainer", m.group(1))
        asyncio.ensure_future(_run_ajouter(jid, m.group(1), [body.get("modele")],
                                           seuil_pca=float(body.get("seuil_energie") or 99)))
        return _json.dumps({"job_id": jid})
    m = re.match(r"^/api/riviere/([^/]+)/entrainer$", p)
    if m and method == "POST":
        jid = _job_nouveau("entrainer", m.group(1))
        asyncio.ensure_future(_run_ajouter(jid, m.group(1), body.get("modeles") or [],
                                           seuil_pca=float(body.get("seuil_energie") or 99)))
        return _json.dumps({"job_id": jid})
    m = re.match(r"^/api/riviere/([^/]+)/sauvegarder$", p)
    if m and method == "POST":
        _sauver(m.group(1))
        return _json.dumps({"ok": True})
    m = re.match(r"^/api/jobs/(\d+)$", p)
    if m:
        job = JOBS.get(int(m.group(1)))
        if not job:
            return _json.dumps({"erreur": "job inconnu"})
        return _json.dumps(_job_dict(job, int(g1("depuis") or 0)))
    m = re.match(r"^/api/jobs/(\d+)/arreter$", p)
    if m and method == "POST":
        job = JOBS.get(int(m.group(1)))
        if job:
            job["arret"] = True
        return _json.dumps({"statut": "arret_demande"})

    return _json.dumps({"erreur": f"(client) endpoint pas encore porté : {method} {p}"})


async def traiter(method, path, body):
    """Enveloppe : capture toute exception et renvoie la trace pour la voir côté navigateur."""
    try:
        return await _traiter(method, path, body)
    except Exception as e:
        import traceback
        return _json.dumps({"erreur": f"{type(e).__name__}: {e}",
                            "trace": traceback.format_exc()[-1200:]})


# Recharge les stations sauvegardées (le worker a déjà monté + synchronisé /persist).
_charger_tout()

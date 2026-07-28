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
import io
import math
import base64
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
VARS_METEO = ["rain_sum", "temperature_2m_mean", "snowfall_sum"]   # journalières natives (défaut)


def _vars_meteo(temp_mode="moyenne"):
    """Colonnes météo journalières (features) selon le mode de température :
    'moyenne' = 1 colonne, 'minmax' = min/max/moyenne, 'horaire' = 24 valeurs/jour."""
    if temp_mode == "minmax":
        return ["rain_sum", "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "snowfall_sum"]
    if temp_mode == "horaire":
        return ["rain_sum", "snowfall_sum"] + [f"temp_h{h}" for h in range(24)]
    return list(VARS_METEO)

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


def _store_leger(st):
    """Store SANS les données d'entraînement brutes (Xtr/Ytr…) : elles pèsent
    plusieurs Mo et provoquaient des plantages mémoire au moment de sauvegarder
    sur téléphone. Prévision / backtest / partage n'en ont pas besoin (le backtest
    retélécharge à la demande si nécessaire). Elles restent en RAM pour la session
    (pour ré-entraîner un autre modèle sans re-télécharger)."""
    return {k: v for k, v in st.items() if k != "data"} if st else None


def _sauver(code):
    """Écrit modèles + zones + sélection d'une station sur le disque du navigateur."""
    try:
        os.makedirs(_PERSIST, exist_ok=True)
        paquet = {"store": _store_leger(STORE.get(code)), "zones": ZONES.get(code), "selection": SELECTION.get(code)}
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


def exporter(code):
    """Sérialise le modèle complet d'une station (modèles + incertitude + zones +
    points) en base64, pour le télécharger en fichier .riverlab et l'envoyer à
    quelqu'un d'autre."""
    if code not in STORE:
        return {"erreur": "Aucun modèle à exporter pour cette station."}
    paquet = {"format": "riverlab-1", "code": code, "store": _store_leger(STORE.get(code)),
              "zones": ZONES.get(code), "selection": SELECTION.get(code)}
    buf = io.BytesIO()
    joblib.dump(paquet, buf, compress=3)
    octets = buf.getvalue()
    nom = (STORE[code]["meta"].get("nom") or code)
    return {"code": code, "nom": nom, "octets": len(octets),
            "b64": base64.b64encode(octets).decode("ascii")}


def _r2_max(store):
    return max((s["r2"] for s in store.get("scores", {}).values()), default=0.0)


def _resume_modele(store):
    """Infos lisibles d'un modèle : liste, meilleur R², horizon, années de test."""
    return {"modeles": [NOMS_MODELE.get(n, n) for n in store.get("scores", {})],
            "r2": round(_r2_max(store) * 100),
            "horizon": store["meta"]["horizon"],
            "annees_test": store["meta"].get("annees_test", [])}


def exporter_bytes(code):
    """Comme exporter() mais renvoie les OCTETS bruts (pas de base64) : le worker
    les transfère tels quels au thread principal (zéro copie), pour ne pas saturer
    la mémoire sur téléphone."""
    if code not in STORE:
        raise ValueError("Aucun modèle à exporter pour cette station.")
    paquet = {"format": "riverlab-1", "code": code, "store": _store_leger(STORE.get(code)),
              "zones": ZONES.get(code), "selection": SELECTION.get(code)}
    buf = io.BytesIO()
    joblib.dump(paquet, buf, compress=3)
    return buf.getvalue()


def importer_bytes(donnees, mode="demander"):
    """Comme importer() mais reçoit les octets bruts (Uint8Array JS) ; renvoie une
    chaîne JSON (le worker la transmet telle quelle au thread principal)."""
    try:
        if hasattr(donnees, "to_py"):
            donnees = donnees.to_py()
        b = bytes(donnees)
    except Exception as ex:
        return _json.dumps({"erreur": f"Données illisibles : {ex}"})
    return _json.dumps(_importer_paquet(b, mode))


def importer(b64, mode="demander"):
    """Variante base64 (compat). Préférer importer_bytes()."""
    try:
        b = base64.b64decode(b64)
    except Exception as e:
        return {"erreur": f"Fichier illisible : {e}"}
    return _importer_paquet(b, mode)


def _importer_paquet(b, mode="demander"):
    """Charge un modèle .riverlab (octets). Si un modèle existe déjà pour cette
    station, renvoie un 'conflit' pour laisser choisir : remplacer, garder le
    meilleur (R²), ou garder les deux (fusion des modèles compatibles)."""
    try:
        paquet = joblib.load(io.BytesIO(b))
    except Exception as e:
        return {"erreur": f"Fichier illisible ou corrompu : {e}"}
    if not isinstance(paquet, dict) or paquet.get("format") != "riverlab-1" or "code" not in paquet:
        return {"erreur": "Ce fichier n'est pas un modèle River Lab."}
    code = paquet["code"]
    nouveau = paquet.get("store")
    if not nouveau:
        return {"erreur": "Fichier sans modèle."}
    existant = STORE.get(code)

    if existant and mode == "demander":
        return {"conflit": True, "code": code, "nom": nouveau["meta"].get("nom", code),
                "actuel": _resume_modele(existant), "importe": _resume_modele(nouveau)}

    def poser_importe():
        STORE[code] = nouveau
        if paquet.get("zones"): ZONES[code] = paquet["zones"]
        if paquet.get("selection"): SELECTION[code] = paquet["selection"]

    if not existant or mode == "remplacer":
        poser_importe()
    elif mode == "meilleur":
        if _r2_max(nouveau) >= _r2_max(existant):
            poser_importe()
        # sinon on garde l'existant tel quel
    elif mode == "fusionner":
        # base = l'importé ; on y ajoute les modèles de l'existant s'ils sont compatibles
        if existant.get("feats") == nouveau.get("feats"):
            for nom in existant.get("modeles", {}):
                if nom not in nouveau["modeles"]:
                    nouveau["modeles"][nom] = existant["modeles"][nom]
                    nouveau["scores"][nom] = existant["scores"][nom]
                    if nom in existant.get("variance", {}):
                        nouveau.setdefault("variance", {})[nom] = existant["variance"][nom]
        poser_importe()
    else:
        poser_importe()

    _sauver(code)
    st = STORE.get(code, existant)
    return {"code": code, "nom": st["meta"].get("nom", code),
            "modeles": [NOMS_MODELE.get(n, n) for n in st["scores"].keys()],
            "r2": round(_r2_max(st) * 100), "horizon": st["meta"]["horizon"]}


def _supprimer(code, cible):
    chemin = f"{_PERSIST}/{code}.joblib"
    octets = os.path.getsize(chemin) if os.path.exists(chemin) else 0
    if cible == "donnees":
        st = STORE.get(code)
        libere = 0
        if st and "data" in st:
            d = st["data"]
            try:
                libere = int(sum(getattr(v, "nbytes", 0) for arr in d.values()
                                 for v in ([arr.values] if hasattr(arr, "values") else [arr])))
            except Exception:
                libere = 0
            st.pop("data", None)
        return {"libelle": "données d'entraînement (en mémoire)", "octets_liberes": libere}
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
            st.get("variance", {}).pop(nom, None)
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


async def meteo_moyenne(coords, start, end, forecast=False, temp_mode="moyenne"):
    """Météo journalière moyennée sur les points `coords` (1 requête batchée).
    En mode 'horaire', télécharge la température HORAIRE et la déplie en 24
    colonnes/jour (temp_h0…temp_h23), moyennées sur les points."""
    end = _cap_archive(end, forecast)
    base = "https://api.open-meteo.com/v1/forecast" if forecast else "https://archive-api.open-meteo.com/v1/archive"
    lat = ",".join(str(c[0]) for c in coords)
    lon = ",".join(str(c[1]) for c in coords)
    if temp_mode == "minmax":
        daily_vars = ["rain_sum", "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max", "snowfall_sum"]
    elif temp_mode == "horaire":
        daily_vars = ["rain_sum", "snowfall_sum"]
    else:
        daily_vars = ["rain_sum", "temperature_2m_mean", "snowfall_sum"]
    url = (f"{base}?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&daily={','.join(daily_vars)}")
    if temp_mode == "horaire":
        url += "&hourly=temperature_2m"
    data = await _fetch_json(url)
    points = data if isinstance(data, list) else [data]

    dailies, hourlies = [], []
    for p in points:
        if "daily" in p:
            dp = pd.DataFrame(p["daily"]); dp["date"] = pd.to_datetime(dp["time"]).dt.normalize()
            dailies.append(dp.drop(columns=["time"]))
        if "hourly" in p:
            hp = pd.DataFrame(p["hourly"]); hp["dt"] = pd.to_datetime(hp["time"])
            hourlies.append(hp.drop(columns=["time"]))
    if not dailies:
        return pd.DataFrame(columns=["date"] + _vars_meteo(temp_mode))
    df = pd.concat(dailies).groupby("date", as_index=False).mean()
    if temp_mode == "horaire" and hourlies:
        hp = pd.concat(hourlies)
        hp["date"] = hp["dt"].dt.normalize(); hp["h"] = hp["dt"].dt.hour
        piv = hp.groupby(["date", "h"])["temperature_2m"].mean().unstack("h")
        piv.columns = [f"temp_h{int(h)}" for h in piv.columns]
        df = pd.merge(df, piv.reset_index(), on="date", how="left")
    return df


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
    prog("Recherche du bassin versant", None); log("Lecture de la surface du bassin (Hub'Eau)…")
    surface = await _surface_bv(code)
    rayon_km = min(300.0, max(25.0, 2.5 * math.sqrt(surface))) if surface else 60.0
    dlat = rayon_km / 111.0
    dlon = rayon_km / (111.0 * math.cos(math.radians(lat0)))
    bbox = f"{lon0-dlon},{lat0-dlat},{lon0+dlon},{lat0+dlat}"

    arret()
    prog("Recherche du bassin versant", None); log("🌍 Récupération des bassins versants (Sandre)…")
    url = ("https://services.sandre.eaufrance.fr/geo/sandre?SERVICE=WFS&VERSION=1.0.0&REQUEST=GetFeature"
           f"&typeName=ZoneHydro_FXX&outputFormat=geojson&bbox={bbox}")
    data = await _fetch_json(url)
    zones = data.get("features", [])
    log(f"{len(zones)} zones hydrographiques trouvées dans le rayon de recherche.")
    await asyncio.sleep(0)

    retenues, _exclues = _zones_amont(zones, surface, code, log)
    geojson = {"type": "FeatureCollection", "features": retenues}

    prog("Repérage des zones amont", None); log("🎯 Génération des points à l'intérieur des zones…")
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
    prog("Analyse du relief", None); log(f"⛰️ Altitude de {n} points candidats (gratuit)…")
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

    prog("Sélection des meilleurs points", None)
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
    prog("Analyse pluie & neige", None)
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
def construire(df_eau, df_meteo, past, horizon, vars_meteo=None):
    vm = vars_meteo or VARS_METEO
    df = pd.merge(df_eau, df_meteo, on="date", how="inner").sort_values("date").reset_index(drop=True)
    feats, targets = [], []
    for i in range(1, past + 1):
        c = f"debit_J-{i}"; df[c] = df["debit_L_s"].shift(i); feats.append(c)
    for i in range(1, past + 1):
        for v in vm:
            c = f"{v}_J-{i}"; df[c] = df[v].shift(i); feats.append(c)
    for i in range(0, horizon + 1):
        for v in vm:
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
            prog("Entraînement du modèle",
                 int((idx + (h + 1) / nh) / nb * 100), idx + 1, nb)
            if h % 2 == 0:
                await asyncio.sleep(0)
        m = MultiGB(ests)
    else:
        return None, None
    pred = m.predict(Xte)
    r2 = r2_score(Yte, pred, multioutput="raw_values")
    return m, {"r2": float(r2_score(Yte, pred)), "detail": [float(x) for x in r2]}


MODELES_VAR = ("gradient_boosting", "ridge")
NOMS_VAR = {"gradient_boosting": "Gradient Boosting quantile", "ridge": "Ridge quantile", "foret": "Forêt aléatoire"}


def _reg_var_neuf(type_var):
    """Régresseur qui apprend la variance (ancien format, conservé pour compat)."""
    if type_var == "ridge":
        return Ridge(alpha=10.0)
    if type_var == "foret":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=60, max_depth=9, min_samples_leaf=20, n_jobs=1, random_state=42)
    return HistGradientBoostingRegressor(max_iter=70, max_depth=3, max_leaf_nodes=15,
                                         learning_rate=0.08, min_samples_leaf=40, l2_regularization=2.0,
                                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=8, random_state=42)


def _reg_quantile_neuf(type_var, q):
    """Régresseur de QUANTILE `q` (bord haut/bas), au choix de l'utilisateur.
    'ridge' = régression quantile linéaire ; sinon Gradient Boosting quantile."""
    if type_var == "ridge":
        from sklearn.linear_model import QuantileRegressor
        return QuantileRegressor(quantile=q, alpha=0.0, solver="highs")
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=100, max_depth=3,
                                         max_leaf_nodes=15, learning_rate=0.08, min_samples_leaf=40,
                                         l2_regularization=2.0, early_stopping=True, validation_fraction=0.1,
                                         n_iter_no_change=8, random_state=42)


def _reg_variance(X, r, cote, type_var):
    """Régresseur de variance d'UN côté (résidus positifs 'up' / négatifs 'down'),
    cible = résidu². Si un côté a trop peu de points, on prend tous les résidus."""
    masque = (r >= 0) if cote == "up" else (r < 0)
    if masque.sum() < 40:
        masque = np.ones(len(r), dtype=bool)
    reg = _reg_var_neuf(type_var)
    reg.fit(X[masque], r[masque] ** 2)
    return reg


# niveaux d'IC et alpha correspondant (couverture visée = 1 - alpha)
_IC_ALPHA = (("50", 0.50), ("95", 0.05), ("99", 0.01))


def _bornes(regs):
    """Plancher/plafond de VARIANCE par horizon (évite σ≈0 qui fait exploser les
    facteurs conformes, et σ énorme qui donne des IC délirants). None si absent."""
    vb = regs.get("var_bornes")
    if vb:
        return np.asarray(vb["floor"], float), np.asarray(vb["cap"], float)
    return None, None


def _borne_var(v, fl, cp, h):
    """Borne UNE variance (scalaire) à l'horizon h."""
    if fl is not None:
        return min(max(float(v), float(fl[h])), float(cp[h]))
    return max(1e-12, float(v))


def _sigmas_ligne(regs, X, horizons):
    """(sigma_haut, sigma_bas) pour UNE ligne X, par horizon (variances bornées)."""
    up, down = regs["up"], regs["down"]
    fl, cp = _bornes(regs)
    return {h: (float(np.sqrt(_borne_var(up[h].predict(X)[0], fl, cp, h))),
                float(np.sqrt(_borne_var(down[h].predict(X)[0], fl, cp, h)))) for h in horizons}


def _sigmas_mat(regs, X):
    """(sigma_haut, sigma_bas) matriciels (n, H) pour tout X (variances bornées)."""
    fl, cp = _bornes(regs)
    up = np.column_stack([r.predict(X) for r in regs["up"]])
    dn = np.column_stack([r.predict(X) for r in regs["down"]])
    if fl is not None:
        up = np.clip(up, fl, cp); dn = np.clip(dn, fl, cp)
    else:
        up = np.clip(up, 1e-12, None); dn = np.clip(dn, 1e-12, None)
    return np.sqrt(up), np.sqrt(dn)


_Z_IC = {"50": 0.674, "95": 1.960, "99": 2.576}


def _ic_par_horizon(regs, X, pred, horizons):
    """{niv: {h: (bas, haut)}} en L/s pour une ligne X. CQR (quantiles bas/haut +
    correction conforme Q CALIBRÉE PAR NIVEAU — 50/95/99 chacun sur les vrais
    résidus, un seul score par horizon) si dispo, sinon anciens formats (sigma
    conforme ou z-score) pour rétro-compat."""
    out = {}
    if regs.get("cqr"):
        Q = regs["Q"]
        for niv, _a in _IC_ALPHA:
            qn = Q[niv] if isinstance(Q, dict) else Q   # rétro-compat : ancien Q par horizon (95% seul)
            d = {}
            for h in horizons:
                lo = float(regs["q_lo"][h].predict(X)[0]); hi = float(regs["q_hi"][h].predict(X)[0])
                bas = lo - qn[h]; haut = hi + qn[h]      # Q<0 au 50% -> resserre ; Q>0 au 99% -> élargit
                if bas > haut:                            # 50% resserré au point de croiser -> point médian
                    bas = haut = 0.5 * (bas + haut)
                d[h] = (max(0.0, bas), max(0.0, haut))
            out[niv] = d
        return out
    # anciens formats
    sig = _sigmas_ligne(regs, X, horizons)
    conforme = "q_up" in regs
    for niv, _a in _IC_ALPHA:
        d = {}
        for h in horizons:
            su, sd = sig[h]
            if conforme:
                bas = pred[h] + regs["q_down"][niv][h] * sd
                haut = pred[h] + regs["q_up"][niv][h] * su
            else:
                z = _Z_IC[niv]
                bas = pred[h] - z * sd; haut = pred[h] + z * su
            d[h] = (max(0.0, float(bas)), float(haut))
        out[niv] = d
    return out


async def _fit_incertitude(base, Xm, Ym, Xc, Yc, Xe, Ye, targets, type_var="gradient_boosting", prog=None, arret=None):
    """Incertitude par CQR (Conformalized Quantile Regression), SANS fuite :
      • deux modèles de QUANTILE — bas 2,5 % et haut 97,5 % — appris sur le jeu
        MODÈLE (choix du type : Gradient Boosting quantile ou Ridge quantile) ;
      • correction conforme Q par horizon sur le jeu de CALIBRATION → garantit la
        couverture 95 % (Q = quantile 95% du score max(q_bas−y, y−q_haut)) ;
      • 50 % et 99 % obtenus par extrapolation gaussienne ASYMÉTRIQUE du 95 % ;
      • couverture + note mesurées sur le jeu d'ÉVALUATION (jamais vu).
    Aucune division par σ → plus d'intervalles qui explosent."""
    prog = prog or (lambda *a, **k: None); arret = arret or (lambda: None)
    nh = len(targets)
    Ym_v = Ym.values
    q_lo, q_hi = [], []
    for h in range(nh):
        arret()
        mlo = _reg_quantile_neuf(type_var, 0.025); mlo.fit(Xm, Ym_v[:, h]); q_lo.append(mlo)
        mhi = _reg_quantile_neuf(type_var, 0.975); mhi.fit(Xm, Ym_v[:, h]); q_hi.append(mhi)
        prog("Incertitude (CQR)", int((h + 1) / nh * 100), h + 1, nh)
        if h % 2 == 0:
            await asyncio.sleep(0)

    # --- correction conforme Q, sur le jeu de calibration : UN seul score par
    #     horizon (E), dont on prend TROIS quantiles (50/95/99). Q<0 au 50%
    #     resserre la bande ; Q>0 au 99% l'élargit. Les 3 niveaux sont donc
    #     calibrés sur les vrais résidus (cohérents), sans hypothèse gaussienne. ---
    Yc_v = Yc.values; nc = max(1, len(Yc_v))
    Q = {niv: [] for niv, _a in _IC_ALPHA}
    for h in range(nh):
        lo = q_lo[h].predict(Xc); hi = q_hi[h].predict(Xc)
        E = np.maximum(lo - Yc_v[:, h], Yc_v[:, h] - hi)   # score de non-conformité (calculé 1 fois)
        for niv, a in _IC_ALPHA:
            niveau = min(1.0, (1 - a) * (nc + 1) / nc)     # quantile conforme du niveau
            Q[niv].append(float(np.quantile(E, niveau)))
    regs = {"cqr": True, "q_lo": q_lo, "q_hi": q_hi, "Q": Q, "type_var": type_var}

    # --- couverture réelle + note, sur le jeu d'ÉVALUATION (jamais vu) ---
    couverture = {}
    if Xe is not None and len(Xe):
        ye = Ye.values
        lo_e = np.column_stack([q_lo[h].predict(Xe) for h in range(nh)])
        hi_e = np.column_stack([q_hi[h].predict(Xe) for h in range(nh)])
        for niv, _a in _IC_ALPHA:
            Qn = np.asarray(Q[niv])
            L = lo_e - Qn; U = hi_e + Qn
            crois = L > U                                  # 50% resserré au point de croiser
            mid = 0.5 * (L + U); L = np.where(crois, mid, L); U = np.where(crois, mid, U)
            couverture[niv] = round(float(((ye >= L) & (ye <= U)).mean()) * 100, 1)
        # NOTE = score d'intervalle de Winkler au 95% (pénalise largeur ET dépassements)
        Q95 = np.asarray(Q["95"]); U95 = hi_e + Q95; L95 = lo_e - Q95
        a95 = 0.05
        largeur = U95 - L95
        interval_score = largeur + (2 / a95) * np.maximum(0, L95 - ye) + (2 / a95) * np.maximum(0, ye - U95)
        echelle = max(1e-9, float(np.mean(np.abs(ye))))
        is_norm = float(np.mean(interval_score)) / echelle
        couverture["note"] = int(round(100.0 / (1.0 + is_norm)))
        couverture["largeur"] = round(float(np.mean(largeur)) / echelle * 100)
    regs["couverture"] = couverture
    return regs


def _decouper3(df, feats, targets, mode_split, part_sigma, part_eval):
    """Découpe en TROIS jeux disjoints pour éviter les fuites : MODÈLE (le reste),
    ÉCART-TYPE (part_sigma) et ÉVALUATION (part_eval). Par années tirées au sort
    ou chronologiquement. Renvoie (Xm,Ym, Xs,Ys, Xe,Ye, eval_df)."""
    part_sigma = min(0.6, max(0.1, float(part_sigma)))
    part_eval = min(0.6, max(0.1, float(part_eval)))
    if part_sigma + part_eval > 0.9:            # garder ≥10 % pour le modèle
        f = 0.9 / (part_sigma + part_eval); part_sigma *= f; part_eval *= f
    n = len(df)
    grp = np.zeros(n, dtype=int)                # 0 = modèle, 1 = écart-type, 2 = éval
    if mode_split == "chronologique":
        i1 = int(n * (1 - part_sigma - part_eval)); i2 = int(n * (1 - part_eval))
        grp[i1:i2] = 1; grp[i2:] = 2
    else:  # annees_aleatoires
        annees = df["date"].dt.year.values
        uniques = np.array(sorted(set(annees)))
        rng = np.random.default_rng(42); rng.shuffle(uniques)
        n_sig = max(1, round(len(uniques) * part_sigma))
        n_ev = max(1, round(len(uniques) * part_eval))
        a_sig, a_ev = set(uniques[:n_sig]), set(uniques[n_sig:n_sig + n_ev])
        grp = np.array([1 if a in a_sig else (2 if a in a_ev else 0) for a in annees])
        if not (grp == 0).any():                # garde-fou : modèle non vide
            grp = np.zeros(n, dtype=int)
            i1 = int(n * (1 - part_sigma - part_eval)); i2 = int(n * (1 - part_eval))
            grp[i1:i2] = 1; grp[i2:] = 2
    mm, ms, me = grp == 0, grp == 1, grp == 2
    return (df[feats][mm], df[targets][mm], df[feats][ms], df[targets][ms],
            df[feats][me], df[targets][me], df[me].reset_index(drop=True))


async def entrainer(code, coords=None, past=15, horizon=15, annees=10,
                    modeles=("ridge", "lineaire", "gradient_boosting"),
                    log=None, prog=None, arret=None, debut_str=None, fin_str=None, seuil_pca=99,
                    mode_split="annees_aleatoires", part_test=0.2, temp_mode="moyenne",
                    var_modele="gradient_boosting", part_sigma=0.3, part_eval=0.2):
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
    vm = _vars_meteo(temp_mode)
    arret()
    prog("Téléchargement de la météo", None); log(f"Téléchargement de la météo sur {len(coords)} points…")
    df_meteo = await meteo_moyenne(coords, s, e, temp_mode=temp_mode)
    t_dl = time.time() - t0
    await asyncio.sleep(0)

    prog("Préparation des données", None); log("Construction des variables explicatives…")
    df, feats, targets = construire(df_eau, df_meteo, past, horizon, vars_meteo=vm)
    n = len(df)
    if n < 50:
        raise ValueError(f"Trop peu de données exploitables ({n} lignes) sur cette période.")
    Xm, Ym, Xs, Ys, Xe, Ye, test_df = _decouper3(df, feats, targets, mode_split, part_sigma, part_eval)
    log(f"Découpage {mode_split} : modèle {len(Xm)} j · écart-type {len(Xs)} j · évaluation {len(Xe)} j.")
    await asyncio.sleep(0)

    resultats, objets, variances = {}, {}, {}
    tf = time.time()
    nb = len(modeles)
    for idx, nom in enumerate(modeles):
        arret()
        prog(f"Entraînement : {NOMS_MODELE.get(nom, nom)}", int(idx / nb * 100), idx + 1, nb)
        log(f"Entraînement : {NOMS_MODELE.get(nom, nom)}…")
        await asyncio.sleep(0)
        m, sc = await _fit_modele(nom, Xm, Ym, Xe, Ye, targets, prog, arret, idx, nb, seuil_pca)
        if m is None:
            continue
        objets[nom] = m
        resultats[nom] = sc
        log(f"Estimation de l'incertitude (conforme, modèle : {NOMS_VAR.get(var_modele, var_modele)})…")
        variances[nom] = await _fit_incertitude(m, Xm, Ym, Xs, Ys, Xe, Ye, targets, var_modele, prog, arret)
        cov = variances[nom].get("couverture", {})
        if cov.get("95") is not None:
            log(f"   couverture réelle de l'IC 95% (sur l'évaluation) : {cov['95']} %")
    t_fit = time.time() - tf
    prog("Entraînement terminé", 100)

    STORE[code] = {"test": test_df, "feats": feats, "targets": targets, "modeles": objets,
                   # données gardées en mémoire pour ré-entraîner un autre modèle sans re-télécharger
                   "data": {"Xm": Xm, "Ym": Ym, "Xs": Xs, "Ys": Ys, "Xe": Xe, "Ye": Ye},
                   "scores": resultats, "variance": variances,
                   "meta": {"nom": infos["nom"], "coords": coords, "past": past, "horizon": horizon,
                            "vars_meteo": vm, "temp_mode": temp_mode,
                            "annees_test": sorted({int(y) for y in test_df["date"].dt.year.unique()})}}
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
                               seuil_pca=float(body.get("seuil_energie") or 99),
                               mode_split=body.get("mode_split", "annees_aleatoires"),
                               part_test=float(body.get("part_test") or 0.2),
                               temp_mode=body.get("temp_mode", "moyenne"),
                               var_modele=body.get("var_modele", "gradient_boosting"),
                               part_sigma=float(body.get("part_sigma") or 0.3),
                               part_eval=float(body.get("part_eval") or 0.2))
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


async def ajouter_modeles(code, noms, log, prog, arret, seuil_pca=99, var_modele="gradient_boosting"):
    """Entraîne un/des modèle(s) SUPPLÉMENTAIRE(S) sur les données déjà en mémoire
    (aucun re-téléchargement) et les ajoute à STORE[code]."""
    st = STORE.get(code)
    if not st or "data" not in st or "Xm" not in st.get("data", {}):
        raise ValueError("Les données d'entraînement ne sont plus en mémoire (modèle rechargé ou importé). "
                         "Ré-analyse la rivière pour entraîner d'autres modèles.")
    d = st["data"]; targets = st["targets"]; nb = len(noms)
    for idx, nom in enumerate(noms):
        arret()
        prog(f"Entraînement : {NOMS_MODELE.get(nom, nom)}", int(idx / nb * 100), idx + 1, nb)
        log(f"Entraînement : {NOMS_MODELE.get(nom, nom)}…")
        await asyncio.sleep(0)
        m, sc = await _fit_modele(nom, d["Xm"], d["Ym"], d["Xe"], d["Ye"], targets, prog, arret, idx, nb, seuil_pca)
        if m is None:
            continue
        st["modeles"][nom] = m; st["scores"][nom] = sc
        st.setdefault("variance", {})[nom] = await _fit_incertitude(m, d["Xm"], d["Ym"], d["Xs"], d["Ys"], d["Xe"], d["Ye"], targets, var_modele, prog, arret)
        log(f"   → {NOMS_MODELE.get(nom, nom)} : fiabilité {sc['r2'] * 100:.0f} %")
    prog("Terminé", 100)
    return {"modeles": list(st["scores"].keys())}


async def _run_ajouter(jid, code, noms, seuil_pca=99, var_modele="gradient_boosting"):
    job = JOBS[jid]
    log, prog, arret = _cbs(job)
    try:
        noms = [n for n in noms if n in MODELES_DISPO]
        if not noms:
            raise ValueError("Aucun modèle valide à entraîner.")
        job["resultat"] = await ajouter_modeles(code, noms, log, prog, arret, seuil_pca=seuil_pca, var_modele=var_modele)
        _sauver(code)
        log("✅ Terminé (sauvegardé sur cet appareil).")
        job["statut"] = "termine"
    except _Arret:
        log("⏹️ Arrêté à la demande."); job["statut"] = "arrete"
    except Exception as e:
        import traceback
        job["erreur"] = f"{type(e).__name__}: {e}"
        log("❌ " + traceback.format_exc()[-700:]); job["statut"] = "erreur"


async def _run_incertitude(jid, code, modele, var_modele):
    """Ré-entraîne SEULEMENT l'incertitude d'un modèle existant (le modèle de débit
    ne change pas), avec le modèle de variance choisi. Nécessite les données en mémoire."""
    job = JOBS[jid]
    log, prog, arret = _cbs(job)
    try:
        st = STORE.get(code)
        if not st or "data" not in st or "Xm" not in st.get("data", {}):
            raise ValueError("Données d'entraînement absentes de la mémoire : ré-analyse la rivière.")
        if modele not in st.get("modeles", {}):
            raise ValueError("Modèle introuvable.")
        d = st["data"]
        log(f"Ré-entraînement de l'incertitude ({NOMS_VAR.get(var_modele, var_modele)}) pour {NOMS_MODELE.get(modele, modele)}…")
        st.setdefault("variance", {})[modele] = await _fit_incertitude(
            st["modeles"][modele], d["Xm"], d["Ym"], d["Xs"], d["Ys"], d["Xe"], d["Ye"], st["targets"], var_modele, prog, arret)
        _sauver(code)
        cov = st["variance"][modele].get("couverture", {})
        log(f"✅ IC 95% couvre {cov.get('95')} % · note {cov.get('note')}/100 · largeur ~{cov.get('largeur')}% du débit.")
        job["resultat"] = {"couverture": cov}
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
                               seuil_pca=float(body.get("seuil_energie") or 99),
                               mode_split=body.get("mode_split", "annees_aleatoires"),
                               part_test=float(body.get("part_test") or 0.2),
                               temp_mode=body.get("temp_mode", "moyenne"),
                               var_modele=body.get("var_modele", "gradient_boosting"),
                               part_sigma=float(body.get("part_sigma") or 0.3),
                               part_eval=float(body.get("part_eval") or 0.2))
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


async def _fenetre_test(code, date, meta):
    """Télécharge juste ce qu'il faut pour tester à `date` (débit + météo sur
    ~[date-past-20 j, date+horizon+3 j]) et construit la ligne de features."""
    past, horizon, coords = meta["past"], meta["horizon"], meta["coords"]
    vm = meta.get("vars_meteo", VARS_METEO); temp_mode = meta.get("temp_mode", "moyenne")
    s = (date - pd.Timedelta(days=past + 20)).strftime("%Y-%m-%d")
    e = (date + pd.Timedelta(days=horizon + 3)).strftime("%Y-%m-%d")
    df_eau = await debit(code, s, e)
    df_meteo = await meteo_moyenne(coords, s, e, temp_mode=temp_mode)
    df, _, _ = construire(df_eau, df_meteo, past, horizon, vars_meteo=vm)
    return df


async def backtest(code, modele, date_str, nb_jours=None, hybride=True):
    """Fan chart (prévision vs réalité) pour une date du jeu de TEST. Vérifie que la
    date est bien une date de test (sinon renvoie les années de test), et si la ligne
    n'est pas déjà en mémoire, télécharge à la demande la petite fenêtre nécessaire."""
    st = STORE.get(code)
    if not st or modele not in st["modeles"]:
        return None
    feats, targets, meta = st["feats"], st["targets"], st["meta"]
    horizon = meta["horizon"]
    h_max = horizon if nb_jours is None else max(1, min(int(nb_jours), horizon))
    date = pd.to_datetime(date_str)
    test = st.get("test")
    annees_test = meta.get("annees_test") or (sorted({int(y) for y in test["date"].dt.year.unique()}) if test is not None else [])
    if annees_test and date.year not in annees_test:
        return {"erreur": f"Le {date.strftime('%d/%m/%Y')} ne fait pas partie du jeu de test. "
                          f"Années de test : {', '.join(map(str, annees_test))}.",
                "annees_test": annees_test}
    # ligne déjà calculée ? sinon on télécharge la fenêtre concernée
    ligne = test[test["date"] == date] if test is not None else None
    contexte_df = test
    if ligne is None or ligne.empty:
        contexte_df = await _fenetre_test(code, date, meta)
        ligne = contexte_df[contexte_df["date"] == date]
        if ligne.empty:
            return {"erreur": "Données indisponibles pour cette date (débit ou météo manquants)."}
    X = ligne[feats]
    pred = st["modeles"][modele].predict(X)[0][:h_max + 1]
    vrai = ligne[[f"cible_J+{h}" for h in range(h_max + 1)]].values[0]
    debit0 = ligne["debit_L_s"].values[0]
    contexte = contexte_df[(contexte_df["date"] >= date - pd.Timedelta(days=15)) & (contexte_df["date"] <= date)]
    D = 1000.0
    dates = [date] + list(pd.date_range(date + pd.Timedelta(days=1), periods=h_max + 1))
    pts = [{"date": d.strftime("%Y-%m-%d"),
            "prev": float(v) / D, "reel": float(r) / D}
           for d, v, r in zip(dates, [debit0] + list(pred), [debit0] + list(vrai))]
    # intervalles de confiance (conformes si dispo, sinon z-score ; asymétriques)
    regs = st.get("variance", {}).get(modele) if hybride else None
    hybride = False
    couverture = None
    if isinstance(regs, dict):
        hybride = True
        couverture = regs.get("couverture")
        ic = _ic_par_horizon(regs, X, pred, range(h_max + 1))
        for niv, _a in _IC_ALPHA:
            for k, p in enumerate(pts):
                if k == 0:
                    p[f"ic{niv}_bas"] = p[f"ic{niv}_haut"] = float(debit0) / D
                else:
                    bas, haut = ic[niv][k - 1]
                    p[f"ic{niv}_bas"] = bas / D; p[f"ic{niv}_haut"] = haut / D
    return {"pivot": date.strftime("%Y-%m-%d"), "hybride": hybride, "couverture": couverture,
            "observe": [{"date": d.strftime("%Y-%m-%d"), "debit": float(x) / D}
                        for d, x in zip(contexte["date"], contexte["debit_L_s"])],
            "points": pts}


async def prevision(code, modele, nb_jours=None, hybride=True):
    """Prévision réelle à partir d'AUJOURD'HUI : débit récent (Hub'Eau) + météo
    PRÉVISIONNELLE (Open-Meteo forecast) sur les points retenus, puis prédiction
    des prochains jours. Renvoie un fan chart (pivot = aujourd'hui)."""
    st = STORE.get(code)
    if not st or modele not in st["modeles"]:
        return {"erreur": "modèle non entraîné"}
    feats, meta = st["feats"], st["meta"]
    coords, past, horizon = meta["coords"], meta["past"], meta["horizon"]
    vm = meta.get("vars_meteo", VARS_METEO)
    temp_mode = meta.get("temp_mode", "moyenne")
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
            df_meteo = await meteo_moyenne(coords, start, end, forecast=True, temp_mode=temp_mode)
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
        for v in vm:
            f[f"{v}_J-{i}"] = float(dm.loc[date_J0 - pd.Timedelta(days=i), v])
    for i in range(0, horizon + 1):
        for v in vm:
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
    # intervalles de confiance (conformes si dispo ; asymétriques)
    regs = st.get("variance", {}).get(modele) if hybride else None
    hybride = False
    couverture = None
    if isinstance(regs, dict):
        hybride = True
        couverture = regs.get("couverture")
        ic = _ic_par_horizon(regs, X, pred, horizons)
        for niv, _a in _IC_ALPHA:
            points[0][f"ic{niv}_bas"] = points[0][f"ic{niv}_haut"] = debit0 / D
            for k, h in enumerate(horizons, start=1):
                bas, haut = ic[niv][h]
                points[k][f"ic{niv}_bas"] = bas / D; points[k][f"ic{niv}_haut"] = haut / D
    sc = st["scores"].get(modele, {})
    return {"pivot": today.strftime("%Y-%m-%d"), "hybride": hybride, "couverture": couverture,
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
import re
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
        variance = st.get("variance", {})
        for nom, r in st.get("scores", {}).items():
            v = variance.get(nom)
            modeles.append({"nom": nom, "score": r["r2"], "espace": "brut",
                            "hybride": isinstance(v, dict),
                            "couverture": v.get("couverture") if isinstance(v, dict) else None})
    # points finaux : la sélection auto/manuelle si elle existe, sinon ceux de l'entraînement
    coords_finales = None
    if sel and sel.get("coords_finales"):
        coords_finales = sel["coords_finales"]
    elif st:
        coords_finales = st["meta"]["coords"]
    # données d'entraînement encore en mémoire (permet d'ajouter modèle / PCA / incertitude)
    donnees_mem = bool(st and "data" in st and "Xm" in st.get("data", {}))
    octets_donnees = 0
    if donnees_mem:
        try:
            octets_donnees = int(sum(v.values.nbytes if hasattr(v, "values") else getattr(v, "nbytes", 0)
                                     for v in st["data"].values()))
        except Exception:
            octets_donnees = 0
    return {"code_station": code, "nom_station": info.get("nom"),
            "donnees_en_memoire": donnees_mem, "octets_donnees": octets_donnees,
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
    dts = sorted(d.strftime("%Y-%m-%d") for d in st["test"]["date"])
    annees = st["meta"].get("annees_test") or sorted({int(y) for y in st["test"]["date"].dt.year.unique()})
    return {"min": dts[0], "max": dts[-1], "dates": dts, "annees_test": annees}


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
    if p == "/api/importer" and method == "POST":
        return _json.dumps(importer(body.get("b64", ""), body.get("mode", "demander")))
    m = re.match(r"^/api/riviere/([^/]+)/exporter$", p)
    if m:
        return _json.dumps(exporter(m.group(1)))

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
                                           int(g1("nb_jours")) if g1("nb_jours") else None,
                                           hybride=g1("hybride", "1") != "0"))
    m = re.match(r"^/api/riviere/([^/]+)/pca$", p)
    if m:
        return _json.dumps(pca_analyse(m.group(1), float(g1("seuil") or 99)))
    m = re.match(r"^/api/riviere/([^/]+)/backtest$", p)
    if m:
        r = await backtest(m.group(1), g1("modele", "gradient_boosting"), g1("date"), int(g1("nb_jours") or 15),
                           hybride=g1("hybride", "1") != "0")
        st = STORE.get(m.group(1))
        if r and st and "points" in r:
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
                                           seuil_pca=float(body.get("seuil_energie") or 99),
                                           var_modele=body.get("var_modele", "gradient_boosting")))
        return _json.dumps({"job_id": jid})
    m = re.match(r"^/api/riviere/([^/]+)/entrainer$", p)
    if m and method == "POST":
        jid = _job_nouveau("entrainer", m.group(1))
        asyncio.ensure_future(_run_ajouter(jid, m.group(1), body.get("modeles") or [],
                                           seuil_pca=float(body.get("seuil_energie") or 99),
                                           var_modele=body.get("var_modele", "gradient_boosting")))
        return _json.dumps({"job_id": jid})
    m = re.match(r"^/api/riviere/([^/]+)/incertitude$", p)
    if m and method == "POST":
        jid = _job_nouveau("incertitude", m.group(1))
        asyncio.ensure_future(_run_incertitude(jid, m.group(1), body.get("modele"),
                                               body.get("var_modele", "gradient_boosting")))
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

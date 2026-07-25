"""
Persistance légère d'une River entraînée.

Principe (voulu pour ne pas garder des Go de données par rivière) :
- On sauvegarde : les modèles entraînés (+ réseaux de variance), le scaler/PCA,
  les points météo retenus, les métadonnées (features, horizons, scores, zones).
- On garde AUSSI le jeu de test (X_test/Y_test/dates_test, ~25 Mo) : c'est lui
  qui permet de rejouer les prédictions passées (backtest) et l'audit
  d'incertitude sans rien retélécharger.
- On peut ensuite supprimer les gros fichiers (X_train ~100 Mo, cache météo
  ~40 Mo) via nettoyer_gros_fichiers() : la prévision des 15 prochains jours
  ne dépend que des modèles + points, pas de ces fichiers.

Structure sur disque :
  data/{code_station}/
    meta.json
    modeles/
      gradient_boosting.joblib, ridge_brut.joblib, lineaire_pca.joblib,
      keras_brut.keras, variance_{nom}.keras, scaler_pca.joblib, pca_model.joblib
    evaluation/
      X_test.csv, Y_test.csv, dates_test.csv
    (+ les fichiers de travail produits par river.py tant qu'on n'a pas nettoyé)
"""

import os
import json
import shutil

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ------------------------------------------------------------------
# Équivalents module-level (donc picklables/reconstructibles) des classes
# locales créées par River.entrainer_incertitude(). Même interface :
# predict(), estimators_[i].pred_dist(X) -> objet avec .params['loc'/'scale'].
# ------------------------------------------------------------------
class DistributionGaussienne:
    def __init__(self, loc, scale):
        self.params = {'loc': loc, 'scale': scale}


class EstimateurHorizon:
    def __init__(self, model_base, nn_model, horizon_idx):
        self.model_base = model_base
        self.nn_model = nn_model
        self.horizon_idx = horizon_idx

    def pred_dist(self, X):
        mu_all = self.model_base.predict(X)
        log_var_all = self.nn_model.predict(X, verbose=0)
        mu = mu_all[:, self.horizon_idx]
        std = np.sqrt(np.exp(log_var_all[:, self.horizon_idx]))
        return DistributionGaussienne(loc=mu, scale=std)


class ModeleHybrideRecharge:
    def __init__(self, model_base, nn_model, n_horizons):
        self.model_base = model_base
        self.nn_model = nn_model
        self.estimators_ = [EstimateurHorizon(model_base, nn_model, i) for i in range(n_horizons)]

    def predict(self, X):
        return self.model_base.predict(X)


def _est_modele_keras(model):
    return isinstance(model, tf.keras.Model)


def sauvegarder_riviere(riviere, dossier, params_hourly=None, params_daily=None, **_ignores):
    """Sauvegarde l'état entraîné d'une River dans `dossier` (voir docstring module)."""
    os.makedirs(dossier, exist_ok=True)
    d_modeles = os.path.join(dossier, "modeles")
    d_eval = os.path.join(dossier, "evaluation")
    os.makedirs(d_modeles, exist_ok=True)
    os.makedirs(d_eval, exist_ok=True)

    meta = {
        "code_station": riviere.code_station,
        "nom_station": riviere.nom_station,
        "lat_station": riviere.lat_station,
        "lon_station": riviere.lon_station,
        "coords_finales": [list(c) for c in riviere.coords_finales] if riviere.coords_finales else None,
        "params_hourly": params_hourly,
        "params_daily": params_daily,
        "agregations_horaires": riviere.agregations_horaires or None,
        "past_day": riviere.past_day2,
        "predict_day": riviere.predict_day2,
        "split_date": str(riviere.split_date2) if riviere.split_date2 is not None else None,
        "features_X2": riviere.features_X2,
        "targets_Y2": riviere.targets_Y2,
        "n_composantes_pca": riviere.n_composantes_pca,
        "noms_zones": riviere.noms_zones,
        "points_par_zone": {nom: [list(p) for p in pts] for nom, pts in riviere.points_par_zone.items()} if riviere.points_par_zone else None,
        "points_preselectionnes": [list(p) for p in riviere.points_preselectionnes] if riviere.points_preselectionnes else None,
        "altitudes_preselection": riviere._altitudes_preselection,
        "modeles": {},
    }

    for nom, info in riviere.modeles.items():
        model = info["model"]
        if _est_modele_keras(model):
            chemin = os.path.join(d_modeles, f"{nom}.keras")
            model.save(chemin)
            format_modele = "keras"
        else:
            chemin = os.path.join(d_modeles, f"{nom}.joblib")
            joblib.dump(model, chemin, compress=3)   # compresse (le GB passe de ~130 Mo à ~30 Mo)
            format_modele = "joblib"

        a_variance = "modele_variance" in info
        if a_variance:
            info["modele_variance"].save(os.path.join(d_modeles, f"variance_{nom}.keras"))

        scores_detail = info.get("scores_detail")
        meta["modeles"][nom] = {
            "format": format_modele,
            "score": float(info["score"]),
            "scores_detail": [float(s) for s in scores_detail] if scores_detail is not None else None,
            "espace": info.get("espace", "brut"),
            "variance": a_variance,
        }

    if riviere.scaler_pca is not None:
        joblib.dump(riviere.scaler_pca, os.path.join(d_modeles, "scaler_pca.joblib"))
    if riviere.pca_model is not None:
        joblib.dump(riviere.pca_model, os.path.join(d_modeles, "pca_model.joblib"))

    if riviere.geojson_bassins is not None:
        with open(os.path.join(dossier, "zones.geojson"), "w", encoding="utf-8") as f:
            json.dump(riviere.geojson_bassins, f)

    # Jeu de test pour les backtests (le train, lui, pourra être supprimé).
    if riviere.X_test2 is not None:
        riviere.X_test2.to_csv(os.path.join(d_eval, "X_test.csv"), index=False)
        riviere.Y_test2.to_csv(os.path.join(d_eval, "Y_test.csv"), index=False)
        if riviere.test2 is not None and 'date' in riviere.test2.columns:
            riviere.test2[['date', 'debit_L_s']].to_csv(os.path.join(d_eval, "dates_test.csv"), index=False)

    with open(os.path.join(dossier, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return meta


def _inferer_agregations(params_hourly, features_X2):
    """Reconstruit le mode d'agrégation de chaque variable horaire à partir des
    NOMS de colonnes de features sauvegardées -- indispensable pour que les
    modèles entraînés AVANT l'ajout du champ 'agregations_horaires' (features
    en `var_h0`..`var_h23`) continuent de prédire correctement. Reconnaît :
    plusieurs `var_hN` -> "horaire" ; seulement `var_h0` -> "minuit" ;
    `var_moy` -> "moyenne"."""
    if not params_hourly or not features_X2:
        return None
    import re
    agg = {}
    for var in params_hourly:
        heures = set()
        a_moyenne = False
        for col in features_X2:
            base = re.sub(r"_J[+-]\d+$", "", col)
            if base == f"{var}_moy":
                a_moyenne = True
            m = re.match(rf"^{re.escape(var)}_h(\d+)$", base)
            if m:
                heures.add(int(m.group(1)))
        if a_moyenne:
            agg[var] = "moyenne"
        elif heures == {0}:
            agg[var] = "minuit"
        elif len(heures) > 1:
            agg[var] = "horaire"
    return agg or None


def charger_riviere(riviere, dossier):
    """Restaure dans une instance River fraîche tout ce qui a été sauvegardé.
    Retourne le meta dict. Après ça : backtest, audit d'incertitude et
    prévision réelle fonctionnent ; le ré-entraînement, lui, nécessiterait de
    reconstruire les données complètes (X_train n'est pas conservé)."""
    with open(os.path.join(dossier, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)

    d_modeles = os.path.join(dossier, "modeles")
    d_eval = os.path.join(dossier, "evaluation")

    riviere.coords_finales = [tuple(c) for c in meta["coords_finales"]] if meta.get("coords_finales") else None
    riviere.past_day2 = meta.get("past_day")
    riviere.predict_day2 = meta.get("predict_day")
    riviere.split_date2 = meta.get("split_date")
    riviere.agregations_horaires = (meta.get("agregations_horaires")
                                    or _inferer_agregations(meta.get("params_hourly"), meta.get("features_X2"))
                                    or {})
    riviere.features_X2 = meta.get("features_X2")
    riviere.targets_Y2 = meta.get("targets_Y2")
    riviere.n_composantes_pca = meta.get("n_composantes_pca")
    riviere.noms_zones = meta.get("noms_zones")
    if meta.get("points_par_zone"):
        riviere.points_par_zone = {nom: [tuple(p) for p in pts] for nom, pts in meta["points_par_zone"].items()}
        riviere.listes_de_points = list(riviere.points_par_zone.values())
    if meta.get("points_preselectionnes"):
        riviere.points_preselectionnes = [tuple(p) for p in meta["points_preselectionnes"]]
        riviere._altitudes_preselection = meta.get("altitudes_preselection")

    chemin_zones = os.path.join(dossier, "zones.geojson")
    if os.path.exists(chemin_zones):
        with open(chemin_zones, encoding="utf-8") as f:
            riviere.geojson_bassins = json.load(f)

    chemin_scaler = os.path.join(d_modeles, "scaler_pca.joblib")
    if os.path.exists(chemin_scaler):
        riviere.scaler_pca = joblib.load(chemin_scaler)
    chemin_pca = os.path.join(d_modeles, "pca_model.joblib")
    if os.path.exists(chemin_pca):
        riviere.pca_model = joblib.load(chemin_pca)

    for nom, infos in meta.get("modeles", {}).items():
        if infos["format"] == "keras":
            model = tf.keras.models.load_model(os.path.join(d_modeles, f"{nom}.keras"), compile=False)
        else:
            model = joblib.load(os.path.join(d_modeles, f"{nom}.joblib"))

        info = {
            "model": model,
            "score": infos["score"],
            "scores_detail": np.array(infos["scores_detail"]) if infos.get("scores_detail") else None,
            "espace": infos.get("espace", "brut"),
        }

        if infos.get("variance"):
            model_variance = tf.keras.models.load_model(
                os.path.join(d_modeles, f"variance_{nom}.keras"), compile=False)
            n_horizons = (riviere.predict_day2 or 15) + 1
            info["modele_variance"] = model_variance
            info["hybride"] = ModeleHybrideRecharge(model, model_variance, n_horizons)

        riviere.modeles[nom] = info

    # Jeu de test pour les backtests.
    chemin_X_test = os.path.join(d_eval, "X_test.csv")
    if os.path.exists(chemin_X_test):
        riviere.X_test2 = pd.read_csv(chemin_X_test)
        riviere.Y_test2 = pd.read_csv(os.path.join(d_eval, "Y_test.csv"))
        chemin_dates = os.path.join(d_eval, "dates_test.csv")
        if os.path.exists(chemin_dates):
            dates_test = pd.read_csv(chemin_dates, parse_dates=['date'])
            test2 = pd.concat([dates_test.reset_index(drop=True),
                               riviere.X_test2.reset_index(drop=True),
                               riviere.Y_test2.reset_index(drop=True)], axis=1)
            riviere.test2 = test2
            riviere.df_final = test2.sort_values('date').reset_index(drop=True)

    return meta


FICHIERS_LOURDS = [
    "cache_eau_{code}.csv",
    "cache_meteo_{code}.csv",
    "X_train_{code}.csv",
    "Y_train_{code}.csv",
    "X_test_{code}.csv",
    "Y_test_{code}.csv",
    "dates_train_{code}.csv",
    "dates_test_{code}.csv",
    "X_train_pca_{code}.csv",
    "X_test_pca_{code}.csv",
]


def taille_gros_fichiers(dossier, code):
    """Liste (nom, octets) des fichiers de travail lourds encore présents."""
    resultats = []
    for gabarit in FICHIERS_LOURDS:
        chemin = os.path.join(dossier, gabarit.format(code=code))
        if os.path.exists(chemin):
            resultats.append((os.path.basename(chemin), os.path.getsize(chemin)))
    return resultats


def nettoyer_gros_fichiers(dossier, code):
    """Supprime les fichiers de travail lourds. Refuse si la sauvegarde des
    modèles n'existe pas encore (sinon on perdrait tout)."""
    if not os.path.exists(os.path.join(dossier, "meta.json")):
        raise Exception("Sauvegarde absente (meta.json) : sauvegarde d'abord les modèles avant de nettoyer.")

    octets_liberes = 0
    supprimes = []
    for gabarit in FICHIERS_LOURDS:
        chemin = os.path.join(dossier, gabarit.format(code=code))
        if os.path.exists(chemin):
            octets_liberes += os.path.getsize(chemin)
            os.remove(chemin)
            supprimes.append(os.path.basename(chemin))
    return supprimes, octets_liberes


# ==========================================================================
# GESTION FINE DU STOCKAGE (inventaire + suppression choisie)
# ==========================================================================
def _taille_dossier(chemin):
    total = 0
    if os.path.isdir(chemin):
        for racine, _, fichiers in os.walk(chemin):
            for f in fichiers:
                try:
                    total += os.path.getsize(os.path.join(racine, f))
                except OSError:
                    pass
    elif os.path.isfile(chemin):
        total = os.path.getsize(chemin)
    return total


def inventaire_station(dossier, code):
    """Détail du stockage d'une station : taille totale + par catégorie
    (modèles un par un, jeu de test, fichiers de travail), pour que
    l'utilisateur choisisse précisément quoi supprimer."""
    if not os.path.isdir(dossier):
        return None
    d_modeles = os.path.join(dossier, "modeles")
    meta = {}
    chemin_meta = os.path.join(dossier, "meta.json")
    if os.path.exists(chemin_meta):
        try:
            with open(chemin_meta, encoding="utf-8") as f:
                meta = json.load(f)
        except ValueError:
            meta = {}

    # Modèles : un modèle = son fichier principal + son réseau de variance.
    modeles = []
    if os.path.isdir(d_modeles):
        noms = set()
        for f in os.listdir(d_modeles):
            for suff in (".joblib", ".keras"):
                if f.endswith(suff) and not f.startswith("variance_") and f not in ("scaler_pca.joblib", "pca_model.joblib"):
                    noms.add(f[: -len(suff)])
        for nom in sorted(noms):
            octets = 0
            for cand in (f"{nom}.joblib", f"{nom}.keras", f"variance_{nom}.keras"):
                octets += _taille_dossier(os.path.join(d_modeles, cand))
            score = (meta.get("modeles", {}).get(nom) or {}).get("score")
            modeles.append({"nom": nom, "octets": octets, "score": score})

    annexes = sum(_taille_dossier(os.path.join(d_modeles, x))
                  for x in ("scaler_pca.joblib", "pca_model.joblib"))
    octets_test = _taille_dossier(os.path.join(dossier, "evaluation"))
    octets_travail = sum(os.path.getsize(os.path.join(dossier, g.format(code=code)))
                         for g in FICHIERS_LOURDS if os.path.exists(os.path.join(dossier, g.format(code=code))))

    return {
        "code": code,
        "nom": meta.get("nom_station"),
        "octets_total": _taille_dossier(dossier),
        "modeles": modeles,
        "octets_annexes_pca": annexes,
        "octets_test": octets_test,
        "octets_travail": octets_travail,
        "a_meta": bool(meta),
    }


def inventaire_global(dossier_base):
    """Inventaire de TOUTES les stations stockées (triées par taille décroissante)."""
    stations = []
    if os.path.isdir(dossier_base):
        for code in os.listdir(dossier_base):
            chemin = os.path.join(dossier_base, code)
            if os.path.isdir(chemin):
                inv = inventaire_station(chemin, code)
                if inv:
                    stations.append(inv)
    stations.sort(key=lambda s: s["octets_total"], reverse=True)
    return {"stations": stations, "octets_total": sum(s["octets_total"] for s in stations)}


def supprimer_element(dossier, code, cible):
    """
    Supprime précisément un élément d'une station.
    cible :
      - "station"        : tout le dossier de la station
      - "travail"        : les fichiers de travail lourds (caches, X/Y CSV)
      - "test"           : le jeu de test (dossier evaluation) -> désactive les backtests
      - "modele:<nom>"   : un modèle précis (+ son réseau de variance)
    Retourne (libellé, octets_libérés).
    """
    import shutil
    d_modeles = os.path.join(dossier, "modeles")

    if cible == "station":
        octets = _taille_dossier(dossier)
        shutil.rmtree(dossier, ignore_errors=True)
        return "station entière", octets

    if cible == "travail":
        octets = 0
        for g in FICHIERS_LOURDS:
            p = os.path.join(dossier, g.format(code=code))
            if os.path.exists(p):
                octets += os.path.getsize(p); os.remove(p)
        return "fichiers de travail", octets

    if cible == "test":
        d_eval = os.path.join(dossier, "evaluation")
        octets = _taille_dossier(d_eval)
        shutil.rmtree(d_eval, ignore_errors=True)
        return "jeu de test", octets

    if cible.startswith("modele:"):
        nom = cible.split(":", 1)[1]
        octets = 0
        for cand in (f"{nom}.joblib", f"{nom}.keras", f"variance_{nom}.keras"):
            p = os.path.join(d_modeles, cand)
            if os.path.exists(p):
                octets += os.path.getsize(p); os.remove(p)
        # Retire aussi le modèle du meta.json pour rester cohérent.
        chemin_meta = os.path.join(dossier, "meta.json")
        if os.path.exists(chemin_meta):
            try:
                with open(chemin_meta, encoding="utf-8") as f:
                    meta = json.load(f)
                if nom in meta.get("modeles", {}):
                    del meta["modeles"][nom]
                    with open(chemin_meta, "w", encoding="utf-8") as f:
                        json.dump(meta, f)
            except ValueError:
                pass
        return f"modèle {nom}", octets

    raise Exception(f"Cible de suppression inconnue : {cible}")

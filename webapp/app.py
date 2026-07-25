"""
Site web d'interaction avec river.py.

Lancement :
    cd webapp && python3 app.py
puis ouvrir http://localhost:5050

Architecture : Flask sert le front (static/) + une API JSON. Les opérations
longues (pipeline, entraînements) tournent en job de fond (jobs.py), une à la
fois ; le front suit leur log en polling. Chaque station a son dossier
data/{code}/ où river.py écrit ses fichiers de travail et où stockage.py
persiste modèles + points (voir stockage.py pour le schéma).
"""

import os
import sys
import io
import json
import base64
import threading
import warnings

import matplotlib
matplotlib.use("Agg")
warnings.filterwarnings("ignore", message=".*non-interactive.*")
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

DOSSIER_WEBAPP = os.path.dirname(os.path.abspath(__file__))
DOSSIER_PROJET = os.path.dirname(DOSSIER_WEBAPP)
sys.path.insert(0, DOSSIER_PROJET)

import matplotlib.pyplot as plt
import pandas as pd
import requests
from flask import Flask, jsonify, request, send_from_directory

from river import River, GestionnaireDonnees, GestionnaireQuota
import stockage
from jobs import GestionnaireJobs, JobOccupe, brancher_log_sur_job

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Mode PUBLIC (démo en ligne) : les rivières livrées sont en lecture seule.
# On bloque tout ce qui écrit/entraîne/coûte du quota (voir bloquer_ecriture_en_public).
MODE_PUBLIC = os.environ.get("RIVER_PUBLIC", "0") == "1"

# Dossier des données/modèles (surchargeable pour livrer un jeu curé en public).
DOSSIER_DATA = os.environ.get("RIVER_DATA_DIR") or os.path.join(DOSSIER_WEBAPP, "data")
os.makedirs(DOSSIER_DATA, exist_ok=True)

# Quota Open-Meteo PARTAGÉ par toutes les stations (il est lié à l'IP). Chemin
# absolu pour rester unique malgré les chdir vers data/{code}/ des jobs.
CHEMIN_QUOTA = os.path.join(DOSSIER_DATA, "quota_openmeteo.json")
GestionnaireDonnees.chemin_quota_openmeteo = CHEMIN_QUOTA
QUOTA = GestionnaireQuota(chemin=CHEMIN_QUOTA)

JOBS = GestionnaireJobs()
RIVIERES = {}                 # code_station -> instance River vivante
PARAMS_RIVIERES = {}          # code_station -> {"params_hourly": [...], "params_daily": [...]}
VERROU_RIVIERES = threading.Lock()
VERROU_GRAPHIQUE = threading.Lock()   # pyplot n'est pas thread-safe

PARAMS_HOURLY_DEFAUT = ["snow_depth", "temperature_2m"]
PARAMS_DAILY_DEFAUT = ["rain_sum"]
# Agrégation par défaut de chaque variable horaire (voir hourly_to_daily_spec) :
# la neige comme un STOCK (photo à minuit), la température comme une ÉNERGIE
# (moyenne journalière). L'UI permet de passer la température en "horaire".
AGREGATIONS_DEFAUT = {"snow_depth": "minuit", "temperature_2m": "moyenne"}


# ----------------------------------------------------------------------
# Aides
# ----------------------------------------------------------------------
def dossier_station(code):
    d = os.path.join(DOSSIER_DATA, code)
    os.makedirs(d, exist_ok=True)
    return d


def obtenir_riviere(code):
    """Retourne l'instance River du code demandé, en la créant (et en
    rechargeant une éventuelle sauvegarde disque) au besoin."""
    with VERROU_RIVIERES:
        if code in RIVIERES:
            return RIVIERES[code]

    riviere = River(code, verbose=True)
    dossier = dossier_station(code)
    if os.path.exists(os.path.join(dossier, "meta.json")):
        meta = stockage.charger_riviere(riviere, dossier)
        PARAMS_RIVIERES[code] = {
            "params_hourly": meta.get("params_hourly") or PARAMS_HOURLY_DEFAUT,
            "params_daily": meta.get("params_daily") or PARAMS_DAILY_DEFAUT,
            # riviere.agregations_horaires est déjà correct (déduit des noms de
            # features pour les modèles d'avant ce champ) -> l'UI reflète le
            # VRAI mode du modèle chargé, pas le défaut.
            "agregations": riviere.agregations_horaires or dict(AGREGATIONS_DEFAUT),
        }

    with VERROU_RIVIERES:
        RIVIERES[code] = riviere
    return riviere


def params_de(code, payload=None):
    payload = payload or {}
    defauts = PARAMS_RIVIERES.get(code, {})
    return {
        "params_hourly": payload.get("params_hourly") or defauts.get("params_hourly") or PARAMS_HOURLY_DEFAUT,
        "params_daily": payload.get("params_daily") or defauts.get("params_daily") or PARAMS_DAILY_DEFAUT,
        "agregations": payload.get("agregations") or defauts.get("agregations") or dict(AGREGATIONS_DEFAUT),
    }


def assurer_donnees_train(riviere, code):
    """Si les données d'entraînement ne sont pas en mémoire mais que les CSV
    écrits par construire_donnees_completes() sont encore sur le disque
    (typiquement après un redémarrage du serveur), les recharge au lieu
    d'exiger un re-téléchargement. Retourne True si les données sont prêtes."""
    if riviere.X_train2 is not None:
        return True
    dossier = dossier_station(code)
    chemins = {p: os.path.join(dossier, f"{p}_{code}.csv")
               for p in ("X_train", "Y_train", "X_test", "Y_test", "dates_train", "dates_test")}
    if not all(os.path.exists(chemins[p]) for p in ("X_train", "Y_train", "X_test", "Y_test")):
        return False
    avec_dates = os.path.exists(chemins["dates_train"]) and os.path.exists(chemins["dates_test"])
    riviere.charger_donnees_completes(
        chemins["X_train"], chemins["Y_train"], chemins["X_test"], chemins["Y_test"],
        dates_train_csv=chemins["dates_train"] if avec_dates else None,
        dates_test_csv=chemins["dates_test"] if avec_dates else None,
        past_day=riviere.past_day2 or 20,
        predict_day=riviere.predict_day2 or 15,
        split_date=riviere.split_date2,
    )
    return True


def capturer_graphique_png():
    """Récupère la dernière figure matplotlib en PNG base64 puis ferme tout."""
    numeros = plt.get_fignums()
    if not numeros:
        return None
    figure = plt.figure(numeros[-1])
    tampon = io.BytesIO()
    figure.savefig(tampon, format="png", dpi=110, bbox_inches="tight")
    plt.close("all")
    return base64.b64encode(tampon.getvalue()).decode()


def etat_riviere(code):
    riviere = obtenir_riviere(code)
    dossier = dossier_station(code)

    modeles = []
    for nom, info in riviere.modeles.items():
        scores_detail = info.get("scores_detail")
        modeles.append({
            "nom": nom,
            "score": float(info["score"]),
            "scores_detail": [float(s) for s in scores_detail] if scores_detail is not None else None,
            "espace": info.get("espace", "brut"),
            "hybride": "hybride" in info,
        })
    modeles.sort(key=lambda m: m["score"], reverse=True)

    cartes = [f for f in ("bassins_versants.html", "carte_optimisation_finale.html", "bassin_versant_apercu.html")
              if os.path.exists(os.path.join(dossier, f))]

    return {
        "code_station": code,
        "nom_station": riviere.nom_station,
        "lat_station": riviere.lat_station,
        "lon_station": riviere.lon_station,
        "zones_definies": riviere.listes_de_points is not None,
        "geojson_bassins": riviere.geojson_bassins,
        "points_par_zone": {n: [list(p) for p in pts] for n, pts in riviere.points_par_zone.items()} if riviere.points_par_zone else None,
        "points_preselectionnes": [list(p) for p in riviere.points_preselectionnes] if riviere.points_preselectionnes else None,
        "altitudes_preselection": riviere._altitudes_preselection,
        "coords_finales": [list(c) for c in riviere.coords_finales] if riviere.coords_finales else None,
        "donnees_pretes": riviere.X_test2 is not None,
        "donnees_train_presentes": riviere.X_train2 is not None,
        "pca_prete": riviere.pca_model is not None,
        "n_composantes_pca": riviere.n_composantes_pca,
        "predict_day": riviere.predict_day2,
        "past_day": riviere.past_day2,
        "modeles": modeles,
        "sauvegarde_existante": os.path.exists(os.path.join(dossier, "meta.json")),
        "gros_fichiers": [{"nom": n, "octets": o} for n, o in stockage.taille_gros_fichiers(dossier, code)],
        "cartes_folium": cartes,
        "params": params_de(code),
    }


def reponse_job(job):
    return jsonify({"job_id": job.id}), 202


@app.before_request
def bloquer_ecriture_en_public():
    """En démo publique : on autorise tout ce qui est LECTURE (carte, prévision,
    backtest, historique, inventaire...) mais on bloque tout POST (entraînement,
    téléchargement, suppression, réglages de quota) — sauf l'arrêt d'un job. Ça
    protège les rivières livrées et évite d'épuiser le quota partagé."""
    if not MODE_PUBLIC or request.method != "POST":
        return
    if request.path.endswith("/arreter"):
        return
    return jsonify({"erreur": "Action désactivée sur la démo en ligne. "
                    "Installe le projet en local (dépôt GitHub) pour analyser et gérer tes propres rivières."}), 403


@app.route("/api/config")
def api_config():
    """Config lue par le front : mode public + éventuel compteur de visites
    (GoatCounter), défini par la variable d'environnement GOATCOUNTER."""
    return jsonify({"public": MODE_PUBLIC, "analytics": os.environ.get("GOATCOUNTER") or None})


# ----------------------------------------------------------------------
# Front
# ----------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/carte/<code>/<nom_fichier>")
def carte_folium(code, nom_fichier):
    """Sert les cartes Folium (HTML) produites par river.py pour une station."""
    if not nom_fichier.endswith(".html"):
        return jsonify({"erreur": "fichier non autorisé"}), 403
    return send_from_directory(dossier_station(code), nom_fichier)


# ----------------------------------------------------------------------
# Stations (référentiel Hub'Eau)
# ----------------------------------------------------------------------
CACHE_STATIONS = os.path.join(DOSSIER_DATA, "stations.json")


@app.route("/api/stations")
def api_stations():
    rafraichir = request.args.get("refresh") == "1"
    if os.path.exists(CACHE_STATIONS) and not rafraichir:
        with open(CACHE_STATIONS, encoding="utf-8") as f:
            return jsonify(json.load(f))

    stations = []
    url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations"
    params = {"format": "json", "size": 10000,
              "fields": "code_station,libelle_station,latitude_station,longitude_station,en_service,libelle_cours_eau,code_departement"}
    for _ in range(5):  # suit la pagination si > 10000
        reponse = requests.get(url, params=params, timeout=30)
        if reponse.status_code not in (200, 206):
            return jsonify({"erreur": f"Hub'Eau a répondu {reponse.status_code}"}), 502
        corps = reponse.json()
        for s in corps.get("data", []):
            if s.get("latitude_station") is None or s.get("longitude_station") is None:
                continue
            stations.append({
                "code": s["code_station"],
                "nom": s.get("libelle_station") or "Sans nom",
                "lat": s["latitude_station"],
                "lon": s["longitude_station"],
                "en_service": bool(s.get("en_service")),
                "cours_eau": s.get("libelle_cours_eau"),
                "departement": s.get("code_departement"),
            })
        suivant = corps.get("next")
        if not suivant:
            break
        url, params = suivant, None

    with open(CACHE_STATIONS, "w", encoding="utf-8") as f:
        json.dump(stations, f, ensure_ascii=False)
    return jsonify(stations)


# ----------------------------------------------------------------------
# État d'une rivière
# ----------------------------------------------------------------------
@app.route("/api/riviere/<code>")
def api_riviere(code):
    try:
        return jsonify(etat_riviere(code))
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500


# ----------------------------------------------------------------------
# Jobs de fond
# ----------------------------------------------------------------------
@app.route("/api/jobs/<int:job_id>")
def api_job(job_id):
    job = JOBS.job(job_id)
    if job is None:
        return jsonify({"erreur": "job inconnu"}), 404
    depuis = request.args.get("depuis", 0, type=int)
    return jsonify(job.vers_dict(depuis_ligne=depuis))


@app.route("/api/jobs/<int:job_id>/arreter", methods=["POST"])
def api_arreter_job(job_id):
    """Demande l'arrêt coopératif du job : il s'interrompt au prochain point de
    contrôle (entre deux lots de téléchargement, deux générations du GA, deux
    modèles...) au lieu de gaspiller temps et quota."""
    job = JOBS.demander_arret(job_id)
    if job is None:
        return jsonify({"erreur": "aucun job en cours à arrêter"}), 404
    return jsonify({"job_id": job_id, "statut": "arret_demande"})


def demarrer_job(type_job, code, fonction):
    try:
        job = JOBS.lancer(type_job, code, dossier_station(code), fonction)
        return reponse_job(job)
    except JobOccupe as e:
        return jsonify({"erreur": str(e)}), 409


@app.route("/api/riviere/<code>/zones", methods=["POST"])
def api_zones(code):
    payload = request.get_json(silent=True) or {}
    riviere = obtenir_riviere(code)

    def fonction(job):
        brancher_log_sur_job(riviere, job)
        riviere.determiner_bassins_versants(
            delta=payload.get("delta", 0.05),
            n_points_grille=payload.get("n_points_grille", 15),
            max_points_par_zone=payload.get("max_points_par_zone", 25),
            afficher=False,
        )
        return {"nb_zones": len(riviere.noms_zones),
                "nb_points": sum(len(z) for z in riviere.listes_de_points)}

    return demarrer_job("zones", code, fonction)


@app.route("/api/riviere/<code>/pipeline", methods=["POST"])
def api_pipeline(code):
    """La fonction qui fait tout jusqu'au Gradient Boosting. Deux méthodes de
    sélection des points météo : "deux_temps" (défaut : présélection gratuite
    altitude+couverture, puis affinage léger corrélation+neige) ou "genetique"
    (l'ancien GA, plus lourd)."""
    payload = request.get_json(silent=True) or {}
    riviere = obtenir_riviere(code)
    params = params_de(code, payload)
    PARAMS_RIVIERES[code] = params
    methode = payload.get("methode", "deux_temps")

    def fonction(job):
        brancher_log_sur_job(riviere, job)
        if methode == "genetique":
            riviere.determiner_bassins_versants(afficher=False)
            kwargs_ga = dict(payload.get("ga") or {})
            kwargs_ga["agregations_horaires"] = params["agregations"]
            if payload.get("start_train"):
                kwargs_ga["start_train"] = payload["start_train"]
            if payload.get("end_train"):
                kwargs_ga["end_train"] = payload["end_train"]
            info = riviere.executer_pipeline_gradient_boosting(
                params_hourly=params["params_hourly"],
                params_daily=params["params_daily"],
                **kwargs_ga,
            )
        else:
            sel = payload.get("selection") or {}
            info = riviere.executer_pipeline_deux_temps(
                params_hourly=params["params_hourly"],
                params_daily=params["params_daily"],
                n_preselection=int(sel.get("n_preselection", 30)),
                n_final=int(sel.get("n_final", 5)),
                fenetre_selection_annees=float(sel.get("fenetre_annees", 2)),
                densite_grille=int(sel.get("densite_grille", 25)),
                poids_altitude=float(sel.get("poids_altitude", 1.0)),
                poids_zones=float(sel.get("poids_zones", 1.0)),
                poids_pluie=float(sel.get("poids_pluie", 1.0)),
                poids_neige=float(sel.get("poids_neige", 1.0)),
                past_day=int(payload.get("past_day", 20)),
                predict_day=int(payload.get("predict_day", 15)),
                mode_split=payload.get("mode_split", "annees_aleatoires"),
                part_test=float(payload.get("part_test", 0.2)),
                start_fetch=payload.get("start_train"),
                end_fetch=payload.get("end_train"),
                agregations_horaires=params["agregations"],
            )
        stockage.sauvegarder_riviere(riviere, dossier_station(code), **params)
        job.ajouter_log("💾 Modèles et points sauvegardés automatiquement.")
        resultat = {"score_gradient_boosting": float(info["score"]),
                    "coords_finales": [list(c) for c in riviere.coords_finales]}
        if riviere.diagnostic_selection:
            resultat["temps_reponse_jours"] = riviere.diagnostic_selection.get("temps_reponse_jours")
        return resultat

    return demarrer_job("pipeline", code, fonction)


@app.route("/api/riviere/<code>/points", methods=["POST"])
def api_points_manuels(code):
    """Points optimaux choisis à la main sur la carte : construit les données
    complètes dessus (sans entraîner -- utiliser ensuite les boutons modèles)."""
    payload = request.get_json(silent=True) or {}
    points = payload.get("points") or []
    if not points:
        return jsonify({"erreur": "aucun point fourni"}), 400

    riviere = obtenir_riviere(code)
    params = params_de(code, payload)
    PARAMS_RIVIERES[code] = params

    def fonction(job):
        brancher_log_sur_job(riviere, job)
        riviere.definir_points_optimaux([(float(p[0]), float(p[1])) for p in points])
        riviere.construire_donnees_completes(
            params_hourly=params["params_hourly"],
            params_daily=params["params_daily"],
            start_fetch=payload.get("start_train") or payload.get("start_fetch"),
            end_fetch=payload.get("end_train") or payload.get("end_fetch"),
            split_date=payload.get("split_date"),
            past_day=int(payload.get("past_day", 20)),
            predict_day=int(payload.get("predict_day", 15)),
            mode_split=payload.get("mode_split", "annees_aleatoires"),
            part_test=float(payload.get("part_test", 0.2)),
            agregations_horaires=params["agregations"],
        )
        return {"lignes_train": int(len(riviere.X_train2)), "lignes_test": int(len(riviere.X_test2)),
                "dimensions": int(riviere.X_train2.shape[1])}

    return demarrer_job("points_manuels", code, fonction)


@app.route("/api/riviere/<code>/donnees", methods=["POST"])
def api_donnees(code):
    """Télécharge/construit les données complètes sur les points actuels
    (coords_finales), SANS entraîner de modèle."""
    payload = request.get_json(silent=True) or {}
    riviere = obtenir_riviere(code)
    if riviere.coords_finales is None:
        return jsonify({"erreur": "Aucun point météo défini : lance la sélection génétique ou choisis des points sur la carte."}), 400
    params = params_de(code, payload)
    PARAMS_RIVIERES[code] = params

    def fonction(job):
        brancher_log_sur_job(riviere, job)
        riviere.construire_donnees_completes(
            params_hourly=params["params_hourly"],
            params_daily=params["params_daily"],
            start_fetch=payload.get("start_train") or payload.get("start_fetch"),
            end_fetch=payload.get("end_train") or payload.get("end_fetch"),
            split_date=payload.get("split_date"),
            past_day=int(payload.get("past_day", 20)),
            predict_day=int(payload.get("predict_day", 15)),
            mode_split=payload.get("mode_split", "annees_aleatoires"),
            part_test=float(payload.get("part_test", 0.2)),
            agregations_horaires=params["agregations"],
        )
        return {"lignes_train": int(len(riviere.X_train2)), "lignes_test": int(len(riviere.X_test2)),
                "dimensions": int(riviere.X_train2.shape[1])}

    return demarrer_job("donnees", code, fonction)


@app.route("/api/riviere/<code>/periode")
def api_periode(code):
    """Période EXPLOITABLE (recouvrement débit Hub'Eau ∩ météo ERA5 depuis 1940)."""
    riviere = obtenir_riviere(code)
    try:
        premiere, debut_exploitable, derniere = riviere._data.periode_exploitable_river(code)
    except Exception as e:
        return jsonify({"erreur": str(e)}), 400
    nb_annees = round((derniere - debut_exploitable).days / 365.25, 1)
    avertissement = None
    if debut_exploitable > premiere:
        avertissement = (f"Débit disponible depuis {premiere.year}, mais la météo (ERA5) ne commence qu'en "
                         f"{debut_exploitable.year} : l'apprentissage utilise {debut_exploitable.year}→{derniere.year}.")
    if nb_annees < 5:
        avertissement = (f"Seulement {nb_annees} an(s) de données exploitables (débit + météo) : c'est peu pour "
                         f"apprendre. Les scores et les intervalles de confiance seront fragiles.")
    return jsonify({"debut": debut_exploitable.strftime("%Y-%m-%d"), "fin": derniere.strftime("%Y-%m-%d"),
                    "debut_debit": premiere.strftime("%Y-%m-%d"), "nb_annees": nb_annees,
                    "avertissement": avertissement})


def _format_duree(secondes):
    secondes = int(secondes)
    if secondes <= 0:
        return "maintenant"
    if secondes < 60:
        return f"{secondes} s"
    if secondes < 3600:
        return f"{secondes // 60} min"
    if secondes < 86400:
        return f"{secondes // 3600} h {(secondes % 3600) // 60} min"
    return f"{secondes // 86400} j {(secondes % 86400) // 3600} h"


@app.route("/api/quota")
def api_quota():
    """État du quota Open-Meteo (partagé toutes stations) + éventuelle
    estimation du coût d'un téléchargement projeté (?points=&jours=&vars=)."""
    import time as _t
    now = _t.time()
    detail = QUOTA.etat(now)
    libelles = {"minute": "par minute", "heure": "par heure", "jour": "par jour", "mois": "par mois"}
    fenetres = []
    for nom, d in detail.items():
        fenetres.append({
            "fenetre": nom,
            "libelle": libelles.get(nom, nom),
            "utilise": d["utilise"],
            "limite": d["limite"],
            "restant": d["restant"],
            "pct": d["pct"],
            "reset_dans_s": d["reset_dans_s"],
            "reset_texte": _format_duree(d["reset_dans_s"]) if d["reset_dans_s"] else "—",
        })

    reponse = {"fenetres": fenetres}

    # Estimation optionnelle du coût d'une requête projetée.
    points = request.args.get("points", type=int)
    jours = request.args.get("jours", type=int)
    n_vars = request.args.get("vars", type=int)
    if points and jours:
        poids = QUOTA.poids_appel(points, n_vars or 3, jours)
        jour = detail["jour"]
        tient = poids <= jour["restant"]
        reponse["estimation"] = {
            "points": points, "jours": jours, "vars": n_vars or 3,
            "poids": poids,
            "budget_jour_restant": jour["restant"],
            "tient_dans_le_jour": tient,
            "message": (f"Coût estimé {poids} sur les {jour['restant']} encore disponibles aujourd'hui."
                        if tient else
                        f"Coût estimé {poids}, or il ne reste que {jour['restant']} aujourd'hui : "
                        f"réduis la période ou attends le rechargement (~{_format_duree(jour['reset_dans_s'])})."),
        }
    return jsonify(reponse)


@app.route("/api/quota/limites", methods=["POST"])
def api_quota_limites():
    """Règle les limites du quota (minute/heure/jour/mois), persistées. Utile
    pour s'adapter à une autre offre Open-Meteo ou pour se donner une marge."""
    payload = request.get_json(silent=True) or {}
    QUOTA.definir_limites({k: payload[k] for k in ("minute", "heure", "jour", "mois") if k in payload})
    return jsonify({"ok": True, "limites": {k: v[1] for k, v in QUOTA.limites.items()}})


@app.route("/api/quota/reinitialiser", methods=["POST"])
def api_quota_reinitialiser():
    """Remet le compteur d'appels à zéro (les limites personnalisées restent)."""
    QUOTA.reinitialiser()
    return jsonify({"ok": True})


ENTRAINEURS = {
    "lineaire_pca": lambda r: r.entrainer_regression_lineaire_pca(),
    "ridge_brut": lambda r: r.entrainer_ridge_brut(),
    "ridge_causal": lambda r: r.entrainer_ridge_causal(),
    "gradient_boosting": lambda r: r.entrainer_gradient_boosting(),
    "keras_brut": lambda r: r.entrainer_keras_brut(),
}


@app.route("/api/riviere/<code>/entrainer-modele", methods=["POST"])
def api_entrainer_modele(code):
    """Entraîne UN modèle (+ son incertitude) puis sauvegarde."""
    payload = request.get_json(silent=True) or {}
    nom = payload.get("modele")
    if nom not in ENTRAINEURS:
        return jsonify({"erreur": f"modèle inconnu : {nom} (choix : {', '.join(ENTRAINEURS)})"}), 400
    riviere = obtenir_riviere(code)
    if not assurer_donnees_train(riviere, code):
        return jsonify({"erreur": "Données d'entraînement absentes : télécharge les données d'abord."}), 400
    params = params_de(code)

    def fonction(job):
        brancher_log_sur_job(riviere, job)
        if nom == "lineaire_pca" and riviere.X_train_pca is None:
            with VERROU_GRAPHIQUE:
                riviere.appliquer_pca(seuil_energie=payload.get("seuil_energie", 99), afficher_graphique=False)
        ENTRAINEURS[nom](riviere)
        if payload.get("incertitude", True):
            riviere.entrainer_incertitude(nom)
        stockage.sauvegarder_riviere(riviere, dossier_station(code), **params)
        job.ajouter_log("💾 Sauvegardé.")
        return {"modele": nom, "score": float(riviere.modeles[nom]["score"])}

    return demarrer_job(f"entrainement_{nom}", code, fonction)


@app.route("/api/riviere/<code>/pca", methods=["POST"])
def api_pca(code):
    payload = request.get_json(silent=True) or {}
    riviere = obtenir_riviere(code)
    if not assurer_donnees_train(riviere, code):
        return jsonify({"erreur": "Données d'entraînement absentes (nettoyées) : relance le téléchargement des données d'abord."}), 400

    def fonction(job):
        brancher_log_sur_job(riviere, job)
        with VERROU_GRAPHIQUE:
            riviere.appliquer_pca(seuil_energie=payload.get("seuil_energie", 99), afficher_graphique=True)
            image = capturer_graphique_png()
        return {"n_composantes": riviere.n_composantes_pca, "image": image}

    return demarrer_job("pca", code, fonction)


@app.route("/api/riviere/<code>/entrainer", methods=["POST"])
def api_entrainer(code):
    """Entraîne les modèles sélectionnés (+ incertitude) puis sauvegarde."""
    payload = request.get_json(silent=True) or {}
    riviere = obtenir_riviere(code)
    if not assurer_donnees_train(riviere, code):
        return jsonify({"erreur": "Données d'entraînement absentes : télécharge les données d'abord."}), 400
    params = params_de(code)
    modeles = payload.get("modeles") or list(ENTRAINEURS.keys())
    incertitude = payload.get("incertitude", True)

    def fonction(job):
        brancher_log_sur_job(riviere, job)
        if "lineaire_pca" in modeles and riviere.X_train_pca is None:
            with VERROU_GRAPHIQUE:
                riviere.appliquer_pca(seuil_energie=payload.get("seuil_energie", 99), afficher_graphique=False)
        riviere.entrainer_tous_les_modeles(entrainer_incertitude_aussi=incertitude, modeles=modeles)
        stockage.sauvegarder_riviere(riviere, dossier_station(code), **params)
        job.ajouter_log("💾 Modèles et points sauvegardés automatiquement.")
        return {"scores": {nom: float(info["score"]) for nom, info in riviere.modeles.items()}}

    return demarrer_job("entrainement", code, fonction)


# ----------------------------------------------------------------------
# Évaluation (backtest) et prévision réelle -- rapides, en synchrone
# ----------------------------------------------------------------------
@app.route("/api/riviere/<code>/dates-test")
def api_dates_test(code):
    riviere = obtenir_riviere(code)
    if riviere.test2 is None or 'date' not in riviere.test2.columns:
        return jsonify({"erreur": "Pas de jeu de test disponible pour cette station."}), 400
    dates = riviere.test2['date'].dt.strftime("%Y-%m-%d").tolist()
    return jsonify({"min": dates[0], "max": dates[-1], "dates": dates})


@app.route("/api/riviere/<code>/backtest")
def api_backtest(code):
    riviere = obtenir_riviere(code)
    nom_modele = request.args.get("modele", "gradient_boosting")
    date = request.args.get("date")
    hybride = request.args.get("hybride", "1") == "1"
    nb_jours = request.args.get("nb_jours", type=int)
    if nom_modele not in riviere.modeles:
        return jsonify({"erreur": f"modèle '{nom_modele}' non entraîné"}), 400
    if not date:
        return jsonify({"erreur": "paramètre 'date' requis (AAAA-MM-JJ)"}), 400

    donnees = riviere.donnees_prediction_passee(nom_modele, date, utiliser_hybride=hybride, nb_jours=nb_jours)
    if donnees is None:
        return jsonify({"erreur": f"La date {date} n'est pas disponible dans le jeu de test."}), 400

    infos = riviere.modeles[nom_modele]
    scores_detail = infos.get("scores_detail")
    donnees["scores_detail"] = [float(s) for s in scores_detail] if scores_detail is not None else None
    donnees["modele"] = nom_modele
    donnees["score"] = float(infos["score"])   # R² global du modèle (jeu de test)
    donnees["unite"] = "m³/s"
    return jsonify(donnees)


@app.route("/api/riviere/<code>/prevision")
def api_prevision(code):
    riviere = obtenir_riviere(code)
    nom_modele = request.args.get("modele", "gradient_boosting")
    hybride = request.args.get("hybride", "1") == "1"
    nb_jours = request.args.get("nb_jours", type=int)
    if nom_modele not in riviere.modeles:
        return jsonify({"erreur": f"modèle '{nom_modele}' non entraîné"}), 400
    params = params_de(code)

    try:
        riviere.predire_futur(
            nom_modele=nom_modele,
            params_hourly=params["params_hourly"],
            params_daily=params["params_daily"],
            nb_jours=nb_jours,
            utiliser_hybride=hybride,
            afficher_graphique=False,   # plus de PNG : le graphique est interactif côté navigateur
        )
    except Exception as e:
        return jsonify({"erreur": str(e)}), 500

    infos = riviere.modeles[nom_modele]
    scores_detail = infos.get("scores_detail")
    donnees = dict(riviere.derniere_prevision or {})
    donnees["modele"] = nom_modele
    donnees["score"] = float(infos["score"])   # R² global (jeu de test)
    donnees["scores_detail"] = [float(s) for s in scores_detail] if scores_detail is not None else None
    donnees["unite"] = "m³/s"
    return jsonify(donnees)


@app.route("/api/riviere/<code>/historique")
def api_historique(code):
    """Débit observé (Hub'Eau) entre deux dates passées : graphique + stats."""
    riviere = obtenir_riviere(code)
    debut = request.args.get("debut")
    fin = request.args.get("fin")
    if not debut or not fin:
        return jsonify({"erreur": "paramètres 'debut' et 'fin' requis (AAAA-MM-JJ)"}), 400
    aujourd_hui = pd.Timestamp.now().normalize()
    if pd.to_datetime(fin) > aujourd_hui:
        fin = aujourd_hui.strftime("%Y-%m-%d")
    if pd.to_datetime(debut) >= pd.to_datetime(fin):
        return jsonify({"erreur": "la date de début doit précéder la date de fin"}), 400

    try:
        df = riviere._data.fetch_river_data(code, debut, fin)
    except Exception as e:
        return jsonify({"erreur": str(e)}), 400

    D = 1000.0   # L/s -> m³/s
    debit_m3 = df['debit_L_s'] / D
    serie = [{"date": d.strftime("%Y-%m-%d"), "debit": float(v)}
             for d, v in zip(df['date'], debit_m3)]

    return jsonify({
        "unite": "m³/s",
        "serie": serie,
        "nb_jours": int(len(df)),
        "debit_min": float(debit_m3.min()),
        "debit_max": float(debit_m3.max()),
        "debit_moyen": float(debit_m3.mean()),
        "date_min": df.loc[df['debit_L_s'].idxmin(), 'date'].strftime("%d/%m/%Y"),
        "date_max": df.loc[df['debit_L_s'].idxmax(), 'date'].strftime("%d/%m/%Y"),
    })


# ----------------------------------------------------------------------
# Stockage
# ----------------------------------------------------------------------
@app.route("/api/riviere/<code>/sauvegarder", methods=["POST"])
def api_sauvegarder(code):
    riviere = obtenir_riviere(code)
    if not riviere.modeles:
        return jsonify({"erreur": "aucun modèle entraîné à sauvegarder"}), 400
    params = params_de(code)
    stockage.sauvegarder_riviere(riviere, dossier_station(code), **params)
    return jsonify({"ok": True})


@app.route("/api/riviere/<code>/nettoyer", methods=["POST"])
def api_nettoyer(code):
    try:
        supprimes, octets = stockage.nettoyer_gros_fichiers(dossier_station(code), code)
    except Exception as e:
        return jsonify({"erreur": str(e)}), 400
    riviere = RIVIERES.get(code)
    if riviere is not None:
        riviere.X_train2 = riviere.Y_train2 = None
        riviere.X_train_pca = riviere.X_test_pca = None
        riviere.train2 = None
    return jsonify({"supprimes": supprimes, "octets_liberes": octets})


# ----------------------------------------------------------------------
# Gestionnaire de stockage (inventaire + suppression choisie)
# ----------------------------------------------------------------------
@app.route("/api/stockage")
def api_stockage():
    """Inventaire de TOUTES les stations stockées, par taille (modèles un par un,
    jeu de test, fichiers de travail) : de quoi choisir précisément quoi supprimer."""
    return jsonify(stockage.inventaire_global(DOSSIER_DATA))


@app.route("/api/stockage/supprimer", methods=["POST"])
def api_stockage_supprimer():
    """Supprime un élément précis : cible = station | travail | test | modele:<nom>."""
    payload = request.get_json(silent=True) or {}
    code = payload.get("code")
    cible = payload.get("cible")
    if not code or not cible:
        return jsonify({"erreur": "'code' et 'cible' requis"}), 400
    dossier = os.path.join(DOSSIER_DATA, code)   # sans créer le dossier (contrairement à dossier_station)
    if not os.path.isdir(dossier):
        return jsonify({"erreur": "station inconnue dans le stockage"}), 404
    try:
        libelle, octets = stockage.supprimer_element(dossier, code, cible)
    except Exception as e:
        return jsonify({"erreur": str(e)}), 400

    # Cohérence de l'instance en mémoire (si chargée).
    with VERROU_RIVIERES:
        riviere = RIVIERES.get(code)
        if riviere is not None:
            if cible == "station":
                RIVIERES.pop(code, None)
            elif cible == "test":
                riviere.X_test2 = riviere.Y_test2 = riviere.test2 = None
            elif cible == "travail":
                riviere.X_train2 = riviere.Y_train2 = riviere.train2 = None
                riviere.X_train_pca = riviere.X_test_pca = None
            elif cible.startswith("modele:"):
                riviere.modeles.pop(cible.split(":", 1)[1], None)
    return jsonify({"libelle": libelle, "octets_liberes": octets})


if __name__ == "__main__":
    # HOST=0.0.0.0 pour être accessible sur le réseau (LAN / hébergeur) ;
    # PORT depuis l'environnement (7860 sur Hugging Face Spaces).
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5050"))
    app.run(host=host, port=port, debug=False, threaded=True)

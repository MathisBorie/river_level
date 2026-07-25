"""
================================================================================
MODULE river.py
================================================================================
Deux classes :

- GestionnaireDonnees : parle aux APIs (Hub'Eau, Open-Meteo), gère les erreurs
  réseau (429, etc.), et fait les transformations de données brutes (horaire
  -> journalier large, encodage cyclique de la date). Ne connaît rien à la
  notion de "River" : elle sait juste collecter et nettoyer.

- River : orchestre tout le pipeline pour UNE station hydrométrique donnée.
  Compose une instance de GestionnaireDonnees (self._data) pour toute la
  collecte, plutôt que d'en hériter -- une River "a" une capacité de collecte
  de données, elle n'"est" pas elle-même un gestionnaire de données. Ça permet
  aussi de réutiliser GestionnaireDonnees ailleurs si besoin, indépendamment
  de River.

⚠️ Limitation connue : DEAP utilise un registre global (`creator.create(...)`)
   pour définir les classes FitnessMax/Individual. C'est une contrainte de la
   bibliothèque elle-même, pas quelque chose de spécifique à ce module : deux
   instances de River ne doivent donc pas lancer lancer_selection_genetique()
   en parallèle (deux threads/processus). En séquentiel, aucun problème : le
   registre est recréé à chaque appel.

⚠️ Ce fichier n'a pas pu être testé contre les vraies APIs (Hub'Eau, Open-Meteo,
   Sandre) car le bac à sable où il a été écrit n'a pas accès réseau à ces
   domaines. Seule la syntaxe Python a été vérifiée (py_compile). Teste-le
   dans ton notebook habituel avant de t'y fier pour de vrai.
================================================================================
"""

import os
import re
import time
import math
import json
import random
import base64
import threading
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import folium
from IPython.display import IFrame, display

from shapely.geometry import shape, Point

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

from deap import base, creator, tools

import tensorflow as tf
from tqdm import tqdm
from tqdm.keras import TqdmCallback


class ArretDemande(Exception):
    """Levée aux points de contrôle quand un arrêt du calcul en cours a été
    demandé (bouton « Arrêter » du site). Permet d'interrompre proprement un
    téléchargement, la sélection génétique ou un entraînement au lieu de
    gaspiller du temps/quota sur une exécution qu'on sait déjà erronée."""
    pass


def _dormir_interruptible(secondes, doit_arreter=None):
    """time.sleep en tranches de 1 s qui s'interrompt (ArretDemande) si un arrêt
    est demandé pendant l'attente. Sans quoi un long délai (attente de quota,
    backoff 429) rendrait le bouton « Arrêter » inopérant pendant tout ce temps."""
    restant = float(secondes)
    while restant > 0:
        if doit_arreter is not None and doit_arreter():
            raise ArretDemande("Calcul interrompu à la demande de l'utilisateur.")
        time.sleep(min(1.0, restant))
        restant -= 1.0


def _callback_arret_keras(doit_arreter):
    """Callback Keras qui stoppe l'entraînement en fin d'epoch si un arrêt a été
    demandé. Retourne None si aucun contrôle d'arrêt n'est branché."""
    if doit_arreter is None:
        return None

    class _ArretKeras(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if doit_arreter():
                self.model.stop_training = True

    return _ArretKeras()


class RegresseurMultiSortie:
    """Multi-sortie « maison » : un régresseur indépendant par colonne de Y.
    Équivalent picklable de MultiOutputRegressor, mais entraîné SÉQUENTIELLEMENT
    horizon par horizon, ce qui permet de reporter l'avancement (barre de
    progression) -- MultiOutputRegressor fait tout en un bloc opaque.
    (Chaque HistGradientBoosting parallélise déjà en interne, la perte de
    parallélisme externe est minime pour une quinzaine d'horizons.)"""

    def __init__(self, estimators):
        self.estimators_ = list(estimators)

    def predict(self, X):
        return np.column_stack([est.predict(X) for est in self.estimators_])


def _callback_progres_keras(hook_progres, phase, total_epochs):
    """Callback Keras qui pousse l'avancement (epoch/total) vers la barre de
    progression du site. Retourne None si aucun hook n'est branché."""
    if hook_progres is None:
        return None

    class _ProgresKeras(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            hook_progres(phase, epoch + 1, total_epochs)

    return _ProgresKeras()


# ==============================================================================
# GESTIONNAIRE DE QUOTA OPEN-METEO (partagé, persistant)
# ==============================================================================
_VERROU_QUOTA = threading.Lock()   # sérialise les accès au fichier de quota


class GestionnaireQuota:
    """
    Suit la consommation de l'API Open-Meteo pour ANTICIPER les 429 au lieu de
    les subir. Open-Meteo facture chaque requête par un "poids" (et non par un
    simple +1) : sa règle est ~ 1 variable × 1 lieu × 14 jours = 1 appel. Une
    requête de 4 points × 3 variables × 66 ans pèse donc ~20 000 appels, ce qui
    explique qu'un seul gros téléchargement épuise le quota journalier (10 000).

    L'état (liste d'événements [timestamp, poids]) est persisté dans un fichier
    JSON PARTAGÉ par toutes les stations (le quota est lié à l'IP, pas à la
    rivière) et rechargé à chaque opération, de sorte que plusieurs instances de
    River voient le même compteur, même après un redémarrage du serveur.

    Fenêtres et limites par défaut = limites documentées de l'offre gratuite.
    """

    # fenetre -> (duree_secondes, limite_poids)
    LIMITES = {
        "minute": (60, 600),
        "heure": (3600, 5000),
        "jour": (86400, 10000),
        "mois": (2592000, 300000),   # 30 jours glissants
    }

    def __init__(self, chemin=None, limites=None):
        self.chemin = chemin or os.path.join(os.path.expanduser("~"), ".river_quota_openmeteo.json")
        self.limites = limites or dict(self.LIMITES)
        # Les limites éventuellement personnalisées par l'utilisateur (persistées
        # dans le fichier) l'emportent sur les valeurs par défaut.
        limites_disque = self._lire_fichier().get("limites")
        if limites_disque:
            for nom, (duree, _defaut) in self.LIMITES.items():
                if nom in limites_disque:
                    self.limites[nom] = (duree, int(limites_disque[nom]))

    def _lire_fichier(self):
        try:
            with open(self.chemin, encoding="utf-8") as f:
                brut = json.load(f)
        except (FileNotFoundError, ValueError):
            return {}
        # Rétro-compat : ancien format = simple liste d'événements.
        if isinstance(brut, list):
            return {"evenements": brut}
        return brut or {}

    def _ecrire_fichier(self, contenu):
        os.makedirs(os.path.dirname(self.chemin) or ".", exist_ok=True)
        tmp = self.chemin + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(contenu, f)
        os.replace(tmp, self.chemin)

    # ---- coût d'une requête -------------------------------------------------
    @staticmethod
    def poids_appel(n_points, n_variables, n_jours):
        """
        Poids Open-Meteo ESTIMÉ d'une requête, d'après la règle documentée par
        Open-Meteo : « plus de 10 variables OU plus de 2 semaines pour UNE
        localisation = plusieurs appels » (ex. 2 semaines + 15 variables = 1,5 ;
        4 semaines = 3,0). Le coût par localisation vaut donc :
            max(1, variables/10) × max(1, jours/14)
        et -- correction d'un ancien bug -- il faut le MULTIPLIER PAR LE NOMBRE
        DE POINTS : chaque coordonnée compte comme un appel distinct, même
        regroupée dans une seule requête HTTP. C'est ce qui explique les 429 sur
        le « cache léger » : 150 points × 3 ans = ~11 000 appels alors que
        l'ancienne formule (qui ignorait les points) estimait ~235.

        Conséquence : les leviers d'économie sont le nombre de POINTS et le
        nombre de JOURS (en dessous de 10 variables, le nombre de variables ne
        change RIEN au coût API -- il ne change que la dimension du modèle).
        """
        cout_par_point = max(1.0, n_variables / 10.0) * max(1.0, n_jours / 14.0)
        return max(1, int(round(max(1, n_points) * cout_par_point)))

    @staticmethod
    def nb_jours(start_date, end_date):
        d0 = pd.to_datetime(start_date)
        d1 = pd.to_datetime(end_date)
        return max(1, int((d1 - d0).days) + 1)

    # ---- persistance --------------------------------------------------------
    def _charger_evenements(self, now):
        contenu = self._lire_fichier()
        evts = contenu.get("evenements", [])
        # On ne garde que le dernier mois glissant (au-delà, plus rien à suivre).
        limite_mois = self.limites["mois"][0]
        return [e for e in evts if now - e[0] <= limite_mois]

    def _sauver_evenements(self, evts):
        contenu = self._lire_fichier()
        contenu["evenements"] = evts
        self._ecrire_fichier(contenu)

    def definir_limites(self, nouvelles):
        """Met à jour (et persiste) les limites par fenêtre choisies par
        l'utilisateur. `nouvelles` : dict {fenetre: limite_entiere}."""
        contenu = self._lire_fichier()
        limites_persistees = contenu.get("limites", {})
        for nom, (duree, _) in self.LIMITES.items():
            if nom in nouvelles and nouvelles[nom]:
                valeur = int(nouvelles[nom])
                self.limites[nom] = (duree, valeur)
                limites_persistees[nom] = valeur
        contenu["limites"] = limites_persistees
        self._ecrire_fichier(contenu)

    def reinitialiser(self):
        """Remet le compteur d'appels à zéro (utile après un redémarrage ou pour
        repartir proprement). Conserve les limites personnalisées."""
        contenu = self._lire_fichier()
        contenu["evenements"] = []
        self._ecrire_fichier(contenu)

    # ---- lecture de l'état --------------------------------------------------
    def etat(self, now=None):
        """État courant par fenêtre : utilisé / limite / restant / % / secondes
        avant le prochain 'jeton' rendu (quand le plus ancien appel de la
        fenêtre en sort)."""
        now = now if now is not None else time.time()
        with _VERROU_QUOTA:
            evts = self._charger_evenements(now)
        detail = {}
        for nom, (duree, limite) in self.limites.items():
            dans_fenetre = [e for e in evts if now - e[0] <= duree]
            utilise = sum(e[1] for e in dans_fenetre)
            if dans_fenetre:
                plus_ancien = min(e[0] for e in dans_fenetre)
                reset_dans = max(0, int(plus_ancien + duree - now))
            else:
                reset_dans = 0
            detail[nom] = {
                "utilise": int(utilise),
                "limite": int(limite),
                "restant": int(max(0, limite - utilise)),
                "pct": round(100 * utilise / limite, 1) if limite else 0.0,
                "reset_dans_s": reset_dans,
            }
        return detail

    # ---- décision -----------------------------------------------------------
    def _attente_fenetres_courtes(self, poids, now, evts, fenetres):
        """Secondes à attendre pour qu'un appel de `poids` tienne dans les
        fenêtres indiquées. 0 si c'est déjà bon."""
        attente = 0
        for nom in fenetres:
            duree, limite = self.limites[nom]
            # Un appel dont le poids dépasse à lui seul la limite de la fenêtre
            # ne tiendra JAMAIS dedans : attendre ne fait qu'expirer les appels
            # PASSÉS, pas réduire celui-ci. Inutile d'attendre (Open-Meteo ne
            # bloque pas une grosse requête unique sur la minute ; c'est le
            # total JOURNALIER qui compte). On laisse passer, le retry gérera.
            if poids > limite:
                continue
            dans_fenetre = sorted((e for e in evts if now - e[0] <= duree), key=lambda e: e[0])
            utilise = sum(e[1] for e in dans_fenetre)
            if utilise + poids <= limite:
                continue
            a_liberer = utilise + poids - limite
            libere = 0
            quand = 0
            for ts, p in dans_fenetre:
                libere += p
                quand = ts + duree - now
                if libere >= a_liberer:
                    break
            attente = max(attente, quand)
        return attente

    def attendre_si_besoin(self, poids, log=None, max_attente_s=65, doit_arreter=None):
        """
        Avant un appel Open-Meteo (poids estimé `poids`) : le seul quota qui
        BLOQUE vite et se recharge vite est le quota PAR MINUTE (Open-Meteo
        répond « Minutely API request limit exceeded, try again in one minute »).
        On attend donc au plus ~1 minute pour rester sous ce quota-là.

        Heure / jour / mois : simple AVERTISSEMENT (pas de blocage) -- l'API
        n'expose aucun compteur, notre estimation est approximative, et c'est le
        429 (géré par le retry) qui tranche vraiment. Inutile d'attendre 3 min
        sur une estimation : ça n'a pas de sens (il n'y a pas de fenêtre de 3 min).
        """
        now = time.time()
        with _VERROU_QUOTA:
            evts = self._charger_evenements(now)

        # IMPORTANT : on ne cale QUE sur la minute. C'est la seule limite qui
        # bloque de façon fiable et OBSERVABLE (le 429 dit « Minutely ... » et
        # se recharge en ~1 min). Les fenêtres heure/jour/mois ne sont PAS
        # estimables correctement (l'API n'expose aucun compteur, et le coût réel
        # d'une requête multi-points diffère de notre estimation -- on voit des
        # requêtes passer alors que l'estimation les donnait perdantes). On ne
        # crie donc plus au loup dessus : c'est le vrai 429 (géré par le retry,
        # qui lit la raison) qui tranche pour ces fenêtres-là.
        attente = min(self._attente_fenetres_courtes(poids, now, evts, ("minute",)), max_attente_s)
        if attente > 0:
            if log:
                log(f"⏳ Quota par minute Open-Meteo : petite pause de {int(attente)}s avant l'appel "
                    f"(poids ~{poids})...", "info")
            _dormir_interruptible(attente + 0.5, doit_arreter)

    def enregistrer(self, poids, now=None):
        """Enregistre un appel RÉUSSI (poids consommé) dans le fichier partagé."""
        now = now if now is not None else time.time()
        with _VERROU_QUOTA:
            evts = self._charger_evenements(now)
            evts.append([now, int(poids)])
            self._sauver_evenements(evts)


def afficher_carte_html(lien):
    """Affiche une carte Folium sauvegardée en HTML directement dans le notebook
    (contournement du sandboxing d'iframe de Jupyter via une Data URI base64)."""
    with open(lien, "r", encoding="utf-8") as f:
        html_data = f.read()
    b64_html = base64.b64encode(html_data.encode('utf-8')).decode('utf-8')
    data_uri = f"data:text/html;base64,{b64_html}"
    display(IFrame(src=data_uri, width="100%", height="600px"))


# ==============================================================================
# CLASSE 1 : GESTIONNAIRE DE DONNÉES (Hub'Eau + Open-Meteo)
# ==============================================================================
# Début de la réanalyse météo ERA5 d'Open-Meteo (archive) : AVANT cette date,
# aucune donnée météo n'existe. Le débit Hub'Eau peut remonter bien plus loin
# (parfois 1900), mais ces années sont inexploitables faute de météo -> on
# borne l'apprentissage au recouvrement eau ∩ météo.
DEBUT_METEO_ARCHIVE = pd.Timestamp("1940-01-01")

# Affichage : les débits sont stockés en L/s (colonne debit_L_s) mais présentés
# en m³/s (unité usuelle en hydrologie). 1 m³/s = 1000 L/s.
L_PAR_S_VERS_M3_PAR_S = 1000.0
FORMAT_DATE_FR = "%d/%m/%Y"

# Variables horaires qui possèdent une MOYENNE JOURNALIÈRE native sur Open-Meteo :
# quand on veut juste la moyenne du jour, autant demander directement la variable
# journalière (1 valeur/jour) au lieu de télécharger les 24 valeurs horaires et
# de les moyenner nous-mêmes -> réponse ~24× plus légère, téléchargements longs
# (60+ ans) qui ne se coupent plus. Le coût quota reste ~identique (compté par
# variable), mais le volume et le temps chutent.
MOYENNE_JOURNALIERE_NATIVE = {
    "temperature_2m": "temperature_2m_mean",
    "relative_humidity_2m": "relative_humidity_2m_mean",
    "dew_point_2m": "dew_point_2m_mean",
    "apparent_temperature": "apparent_temperature_mean",
    "wind_speed_10m": "wind_speed_10m_mean",
    "cloud_cover": "cloud_cover_mean",
    "surface_pressure": "surface_pressure_mean",
}


class GestionnaireDonnees:
    """
    Toute la collecte de données brutes, indépendante de la notion de rivière.
    Garde un historique des erreurs réseau rencontrées (utile pour un audit
    après une longue session de téléchargement).
    """

    # Chemin du fichier de quota Open-Meteo, PARTAGÉ par toutes les instances.
    # Le serveur web le fixe sur un chemin absolu commun (webapp/data/...) pour
    # que le compteur soit unique quelle que soit la station courante.
    chemin_quota_openmeteo = None

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.erreurs_reseau = []
        self.quota = GestionnaireQuota(chemin=type(self).chemin_quota_openmeteo)
        self._doit_arreter = None   # callable -> bool, branché par le site
        self._hook_progres = None   # callable(phase, courant, total), branché par le site

    def _log(self, message, niveau="info"):
        if not self.verbose:
            return
        prefixes = {"info": "ℹ️ ", "warning": "⚠️ ", "erreur": "❌ ", "succes": "✅ "}
        print(f"{prefixes.get(niveau, '')}{message}")

    def _verifier_arret(self):
        """Point de contrôle coopératif : lève ArretDemande si un arrêt a été
        demandé. Appelé aux endroits où l'interruption est sûre (entre deux
        lots de téléchargement, avant une longue attente...)."""
        if self._doit_arreter is not None and self._doit_arreter():
            raise ArretDemande("Calcul interrompu à la demande de l'utilisateur.")

    # --------------------------------------------------------------------
    # RIVIÈRE (Hub'Eau)
    # --------------------------------------------------------------------
    def fetch_river_data(self, station_code, start_date, end_date, max_retries=4):
        """Récupère le débit moyen journalier (QmnJ). Réessaie sur les erreurs
        serveur TRANSITOIRES (502/503/504, coupures) -- fréquentes sur Hub'Eau
        -- au lieu de faire échouer tout le pipeline pour un incident passager."""
        url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
        params = {
            "code_entite": station_code,
            "grandeur_hydro_elab": "QmnJ",
            "date_debut_obs_elab": pd.to_datetime(start_date).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "date_fin_obs_elab": pd.to_datetime(end_date).strftime('%Y-%m-%dT%H:%M:%SZ'),
            "size": 20000,
            "sort": "asc"
        }
        for attempt in range(max_retries):
            self._verifier_arret()
            try:
                resp = requests.get(url, params=params, timeout=60)
            except requests.exceptions.RequestException as e:
                wait = 2 ** attempt
                self._log(f"Hub'Eau injoignable ({type(e).__name__}), retry dans {wait}s "
                          f"(tentative {attempt+1}/{max_retries})...", "warning")
                _dormir_interruptible(wait, self._doit_arreter)
                continue

            if resp.status_code in [200, 206]:
                data = resp.json().get('data', [])
                if not data:
                    raise Exception(f"Aucune donnée QmnJ Hub'Eau pour {station_code} entre "
                                    f"{start_date} et {end_date} -- vérifie la période disponible.")
                df = pd.DataFrame(data)[['date_obs_elab', 'resultat_obs_elab']]
                df = df.rename(columns={'date_obs_elab': 'date', 'resultat_obs_elab': 'debit_L_s'})
                df['date'] = pd.to_datetime(df['date'])
                return df
            elif resp.status_code in [429, 500, 502, 503, 504]:
                # Transitoire (surcharge/maintenance côté Hub'Eau) : on retente.
                wait = min(30, 3 * (attempt + 1))
                self.erreurs_reseau.append({"api": "hubeau", "code": resp.status_code, "moment": datetime.now()})
                self._log(f"Hub'Eau {resp.status_code} (transitoire), retry dans {wait}s "
                          f"(tentative {attempt+1}/{max_retries})...", "warning")
                _dormir_interruptible(wait, self._doit_arreter)
            else:
                self.erreurs_reseau.append({"api": "hubeau", "code": resp.status_code, "moment": datetime.now()})
                raise Exception(f"Erreur Hub'Eau : {resp.status_code}")

        raise Exception(f"Hub'Eau injoignable après {max_retries} tentatives (erreurs serveur persistantes).")

    def periode_disponible_river(self, station_code):
        """Détermine (première date, dernière date) des débits QmnJ réellement
        disponibles sur Hub'Eau pour cette station. Permet d'élargir au maximum
        la période de téléchargement plutôt que d'utiliser des dates en dur.
        Note : l'API ne supporte pas sort=desc, donc la dernière date est
        cherchée en interrogeant des fenêtres récentes de plus en plus larges."""
        url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
        base = {"code_entite": station_code, "grandeur_hydro_elab": "QmnJ", "sort": "asc"}

        resp = requests.get(url, params={**base, "size": 1})
        if resp.status_code not in [200, 206] or not resp.json().get('data'):
            raise Exception(f"Aucune donnée QmnJ trouvée pour la station {station_code}.")
        premiere = pd.to_datetime(resp.json()['data'][0]['date_obs_elab'])

        derniere = None
        for jours in [45, 200, 1500, 20000]:
            debut_fenetre = (datetime.now() - pd.Timedelta(days=jours)).strftime('%Y-%m-%d')
            resp = requests.get(url, params={**base, "size": 20000, "date_debut_obs_elab": debut_fenetre})
            data = resp.json().get('data', []) if resp.status_code in [200, 206] else []
            if data:
                derniere = pd.to_datetime(data[-1]['date_obs_elab'])
                break
        if derniere is None:
            raise Exception(f"Impossible de déterminer la dernière date disponible pour {station_code}.")

        self._log(f"Période QmnJ disponible : {premiere.strftime('%Y-%m-%d')} -> {derniere.strftime('%Y-%m-%d')}", "succes")
        return premiere, derniere

    def periode_exploitable_river(self, station_code):
        """Période réellement EXPLOITABLE = recouvrement entre le débit Hub'Eau
        et la météo ERA5 (qui commence en 1940). Renvoie
        (debut_eau, debut_exploitable, fin) : le débit peut commencer avant
        `debut_exploitable`, mais sans météo ces années sont inutilisables."""
        premiere, derniere = self.periode_disponible_river(station_code)
        debut_exploitable = max(premiere, DEBUT_METEO_ARCHIVE)
        if debut_exploitable > premiere:
            self._log(f"Débit disponible depuis {premiere.strftime('%Y-%m-%d')}, mais la météo (ERA5) commence en "
                      f"{DEBUT_METEO_ARCHIVE.year} : apprentissage borné à {debut_exploitable.strftime('%Y-%m-%d')} -> "
                      f"{derniere.strftime('%Y-%m-%d')}.", "info")
        return premiere, debut_exploitable, derniere

    # --------------------------------------------------------------------
    # MÉTÉO (Open-Meteo) -- version un point à la fois
    # --------------------------------------------------------------------
    def fetch_weather_data(self, lat, lon, start_date, end_date,
                            daily_vars=["temperature_2m_max", "rain_sum"],
                            hourly_vars=["snow_depth"]):
        """Récupère Température, Précipitations et Neige en UN SEUL appel"""
        url = "https://archive-api.open-meteo.com/v1/archive" if datetime.strptime(end_date, "%Y-%m-%d") < datetime.now() else "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": daily_vars,
            "hourly": hourly_vars,
            "start_date": start_date,
            "end_date": end_date,
        }
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            self.erreurs_reseau.append({"api": "open-meteo", "code": resp.status_code, "moment": datetime.now(), "point": (lat, lon)})
            raise Exception(f"Erreur Open-Meteo : {resp.status_code}")
        data = resp.json()
        df_daily = pd.DataFrame()
        if 'daily' in data:
            df_daily = pd.DataFrame(data['daily'])
            df_daily['date'] = pd.to_datetime(df_daily['time'])
            df_daily.drop(columns=['time'], inplace=True)
        df_hourly = pd.DataFrame()
        if 'hourly' in data:
            df_hourly = pd.DataFrame(data['hourly'])
            df_hourly['date'] = pd.to_datetime(df_hourly['time'])
            df_hourly.drop(columns=['time'], inplace=True)
        return df_hourly, df_daily

    # --------------------------------------------------------------------
    # MÉTÉO (Open-Meteo) -- version multi-points batchée (1 seul appel HTTP)
    # --------------------------------------------------------------------
    def fetch_weather_data_multi(self, coords_list, start_date, end_date, daily_vars, hourly_vars, max_retries=5):
        """
        Une seule requête HTTP pour tous les points de coords_list (jusqu'à 1000).
        Retourne une liste de tuples (df_hourly, df_daily), dans le même ordre
        que coords_list. Retry avec backoff exponentiel sur 429.
        """
        url = "https://archive-api.open-meteo.com/v1/archive" if datetime.strptime(end_date, "%Y-%m-%d") < datetime.now() else "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": [c[0] for c in coords_list],
            "longitude": [c[1] for c in coords_list],
            "start_date": start_date,
            "end_date": end_date,
        }
        # On n'envoie 'daily'/'hourly' que s'ils sont non vides (Open-Meteo
        # refuse une liste vide) -- utile quand toutes les variables horaires
        # ont été déplacées en journalier (temperature_2m_mean, etc.).
        if daily_vars:
            params["daily"] = list(daily_vars)
        if hourly_vars:
            params["hourly"] = list(hourly_vars)

        # Anticipation du quota : on attend (ou on refuse proprement) AVANT
        # d'envoyer, pour éviter la plupart des 429 plutôt que les subir.
        self._verifier_arret()
        n_vars = len(daily_vars or []) + len(hourly_vars or [])
        poids = self.quota.poids_appel(len(coords_list), n_vars, self.quota.nb_jours(start_date, end_date))
        self.quota.attendre_si_besoin(poids, self._log, doit_arreter=self._doit_arreter)

        data = None
        for attempt in range(max_retries):
            self._verifier_arret()   # interruption possible entre deux tentatives
            try:
                resp = requests.get(url, params=params, timeout=180)
            except requests.exceptions.RequestException as e:
                # Coupure en plein transfert (fréquent sur les très grosses
                # réponses, ex. 60+ ans horaires) ou timeout : on retente.
                wait = 2 ** attempt
                self._log(f"Connexion interrompue ({type(e).__name__}), retry dans {wait}s (tentative {attempt+1}/{max_retries})...", "warning")
                self.erreurs_reseau.append({"api": "open-meteo", "code": "connexion", "moment": datetime.now(), "tentative": attempt + 1})
                _dormir_interruptible(wait, self._doit_arreter)
                continue

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    self.quota.enregistrer(poids)   # appel réussi -> poids consommé
                    break
                except ValueError as e:
                    # Corps tronqué malgré un statut 200 : réponse inexploitable, on retente.
                    wait = 2 ** attempt
                    self._log(f"Réponse tronquée/illisible, retry dans {wait}s (tentative {attempt+1}/{max_retries})...", "warning")
                    self.erreurs_reseau.append({"api": "open-meteo", "code": "json", "moment": datetime.now(), "tentative": attempt + 1})
                    _dormir_interruptible(wait, self._doit_arreter)
            elif resp.status_code == 429:
                # Open-Meteo précise DANS LE CORPS quelle fenêtre est dépassée
                # (ex. « Minutely API request limit exceeded, try again in one
                # minute »). On adapte : minute -> on attend ~1 min ; jour/mois
                # -> inutile d'insister (ça ne se libère pas avant demain).
                raison = ""
                try:
                    raison = (resp.json().get("reason") or "").lower()
                except ValueError:
                    pass
                self.erreurs_reseau.append({"api": "open-meteo", "code": 429, "raison": raison, "moment": datetime.now(), "tentative": attempt + 1})
                if any(mot in raison for mot in ("daily", "day", "hourly", "hour")):
                    raise Exception(f"Quota Open-Meteo épuisé ({raison or 'jour/heure'}) -- relance plus tard "
                                    f"(ce quota ne se recharge pas dans la minute).")
                wait = 60 if attempt == 0 else min(60, 20 * attempt)
                self._log(f"Quota par minute Open-Meteo atteint : nouvel essai dans {wait}s "
                          f"(tentative {attempt+1}/{max_retries})...", "warning")
                _dormir_interruptible(wait, self._doit_arreter)
            elif resp.status_code in (500, 502, 503, 504):
                # Erreur serveur TRANSITOIRE (passerelle Open-Meteo surchargée,
                # fréquent après une coupure sur une grosse réponse) : on retente.
                wait = min(30, 4 * (attempt + 1))
                self._log(f"Open-Meteo {resp.status_code} (erreur serveur transitoire), retry dans {wait}s "
                          f"(tentative {attempt+1}/{max_retries})...", "warning")
                self.erreurs_reseau.append({"api": "open-meteo", "code": resp.status_code, "moment": datetime.now(), "tentative": attempt + 1})
                _dormir_interruptible(wait, self._doit_arreter)
            else:
                self.erreurs_reseau.append({"api": "open-meteo", "code": resp.status_code, "moment": datetime.now()})
                raise Exception(f"Erreur Open-Meteo : {resp.status_code}")

        if data is None:
            raise Exception("Échec après plusieurs tentatives (429/coupures persistants)")
        if isinstance(data, dict):
            data = [data]

        resultats = []
        for point_data in data:
            df_daily = pd.DataFrame()
            if 'daily' in point_data:
                df_daily = pd.DataFrame(point_data['daily'])
                df_daily['date'] = pd.to_datetime(df_daily['time'])
                df_daily.drop(columns=['time'], inplace=True)

            df_hourly = pd.DataFrame()
            if 'hourly' in point_data:
                df_hourly = pd.DataFrame(point_data['hourly'])
                df_hourly['date'] = pd.to_datetime(df_hourly['time'])
                df_hourly.drop(columns=['time'], inplace=True)

            resultats.append((df_hourly, df_daily))

        return resultats

    def fetch_elevations(self, coords_list, taille_lot=100, max_retries=4):
        """
        Altitude (m) de chaque point via l'endpoint ÉLÉVATION d'Open-Meteo
        (modèle de terrain Copernicus 90 m). C'est un endpoint distinct de la
        météo, très léger (jusqu'à 100 points par requête) : on l'utilise pour
        la PRÉSÉLECTION gratuite (étape 1), sans toucher au quota météo.
        Retourne une liste d'altitudes alignée sur coords_list (None si échec).
        """
        url = "https://api.open-meteo.com/v1/elevation"
        elevations = []
        for i in range(0, len(coords_list), taille_lot):
            self._verifier_arret()
            lot = coords_list[i:i + taille_lot]
            params = {"latitude": [c[0] for c in lot], "longitude": [c[1] for c in lot]}
            data = None
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, params=params, timeout=60)
                except requests.exceptions.RequestException:
                    _dormir_interruptible(2 ** attempt, self._doit_arreter)
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    break
                elif resp.status_code == 429:
                    _dormir_interruptible(min(30, 5 * (attempt + 1)), self._doit_arreter)
                else:
                    break
            if data and "elevation" in data:
                elevations.extend(data["elevation"])
            else:
                elevations.extend([None] * len(lot))
        return elevations

    # --------------------------------------------------------------------
    # TRANSFORMATIONS
    # --------------------------------------------------------------------
    @staticmethod
    def hourly_to_wide_daily(df_hourly):
        """
        Transforme l'horaire en format large : une ligne par jour,
        avec 24 colonnes par variable (var_h0 ... var_h23). Aucune perte d'info.
        """
        if df_hourly.empty:
            return pd.DataFrame(columns=['date'])

        df_hourly = df_hourly.copy()
        df_hourly = df_hourly.sort_values('date')
        df_hourly['jour'] = df_hourly['date'].dt.floor('D')
        df_hourly['heure'] = df_hourly['date'].dt.hour

        hourly_vars = [c for c in df_hourly.columns if c not in ['date', 'jour', 'heure']]

        df_wide = df_hourly.pivot_table(index='jour', columns='heure', values=hourly_vars)
        df_wide.columns = [f"{var}_h{heure}" for var, heure in df_wide.columns]
        df_wide = df_wide.reset_index().rename(columns={'jour': 'date'})

        return df_wide

    @staticmethod
    def hourly_to_daily_mean(df_hourly):
        """
        Agrège l'horaire en moyenne journalière : 1 colonne par variable au lieu
        de 24. On perd le profil intra-journée mais on divise la dimension par
        24 -- parfait pour l'évaluateur rapide de la sélection génétique, qui
        n'a besoin que de classer des configurations de points entre elles.
        """
        if df_hourly.empty:
            return pd.DataFrame(columns=['date'])

        df_hourly = df_hourly.copy()
        df_hourly['date'] = df_hourly['date'].dt.floor('D')
        hourly_vars = [c for c in df_hourly.columns if c != 'date']
        return df_hourly.groupby('date')[hourly_vars].mean().reset_index()

    # Modes d'agrégation d'une variable horaire vers des colonnes journalières.
    MODES_AGREGATION = ("moyenne", "minuit", "horaire")

    @staticmethod
    def hourly_to_daily_spec(df_hourly, agregations=None):
        """
        Réduit l'horaire en colonnes journalières AVEC UN MODE PAR VARIABLE, pour
        maîtriser la dimension du problème sans perdre l'information utile :

        - "moyenne"  : moyenne journalière (1 colonne `var_moy`). Physiquement,
          c'est ~l'intégrale de l'énergie reçue sur la journée -- pertinent pour
          la température (fonte du manteau neigeux) ou le rayonnement.
        - "minuit"   : la valeur à 0h (1 colonne `var_h0`). Une PHOTO de l'état,
          idéale pour un STOCK comme la hauteur de neige : inutile de la suivre
          heure par heure, sa valeur à minuit résume le stock ; s'il fond, ça se
          lira dans la température et les précipitations.
        - "horaire"  : les 24 heures conservées (`var_h0`..`var_h23`), aucune
          perte mais 24× plus de colonnes.

        `agregations` : dict {nom_variable: mode}. Une variable absente est
        traitée en "moyenne". Retourne un DataFrame indexé par 'date'.
        """
        if df_hourly.empty:
            return pd.DataFrame(columns=['date'])

        if agregations is None:
            agregations = {}
        df = df_hourly.copy().sort_values('date')
        df['jour'] = df['date'].dt.floor('D')
        df['heure'] = df['date'].dt.hour
        hourly_vars = [c for c in df.columns if c not in ('date', 'jour', 'heure')]

        morceaux = []
        for var in hourly_vars:
            mode = agregations.get(var, "moyenne")
            if mode == "horaire":
                wide = df.pivot_table(index='jour', columns='heure', values=var)
                wide.columns = [f"{var}_h{int(h)}" for h in wide.columns]
                morceaux.append(wide)
            elif mode == "minuit":
                minuit = df[df['heure'] == 0].groupby('jour')[var].mean().rename(f"{var}_h0")
                morceaux.append(minuit)
            else:  # "moyenne" (défaut)
                moy = df.groupby('jour')[var].mean().rename(f"{var}_moy")
                morceaux.append(moy)

        if not morceaux:
            return pd.DataFrame(columns=['date'])

        resultat = pd.concat(morceaux, axis=1).reset_index().rename(columns={'jour': 'date'})
        return resultat

    @staticmethod
    def repartir_variables_meteo(params_hourly, params_daily, agregations):
        """
        Optimisation des requêtes : une variable horaire demandée en moyenne
        journalière et qui a une moyenne NATIVE côté Open-Meteo (voir
        MOYENNE_JOURNALIERE_NATIVE) est déplacée dans les variables JOURNALIÈRES
        (1 valeur/jour au lieu de 24). Retourne (hourly_a_telecharger,
        daily_a_telecharger, renommage) où `renommage` ramène la variable native
        (ex. temperature_2m_mean) au nom de colonne habituel (temperature_2m_moy),
        pour que le reste du pipeline soit inchangé.
        """
        agregations = agregations or {}
        hourly_fetch, daily_extra, renommage = [], [], {}
        for var in (params_hourly or []):
            mode = agregations.get(var, "moyenne")
            natif = MOYENNE_JOURNALIERE_NATIVE.get(var)
            if mode == "moyenne" and natif:
                daily_extra.append(natif)
                renommage[natif] = f"{var}_moy"   # même nom que hourly_to_daily_spec("moyenne")
            else:
                hourly_fetch.append(var)
        daily_fetch = list(params_daily or []) + daily_extra
        return hourly_fetch, daily_fetch, renommage

    @staticmethod
    def encode_cyclical_date(date_obj):
        """Encode une date en deux composantes cycliques (sin, cos) pour éviter
        la cassure du Nouvel An."""
        day_of_year = date_obj.dayofyear
        days_in_year = 365.25
        sin_date = np.sin(2 * np.pi * day_of_year / days_in_year)
        cos_date = np.cos(2 * np.pi * day_of_year / days_in_year)
        return [sin_date, cos_date]

    @staticmethod
    def visualiser_encodage_cyclique(annee=2025):
        """Trace le cercle de l'encodage cyclique (sin/cos) sur une année, coloré par date."""
        date_range = pd.date_range(start=f"{annee}-01-01", end=f"{annee}-12-31")

        cyclical_features = np.array([GestionnaireDonnees.encode_cyclical_date(d) for d in date_range])
        x_cos = cyclical_features[:, 1]
        y_sin = cyclical_features[:, 0]

        colors = mdates.date2num(date_range)

        fig, ax = plt.subplots(figsize=(8, 8))
        scatter = ax.scatter(x_cos, y_sin, c=colors, cmap='twilight', s=15, edgecolors='none')
        ax.plot(x_cos, y_sin, color='gray', alpha=0.3, linestyle='-')

        indices_labels = [0, 90, 181, 273]
        for idx in indices_labels:
            date_label = date_range[idx].strftime('%d %b')
            ax.text(x_cos[idx] * 1.15, y_sin[idx] * 1.15, date_label,
                    fontsize=10, fontweight='bold', ha='center', va='center')

        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_xlim([-1.3, 1.3])
        ax.set_ylim([-1.3, 1.3])
        ax.set_aspect('equal')
        ax.set_title("Représentation Spatiale 2D de l'Encodage Cyclique", fontweight='bold', fontsize=14)
        ax.set_xlabel("Composante Cosinus (Cos)")
        ax.set_ylabel("Composante Sinus (Sin)")
        ax.grid(True, linestyle=':', alpha=0.5)

        cbar = fig.colorbar(scatter, ax=ax, orientation='vertical', shrink=0.8)
        cbar.ax.yaxis.set_major_locator(mdates.MonthLocator(interval=2))
        cbar.ax.yaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        plt.tight_layout()
        plt.show()


# ==============================================================================
# RÉGRESSEUR MULTI-HORIZON CAUSAL
# ==============================================================================
class RegresseurMultiHorizonCausal:
    """
    Multi-sortie avec masquage CAUSAL des features : pour prédire J+k, seules
    les features météo jusqu'à J+min(k, cap_futur_meteo) sont utilisées (la
    pluie de J+8 ne peut pas physiquement influencer le débit de J+3). Un
    sous-modèle est entraîné par horizon, chacun sur son propre sous-ensemble
    de colonnes.

    Double intérêt :
    - retire du bruit non-causal des horizons courts ;
    - permet de prédire à des horizons LOINTAINS (k > 15) : le sous-modèle de
      J+60 n'utilise que la météo <= J+cap (disponible en prévision réelle,
      Open-Meteo s'arrêtant à ~15 jours) + l'historique du débit + la date.

    Interface sklearn minimale (fit/predict), donc compatible avec
    entrainer_incertitude() et la sauvegarde joblib (classe module-level).
    """

    def __init__(self, classe_modele, kwargs_modele, features, cap_futur_meteo=15):
        # (classe, kwargs) plutôt qu'une lambda : indispensable pour que le
        # modèle reste picklable (sauvegarde joblib).
        self.classe_modele = classe_modele
        self.kwargs_modele = dict(kwargs_modele)
        self.features = list(features)
        self.cap_futur_meteo = cap_futur_meteo
        self.modeles_ = []
        self.masques_ = []

    def _indices_pour_horizon(self, k):
        limite = min(k, self.cap_futur_meteo)
        indices = []
        for i, col in enumerate(self.features):
            correspondance = re.search(r"_J\+(\d+)$", col)
            if correspondance is None or int(correspondance.group(1)) <= limite:
                indices.append(i)
        return indices

    @staticmethod
    def _en_matrice(X):
        return X.values if hasattr(X, "values") else np.asarray(X)

    def fit(self, X, Y):
        X = self._en_matrice(X)
        Y = self._en_matrice(Y)
        self.modeles_ = []
        self.masques_ = []
        for k in range(Y.shape[1]):
            indices = self._indices_pour_horizon(k)
            modele = self.classe_modele(**self.kwargs_modele)
            modele.fit(X[:, indices], Y[:, k])
            self.modeles_.append(modele)
            self.masques_.append(indices)
        return self

    def predict(self, X):
        X = self._en_matrice(X)
        colonnes = [modele.predict(X[:, indices])
                    for modele, indices in zip(self.modeles_, self.masques_)]
        return np.column_stack(colonnes)


# ==============================================================================
# CLASSE 2 : RIVER -- pipeline complet pour une station donnée
# ==============================================================================
class River:
    """
    Pipeline complet pour une station hydrométrique :
    1. Identification de la station (nom, position) via le référentiel Hub'Eau.
    2. determiner_bassins_versants() : zones hydro Sandre + grille de points candidats.
    3. preparer_donnees_legeres() : cache local léger (quelques années) pour la
       sélection génétique.
    4. lancer_selection_genetique() : DEAP sélectionne les meilleurs points, puis
       télécharge automatiquement les données COMPLÈTES uniquement sur ces points.
    5. appliquer_pca() : réduction de dimension.
    6. entrainer_*() : Ridge (brut), Linéaire (PCA), Gradient Boosting (brut), Keras (brut).
    7. entrainer_incertitude() : généralise l'approche "résidus + réseau de
       variance" à n'importe quel modèle déjà entraîné.
    """

    def __init__(self, code_station, verbose=True):
        self.code_station = code_station
        self.verbose = verbose
        self._data = GestionnaireDonnees(verbose=verbose)

        # Infos station
        self.nom_station = None
        self.lat_station = None
        self.lon_station = None

        # Bassins versants / grille de points
        self.geojson_bassins = None
        self.points_par_zone = None
        self.listes_de_points = None
        self.noms_zones = None
        self.graphes_par_zone = None

        # Résultats de la sélection génétique
        self.meilleur_individu = None
        self.coords_finales = None
        self.historique_ga = []

        # Sélection en deux temps (remplaçant du GA)
        self.points_preselectionnes = None
        self._altitudes_preselection = None
        self.diagnostic_selection = None
        self.derniere_prevision = None   # données du dernier graphe de prévision (m³/s)

        # Données d'entraînement finales (après sélection GA + téléchargement complet)
        self.df_final = None
        self.train2 = None
        self.test2 = None
        self.X_train2 = self.Y_train2 = self.X_test2 = self.Y_test2 = None
        self.features_X2 = self.targets_Y2 = None
        self.past_day2 = None
        self.predict_day2 = None
        self.split_date2 = None
        self.agregations_horaires = {}   # {var_horaire: "moyenne"|"minuit"|"horaire"}

        # PCA
        self.scaler_pca = None
        self.pca_model = None
        self.n_composantes_pca = None
        self.X_train_pca = self.X_test_pca = None

        # Modèles entraînés : {"nom": {"model":..., "score":..., "scores_detail":...,
        #                               "espace": "brut"|"pca", "hybride":..., "modele_variance":...}}
        self.modeles = {}

        # Caches internes pour les évaluations rapides du GA (remplacent les
        # variables globales du notebook -- indispensable pour que plusieurs
        # instances de River ne se marchent pas dessus)
        self._cache_eau_precalcule = None
        self._dict_meteo_offline = None
        self._fichier_cache_eau = None
        self._fichier_cache_meteo = None

        self._doit_arreter = None   # callable -> bool, branché par le site
        self._hook_progres = None   # callable(phase, courant, total), branché par le site

        self._recuperer_infos_station()

    def _log(self, message, niveau="info"):
        if not self.verbose:
            return
        prefixes = {"info": "ℹ️ ", "warning": "⚠️ ", "erreur": "❌ ", "succes": "✅ "}
        print(f"{prefixes.get(niveau, '')}{message}")

    def _progres(self, phase, courant=None, total=None):
        """Signale l'avancement de l'étape en cours au site (barre de
        progression). courant/total=None => barre indéterminée (étape dont la
        durée n'est pas quantifiable, ex. un gros téléchargement en un bloc)."""
        if self._hook_progres is not None:
            self._hook_progres(phase, courant, total)

    def _verifier_arret(self):
        """Point de contrôle coopératif : lève ArretDemande si l'utilisateur a
        demandé l'arrêt (propagé aussi au gestionnaire de données)."""
        if self._doit_arreter is not None and self._doit_arreter():
            raise ArretDemande("Calcul interrompu à la demande de l'utilisateur.")

    # ==========================================================================
    # 1. IDENTIFICATION DE LA STATION
    # ==========================================================================
    def _recuperer_infos_station(self):
        """Interroge le référentiel Hub'Eau pour récupérer le nom et la position
        de la station à partir de son seul code."""
        url = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/stations"
        params = {"code_station": self.code_station, "format": "json", "size": 1}
        try:
            resp = requests.get(url, params=params, timeout=10)
        except Exception as e:
            self._log(f"Impossible de joindre le référentiel Hub'Eau : {e}", "erreur")
            return

        if resp.status_code != 200:
            self._log(f"Erreur référentiel Hub'Eau (code {resp.status_code})", "erreur")
            return

        data = resp.json().get('data', [])
        if not data:
            self._log(f"Aucune station trouvée pour le code '{self.code_station}'.", "warning")
            return

        info = data[0]
        self.nom_station = info.get('libelle_station', 'Nom inconnu')
        self.lat_station = info.get('latitude_station')
        self.lon_station = info.get('longitude_station')

        self._log(f"Station '{self.nom_station}' localisée ({self.lat_station}, {self.lon_station})", "succes")

    # ==========================================================================
    # 2. BASSINS VERSANTS
    # ==========================================================================
    def previsualiser_bassin(self, lat=None, lon=None, sauvegarder_sous="bassin_versant_apercu.html", afficher=True):
        """
        Aperçu rapide : affiche le(s) polygone(s) de zone hydrographique Sandre
        contenant le point donné (par défaut la station). Diagnostic visuel avant
        de lancer determiner_bassins_versants() qui génère la grille complète.
        """
        lat = self.lat_station if lat is None else lat
        lon = self.lon_station if lon is None else lon

        if lat is None or lon is None:
            self._log("Coordonnées de la station indisponibles, impossible de prévisualiser.", "erreur")
            return None

        delta = 0.05
        bbox = f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}"

        url = "https://services.sandre.eaufrance.fr/geo/sandre"
        params = {
            "SERVICE": "WFS",
            "VERSION": "1.0.0",
            "REQUEST": "GetFeature",
            "typeName": "ZoneHydro_FXX",
            "outputFormat": "geojson",
            "srsName": "EPSG:4326",
            "bbox": bbox
        }

        self._log("Interrogation de l'API Sandre...")
        try:
            rep = requests.get(url, params=params, timeout=10)
        except Exception as e:
            self._log(f"Impossible de joindre l'API : {e}", "erreur")
            return None

        if rep.status_code != 200:
            self._log(f"Erreur Serveur (Code {rep.status_code})", "erreur")
            return None

        try:
            geojson_data = rep.json()
        except Exception:
            self._log("L'API n'a pas renvoyé de JSON valide.", "erreur")
            return None

        if 'features' not in geojson_data or len(geojson_data['features']) == 0:
            self._log("Aucun bassin versant trouvé à ces coordonnées exactes.", "warning")
            self._log(f"Lien de la requête brute : {rep.url}")
            return None

        self._log(f"Bassin versant trouvé ! (ID: {geojson_data['features'][0]['properties'].get('CdSecteurHydro', 'Inconnu')})", "succes")

        carte = folium.Map(location=[lat, lon], zoom_start=11)

        folium.GeoJson(
            geojson_data,
            name="Secteur Hydrographique",
            style_function=lambda feature: {
                'fillColor': '#1f77b4',
                'color': '#0d47a1',
                'weight': 2,
                'fillOpacity': 0.3
            }
        ).add_to(carte)

        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color="red", icon="info-sign"),
            popup="📍 Station"
        ).add_to(carte)

        carte.save(sauvegarder_sous)
        self._log(f"Carte sauvegardée sous '{sauvegarder_sous}'", "succes")

        if afficher:
            afficher_carte_html(sauvegarder_sous)

        return carte

    def _surface_bassin_versant(self):
        """Surface du bassin versant amont (km²) déclarée dans le référentiel
        Hub'Eau, sur la fiche du SITE (8 premiers caractères du code station)."""
        try:
            resp = requests.get(
                "https://hubeau.eaufrance.fr/api/v2/hydrometrie/referentiel/sites",
                params={"code_site": self.code_station[:8], "format": "json", "size": 1},
                timeout=15,
            )
            data = resp.json().get('data', [])
            if data and data[0].get('surface_bv'):
                return float(data[0]['surface_bv'])
        except Exception:
            pass
        return None

    @staticmethod
    def _surface_km2(poly):
        """Surface approximative (km²) d'un polygone en degrés WGS84."""
        return poly.area * (111.32 ** 2) * math.cos(math.radians(poly.centroid.y))

    def _filtrer_zones_amont(self, zones, surface_bv):
        """
        Ne garde que les zones hydrographiques réellement EN AMONT de la
        station, en exploitant la codification hydrographique française : le
        code de zone (4 caractères, ex. X045) est hiérarchique
        (région X > secteur X0 > sous-secteur X04 = la rivière > zone X045)
        et les zones sont numérotées de la source vers l'aval.

        - Priorité 1 : les zones du même sous-secteur (= le bassin de la
          rivière elle-même) de numéro <= celui de la station (donc en amont).
        - Priorités 2/3 : sous-secteurs puis secteurs plus en amont du même
          bassin -- utiles uniquement pour une station sur un grand cours
          d'eau dont le bassin dépasse son sous-secteur.
        On s'arrête dès que la surface cumulée des zones retenues atteint la
        surface de bassin versant officielle de la station (surface_bv,
        Hub'Eau) : c'est elle qui départage les zones voisines qui ne drainent
        PAS vers la station (ex. la Durance autour de l'Ubaye).
        Retourne (zones_retenues, zones_exclues).
        """
        code_zone = self.code_station[:4]
        region, secteur, sous_secteur = code_zone[:1], code_zone[:2], code_zone[:3]

        def code(f):
            return f['properties'].get('CdZoneHydro') or ''

        tier1 = sorted([f for f in zones if code(f)[:3] == sous_secteur and code(f) <= code_zone],
                       key=code, reverse=True)
        tier2 = sorted([f for f in zones if code(f)[:2] == secteur and code(f)[:3] < sous_secteur],
                       key=code, reverse=True)
        tier3 = sorted([f for f in zones if code(f)[:1] == region and code(f)[:2] < secteur],
                       key=code, reverse=True)

        if not tier1:
            self._log("Aucune zone amont identifiable via le code hydrographique -- toutes les zones proches sont conservées.", "warning")
            return zones, []

        # Sans surface officielle, impossible de savoir si le bassin déborde
        # du sous-secteur : on s'en tient au sous-secteur (le cas courant).
        candidats = tier1 + (tier2 + tier3 if surface_bv else [])

        retenues, surface_cumulee = [], 0.0
        for f in candidats:
            if retenues and surface_bv and surface_cumulee >= 0.95 * surface_bv:
                break
            retenues.append(f)
            surface_cumulee += self._surface_km2(shape(f['geometry']))

        exclues = [f for f in zones if f not in retenues]
        if surface_bv:
            self._log(f"Bassin amont reconstitué : {len(retenues)} zone(s), {surface_cumulee:.0f} km² "
                      f"(surface officielle du bassin : {surface_bv:.0f} km²).", "succes")
        else:
            self._log(f"Bassin amont : {len(retenues)} zone(s) du sous-secteur {sous_secteur} (surface officielle inconnue).", "succes")
        for f in retenues:
            self._log(f"   ✔ {code(f)} -- {f['properties'].get('LbZoneHydro')}")
        for f in exclues:
            self._log(f"   ✘ écartée (pas en amont) : {code(f)} -- {f['properties'].get('LbZoneHydro')}")
        return retenues, exclues

    def determiner_bassins_versants(self, delta=0.05, n_points_grille=15, max_points_par_zone=25,
                                     sauvegarder_sous="bassins_versants.html", afficher=True,
                                     filtrer_amont=True):
        """
        Récupère les zones hydrographiques (bassins versants) Sandre autour de la
        station, ne garde que celles réellement EN AMONT de la station
        (filtrer_amont=True, voir _filtrer_zones_amont), puis génère une grille
        de points candidats strictement à l'intérieur de chaque zone retenue.
        Remplit self.points_par_zone, self.listes_de_points, self.noms_zones.
        """
        if self.lat_station is None or self.lon_station is None:
            raise Exception("Position de la station inconnue -- la récupération des infos station a-t-elle échoué ?")

        surface_bv = self._surface_bassin_versant() if filtrer_amont else None

        if filtrer_amont:
            # La zone de recherche doit couvrir tout le bassin amont, qui peut
            # s'étendre loin de la station : rayon déduit de la surface du bassin.
            rayon_km = min(300.0, max(25.0, 2.5 * math.sqrt(surface_bv))) if surface_bv else 60.0
            dlat = rayon_km / 111.0
            dlon = rayon_km / (111.0 * math.cos(math.radians(self.lat_station)))
        else:
            dlat = dlon = delta
        bbox = f"{self.lon_station-dlon},{self.lat_station-dlat},{self.lon_station+dlon},{self.lat_station+dlat}"

        self._log("🌍 Récupération des bassins versants depuis l'API Sandre...")
        url = "https://services.sandre.eaufrance.fr/geo/sandre"
        params = {
            "SERVICE": "WFS",
            "VERSION": "1.0.0",
            "REQUEST": "GetFeature",
            "typeName": "ZoneHydro_FXX",
            "outputFormat": "geojson",
            "bbox": bbox
        }

        rep = requests.get(url, params=params)
        if rep.status_code != 200:
            raise Exception(f"Erreur API Sandre (Code {rep.status_code})")

        geojson_data = rep.json()
        zones = geojson_data.get('features', [])
        self._log(f"{len(zones)} zones hydrographiques trouvées dans la zone de recherche.", "succes")

        zones_exclues = []
        if filtrer_amont:
            zones, zones_exclues = self._filtrer_zones_amont(zones, surface_bv)
            geojson_data = {"type": "FeatureCollection", "features": zones}

        points_par_zone = {}
        self._log("🎯 Génération des points strictement à l'intérieur des zones...")

        for i, feature in enumerate(zones):
            nom_zone = feature['properties'].get('LbZoneHydro', f"Zone_{i}")
            poly = shape(feature['geometry'])
            min_lon, min_lat, max_lon, max_lat = poly.bounds

            lats = np.linspace(min_lat, max_lat, n_points_grille)
            lons = np.linspace(min_lon, max_lon, n_points_grille)

            points_valides = []
            for lat in lats:
                for lon in lons:
                    point_gps = Point(lon, lat)
                    if poly.contains(point_gps):
                        # float() natif : évite les repr "np.float64(...)" dans
                        # les logs et les caches CSV.
                        points_valides.append((round(float(lat), 4), round(float(lon), 4)))

            if len(points_valides) > max_points_par_zone:
                indices = np.linspace(0, len(points_valides) - 1, max_points_par_zone, dtype=int)
                points_valides = [points_valides[idx] for idx in indices]

            points_par_zone[nom_zone] = points_valides
            self._log(f"   -> {nom_zone} : {len(points_valides)} points validés.")

        self.geojson_bassins = geojson_data
        self.points_par_zone = points_par_zone
        self.listes_de_points = list(points_par_zone.values())
        self.noms_zones = list(points_par_zone.keys())

        # ---- Visualisation ----
        carte = folium.Map(location=[self.lat_station, self.lon_station], zoom_start=11, tiles="CartoDB positron")

        for feature in geojson_data.get('features', []) + zones_exclues:
            if 'properties' in feature:
                if not feature['properties'].get('LbZoneHydro'):
                    feature['properties']['LbZoneHydro'] = "Zone Hydro Inconnue"

        if zones_exclues:
            folium.GeoJson(
                {"type": "FeatureCollection", "features": zones_exclues},
                style_function=lambda x: {
                    'fillColor': '#9e9e9e',
                    'color': '#757575',
                    'weight': 1,
                    'dashArray': '4',
                    'fillOpacity': 0.07
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=['LbZoneHydro'],
                    aliases=['Écartée (pas en amont) :'],
                    localize=True
                )
            ).add_to(carte)

        folium.GeoJson(
            geojson_data,
            style_function=lambda x: {
                'fillColor': '#2ca02c',
                'color': '#006400',
                'weight': 2,
                'fillOpacity': 0.15
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['LbZoneHydro'],
                aliases=['Bassin Versant:'],
                localize=True
            )
        ).add_to(carte)

        couleurs = ['blue', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black']
        for idx, (nom_zone, points) in enumerate(points_par_zone.items()):
            coul = couleurs[idx % len(couleurs)]
            for lat, lon in points:
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color=coul,
                    fill=True,
                    fill_color=coul,
                    fill_opacity=0.7,
                    popup=f"<b>Zone :</b> {nom_zone}<br>Lat: {lat}<br>Lon: {lon}"
                ).add_to(carte)

        folium.Marker(
            [self.lat_station, self.lon_station],
            icon=folium.Icon(color="red", icon="info-sign"),
            popup="<b>📍 Station Hub'Eau de Référence</b>"
        ).add_to(carte)

        carte.save(sauvegarder_sous)
        self._log(f"Carte sauvegardée sous '{sauvegarder_sous}'", "succes")

        if afficher:
            afficher_carte_html(sauvegarder_sous)

        return carte

    def visualiser_points_sur_carte(self, liste_coords, sauvegarder_sous="points_grille.html", afficher=True):
        """Génère une carte interactive avec des points donnés + la station."""
        if not liste_coords:
            self._log("La liste de coordonnées est vide.", "erreur")
            return None

        lats = [coord[0] for coord in liste_coords]
        lons = [coord[1] for coord in liste_coords]
        centre_lat = sum(lats) / len(lats)
        centre_lon = sum(lons) / len(lons)

        carte = folium.Map(location=[centre_lat, centre_lon], zoom_start=10)

        for lat, lon in liste_coords:
            folium.CircleMarker(
                location=[lat, lon],
                radius=4,
                color="#1f77b4",
                fill=True,
                fill_color="#1f77b4",
                fill_opacity=0.7,
                popup=f"Lat: {lat}<br>Lon: {lon}"
            ).add_to(carte)

        if self.lat_station and self.lon_station:
            folium.Marker(
                location=[self.lat_station, self.lon_station],
                icon=folium.Icon(color="red", icon="info-sign"),
                popup="STATION HUB'EAU"
            ).add_to(carte)

        carte.save(sauvegarder_sous)
        if afficher:
            afficher_carte_html(sauvegarder_sous)

        return carte

    # ==========================================================================
    # 3. DONNÉES LÉGÈRES (cache local pour le GA)
    # ==========================================================================
    def preparer_donnees_legeres(self, params_hourly, params_daily,
                                  start_fetch=None, end_fetch=None,
                                  taille_lot=10, pause_entre_lots=20,
                                  agreger_en_journalier=True, duree_cache_annees=3):
        """
        Télécharge TOUTES les données de la grille (self.listes_de_points) une
        seule fois et les sauvegarde sur le disque (cache_eau_{station}.csv /
        cache_meteo_{station}.csv) pour une optimisation hors-ligne rapide par le GA.

        Ce cache est volontairement LÉGER (il ne sert qu'à classer des
        configurations de points entre elles, pas à entraîner le modèle final) :
        - agreger_en_journalier=True : les variables horaires sont réduites en
          moyenne journalière (1 colonne au lieu de 24) -> l'évaluateur Ridge du
          GA passe de ~1800 à ~130 dimensions et devient quasi instantané ;
        - période par défaut (start/end_fetch=None) : les `duree_cache_annees`
          dernières années réellement disponibles sur Hub'Eau (déterminées via
          periode_disponible_river), au lieu de dates en dur qui plantent sur
          les stations sans données à cette période.

        La météo est récupérée par LOTS de `taille_lot` points (1 requête HTTP
        par lot, via fetch_weather_data_multi, qui gère déjà le retry/backoff
        sur 429), avec une pause de `pause_entre_lots` secondes entre deux lots.
        """
        if self.listes_de_points is None:
            raise Exception("Appelle determiner_bassins_versants() avant preparer_donnees_legeres().")

        if start_fetch is None or end_fetch is None:
            premiere, derniere = self._data.periode_disponible_river(self.code_station)
            if end_fetch is None:
                end_fetch = derniere.strftime('%Y-%m-%d')
            if start_fetch is None:
                debut = max(premiere, pd.to_datetime(end_fetch) - pd.Timedelta(days=365 * duree_cache_annees))
                start_fetch = debut.strftime('%Y-%m-%d')
            self._log(f"Période du cache léger : {start_fetch} -> {end_fetch}")

        tous_les_points = [pt for sous_liste in self.listes_de_points for pt in sous_liste]

        fichier_eau = f"cache_eau_{self.code_station}.csv"
        fichier_meteo = f"cache_meteo_{self.code_station}.csv"

        self._log("==========================================")
        self._log("🔄 ÉTAPE 1 : MOISSONNAGE DES DONNÉES")
        self._log("==========================================")

        self._log(f"💧 Téléchargement Hub'Eau ({start_fetch} à {end_fetch})...")
        df_eau = self._data.fetch_river_data(self.code_station, start_fetch, end_fetch)
        df_eau.to_csv(fichier_eau, index=False)
        self._log(f"Sauvegardé -> {fichier_eau}", "succes")

        nb_lots = math.ceil(len(tous_les_points) / taille_lot)
        self._log(f"🌤 Téléchargement Météo pour {len(tous_les_points)} points en {nb_lots} lot(s) de {taille_lot} (1 requête HTTP par lot)...")

        list_df_meteo = []
        erreurs_points = []

        def telecharger_lot(lot):
            resultats = self._data.fetch_weather_data_multi(
                lot, start_fetch, end_fetch,
                daily_vars=params_daily, hourly_vars=params_hourly
            )
            for (lat, lon), (df_h, df_d) in zip(lot, resultats):
                if not df_h.empty:
                    if agreger_en_journalier:
                        df_h_jour = self._data.hourly_to_daily_mean(df_h)
                    else:
                        df_h_jour = self._data.hourly_to_wide_daily(df_h)
                    df_pt = pd.merge(df_d, df_h_jour, on='date', how='outer')
                else:
                    df_pt = df_d.copy()

                df_pt['lat'] = lat
                df_pt['lon'] = lon
                list_df_meteo.append(df_pt)

        lots_echoues = []
        for i in range(0, len(tous_les_points), taille_lot):
            self._verifier_arret()   # interruption possible entre deux lots
            lot = tous_les_points[i:i + taille_lot]
            self._log(f"   -> Lot [{i+1}-{i+len(lot)}/{len(tous_les_points)}]...")
            self._progres("Téléchargement météo (cache léger)", i, len(tous_les_points))

            try:
                telecharger_lot(lot)
                self._progres("Téléchargement météo (cache léger)", i + len(lot), len(tous_les_points))
                self._log(f"      OK ({len(lot)} points)", "succes")
                if i + taille_lot < len(tous_les_points):
                    _dormir_interruptible(pause_entre_lots, self._doit_arreter)
            except ArretDemande:
                raise
            except Exception as e:
                self._log(f"      ERREUR sur ce lot : {e}", "erreur")
                lots_echoues.append(lot)

        # Seconde chance pour les lots tombés sur un quota persistant (le
        # quota HORAIRE d'Open-Meteo, contrairement au quota minute, ne se
        # libère pas en quelques secondes) : une seule repasse après une vraie
        # pause. Au-delà, on continue avec un cache partiel plutôt que planter.
        if lots_echoues:
            self._log(f"{len(lots_echoues)} lot(s) en échec -- nouvelle tentative unique dans 90s (quota horaire ?)...", "warning")
            _dormir_interruptible(90, self._doit_arreter)
            for lot in lots_echoues:
                try:
                    telecharger_lot(lot)
                    self._log(f"      Rattrapage OK ({len(lot)} points)", "succes")
                    _dormir_interruptible(pause_entre_lots, self._doit_arreter)
                except ArretDemande:
                    raise
                except Exception as e:
                    self._log(f"      Rattrapage échoué : {e}", "erreur")
                    erreurs_points.extend([(lat, lon, str(e)) for lat, lon in lot])
            if erreurs_points:
                self._log("Le quota horaire d'Open-Meteo semble épuisé : le GA travaillera sur un cache partiel "
                          "(les points manquants sont simplement ignorés). Relance plus tard pour un cache complet.", "warning")

        if not list_df_meteo:
            raise Exception("Aucune donnée météo n'a pu être téléchargée : le quota JOURNALIER d'Open-Meteo est "
                            "probablement épuisé (il ne se libère pas dans l'heure). Relance demain -- le cache "
                            "sera reconstruit automatiquement.")

        self._log("💾 Assemblage et sauvegarde de la base de données météo géante...")
        df_meteo_total = pd.concat(list_df_meteo, ignore_index=True)
        df_meteo_total.to_csv(fichier_meteo, index=False)

        self._log(f"Sauvegardé -> {fichier_meteo}", "succes")
        self._log(f"🎯 Prêt pour la sélection génétique ! (Taille de la base : {len(df_meteo_total)} lignes)")

        if erreurs_points:
            self._log(f"{len(erreurs_points)} point(s) ont échoué durant le moissonnage :", "warning")
            for lat, lon, err in erreurs_points:
                self._log(f"   - ({lat}, {lon}) : {err}", "warning")

        self._fichier_cache_eau = fichier_eau
        self._fichier_cache_meteo = fichier_meteo

        return fichier_eau, fichier_meteo

    def charger_cache_offline(self, fichier_eau="cache_eau_offline.csv", fichier_meteo="cache_meteo_offline.csv"):
        """
        Pointe l'instance vers un cache eau/météo déjà présent sur le disque
        (par exemple sauvegardé lors d'une session précédente) au lieu de le retélécharger via
        preparer_donnees_legeres(). Permet de lancer lancer_selection_genetique()
        / evaluer_configuration_opti_offline() sans toucher au réseau.
        """
        if not os.path.exists(fichier_eau):
            raise FileNotFoundError(f"Fichier introuvable : {fichier_eau}")
        if not os.path.exists(fichier_meteo):
            raise FileNotFoundError(f"Fichier introuvable : {fichier_meteo}")

        self._fichier_cache_eau = fichier_eau
        self._fichier_cache_meteo = fichier_meteo
        # Invalide les caches en mémoire pour forcer un rechargement depuis ces fichiers.
        self._cache_eau_precalcule = None
        self._dict_meteo_offline = None

        if self.listes_de_points is not None:
            points_dispos = set(map(tuple, pd.read_csv(fichier_meteo, usecols=['lat', 'lon']).round(4).drop_duplicates().values))
            points_grille = set((round(lat, 4), round(lon, 4)) for zone in self.listes_de_points for lat, lon in zone)
            manquants = points_grille - points_dispos
            if manquants:
                self._log(f"{len(manquants)}/{len(points_grille)} points de la grille actuelle sont absents du cache chargé (ignorés par le GA).", "warning")
            else:
                self._log(f"Cache chargé : les {len(points_dispos)} points du cache couvrent toute la grille actuelle.", "succes")

        self._log(f"Cache offline chargé -> {fichier_eau} / {fichier_meteo}", "succes")
        return fichier_eau, fichier_meteo

    # ==========================================================================
    # 4. ÉVALUATEUR RAPIDE HORS-LIGNE (utilisé par le GA)
    # ==========================================================================
    def evaluer_configuration_opti_offline(self, coords_list, split_date="2019-12-31",
                                            past_day_opti=20, predict_day_opti=3, alpha_ridge=10.0):
        """
        Évalue une configuration de capteurs à partir des données pré-téléchargées
        (cache local). Vitesse fulgurante -- utilisée par le GA pour tester des
        milliers de configurations. Ridge (pas LinearRegression) : la régularisation
        L2 gère la colinéarité massive entre les features horaires et empêche
        les coefficients d'exploser sur les splits à petit jeu d'entraînement.
        """
        if self._cache_eau_precalcule is None:
            self._log("🚀 Premier appel : mise en cache et pré-calculs ultra-rapides...")
            fichier_eau = self._fichier_cache_eau or f"cache_eau_{self.code_station}.csv"
            fichier_meteo = self._fichier_cache_meteo or f"cache_meteo_{self.code_station}.csv"

            df_eau = pd.read_csv(fichier_eau, parse_dates=['date']).sort_values('date').reset_index(drop=True)
            df_meteo = pd.read_csv(fichier_meteo, parse_dates=['date'])

            dict_eau_shifts = {}
            for i in range(1, past_day_opti + 1):
                dict_eau_shifts[f"debit_J-{i}"] = df_eau['debit_L_s'].shift(i)
            for i in range(0, predict_day_opti + 1):
                dict_eau_shifts[f"cible_debit_J+{i}"] = df_eau['debit_L_s'].shift(-i)

            dict_eau_shifts['sin_date'] = np.sin(2 * np.pi * df_eau['date'].dt.dayofyear / 365.25)
            dict_eau_shifts['cos_date'] = np.cos(2 * np.pi * df_eau['date'].dt.dayofyear / 365.25)

            self._cache_eau_precalcule = pd.concat([df_eau, pd.DataFrame(dict_eau_shifts)], axis=1)

            df_meteo['lat'] = df_meteo['lat'].round(4)
            df_meteo['lon'] = df_meteo['lon'].round(4)
            self._dict_meteo_offline = {}

            for (lat, lon), group in df_meteo.groupby(['lat', 'lon']):
                self._dict_meteo_offline[(lat, lon)] = group.sort_values('date').set_index('date').drop(columns=['lat', 'lon'])

            self._log("Pré-calcul terminé ! La sélection génétique va maintenant voler.", "succes")

        dfs_a_moyenner = []
        for lat, lon in coords_list:
            coords = (round(lat, 4), round(lon, 4))
            if coords in self._dict_meteo_offline:
                dfs_a_moyenner.append(self._dict_meteo_offline[coords])

        if not dfs_a_moyenner:
            return None, -9999

        df_meteo_mean = sum(dfs_a_moyenner) / len(dfs_a_moyenner)
        colonnes_meteo_opti = df_meteo_mean.columns

        dict_m_shifts = {}
        for i in range(1, past_day_opti + 1):
            shifted = df_meteo_mean.shift(i)
            for col in colonnes_meteo_opti:
                dict_m_shifts[f"{col}_J-{i}"] = shifted[col]

        for i in range(0, predict_day_opti + 1):
            shifted = df_meteo_mean.shift(-i)
            for col in colonnes_meteo_opti:
                dict_m_shifts[f"{col}_J+{i}"] = shifted[col]

        df_meteo_opti = pd.DataFrame(dict_m_shifts, index=df_meteo_mean.index)

        df_opti = self._cache_eau_precalcule.merge(df_meteo_opti, left_on='date', right_index=True, how='inner').dropna()

        if len(df_opti) == 0:
            return None, -9999

        train_opti = df_opti[df_opti['date'] <= split_date]
        test_opti = df_opti[df_opti['date'] > split_date]

        if len(train_opti) == 0 or len(test_opti) == 0:
            return None, -9999

        features_X_opti = [f"debit_J-{i}" for i in range(1, past_day_opti + 1)] + ['sin_date', 'cos_date']
        for i in range(1, past_day_opti + 1):
            features_X_opti.extend([f"{col}_J-{i}" for col in colonnes_meteo_opti])
        for i in range(0, predict_day_opti + 1):
            features_X_opti.extend([f"{col}_J+{i}" for col in colonnes_meteo_opti])

        targets_Y_opti = [f"cible_debit_J+{i}" for i in range(0, predict_day_opti + 1)]
        features_X_opti = [col for col in features_X_opti if col in train_opti.columns]

        X_train_opti, Y_train_opti = train_opti[features_X_opti], train_opti[targets_Y_opti]
        X_test_opti, Y_test_opti = test_opti[features_X_opti], test_opti[targets_Y_opti]

        model_opti = Ridge(alpha=alpha_ridge)
        model_opti.fit(X_train_opti, Y_train_opti)

        predictions_opti = model_opti.predict(X_test_opti)
        scores_r2_simple = r2_score(Y_test_opti, predictions_opti)

        return (model_opti, scores_r2_simple)

    # ==========================================================================
    # 5. SÉLECTION GÉNÉTIQUE (DEAP)
    # ==========================================================================
    def lancer_selection_genetique(self,
                                    taille_population=20,
                                    nombre_generations=50,
                                    proba_croisement=0.5,
                                    proba_mutation=0.7,
                                    nombre_enfants_par_couple=3,
                                    nombre_enfants_gardes=2,
                                    activer_hall_of_fame=True,
                                    tuer_les_mauvais=True,
                                    pourcentage_a_tuer=0.2,
                                    methode_selection="sus",
                                    k_voisins_graphe=4,
                                    split_date_ga=None,
                                    past_day_opti=20,
                                    predict_day_opti=3,
                                    telecharger_donnees_completes=True,
                                    start_fetch_final=None,
                                    end_fetch_final=None,
                                    split_date_final=None,
                                    past_day_final=20,
                                    predict_day_final=15,
                                    mode_split="annees_aleatoires",
                                    part_test=0.2,
                                    agregations_horaires=None,
                                    start_train=None,
                                    end_train=None,
                                    params_hourly=None,
                                    params_daily=None):
        """
        Algorithme génétique (DEAP, ADN à taille variable) pour sélectionner les
        meilleurs points météo dans les bassins versants identifiés. À chaque
        couple de parents, plusieurs enfants sont générés et seuls les meilleurs
        sont gardés. À la fin : nettoyage des doublons, carte des résultats, puis
        téléchargement COMPLET (pas le cache léger) uniquement sur les points
        retenus, et construction de X_train2/Y_train2/X_test2/Y_test2.

        predict_day_opti (défaut 3) : horizon utilisé UNIQUEMENT pour classer
        les configurations de points entre elles pendant le GA -- prédire à
        3 jours suffit pour départager des emplacements de capteurs et divise
        la taille du problème. L'horizon du VRAI modèle final est
        predict_day_final (défaut 15), indépendant.

        split_date_ga=None (défaut) : coupe automatiquement le cache léger à
        1 an avant sa fin (train = le reste, test = la dernière année).
        start/end_fetch_final et split_date_final à None : période finale
        élargie au maximum disponible (voir construire_donnees_completes).
        """
        if self.listes_de_points is None:
            raise Exception("Appelle determiner_bassins_versants() avant lancer_selection_genetique().")

        if split_date_ga is None:
            fichier_eau = self._fichier_cache_eau or f"cache_eau_{self.code_station}.csv"
            dates_cache = pd.read_csv(fichier_eau, usecols=['date'], parse_dates=['date'])['date']
            split_date_ga = (dates_cache.max() - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
            self._log(f"Split automatique du GA : {split_date_ga} (dernière année du cache = test)")

        listes_de_points = self.listes_de_points
        noms_zones = self.noms_zones

        # ---- 1. Graphes d'adjacence ----
        def construire_matrice_adjacence(points, k=4):
            graphe = {}
            n = len(points)
            for i in range(n):
                distances = []
                for j in range(n):
                    if i != j:
                        dist = math.sqrt((points[i][0] - points[j][0]) ** 2 + (points[i][1] - points[j][1]) ** 2)
                        distances.append((j, dist))
                distances.sort(key=lambda x: x[1])
                graphe[i] = [idx for idx, d in distances[:k]]
            return graphe

        self._log("🕸️ Construction des matrices d'adjacence pour les zones...")
        graphes_par_zone = [construire_matrice_adjacence(zone, k=k_voisins_graphe) for zone in listes_de_points]
        self.graphes_par_zone = graphes_par_zone

        # ---- 2. Moteur DEAP (ADN taille variable) ----
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        def init_individu():
            return [(i, random.randint(0, len(zone) - 1)) for i, zone in enumerate(listes_de_points)]

        toolbox.register("individu", tools.initIterate, creator.Individual, init_individu)
        toolbox.register("population", tools.initRepeat, list, toolbox.individu)

        # ---- 3. Évaluation et mutation ----
        def evaluation(individu):
            coords = [listes_de_points[zone_id][pt_idx] for zone_id, pt_idx in individu]
            _, score = self.evaluer_configuration_opti_offline(
                coords_list=coords,
                split_date=split_date_ga,
                past_day_opti=past_day_opti,
                predict_day_opti=predict_day_opti
            )
            return (score if score is not None else -9999,)

        def mutation_variable(individu):
            """ Gère le déplacement, la duplication et la délétion """
            choix = random.random()

            if choix < 0.33 and len(individu) > 0:
                gene_idx = random.randrange(len(individu))
                zone_id, pt_idx = individu[gene_idx]
                voisins = graphes_par_zone[zone_id][pt_idx]
                if voisins:
                    individu[gene_idx] = (zone_id, random.choice(voisins))

            elif choix < 0.66:
                zone_id = random.randrange(len(listes_de_points))
                pt_idx = random.randint(0, len(listes_de_points[zone_id]) - 1)
                individu.append((zone_id, pt_idx))

            elif len(individu) > 1:
                gene_idx = random.randrange(len(individu))
                individu.pop(gene_idx)

            return individu,

        toolbox.register("evaluate", evaluation)
        toolbox.register("mutate", mutation_variable)
        toolbox.register("mate", tools.cxTwoPoint)

        if methode_selection == "tournoi":
            toolbox.register("select", tools.selTournament, tournsize=3)
        elif methode_selection == "elitisme":
            toolbox.register("select", tools.selBest)
        elif methode_selection == "roulette":
            toolbox.register("select", tools.selRoulette)
        elif methode_selection == "sus":
            toolbox.register("select", tools.selStochasticUniversalSampling)

        # ---- 4. Boucle d'évolution ----
        self._log(f"🧬 Lancement de l'Algorithme (Sélection : {methode_selection.upper()} | ADN Variable)...")

        pop = toolbox.population(n=taille_population)
        hof = tools.HallOfFame(1)

        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        if activer_hall_of_fame:
            hof.update(pop)

        pbar = tqdm(total=nombre_generations, desc="Évolution en cours", unit="gén")
        self.historique_ga = []

        for g in range(nombre_generations):
            self._verifier_arret()   # interruption possible à chaque génération
            self._progres("Sélection génétique des points", g, nombre_generations)

            if tuer_les_mauvais:
                nb_survivants = int(len(pop) * (1 - pourcentage_a_tuer))
                pool_reproduction = tools.selBest(pop, nb_survivants)
            else:
                pool_reproduction = pop

            if methode_selection in ["roulette", "sus"]:
                pire_score = min([ind.fitness.values[0] for ind in pool_reproduction])
                if pire_score < 0:
                    for ind in pool_reproduction:
                        ind.fitness.values = (ind.fitness.values[0] + abs(pire_score) + 0.001,)

                parents = toolbox.select(pool_reproduction, len(pop))

                if pire_score < 0:
                    for ind in pool_reproduction:
                        ind.fitness.values = (ind.fitness.values[0] - abs(pire_score) - 0.001,)
            else:
                parents = toolbox.select(pool_reproduction, len(pop))

            parents = list(map(toolbox.clone, parents))

            if len(parents) % 2 == 1:
                # Nombre impair de parents : le dernier serait sinon purement
                # ignoré par le pairing zip() ci-dessous, ce qui ferait
                # rétrécir la population de façon irréversible.
                parents.append(toolbox.clone(random.choice(parents)))

            offspring = []
            for parent1, parent2 in zip(parents[::2], parents[1::2]):

                # cxTwoPoint plante si l'un des deux ADN n'a qu'un seul gène
                # (randrange(1, 1)) -- cas fréquent quand il y a peu de zones
                # et que les mutations-délétions raccourcissent les individus.
                croisement_possible = min(len(parent1), len(parent2)) >= 2

                if random.random() < proba_croisement and croisement_possible:
                    candidats = []
                    for _ in range(nombre_enfants_par_couple):
                        child1, child2 = toolbox.clone(parent1), toolbox.clone(parent2)
                        toolbox.mate(child1, child2)

                        if random.random() < proba_mutation:
                            toolbox.mutate(child1)
                        if random.random() < proba_mutation:
                            toolbox.mutate(child2)

                        candidats.extend([child1, child2])

                    fitnesses = list(map(toolbox.evaluate, candidats))
                    for c, f in zip(candidats, fitnesses):
                        c.fitness.values = f

                    candidats.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
                    offspring.extend(candidats[:nombre_enfants_gardes])

                else:
                    if random.random() < proba_mutation:
                        toolbox.mutate(parent1)
                        del parent1.fitness.values
                    if random.random() < proba_mutation:
                        toolbox.mutate(parent2)
                        del parent2.fitness.values
                    offspring.extend([parent1, parent2])

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # `nombre_enfants_gardes` (par couple) ne correspond pas forcément
            # au nombre de couples : sans ce recadrage, la taille de la
            # population dérive (rétrécit ou explose) génération après
            # génération au lieu de rester égale à taille_population.
            if len(offspring) > len(pop):
                offspring.sort(key=lambda ind: ind.fitness.values[0], reverse=True)
                offspring = offspring[:len(pop)]
            elif len(offspring) < len(pop):
                while len(offspring) < len(pop):
                    offspring.append(toolbox.clone(random.choice(offspring)))

            pop[:] = offspring

            if activer_hall_of_fame:
                hof.update(pop)
                pire_enfant_index = pop.index(min(pop, key=lambda ind: ind.fitness.values[0]))
                pop[pire_enfant_index] = toolbox.clone(hof[0])
                meilleur_actuel = hof[0]
            else:
                meilleur_actuel = tools.selBest(pop, k=1)[0]

            self.historique_ga.append({
                "generation": g,
                "meilleur_score": meilleur_actuel.fitness.values[0],
                "nb_capteurs": len(meilleur_actuel)
            })

            pbar.set_postfix({
                "Meilleur R²": f"{meilleur_actuel.fitness.values[0] * 100:.2f}%",
                "Capteurs": len(meilleur_actuel)
            })
            pbar.update(1)
            self._progres("Sélection génétique des points", g + 1, nombre_generations)

        pbar.close()

        # ---- 5. Résultats finaux (avec suppression des doublons) ----
        meilleur_individu_brut = hof[0] if activer_hall_of_fame else tools.selBest(pop, k=1)[0]
        meilleur_score = meilleur_individu_brut.fitness.values[0]

        meilleur_individu = []
        vus = set()
        for gene in meilleur_individu_brut:
            if gene not in vus:
                vus.add(gene)
                meilleur_individu.append(gene)

        self.meilleur_individu = meilleur_individu

        self._log("==========================================")
        self._log("🏆 OPTIMISATION GLOBALE TERMINÉE !")
        self._log(f"Meilleur score R² trouvé : {meilleur_score * 100:.2f} %")
        self._log(f"Nombre final de capteurs retenus (après nettoyage) : {len(meilleur_individu)}")
        self._log("==========================================")

        self._log("🌍 COORDONNÉES DES CAPTEURS SURVIVANTS :")
        for zone_id, pt_idx in meilleur_individu:
            coord = listes_de_points[zone_id][pt_idx]
            self._log(f" - {noms_zones[zone_id]} : {coord}")

        self.coords_finales = [listes_de_points[zone_id][pt_idx] for zone_id, pt_idx in meilleur_individu]

        # ---- 6. Carte des résultats ----
        self.carte_resultats_ga()

        # ---- 7. Téléchargement complet sur les points retenus ----
        if telecharger_donnees_completes:
            if params_hourly is None or params_daily is None:
                self._log("params_hourly/params_daily non fournis : téléchargement complet ignoré.", "warning")
            else:
                self.construire_donnees_completes(
                    params_hourly=params_hourly,
                    params_daily=params_daily,
                    start_fetch=start_train if start_train is not None else start_fetch_final,
                    end_fetch=end_train if end_train is not None else end_fetch_final,
                    split_date=split_date_final,
                    past_day=past_day_final,
                    predict_day=predict_day_final,
                    mode_split=mode_split,
                    part_test=part_test,
                    agregations_horaires=agregations_horaires,
                )

        return meilleur_individu, meilleur_score

    def carte_resultats_ga(self, sauvegarder_sous="carte_optimisation_finale.html", afficher=True):
        """Carte des zones explorées (grille en fond) + capteurs optimaux retenus par le GA."""
        if self.meilleur_individu is None:
            self._log("Aucun résultat de sélection génétique à cartographier.", "warning")
            return None

        self._log("🗺️ Création de la carte des capteurs optimaux...")

        coords_gagnantes = []
        zones_gagnantes = []
        vus = set()

        for zone_id, pt_idx in self.meilleur_individu:
            if (zone_id, pt_idx) not in vus:
                vus.add((zone_id, pt_idx))
                coords_gagnantes.append(self.listes_de_points[zone_id][pt_idx])
                zones_gagnantes.append(zone_id)

        if coords_gagnantes:
            lat_moyenne = sum([c[0] for c in coords_gagnantes]) / len(coords_gagnantes)
            lon_moyenne = sum([c[1] for c in coords_gagnantes]) / len(coords_gagnantes)
        else:
            lat_moyenne, lon_moyenne = self.lat_station, self.lon_station

        carte_opti = folium.Map(location=[lat_moyenne, lon_moyenne], zoom_start=11, tiles="CartoDB positron")

        couleurs_zones = ['blue', 'green', 'purple', 'orange', 'cadetblue', 'darkred', 'black']

        for i, zone_points in enumerate(self.listes_de_points):
            coul = couleurs_zones[i % len(couleurs_zones)]
            nom = self.noms_zones[i]
            for pt in zone_points:
                folium.CircleMarker(
                    location=[pt[0], pt[1]],
                    radius=3,
                    color=coul,
                    fill=True,
                    fill_opacity=0.2,
                    weight=1,
                    tooltip=f"{nom} (Explorable)"
                ).add_to(carte_opti)

        for i, coord in enumerate(coords_gagnantes):
            zone_id = zones_gagnantes[i]
            folium.Marker(
                location=[coord[0], coord[1]],
                popup=f"<b>{self.noms_zones[zone_id]}</b><br>Lat: {coord[0]}<br>Lon: {coord[1]}",
                tooltip=f"🏆 Capteur Optimal - {self.noms_zones[zone_id]}",
                icon=folium.Icon(color="red", icon="star")
            ).add_to(carte_opti)

        carte_opti.save(sauvegarder_sous)

        self._log(f"Carte générée avec {len(coords_gagnantes)} capteurs uniques !", "succes")
        if afficher:
            afficher_carte_html(sauvegarder_sous)

        return carte_opti

    def definir_points_optimaux(self, coords_finales):
        """
        Définit directement les capteurs à utiliser pour
        construire_donnees_completes(), sans passer par
        lancer_selection_genetique() -- utile pour réutiliser des points déjà
        trouvés par un run précédent du GA, ou choisis à la main.
        """
        self.coords_finales = list(coords_finales)
        self._log(f"{len(self.coords_finales)} points optimaux définis manuellement.", "succes")
        for c in self.coords_finales:
            self._log(f"  {c}")
        return self.coords_finales

    # ==========================================================================
    # 5bis. SÉLECTION EN DEUX TEMPS (remplace le GA)
    #   Étape 1 (GRATUITE) : glouton sous-modulaire sur altitude + couverture
    #                        spatiale + équilibre entre zones hydro.
    #   Étape 2 (API légère) : glouton sur corrélation pluie->débit + stock de
    #                          neige, sur les points présélectionnés seulement.
    # ==========================================================================
    @staticmethod
    def _coords_en_km(coords):
        """Projette (lat, lon) en km locaux (plan tangent) pour des distances
        euclidiennes correctes. Origine = barycentre."""
        lat = np.array([c[0] for c in coords], dtype=float)
        lon = np.array([c[1] for c in coords], dtype=float)
        lat0 = lat.mean()
        x = (lon - lon.mean()) * 111.32 * math.cos(math.radians(lat0))
        y = (lat - lat.mean()) * 111.32
        return np.column_stack([x, y])

    @staticmethod
    def _rangs_normalises(valeurs):
        """Remplace chaque valeur par son rang ramené dans [0, 1]. 
        Insensible aux unités et aux valeurs extrêmes :
        c'est ce qui permet de comparer des grandeurs d'échelles différentes
        (ex. hauteur de neige en m vs somme de corrélations sans unité)."""
        v = np.asarray(valeurs, dtype=float)
        n = len(v)
        if n <= 1:
            return np.zeros(n)
        ordre = np.argsort(np.argsort(v))          # rang de chaque élément (0..n-1)
        return ordre / (n - 1)

    def preselection_altitude_couverture(self, n_points=30, poids_altitude=1.0,
                                          poids_zones=1.0, poids_altitude_dans_distance=1.0,
                                          sigma_km=None, altitudes=None):
        """
        ÉTAPE 1 -- sélection SANS API météo. Choisit n_points parmi la grille de
        candidats (self.listes_de_points) par un algorithme GLOUTON qui maximise
        une fonction SOUS-MODULAIRE (rendements décroissants) :

            F(S) = couverture(S) + poids_altitude * altitude(S)
                                 + poids_zones    * equilibre_zones(S)

        - couverture(S) = somme sur tous les candidats j de   max_{i in S} exp(-d(i,j)^2 / sigma^2)
          où d est la distance dans l'espace (x_km, y_km, altitude_normalisée)
          -> pousse les points retenus à bien REPRÉSENTER tout le bassin, y
          compris le long de l'axe altitude (poids_altitude_dans_distance).
        - altitude(S) = somme des altitudes normalisées des points retenus
          -> favorise les points HAUTS (pour mieux comprendre les stocks de neige).
        - equilibre_zones(S) = somme sur les zones de sqrt(nb de points retenus
          dans la zone) -> fonction CONCAVE : le énième point d'une même zone
          rapporte de moins en moins, donc les points se répartissent entre zones.

        Toutes ces briques sont sous-modulaires croissantes : le glouton est
        alors garanti d'atteindre >= (1 - 1/e) ~ 63 % de l'optimum (théorème de
        Nemhauser-Wolsey-Fisher), en n_points étapes, quasi instantanément.

        altitudes : liste d'altitudes alignée sur les candidats (si None, elles
        sont téléchargées via l'endpoint élévation d'Open-Meteo -- gratuit).
        Remplit self.points_preselectionnes et renvoie la liste des coordonnées.
        """
        if self.listes_de_points is None:
            raise Exception("Appelle determiner_bassins_versants() avant preselection_altitude_couverture().")

        # --- Candidats : coordonnées + indice de zone ---
        candidats, zone_de = [], []
        for idx_zone, zone in enumerate(self.listes_de_points):
            for pt in zone:
                candidats.append((round(float(pt[0]), 4), round(float(pt[1]), 4)))
                zone_de.append(idx_zone)
        n = len(candidats)
        if n == 0:
            raise Exception("Aucun candidat dans la grille.")
        if n_points >= n:
            self._log(f"Moins de candidats ({n}) que de points demandés ({n_points}) : tous gardés.", "warning")
            self.points_preselectionnes = candidats
            return candidats

        zone_de = np.array(zone_de)
        n_zones = len(self.listes_de_points)

        # --- Altitudes (téléchargées si absentes) ---
        if altitudes is None:
            self._log(f"⛰️ Récupération de l'altitude de {n} points candidats (endpoint élévation, gratuit)...")
            altitudes = self._data.fetch_elevations(candidats)
        altitudes = np.array([a if a is not None else np.nan for a in altitudes], dtype=float)
        if np.isnan(altitudes).all():
            self._log("Altitudes indisponibles : présélection sur la seule couverture spatiale.", "warning")
            altitudes = np.zeros(n)
        else:
            altitudes = np.where(np.isnan(altitudes), np.nanmean(altitudes), altitudes)

        # Normalisation min-max de l'altitude dans [0, 1].
        h_min, h_max = altitudes.min(), altitudes.max()
        alt_norm = (altitudes - h_min) / (h_max - h_min) if h_max > h_min else np.zeros(n)

        # --- Matrice de similarité (couverture) dans l'espace (x, y, altitude) ---
        xy = self._coords_en_km(candidats)                       # (n, 2) en km
        etendue_km = max(1.0, np.sqrt(((xy - xy.mean(0)) ** 2).sum(1)).mean())
        # Altitude ramenée en "km équivalents" pour peser dans la distance.
        alt_km = alt_norm * etendue_km * poids_altitude_dans_distance
        pts3d = np.column_stack([xy, alt_km])                    # (n, 3)
        if sigma_km is None:
            sigma_km = 0.5 * etendue_km                          # rayon de couverture par défaut
        d2 = ((pts3d[:, None, :] - pts3d[None, :, :]) ** 2).sum(-1)   # (n, n)
        similarite = np.exp(-d2 / (sigma_km ** 2))               # 1 sur la diagonale

        # --- Glouton sous-modulaire ---
        self._log(f"🧮 Présélection gloutonne de {n_points} points parmi {n} (altitude + couverture + zones)...")
        meilleure_sim = np.zeros(n)          # pour chaque candidat j : max_{i in S} sim(i, j)
        compte_zone = np.zeros(n_zones)
        choisis = []
        disponibles = set(range(n))

        for _ in range(n_points):
            self._verifier_arret()
            meilleur_gain, meilleur_i = -np.inf, None
            for i in disponibles:
                gain_couverture = np.maximum(similarite[i] - meilleure_sim, 0).sum()
                gain_altitude = poids_altitude * alt_norm[i]
                nz = compte_zone[zone_de[i]]
                gain_zone = poids_zones * (math.sqrt(nz + 1) - math.sqrt(nz))
                gain = gain_couverture + gain_altitude + gain_zone
                if gain > meilleur_gain:
                    meilleur_gain, meilleur_i = gain, i
            choisis.append(meilleur_i)
            disponibles.discard(meilleur_i)
            meilleure_sim = np.maximum(meilleure_sim, similarite[meilleur_i])
            compte_zone[zone_de[meilleur_i]] += 1

        self.points_preselectionnes = [candidats[i] for i in choisis]
        self._altitudes_preselection = [float(altitudes[i]) for i in choisis]
        alt_ret = np.array(self._altitudes_preselection)
        self._log(f"✅ {len(choisis)} points présélectionnés | altitude {alt_ret.min():.0f}-{alt_ret.max():.0f} m "
                  f"| répartition zones : {compte_zone.astype(int).tolist()}", "succes")
        return self.points_preselectionnes

    def selection_correlation_neige(self, n_final=5, fenetre_annees=2,
                                    lag_max=20, poids_pluie=1.0, poids_neige=1.0,
                                    sigma_redondance_km=None, points=None):
        """
        ÉTAPE 2 -- affine les points présélectionnés (self.points_preselectionnes)
        avec une petite dose de météo. Ne télécharge QUE ces points, sur une
        fenêtre courte (fenetre_annees), donc quota léger.

        Pour chaque point P :
        - score_pluie(P) = somme sur les décalages k=0..lag_max de max(0, r_k),
          où r_k = corrélation (Pearson) entre la pluie de P décalée de k jours
          et le débit. On SOMME (pas le max) pour capter une influence étalée
          dans le temps (rétention du sol), et on ne garde que les corrélations
          POSITIVES pour ne pas additionner du bruit.
        - score_neige(P) = hauteur de neige moyenne (stock ; la fonte viendra
          plus tard, gérée par le modèle -- ici on repère juste les réservoirs).

        Les deux scores sont normalisés PAR RANG (dans [0,1], insensible aux
        unités), combinés : pertinence = poids_pluie*rang_pluie + poids_neige*rang_neige.
        Puis sélection GLOUTONNE avec pénalité de proximité (redondance) :
            score_effectif(P) = pertinence(P) * (1 - max_{Q déjà pris} exp(-d(P,Q)^2/sigma^2))
        -> les points retenus sont pertinents ET étalés.

        Remplit self.coords_finales. Renvoie un dict de diagnostic (scores par
        point, meilleur décalage = temps de réponse du bassin).
        """
        points = points or self.points_preselectionnes
        if not points:
            raise Exception("Aucun point présélectionné -- lance preselection_altitude_couverture() avant.")
        if n_final >= len(points):
            self._log(f"n_final ({n_final}) >= présélection ({len(points)}) : tous gardés.", "warning")
            self.coords_finales = list(points)
            return {"points": self.coords_finales}

        # --- Fenêtre de téléchargement (courte) ---
        premiere, derniere = self._data.periode_disponible_river(self.code_station)
        debut = max(premiere, derniere - pd.Timedelta(days=int(365.25 * fenetre_annees)))
        start = debut.strftime('%Y-%m-%d')
        fin = derniere.strftime('%Y-%m-%d')
        self._log(f"📥 Étape 2 : téléchargement léger de {len(points)} points sur {start} -> {fin} "
                  f"(pluie quotidienne + neige)...")

        df_eau = self._data.fetch_river_data(self.code_station, start, fin)
        serie_debit = df_eau.dropna(subset=['debit_L_s']).set_index('date')['debit_L_s'].sort_index()

        resultats = self._data.fetch_weather_data_multi(
            points, start, fin, daily_vars=["rain_sum"], hourly_vars=["snow_depth"]
        )

        score_pluie, score_neige, meilleurs_lags = [], [], []
        for (df_h, df_d) in resultats:
            self._verifier_arret()
            # Pluie quotidienne
            if df_d is not None and 'rain_sum' in df_d and not df_d.empty:
                pluie = df_d.set_index('date')['rain_sum'].sort_index()
            else:
                pluie = pd.Series(dtype=float)
            # Neige : moyenne journalière de la hauteur (stock)
            if df_h is not None and 'snow_depth' in df_h and not df_h.empty:
                neige_j = self._data.hourly_to_daily_mean(df_h).set_index('date')['snow_depth']
                score_neige.append(float(neige_j.mean()))
            else:
                score_neige.append(0.0)

            # Corrélations croisées pluie(t) vs débit(t+k)
            somme_corr, meilleur_r, meilleur_k = 0.0, -1.0, 0
            if not pluie.empty:
                for k in range(lag_max + 1):
                    debit_futur = serie_debit.shift(-k)
                    r = pluie.corr(debit_futur)     # Pearson, NaN gérés par paires
                    if pd.notna(r):
                        somme_corr += max(0.0, r)
                        if r > meilleur_r:
                            meilleur_r, meilleur_k = r, k
            score_pluie.append(somme_corr)
            meilleurs_lags.append(meilleur_k)

        score_pluie = np.array(score_pluie)
        score_neige = np.array(score_neige)
        rang_pluie = self._rangs_normalises(score_pluie)
        rang_neige = self._rangs_normalises(score_neige)
        pertinence = poids_pluie * rang_pluie + poids_neige * rang_neige

        # --- Glouton pertinence + redondance ---
        xy = self._coords_en_km(points)
        if sigma_redondance_km is None:
            etendue = max(1.0, np.sqrt(((xy - xy.mean(0)) ** 2).sum(1)).mean())
            sigma_redondance_km = 0.5 * etendue
        d2 = ((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)
        proximite = np.exp(-d2 / (sigma_redondance_km ** 2))

        choisis, redondance = [], np.zeros(len(points))
        for _ in range(n_final):
            self._verifier_arret()
            eff = pertinence * (1 - redondance)
            eff[choisis] = -np.inf
            i = int(np.argmax(eff))
            choisis.append(i)
            redondance = np.maximum(redondance, proximite[i])

        self.coords_finales = [points[i] for i in choisis]

        temps_reponse = int(np.median([meilleurs_lags[i] for i in choisis]))
        self._log(f"✅ Étape 2 : {len(choisis)} points retenus. Temps de réponse médian du bassin : "
                  f"~{temps_reponse} jour(s) (décalage pluie->débit).", "succes")
        self.diagnostic_selection = {
            "temps_reponse_jours": temps_reponse,
            "points": [{"lat": points[i][0], "lon": points[i][1],
                        "score_pluie": float(score_pluie[i]), "score_neige": float(score_neige[i]),
                        "meilleur_lag": int(meilleurs_lags[i]), "retenu": i in choisis}
                       for i in range(len(points))],
        }
        return self.diagnostic_selection

    def executer_selection_deux_temps(self, n_preselection=30, n_final=5,
                                      fenetre_annees=2, densite_grille=25, max_par_zone=60,
                                      **kwargs_selection):
        """
        Enchaîne la sélection en DEUX TEMPS (remplaçant du GA) :
          0. determiner_bassins_versants() avec une grille DENSE (gratuit) ;
          1. preselection_altitude_couverture() -> n_preselection points ;
          2. selection_correlation_neige() -> n_final points (self.coords_finales).
        Renvoie le diagnostic de l'étape 2.
        """
        self.determiner_bassins_versants(
            n_points_grille=densite_grille, max_points_par_zone=max_par_zone, afficher=False)
        self.preselection_altitude_couverture(
            n_points=n_preselection,
            **{k: v for k, v in kwargs_selection.items()
               if k in ("poids_altitude", "poids_zones", "poids_altitude_dans_distance", "sigma_km")})
        diag = self.selection_correlation_neige(
            n_final=n_final, fenetre_annees=fenetre_annees,
            **{k: v for k, v in kwargs_selection.items()
               if k in ("lag_max", "poids_pluie", "poids_neige", "sigma_redondance_km")})
        return diag

    def construire_donnees_completes(self, params_hourly=None, params_daily=None,
                                      start_fetch=None, end_fetch=None,
                                      split_date=None, past_day=20, predict_day=30,
                                      mode_split="annees_aleatoires", part_test=0.2, graine_split=42,
                                      agregations_horaires=None,
                                      df_eau_precharge=None, df_meteo_precharge=None):
        """
        Télécharge les données complètes (toute la période, pas le cache léger)
        uniquement sur self.coords_finales (défini par
        lancer_selection_genetique() ou definir_points_optimaux()), et
        construit X_train2/Y_train2/X_test2/Y_test2. Sauvegarde aussi les CSV
        (avant PCA), y compris un CSV date+debit_L_s par split qui permet à
        tracer_prediction() de fonctionner même après un rechargement via
        charger_donnees_completes().

        start_fetch/end_fetch=None (défaut) : période élargie AU MAXIMUM de ce
        qui existe réellement sur Hub'Eau pour cette station (via
        periode_disponible_river), plutôt que des dates en dur.

        Split train/test :
        - mode_split="annees_aleatoires" (défaut) : part_test des ANNÉES,
          tirées au sort (graine_split fixe la reproductibilité), vont au
          test, le reste au train. Évite le biais d'un split chronologique
          (apprendre sur 1960-2003 pour tester sur 2004-2026 pénalise le
          modèle si le climat a changé). Anti-fuite : toute ligne dont la
          fenêtre [J-past_day, J+predict_day] chevauche une année de l'autre
          bloc est écartée (sinon les mêmes débits observés apparaîtraient à
          la fois dans les features/cibles du train et du test).
        - mode_split="chronologique" : coupe à 80% de la période.
        - split_date="AAAA-MM-JJ" : coupe chronologique à cette date (prioritaire).

        df_eau_precharge / df_meteo_precharge : si fournis (DataFrame avec une
        colonne 'date' + les colonnes habituelles -- respectivement le format
        de fetch_river_data() et celui de l'assemblage météo par point), le
        téléchargement réseau correspondant est sauté. Utile pour rejouer la
        construction des features à partir de données déjà collectées (par
        exemple si le quota de l'API est épuisé).
        """
        if self.coords_finales is None:
            raise Exception("Aucun point optimal défini -- appelle lancer_selection_genetique() ou definir_points_optimaux() avant construire_donnees_completes().")

        # Agrégation par variable horaire (voir hourly_to_daily_spec) : par
        # défaut la neige à minuit (stock) et tout le reste en moyenne
        # journalière. On la mémorise pour que predire_futur reconstruise
        # EXACTEMENT les mêmes colonnes de features.
        if agregations_horaires is None:
            agregations_horaires = {v: ("minuit" if "snow" in v else "moyenne")
                                    for v in (params_hourly or [])}
        self.agregations_horaires = dict(agregations_horaires)

        if df_eau_precharge is None and (start_fetch is None or end_fetch is None):
            # Recouvrement eau ∩ météo : on ne descend PAS sous 1940 (début ERA5),
            # sinon la requête météo échoue et les années sans météo sont inutiles.
            premiere, debut_exploitable, derniere = self._data.periode_exploitable_river(self.code_station)
            if start_fetch is None:
                start_fetch = debut_exploitable.strftime('%Y-%m-%d')
            if end_fetch is None:
                end_fetch = derniere.strftime('%Y-%m-%d')
            self._log(f"Période d'apprentissage (eau ∩ météo) : {start_fetch} -> {end_fetch}")

        coords_finales = self.coords_finales
        self._log(f"📍 {len(coords_finales)} points retenus après suppression des doublons :")
        for c in coords_finales:
            self._log(f"  {c}")

        if df_eau_precharge is not None:
            self._log("💧 Données rivière préchargées fournies -- téléchargement Hub'Eau sauté.", "succes")
            df_eau2 = df_eau_precharge
        else:
            df_eau2 = self._data.fetch_river_data(self.code_station, start_fetch, end_fetch)

        if df_meteo_precharge is not None:
            self._log("🌤 Données météo préchargées fournies -- téléchargement Open-Meteo sauté.", "succes")
            df_meteo2 = df_meteo_precharge
        else:
            if params_hourly is None or params_daily is None:
                raise Exception("params_hourly/params_daily requis pour télécharger la météo (ou fournis df_meteo_precharge).")

            # Les variables en moyenne journalière avec un équivalent natif
            # (temperature_2m -> temperature_2m_mean) sont demandées en JOURNALIER
            # (1 valeur/jour) plutôt qu'en horaire : réponse bien plus légère sur
            # 60+ ans, plus de coupures de connexion.
            hourly_fetch, daily_fetch, renommage = self._data.repartir_variables_meteo(
                params_hourly, params_daily, self.agregations_horaires)

            self._progres("Téléchargement météo complet (toutes les années)", None, None)
            try:
                resultats2 = self._data.fetch_weather_data_multi(
                    coords_finales, start_fetch, end_fetch,
                    daily_vars=daily_fetch, hourly_vars=hourly_fetch
                )
            except ArretDemande:
                raise   # un arrêt demandé n'est pas un échec réseau
            except Exception as e:
                # Erreur transitoire malgré les retries internes : une pause
                # puis une dernière tentative avant d'abandonner.
                self._log(f"Téléchargement complet interrompu ({e}) -- nouvelle tentative dans 30s...", "warning")
                _dormir_interruptible(30, self._doit_arreter)
                resultats2 = self._data.fetch_weather_data_multi(
                    coords_finales, start_fetch, end_fetch,
                    daily_vars=daily_fetch, hourly_vars=hourly_fetch
                )

            list_df_meteo2 = []
            for df_hourly_pt, df_daily_pt in resultats2:
                df_daily_pt = df_daily_pt.rename(columns=renommage)
                df_hourly_wide_pt = self._data.hourly_to_daily_spec(df_hourly_pt, self.agregations_horaires)
                if df_hourly_wide_pt.empty or df_hourly_wide_pt.columns.tolist() == ['date']:
                    df_pt = df_daily_pt
                else:
                    df_pt = pd.merge(df_daily_pt, df_hourly_wide_pt, on='date', how='outer')
                list_df_meteo2.append(df_pt)

            df_meteo_concat2 = pd.concat(list_df_meteo2)
            df_meteo2 = df_meteo_concat2.groupby('date').mean().reset_index()

        df2 = pd.merge(df_eau2, df_meteo2, on='date', how='inner')
        df2 = df2.sort_values('date').reset_index(drop=True)

        colonnes_meteo2 = [col for col in df_meteo2.columns if col != 'date']

        features_X2 = []
        targets_Y2 = []

        for i in range(1, past_day + 1):
            nom_col = f"debit_J-{i}"
            df2[nom_col] = df2['debit_L_s'].shift(i)
            features_X2.append(nom_col)

        for i in range(1, past_day + 1):
            for var in colonnes_meteo2:
                nom_col = f"{var}_J-{i}"
                df2[nom_col] = df2[var].shift(i)
                features_X2.append(nom_col)

        for i in range(0, predict_day + 1):
            for var in colonnes_meteo2:
                nom_col = f"{var}_J+{i}"
                df2[nom_col] = df2[var].shift(-i)
                features_X2.append(nom_col)

        df2['sin_date'] = np.sin(2 * np.pi * df2['date'].dt.dayofyear / 365.25)
        df2['cos_date'] = np.cos(2 * np.pi * df2['date'].dt.dayofyear / 365.25)
        features_X2.extend(['sin_date', 'cos_date'])

        for i in range(0, predict_day + 1):
            nom_col = f"cible_debit_J+{i}"
            df2[nom_col] = df2['debit_L_s'].shift(-i)
            targets_Y2.append(nom_col)

        df2 = df2.dropna()
        if len(df2) == 0:
            raise Exception("Aucune ligne exploitable après construction des features (période trop courte ?).")

        duree_annees = (df2['date'].max() - df2['date'].min()).days / 365.25
        if duree_annees < 5:
            self._log(f"ATTENTION : seulement {duree_annees:.1f} an(s) de données exploitables sur cette station. "
                      f"C'est peu pour apprendre (peu de crues/saisons vues) : scores et intervalles de confiance à prendre avec prudence.", "warning")

        annees_dispo = sorted(df2['date'].dt.year.unique().tolist())
        if split_date is None and mode_split == "annees_aleatoires" and len(annees_dispo) < 4:
            self._log(f"Seulement {len(annees_dispo)} année(s) de données : le tirage au sort d'années n'a pas de sens, split chronologique utilisé.", "warning")
            mode_split = "chronologique"

        if split_date is None and mode_split == "annees_aleatoires":
            rng = random.Random(graine_split)
            nb_test = max(1, round(len(annees_dispo) * part_test))
            annees_test = set(rng.sample(annees_dispo, nb_test))
            bloc_test_par_annee = {a: (a in annees_test) for a in annees_dispo}

            # Anti-fuite : une ligne n'est gardée que si TOUTE sa fenêtre
            # [J-past_day, J+predict_day] tombe dans des années du même bloc
            # qu'elle. Au bord d'une frontière train/test, les features des
            # unes recouvriraient les cibles des autres.
            delta_passe = pd.Timedelta(days=past_day)
            delta_futur = pd.Timedelta(days=predict_day)

            def fenetre_dans_le_meme_bloc(d):
                bloc = bloc_test_par_annee[d.year]
                a_min = (d - delta_passe).year
                a_max = (d + delta_futur).year
                return all(bloc_test_par_annee.get(a, bloc) == bloc for a in range(a_min, a_max + 1))

            garde = df2['date'].map(fenetre_dans_le_meme_bloc)
            est_test = df2['date'].dt.year.isin(annees_test)

            train2 = df2[garde & ~est_test]
            test2 = df2[garde & est_test]
            nb_ecartees = int((~garde).sum())

            split_date = f"annees_aleatoires (graine {graine_split})"
            self._log(f"Split par années tirées au sort : {len(annees_test)}/{len(annees_dispo)} années en test "
                      f"({', '.join(str(a) for a in sorted(annees_test))}).")
            self._log(f"{nb_ecartees} ligne(s) écartée(s) aux frontières train/test pour éviter toute fuite de données "
                      f"(fenêtre J-{past_day} -> J+{predict_day}).")
        else:
            if split_date is None:
                split_date = df2['date'].iloc[int(len(df2) * 0.8)].strftime('%Y-%m-%d')
                self._log(f"Split automatique train/test à 80% de la période : {split_date}")
            train2 = df2[df2['date'] <= split_date]
            test2 = df2[df2['date'] > split_date]

        if len(train2) == 0 or len(test2) == 0:
            raise Exception("Split train/test vide -- période trop courte ou split_date hors période.")

        self.X_train2, self.Y_train2 = train2[features_X2], train2[targets_Y2]
        self.X_test2, self.Y_test2 = test2[features_X2], test2[targets_Y2]

        self.df_final = df2
        self.train2 = train2
        self.test2 = test2
        self.features_X2 = features_X2
        self.targets_Y2 = targets_Y2
        self.past_day2 = past_day
        self.predict_day2 = predict_day
        self.split_date2 = split_date

        self._log(f"X_train2 : {self.X_train2.shape}, X_test2 : {self.X_test2.shape}", "succes")

        self.X_train2.to_csv(f"X_train_{self.code_station}.csv", index=False)
        self.Y_train2.to_csv(f"Y_train_{self.code_station}.csv", index=False)
        self.X_test2.to_csv(f"X_test_{self.code_station}.csv", index=False)
        self.Y_test2.to_csv(f"Y_test_{self.code_station}.csv", index=False)
        train2[['date', 'debit_L_s']].to_csv(f"dates_train_{self.code_station}.csv", index=False)
        test2[['date', 'debit_L_s']].to_csv(f"dates_test_{self.code_station}.csv", index=False)
        self._log("CSV X/Y (avant PCA) + dates/debit_L_s sauvegardés.", "succes")

        return self.X_train2, self.Y_train2, self.X_test2, self.Y_test2

    def charger_donnees_completes(self, X_train_csv, Y_train_csv, X_test_csv, Y_test_csv,
                                   dates_train_csv=None, dates_test_csv=None,
                                   past_day=20, predict_day=15, split_date=None,
                                   coords_finales=None):
        """
        Charge un jeu d'entraînement final (X_train2/Y_train2/X_test2/Y_test2)
        déjà construit et sauvegardé en CSV (par un run précédent de
        construire_donnees_completes(), ou récupéré depuis niveau_bonne.ipynb),
        sans repasser par le réseau ni par la sélection génétique. Permet
        d'enchaîner directement sur appliquer_pca() / entrainer_*() /
        auditer_incertitude().

        dates_train_csv / dates_test_csv (colonnes 'date' + 'debit_L_s', dans
        le même ordre de lignes que X_train_csv/X_test_csv) : si fournis,
        tracer_prediction() redevient utilisable. Sans eux, seuls
        appliquer_pca() / entrainer_*() / auditer_incertitude() fonctionnent.
        """
        self.X_train2 = pd.read_csv(X_train_csv)
        self.Y_train2 = pd.read_csv(Y_train_csv)
        self.X_test2 = pd.read_csv(X_test_csv)
        self.Y_test2 = pd.read_csv(Y_test_csv)

        self.features_X2 = list(self.X_train2.columns)
        self.targets_Y2 = list(self.Y_train2.columns)
        self.past_day2 = past_day
        self.predict_day2 = predict_day
        self.split_date2 = split_date

        if coords_finales is not None:
            self.coords_finales = list(coords_finales)

        self.df_final = None
        self.train2 = None
        self.test2 = None

        if dates_train_csv is not None and dates_test_csv is not None:
            dates_train = pd.read_csv(dates_train_csv, parse_dates=['date'])
            dates_test = pd.read_csv(dates_test_csv, parse_dates=['date'])

            if len(dates_train) != len(self.X_train2) or len(dates_test) != len(self.X_test2):
                self._log("Les fichiers de dates n'ont pas le même nombre de lignes que X_train2/X_test2 -- tracer_prediction() risque d'être incohérent.", "warning")

            train2 = pd.concat([dates_train.reset_index(drop=True),
                                 self.X_train2.reset_index(drop=True),
                                 self.Y_train2.reset_index(drop=True)], axis=1)
            test2 = pd.concat([dates_test.reset_index(drop=True),
                                self.X_test2.reset_index(drop=True),
                                self.Y_test2.reset_index(drop=True)], axis=1)

            self.train2 = train2
            self.test2 = test2
            self.df_final = pd.concat([train2, test2], ignore_index=True).sort_values('date').reset_index(drop=True)
            self._log("Dates rechargées : tracer_prediction() est utilisable.", "succes")
        else:
            self._log("Pas de fichiers de dates fournis : tracer_prediction() ne sera pas utilisable (PCA/entraînement/incertitude restent OK).", "warning")

        self._log(f"Données finales chargées depuis CSV : X_train2 {self.X_train2.shape}, X_test2 {self.X_test2.shape}", "succes")

        return self.X_train2, self.Y_train2, self.X_test2, self.Y_test2

    # ==========================================================================
    # 6. PCA
    # ==========================================================================
    def appliquer_pca(self, seuil_energie=99, afficher_graphique=True):
        """
        Standardise X_train2/X_test2 (StandardScaler -- nécessaire avant toute
        PCA, pas fourni littéralement mais indispensable) puis réduit leur
        dimension. Le nombre de composantes retenu correspond au seuil
        d'énergie cumulée choisi (99% par défaut). N'est utilisée aujourd'hui
        que par entrainer_regression_lineaire_pca() -- Keras et Gradient
        Boosting tournent en espace brut. 95% ne gardait que 19 composantes
        sur ~1786 et plombait le R² (mesuré : 51% à 95%, 70% à 99%, 73% à
        99.9% -- au-delà de 99%, les gains marginaux ne compensent plus le
        nombre de composantes en plus). Sauvegarde X_train_pca/X_test_pca en
        CSV (Y ne change pas avec la PCA, donc pas resauvegardé).
        """
        if self.X_train2 is None:
            raise Exception("Appelle lancer_selection_genetique() (avec téléchargement complet) avant appliquer_pca().")

        self.scaler_pca = StandardScaler()
        X_train_scaled = self.scaler_pca.fit_transform(self.X_train2)
        X_test_scaled = self.scaler_pca.transform(self.X_test2)

        pca_full = PCA(random_state=42)
        pca_full.fit(X_train_scaled)

        valeurs_singulieres = pca_full.singular_values_
        variance_expliquee = pca_full.explained_variance_ratio_
        energie_cumulee = np.cumsum(variance_expliquee) * 100

        idx_seuil = int(np.argmax(energie_cumulee >= seuil_energie) + 1)

        if afficher_graphique:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

            ax1.plot(range(1, len(valeurs_singulieres) + 1), valeurs_singulieres, marker='o', color='#1f77b4', linewidth=2)
            ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.6, label="Seuil de Kaiser ($\\sigma = 1$)")
            ax1.set_title("Décroissance des Valeurs Singulières", fontweight='bold', fontsize=12)
            ax1.set_xlabel("Nombre de Composantes", fontweight='bold')
            ax1.set_ylabel("Valeur Singulière ($\\sigma$)", fontweight='bold')
            ax1.grid(True, linestyle=':', alpha=0.6)
            ax1.legend()

            ax2.plot(range(1, len(energie_cumulee) + 1), energie_cumulee, marker='s', color='#2ca02c', linewidth=2)
            ax2.axhline(y=seuil_energie, color='orange', linestyle='--', alpha=0.7, label=f"Seuil d'Énergie à {seuil_energie}%")
            ax2.axvline(x=idx_seuil, color='orange', linestyle=':', alpha=0.7)

            ax2.set_title("Énergie Cumulée Restituée", fontweight='bold', fontsize=12)
            ax2.set_xlabel("Nombre de Composantes", fontweight='bold')
            ax2.set_ylabel("Variance Expliquée Cumulée (%)", fontweight='bold')
            ax2.set_ylim(bottom=0, top=105)
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend()

            plt.suptitle("Analyse Spectrale de la PCA (Réduction de Modèle)", fontweight='bold', fontsize=14, y=1.02)
            plt.tight_layout()
            plt.show()

        self._log("📊 ANALYSE MACROSCOPIQUE :")
        self._log(f" - Nombre total de dimensions initiales : {len(valeurs_singulieres)}")
        self._log(f" - Nombre de composantes pour atteindre {seuil_energie}% d'énergie : {idx_seuil}")
        self._log(f" - Énergie totale contenue dans les 3 premières composantes : {energie_cumulee[2]:.1f} %")

        self.pca_model = PCA(n_components=idx_seuil, random_state=42)
        self.X_train_pca = self.pca_model.fit_transform(X_train_scaled)
        self.X_test_pca = self.pca_model.transform(X_test_scaled)
        self.n_composantes_pca = idx_seuil

        colonnes_pca = [f"PC{i+1}" for i in range(idx_seuil)]
        pd.DataFrame(self.X_train_pca, columns=colonnes_pca).to_csv(f"X_train_pca_{self.code_station}.csv", index=False)
        pd.DataFrame(self.X_test_pca, columns=colonnes_pca).to_csv(f"X_test_pca_{self.code_station}.csv", index=False)
        self._log("CSV X (après PCA) sauvegardés.", "succes")

        return self.X_train_pca, self.X_test_pca

    def _preparer_X_input(self, X_input_brut, nom_modele):
        """Convertit un X brut (colonnes features_X2) dans l'espace attendu par
        le modèle (brut ou PCA), via le même scaler/PCA que l'entraînement."""
        espace = self.modeles.get(nom_modele, {}).get("espace", "brut")
        if espace == "pca":
            X_scaled = self.scaler_pca.transform(X_input_brut)
            return self.pca_model.transform(X_scaled)
        return X_input_brut.values if hasattr(X_input_brut, "values") else X_input_brut

    # ==========================================================================
    # 7. MODÈLES
    # ==========================================================================
    def entrainer_regression_lineaire_pca(self):
        """LinearRegression sur les données réduites par PCA (pas de Ridge ici :
        la PCA décorrèle déjà les features, donc l'OLS classique reste stable)."""
        if self.X_train_pca is None:
            raise Exception("Appelle appliquer_pca() avant entrainer_regression_lineaire_pca().")

        model_lin_pca = LinearRegression()

        self._log("Entraînement de la Régression Linéaire (PCA) en cours...")
        self._progres("Entraînement Linéaire (PCA)", None, None)
        model_lin_pca.fit(self.X_train_pca, self.Y_train2)

        predictions_lin_pca = model_lin_pca.predict(self.X_test_pca)

        scores_r2_lin_pca = r2_score(self.Y_test2, predictions_lin_pca, multioutput='raw_values')
        scores_r2_lin_pca_simple = r2_score(self.Y_test2, predictions_lin_pca)

        self._log("--- RÉSULTATS RÉGRESSION LINÉAIRE (PCA) ---")
        for i, score in enumerate(scores_r2_lin_pca):
            jour_label = "J0 (Aujourd'hui)" if i == 0 else f"J+{i}"
            self._log(f"Prévision {jour_label:<15} : R² = {score * 100:>5.1f} %")
        self._log(f"Score R² pour cette configuration : {scores_r2_lin_pca_simple * 100:.2f} %")

        self.modeles["lineaire_pca"] = {
            "model": model_lin_pca,
            "score": scores_r2_lin_pca_simple,
            "scores_detail": scores_r2_lin_pca,
            "espace": "pca"
        }
        return model_lin_pca, scores_r2_lin_pca_simple

    def entrainer_ridge_brut(self, alpha=10000.0):
        """Ridge sur les données brutes (non réduites) -- alpha fort pour gérer
        la colinéarité massive entre les features horaires."""
        if self.X_train2 is None:
            raise Exception("Données d'entraînement manquantes. Lance lancer_selection_genetique() d'abord.")

        model_lin2 = Ridge(alpha=alpha)

        self._log("Entraînement de la Régression Ridge en cours...")
        self._progres("Entraînement Ridge", None, None)
        model_lin2.fit(self.X_train2, self.Y_train2)

        predictions_lin2 = model_lin2.predict(self.X_test2)

        scores_r2_lin2 = r2_score(self.Y_test2, predictions_lin2, multioutput='raw_values')
        scores_r2_lin2_simple = r2_score(self.Y_test2, predictions_lin2)

        self._log("--- RÉSULTATS RÉGRESSION RIDGE ---")
        for i, score in enumerate(scores_r2_lin2):
            jour_label = "J0 (Aujourd'hui)" if i == 0 else f"J+{i}"
            self._log(f"Prévision {jour_label:<15} : R² = {score * 100:>5.1f} %")
        self._log(f"Score R² pour cette configuration : {scores_r2_lin2_simple * 100:.2f} %")

        self.modeles["ridge_brut"] = {
            "model": model_lin2,
            "score": scores_r2_lin2_simple,
            "scores_detail": scores_r2_lin2,
            "espace": "brut"
        }
        return model_lin2, scores_r2_lin2_simple

    def entrainer_ridge_causal(self, alpha=10000.0, cap_futur_meteo=15):
        """Ridge multi-horizon avec masquage CAUSAL (RegresseurMultiHorizonCausal) :
        pour prédire J+k, seules les features météo <= J+min(k, cap_futur_meteo)
        sont utilisées. C'est le modèle à privilégier pour prédire AU-DELÀ de
        l'horizon de la météo prévisionnelle (~15 jours) : ses sous-modèles
        lointains n'utilisent que la météo proche + l'historique + la date."""
        if self.X_train2 is None:
            raise Exception("Données d'entraînement manquantes. Lance lancer_selection_genetique() d'abord.")

        model_causal = RegresseurMultiHorizonCausal(
            classe_modele=Ridge,
            kwargs_modele={"alpha": alpha},
            features=self.features_X2,
            cap_futur_meteo=cap_futur_meteo,
        )

        self._log(f"Entraînement du Ridge CAUSAL ({self.Y_train2.shape[1]} sous-modèles, météo future <= J+{cap_futur_meteo})...")
        self._progres("Entraînement Ridge causal", None, None)
        model_causal.fit(self.X_train2, self.Y_train2)

        predictions_causal = model_causal.predict(self.X_test2)

        scores_r2_causal = r2_score(self.Y_test2, predictions_causal, multioutput='raw_values')
        scores_r2_causal_simple = r2_score(self.Y_test2, predictions_causal)

        self._log("--- RÉSULTATS RIDGE CAUSAL ---")
        for i, score in enumerate(scores_r2_causal):
            jour_label = "J0 (Aujourd'hui)" if i == 0 else f"J+{i}"
            self._log(f"Prévision {jour_label:<15} : R² = {score * 100:>5.1f} %")
        self._log(f"Score R² pour cette configuration : {scores_r2_causal_simple * 100:.2f} %")

        self.modeles["ridge_causal"] = {
            "model": model_causal,
            "score": scores_r2_causal_simple,
            "scores_detail": scores_r2_causal,
            "espace": "brut"
        }
        return model_causal, scores_r2_causal_simple

    def entrainer_gradient_boosting(self):
        """HistGradientBoostingRegressor enveloppé en MultiOutputRegressor, sur données brutes."""
        if self.X_train2 is None:
            raise Exception("Données d'entraînement manquantes. Lance lancer_selection_genetique() d'abord.")

        modele_de_base_gb2 = HistGradientBoostingRegressor(
            random_state=42,
            max_depth=8,
            max_leaf_nodes=31,
            learning_rate=0.05,
            max_iter=300,
            l2_regularization=10,
            min_samples_leaf=7,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )
        self._log("Entraînement du Gradient Boosting en cours...")
        # Un HistGradientBoosting par horizon, entraînés EN PARALLÈLE (joblib) ;
        # `return_as="generator"` livre chaque horizon dès qu'il est fini, ce qui
        # permet d'avancer la barre de progression sans perdre le parallélisme.
        X_train = self.X_train2
        Y_train_np = self.Y_train2.values
        n_horizons = Y_train_np.shape[1]

        def _entrainer_horizon(k):
            est = clone(modele_de_base_gb2)
            est.fit(X_train, Y_train_np[:, k])
            return k, est

        self._progres("Entraînement Gradient Boosting", 0, n_horizons)
        resultats = Parallel(n_jobs=-1, return_as="generator")(
            delayed(_entrainer_horizon)(k) for k in range(n_horizons)
        )
        estimateurs = [None] * n_horizons
        faits = 0
        for k, est in resultats:
            estimateurs[k] = est
            faits += 1
            self._progres("Entraînement Gradient Boosting", faits, n_horizons)
            self._verifier_arret()   # abandonne les horizons restants si demandé
        model_gb2 = RegresseurMultiSortie(estimateurs)

        predictions_gb2 = model_gb2.predict(self.X_test2)

        scores_r2_gb2 = r2_score(self.Y_test2, predictions_gb2, multioutput='raw_values')
        scores_r2_gb2_simple = r2_score(self.Y_test2, predictions_gb2)

        self._log("--- RÉSULTATS GRADIENT BOOSTING ---")
        for i, score in enumerate(scores_r2_gb2):
            jour_label = "J0 (Aujourd'hui)" if i == 0 else f"J+{i}"
            self._log(f"Prévision {jour_label:<15} : R² = {score * 100:>5.1f} %")
        self._log(f"Score R² pour cette configuration : {scores_r2_gb2_simple * 100:.2f} %")

        self.modeles["gradient_boosting"] = {
            "model": model_gb2,
            "score": scores_r2_gb2_simple,
            "scores_detail": scores_r2_gb2,
            "espace": "brut"
        }
        return model_gb2, scores_r2_gb2_simple

    def entrainer_keras_brut(self):
        """Réseau résiduel (branche linéaire + branche non-linéaire additionnées,
        connexion résiduelle) sur données BRUTES (pas PCA). Un réseau de neurones
        n'a pas besoin d'une PCA pour rester stable comme l'OLS classique -- une
        couche de Normalization (adaptée sur X_train2) en entrée suffit, et
        laisser le réseau voir les ~1786 features complètes lui donne accès à
        des informations que la PCA compressait/perdait (mesuré : +17 points de
        R² par rapport à la version PCA à 19 composantes)."""
        if self.X_train2 is None:
            raise Exception("Données d'entraînement manquantes. Lance lancer_selection_genetique() d'abord.")

        tf.random.set_seed(42)

        X_train_np2 = self.X_train2.values.astype(np.float32)
        Y_train_np2 = self.Y_train2.values.astype(np.float32)
        X_test_np2 = self.X_test2.values.astype(np.float32)
        Y_test_np2 = self.Y_test2.values.astype(np.float32)

        n_features = X_train_np2.shape[1]
        n_outputs = Y_train_np2.shape[1]

        norm_layer = tf.keras.layers.Normalization(axis=-1)
        norm_layer.adapt(X_train_np2)

        inputs = tf.keras.Input(shape=(n_features,))
        reg = tf.keras.regularizers.l2(1e-3)

        x = norm_layer(inputs)
        x = tf.keras.layers.Dense(256, activation="swish", kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)

        res = x
        x = tf.keras.layers.Dense(256, activation="swish", kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Add()([res, x])

        x = tf.keras.layers.Dense(128, activation="swish", kernel_regularizer=reg)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.1)(x)

        outputs = tf.keras.layers.Dense(n_outputs, activation="linear")(x)

        model_keras2 = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer2 = tf.keras.optimizers.AdamW(learning_rate=3e-3, weight_decay=1e-4, clipnorm=1.0)
        model_keras2.compile(loss="mse", optimizer=optimizer2)

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6)

        self._log("🚀 Début de l'entraînement Keras (brut)...")
        callbacks2 = [early_stop, reduce_lr, TqdmCallback(verbose=1)]
        cb_arret = _callback_arret_keras(self._doit_arreter)
        if cb_arret is not None:
            callbacks2.append(cb_arret)
        cb_progres = _callback_progres_keras(self._hook_progres, "Entraînement Keras (réseau)", 300)
        if cb_progres is not None:
            callbacks2.append(cb_progres)
        model_keras2.fit(
            X_train_np2,
            Y_train_np2,
            epochs=300,
            batch_size=64,
            validation_split=0.15,
            callbacks=callbacks2,
            verbose=0
        )
        self._verifier_arret()   # si l'entraînement a été coupé, on remonte l'arrêt

        predictions_keras2 = model_keras2.predict(X_test_np2)

        scores_r2_keras2 = r2_score(Y_test_np2, predictions_keras2, multioutput='raw_values')
        scores_r2_keras2_simple = r2_score(Y_test_np2, predictions_keras2)

        self._log("--- RÉSULTATS TENSORFLOW / KERAS (brut) ---")
        for i, score in enumerate(scores_r2_keras2):
            jour_label = "J0 (Aujourd'hui)" if i == 0 else f"J+{i}"
            self._log(f"Prévision {jour_label:<15} : R² = {score * 100:>5.1f} %")
        self._log(f"🏆 Score R² Global : {scores_r2_keras2_simple * 100:.2f} %")

        self.modeles["keras_brut"] = {
            "model": model_keras2,
            "score": scores_r2_keras2_simple,
            "scores_detail": scores_r2_keras2,
            "espace": "brut"
        }
        return model_keras2, scores_r2_keras2_simple

    def entrainer_tous_les_modeles(self, entrainer_incertitude_aussi=True, modeles=None):
        """Entraîne les modèles demandés (par défaut les 5), puis affiche un
        tableau comparatif des scores. `modeles` : sous-ensemble de
        {'lineaire_pca','ridge_brut','ridge_causal','gradient_boosting','keras_brut'}."""
        entraineurs = {
            "lineaire_pca": self.entrainer_regression_lineaire_pca,
            "ridge_brut": self.entrainer_ridge_brut,
            "ridge_causal": self.entrainer_ridge_causal,
            "gradient_boosting": self.entrainer_gradient_boosting,
            "keras_brut": self.entrainer_keras_brut,
        }
        a_entrainer = [m for m in (modeles or list(entraineurs.keys())) if m in entraineurs]
        for idx, nom in enumerate(a_entrainer):
            self._verifier_arret()   # interruption possible entre deux modèles
            self._progres(f"Entraînement des modèles ({nom})", idx, len(a_entrainer))
            entraineurs[nom]()
        self._progres("Entraînement des modèles", len(a_entrainer), len(a_entrainer))

        if entrainer_incertitude_aussi:
            noms = list(self.modeles.keys())
            for idx, nom in enumerate(noms):
                self._verifier_arret()
                self._progres(f"Incertitude ({nom})", idx, len(noms))
                self.entrainer_incertitude(nom)

        self._log("")
        self._log("=" * 50)
        self._log("📊 TABLEAU RÉCAPITULATIF DES SCORES R²")
        self._log("=" * 50)
        for nom, info in sorted(self.modeles.items(), key=lambda kv: kv[1]["score"], reverse=True):
            self._log(f"  {nom:<22} R² = {info['score'] * 100:>6.2f} %")
        self._log("=" * 50)

        return self.modeles

    def executer_pipeline_gradient_boosting(self, params_hourly, params_daily,
                                             utiliser_cache_offline_existant=True,
                                             fichier_cache_eau=None,
                                             fichier_cache_meteo=None,
                                             entrainer_incertitude_aussi=True,
                                             **kwargs_ga):
        """
        Enchaîne tout le pipeline, dans l'ordre, jusqu'à obtenir un
        HistGradientBoostingRegressor entraîné et prêt à l'emploi :
        determiner_bassins_versants() (si pas déjà fait) -> cache offline
        existant ou preparer_donnees_legeres() -> lancer_selection_genetique()
        (avec téléchargement complet) -> entrainer_gradient_boosting() (+
        entrainer_incertitude() par défaut).

        Ne fait PAS appliquer_pca() : gradient_boosting tourne en espace brut
        et n'en a pas besoin -- appelle appliquer_pca() toi-même ensuite si tu
        veux aussi les modèles PCA (lineaire_pca).

        utiliser_cache_offline_existant : si True et que fichier_cache_eau /
        fichier_cache_meteo existent déjà sur le disque, le GA les réutilise
        directement (charger_cache_offline()) au lieu de retélécharger --
        économise du temps et du quota API. Par défaut (None), les noms de
        fichiers sont ceux qu'écrit preparer_donnees_legeres()
        (cache_eau_{station}.csv / cache_meteo_{station}.csv), pour que le
        cache d'un premier run soit bien retrouvé au run suivant.

        **kwargs_ga : transmis tels quels à lancer_selection_genetique() (ex.
        taille_population, nombre_generations, split_date_final, past_day_opti,
        predict_day_opti...).
        """
        if fichier_cache_eau is None:
            fichier_cache_eau = f"cache_eau_{self.code_station}.csv"
        if fichier_cache_meteo is None:
            fichier_cache_meteo = f"cache_meteo_{self.code_station}.csv"

        if self.listes_de_points is None:
            self.determiner_bassins_versants(afficher=False)

        # Le cache léger n'est réutilisable que s'il couvre (presque) toute la
        # grille ACTUELLE : si les zones ont changé (ex. nouveau filtre amont),
        # un vieux cache biaiserait le GA vers les seuls points qu'il connaît.
        cache_utilisable = False
        if utiliser_cache_offline_existant and os.path.exists(fichier_cache_eau) and os.path.exists(fichier_cache_meteo):
            points_dispos = set(map(tuple, pd.read_csv(fichier_cache_meteo, usecols=['lat', 'lon']).round(4).drop_duplicates().values))
            points_grille = set((round(lat, 4), round(lon, 4)) for zone in self.listes_de_points for lat, lon in zone)
            manquants = points_grille - points_dispos
            cache_utilisable = len(manquants) <= 0.1 * len(points_grille)
            if not cache_utilisable:
                self._log(f"Cache léger existant mais {len(manquants)}/{len(points_grille)} points de la grille actuelle "
                          f"en sont absents (la grille a changé) : retéléchargement du cache.", "warning")

        if cache_utilisable:
            self.charger_cache_offline(fichier_cache_eau, fichier_cache_meteo)
        else:
            self.preparer_donnees_legeres(params_hourly=params_hourly, params_daily=params_daily)

        self.lancer_selection_genetique(
            params_hourly=params_hourly,
            params_daily=params_daily,
            telecharger_donnees_completes=True,
            **kwargs_ga,
        )

        self.entrainer_gradient_boosting()
        if entrainer_incertitude_aussi:
            self.entrainer_incertitude("gradient_boosting")

        self._log("🏆 Pipeline terminé : 'gradient_boosting' est prêt (riviere.modeles['gradient_boosting']).", "succes")
        return self.modeles["gradient_boosting"]

    def carte_selection_deux_temps(self, sauvegarder_sous="carte_selection_deux_temps.html", afficher=False):
        """Carte : présélection (petits points gris) + points finaux retenus
        (gros points rouges) + station. Illustre la sélection en deux temps."""
        if self.coords_finales is None:
            return None
        centre = [self.lat_station or self.coords_finales[0][0], self.lon_station or self.coords_finales[0][1]]
        carte = folium.Map(location=centre, zoom_start=10, tiles="CartoDB positron")
        if self.geojson_bassins:
            folium.GeoJson(self.geojson_bassins,
                           style_function=lambda x: {'fillColor': '#2ca02c', 'color': '#006400',
                                                     'weight': 1, 'fillOpacity': 0.08}).add_to(carte)
        for i, pt in enumerate(self.points_preselectionnes or []):
            alt = (self._altitudes_preselection or [None] * len(self.points_preselectionnes))[i]
            folium.CircleMarker(location=[pt[0], pt[1]], radius=3, color="#888", fill=True,
                                fill_opacity=0.6,
                                popup=f"Présélection{'' if alt is None else f' — {alt:.0f} m'}").add_to(carte)
        for pt in self.coords_finales:
            folium.CircleMarker(location=[pt[0], pt[1]], radius=9, color="#c40000", fill=True,
                                fill_color="#ff3333", fill_opacity=0.9, weight=2,
                                popup="🏆 Point retenu").add_to(carte)
        if self.lat_station and self.lon_station:
            folium.Marker([self.lat_station, self.lon_station],
                          icon=folium.Icon(color="red", icon="info-sign"), popup="Station").add_to(carte)
        carte.save(sauvegarder_sous)
        if afficher:
            afficher_carte_html(sauvegarder_sous)
        return carte

    def executer_pipeline_deux_temps(self, params_hourly, params_daily,
                                     n_preselection=30, n_final=5, fenetre_selection_annees=2,
                                     densite_grille=25, max_par_zone=60,
                                     poids_altitude=1.0, poids_zones=1.0,
                                     poids_pluie=1.0, poids_neige=1.0,
                                     entrainer_incertitude_aussi=True,
                                     start_fetch=None, end_fetch=None,
                                     past_day=20, predict_day=15,
                                     mode_split="annees_aleatoires", part_test=0.2,
                                     agregations_horaires=None):
        """
        Pipeline complet AVEC SÉLECTION EN DEUX TEMPS (remplaçant du GA, bien
        plus léger en API et en calcul) :
          1. présélection GRATUITE (altitude + couverture + zones) -> n_preselection ;
          2. affinage météo LÉGER (corrélation pluie->débit + neige) -> n_final ;
          3. téléchargement complet sur les n_final points -> Gradient Boosting.
        """
        self.executer_selection_deux_temps(
            n_preselection=n_preselection, n_final=n_final, fenetre_annees=fenetre_selection_annees,
            densite_grille=densite_grille, max_par_zone=max_par_zone,
            poids_altitude=poids_altitude, poids_zones=poids_zones,
            poids_pluie=poids_pluie, poids_neige=poids_neige,
        )
        try:
            self.carte_selection_deux_temps()
        except Exception as e:
            self._log(f"Carte de sélection non générée : {e}", "warning")

        self.construire_donnees_completes(
            params_hourly=params_hourly, params_daily=params_daily,
            start_fetch=start_fetch, end_fetch=end_fetch,
            past_day=past_day, predict_day=predict_day,
            mode_split=mode_split, part_test=part_test,
            agregations_horaires=agregations_horaires,
        )

        self.entrainer_gradient_boosting()
        if entrainer_incertitude_aussi:
            self.entrainer_incertitude("gradient_boosting")

        self._log("🏆 Pipeline (deux temps) terminé : 'gradient_boosting' est prêt.", "succes")
        return self.modeles["gradient_boosting"]

    # ==========================================================================
    # 8. MODÈLE PROBABILISTE (généralisé à N'IMPORTE QUEL modèle déjà entraîné)
    # ==========================================================================
    def entrainer_incertitude(self, nom_modele, epochs=150, batch_size=64):
        """
        Généralisation de l'approche "résidus + réseau de variance" à n'importe
        quel modèle déjà entraîné dans self.modeles (Ridge, PCA, Gradient
        Boosting, Keras...). L'espérance reste 100% celle du modèle de base ;
        seule la variance est apprise par un petit réseau séparé.
        """
        if nom_modele not in self.modeles:
            self._log(f"Modèle '{nom_modele}' introuvable. Entraîne-le d'abord.", "erreur")
            return None

        model_base = self.modeles[nom_modele]["model"]
        espace = self.modeles[nom_modele].get("espace", "brut")

        X_train_espace = self.X_train_pca if espace == "pca" else self.X_train2
        X_test_espace = self.X_test_pca if espace == "pca" else self.X_test2

        self._log(f"Calcul des résidus de '{nom_modele}'...")
        predictions_train = model_base.predict(X_train_espace)

        residus_train = self.Y_train2.values - predictions_train

        X_train_np = np.asarray(X_train_espace, dtype=np.float32)
        residus_train_np = residus_train.astype(np.float32)

        n_features = X_train_np.shape[1]
        n_outputs = residus_train_np.shape[1]

        def gaussian_nll_variance_only(y_true, y_pred):
            """
            y_true : L'erreur réelle du modèle de base.
            y_pred : La log-variance prédite par le réseau.
            On suppose mathématiquement que mu = 0 (le modèle de base a déjà fait le job).
            """
            log_var = y_pred
            var = tf.exp(log_var) + 1e-6
            nll = 0.5 * log_var + 0.5 * tf.square(y_true) / var
            return tf.reduce_mean(nll)

        norm_layer = tf.keras.layers.Normalization(axis=-1)
        norm_layer.adapt(X_train_np)

        inputs = tf.keras.Input(shape=(n_features,))
        x = norm_layer(inputs)

        x = tf.keras.layers.Dense(64, activation="silu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Dense(32, activation="silu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        outputs = tf.keras.layers.Dense(n_outputs, activation="linear")(x)

        model_variance = tf.keras.Model(inputs=inputs, outputs=outputs)
        model_variance.compile(loss=gaussian_nll_variance_only, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3))

        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6)

        self._log(f"Entraînement du réseau de variance pour '{nom_modele}'...")
        callbacks_var = [early_stop, reduce_lr, TqdmCallback(verbose=1)]
        cb_arret = _callback_arret_keras(self._doit_arreter)
        if cb_arret is not None:
            callbacks_var.append(cb_arret)
        cb_progres = _callback_progres_keras(self._hook_progres, f"Incertitude de « {nom_modele} »", epochs)
        if cb_progres is not None:
            callbacks_var.append(cb_progres)
        model_variance.fit(
            X_train_np, residus_train_np,
            epochs=epochs, batch_size=batch_size, validation_split=0.1,
            callbacks=callbacks_var,
            verbose=0
        )

        class DummyDist:
            def __init__(self, loc, scale):
                self.params = {'loc': loc, 'scale': scale}

        class DummyEstimator:
            def __init__(self, model_base, nn_model, horizon_idx):
                self.model_base = model_base
                self.nn_model = nn_model
                self.horizon_idx = horizon_idx

            def pred_dist(self, X):
                mu_all = self.model_base.predict(X)
                log_var_all = self.nn_model.predict(X, verbose=0)
                mu = mu_all[:, self.horizon_idx]
                std = np.sqrt(np.exp(log_var_all[:, self.horizon_idx]))
                return DummyDist(loc=mu, scale=std)

        class ModeleHybride:
            """
            Fusionne un modèle de base (espérance) et un réseau de variance pour
            agir comme un modèle probabiliste complet. Implémente predict() et
            pred_dist() via ses estimators_, comme attendu par tracer_prediction().
            """
            def __init__(self, model_base, nn_model, n_horizons):
                self.model_base = model_base
                self.nn_model = nn_model
                self.estimators_ = [DummyEstimator(model_base, nn_model, i) for i in range(n_horizons)]

            def predict(self, X):
                return self.model_base.predict(X)

        model_hybride = ModeleHybride(model_base, model_variance, n_outputs)

        self.modeles[nom_modele]["hybride"] = model_hybride
        self.modeles[nom_modele]["modele_variance"] = model_variance

        self._log(f"Modèle hybride prêt pour '{nom_modele}' !", "succes")

        return model_hybride

    def auditer_incertitude(self, nom_modele):
        """Évalue si les intervalles de confiance (la variance) du modèle hybride
        sont honnêtes et utiles (couverture empirique, largeur, NLL)."""
        if nom_modele not in self.modeles or "hybride" not in self.modeles[nom_modele]:
            self._log(f"Pas de modèle hybride entraîné pour '{nom_modele}'. Appelle entrainer_incertitude() d'abord.", "erreur")
            return None

        model = self.modeles[nom_modele]["hybride"]
        predict_day = self.predict_day2

        X_test_prepare = self._preparer_X_input(self.X_test2, nom_modele)
        Y_test_np = self.Y_test2.values
        n_lignes = Y_test_np.shape[0]
        n_horizons = predict_day + 1

        mu_mat = np.zeros((n_lignes, n_horizons))
        sigma_mat = np.zeros((n_lignes, n_horizons))

        for i in range(n_horizons):
            dist = model.estimators_[i].pred_dist(X_test_prepare)
            mu_mat[:, i] = dist.params['loc']
            sigma_mat[:, i] = dist.params['scale']

        borne_basse_95 = mu_mat - 1.96 * sigma_mat
        borne_haute_95 = mu_mat + 1.96 * sigma_mat

        succes = (Y_test_np >= borne_basse_95) & (Y_test_np <= borne_haute_95)
        couverture_globale = np.mean(succes) * 100

        largeur_moyenne = np.mean(borne_haute_95 - np.maximum(0, borne_basse_95))

        var_mat = sigma_mat ** 2 + 1e-6
        nll_mat = 0.5 * np.log(2 * np.pi * var_mat) + (0.5 * (Y_test_np - mu_mat) ** 2 / var_mat)
        nll_moyen = np.mean(nll_mat)

        self._log("🔍 Audit probabiliste en cours...")
        self._log("📊 RAPPORT DE QUALITÉ DE LA VARIANCE (Intervalle visé : 95%)")
        self._log("-" * 50)
        self._log(f"🎯 Couverture Empirique : {couverture_globale:.1f} %")

        if couverture_globale > 98:
            self._log("   ⚠️ Diagnostic : Modèle sous-confiant (Intervalles trop larges, peureux).")
        elif couverture_globale < 90:
            self._log("   ⚠️ Diagnostic : Modèle sur-confiant (Intervalles trop étroits, arrogant).")
        else:
            self._log("   ✅ Diagnostic : Excellente calibration. La variance est honnête.")

        self._log(f"📏 Largeur moyenne de l'IC 95% : {largeur_moyenne / L_PAR_S_VERS_M3_PAR_S:.3f} m³/s")
        self._log(f"📉 Score NLL Global            : {nll_moyen:.3f} (le plus bas possible est le meilleur)")
        self._log("-" * 50)

        return couverture_globale, largeur_moyenne, nll_moyen

    # ==========================================================================
    # 9. VISUALISATION DES PRÉDICTIONS
    # ==========================================================================
    def _tracer_fan_chart(self, contexte, dates_futures, valeurs_pred, intervalles,
                           date_pivot, titre, serie_reelle=None):
        """
        Tracé UNIQUE partagé par les prédictions sur le passé
        (tracer_prediction) et les prévisions réelles (predire_futur) :
        passé observé en noir, prédiction en bleu nuit, 3 bandes de confiance
        50/95/99 si `intervalles` est fourni ({"50": (bas, haut), ...} --
        listes alignées sur dates_futures), courbe réelle en rouge si
        `serie_reelle` est fournie (backtest uniquement).
        """
        # Conversion L/s -> m³/s pour l'affichage.
        D = L_PAR_S_VERS_M3_PAR_S
        conv = lambda seq: [v / D for v in seq]
        valeurs_pred = conv(valeurs_pred)
        serie_reelle = conv(serie_reelle) if serie_reelle is not None else None
        intervalles = ({niveau: (conv(bas), conv(haut)) for niveau, (bas, haut) in intervalles.items()}
                       if intervalles else intervalles)
        contexte_debit = (contexte['debit_L_s'] / D) if not contexte.empty else None

        max_bandes = max(intervalles["99"][1]) if intervalles else 0
        max_reel = max(serie_reelle) if serie_reelle is not None else 0
        max_affiche = max(
            contexte_debit.max() if contexte_debit is not None else 0,
            max(valeurs_pred), max_bandes, max_reel,
        )
        limite_y = max_affiche * 1.25

        plt.figure(figsize=(12, 6))

        if intervalles:
            for niveau, alpha, libelle in [("99", 0.15, "IC 99% (Très rare)"),
                                            ("95", 0.35, "IC 95% (Probable)"),
                                            ("50", 0.60, "IC 50% (Cœur de la cible)")]:
                plt.fill_between(dates_futures, intervalles[niveau][0], intervalles[niveau][1],
                                 color='dodgerblue', alpha=alpha, label=libelle, edgecolor='none')

        if not contexte.empty:
            plt.plot(contexte['date'], contexte_debit, color='black', linewidth=2.5, label="Passé (Hub'Eau)")

        plt.plot(dates_futures, valeurs_pred, color='#003366', marker='X', linestyle='-',
                 linewidth=2.5, label="Prédiction (Moyenne)" if intervalles else "Prédiction")

        if serie_reelle is not None:
            plt.plot(dates_futures, serie_reelle, color='red', marker='o', linestyle=':',
                     linewidth=2.5, label="Vrai Futur (Réalité)")

        plt.axvline(date_pivot, color='grey', linestyle='--', alpha=0.8)
        gauche = min(contexte['date'].min(), dates_futures[0]) if not contexte.empty else dates_futures[0]
        plt.xlim(left=gauche, right=dates_futures[-1])
        plt.ylim(bottom=0, top=limite_y)

        plt.xticks(rotation=45, ha='right')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(FORMAT_DATE_FR))

        plt.title(titre, fontweight='bold', fontsize=14)
        plt.ylabel("Débit (m³/s)", fontweight='bold')

        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.grid(True, axis='x', linestyle=':', alpha=0.4)

        plt.legend(loc='upper left', frameon=True, shadow=True, facecolor='white')
        plt.tight_layout()
        plt.show()

    def tracer_prediction(self, nom_modele, date_cible_str, utiliser_hybride=True, titre=None, nb_jours=None):
        """
        Trace le débit avec bandes de confiance (Fan Chart) si un modèle hybride
        est disponible pour ce modèle, sinon un tracé classique point-à-point.
        nb_jours : nombre de jours d'horizon à AFFICHER (défaut = tout l'horizon
        du modèle, predict_day). Utile pour zoomer sur les premiers jours.
        """
        if nom_modele not in self.modeles:
            self._log(f"Modèle '{nom_modele}' introuvable.", "erreur")
            return

        info = self.modeles[nom_modele]
        model = info.get("hybride") if (utiliser_hybride and "hybride" in info) else info["model"]
        predict_day = self.predict_day2
        # Nombre d'horizons affichés (0..h_max), borné à l'horizon du modèle.
        h_max = predict_day if nb_jours is None else max(1, min(int(nb_jours), predict_day))
        titre = titre or nom_modele

        date_cible = pd.to_datetime(date_cible_str)

        data_split = self.test2 if self.test2 is not None else self.df_final
        ligne_J0 = data_split[data_split['date'] == date_cible]
        if ligne_J0.empty:
            self._log(f"La date '{date_cible_str}' n'est pas disponible dans ce bloc ({titre}).", "erreur")
            return

        targets_Y = [f"cible_debit_J+{i}" for i in range(h_max + 1)]

        contexte = self.df_final[(self.df_final['date'] >= date_cible - pd.Timedelta(days=15)) &
                                  (self.df_final['date'] <= date_cible)]

        X_input_brut = ligne_J0[self.features_X2]
        X_input = self._preparer_X_input(X_input_brut, nom_modele)

        Y_vrai_futur = ligne_J0[targets_Y].values[0]
        Y_pred_futur = model.predict(X_input)[0][:h_max + 1]

        debit_actuel = ligne_J0['debit_L_s'].values[0]
        dates_futures = [date_cible] + list(pd.date_range(start=date_cible + pd.Timedelta(days=1), periods=h_max + 1))

        Y_vrai_continu = [debit_actuel] + list(Y_vrai_futur)
        Y_pred_continu = [debit_actuel] + list(Y_pred_futur)

        est_probabiliste = hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'pred_dist')

        intervalles = {}
        if est_probabiliste:
            mus, sigmas = [], []
            for i in range(h_max + 1):
                dist = model.estimators_[i].pred_dist(X_input)
                mus.append(float(dist.params['loc'][0]))
                sigmas.append(float(dist.params['scale'][0]))
            sigmas = pd.Series(sigmas).ffill().bfill().fillna(0).tolist()

            for niveau, z in [("50", 0.674), ("95", 1.960), ("99", 2.576)]:
                intervalles[niveau] = (
                    [debit_actuel] + [max(0, m - z * s) for m, s in zip(mus, sigmas)],
                    [debit_actuel] + [m + z * s for m, s in zip(mus, sigmas)],
                )

        titre_final = f"{titre} (Probabilités & Intervalles)" if est_probabiliste else titre
        self._tracer_fan_chart(
            contexte=contexte,
            dates_futures=dates_futures,
            valeurs_pred=Y_pred_continu,
            intervalles=intervalles,
            date_pivot=date_cible,
            titre=f"{titre_final} (Départ : {date_cible.strftime(FORMAT_DATE_FR)})",
            serie_reelle=Y_vrai_continu,
        )

    def donnees_prediction_passee(self, nom_modele, date_cible_str, utiliser_hybride=True, nb_jours=None):
        """
        Même calcul que tracer_prediction() mais renvoie les DONNÉES (en m³/s,
        prêtes pour un graphique interactif côté navigateur) au lieu de tracer :
        {pivot, hybride, observe: [{date, debit}], points: [{date, prev, reel,
        ic50_bas, ic50_haut, ic95_bas, ic95_haut, ic99_bas, ic99_haut}]}.
        """
        if nom_modele not in self.modeles:
            raise Exception(f"Modèle '{nom_modele}' introuvable.")
        info = self.modeles[nom_modele]
        model = info.get("hybride") if (utiliser_hybride and "hybride" in info) else info["model"]
        predict_day = self.predict_day2
        h_max = predict_day if nb_jours is None else max(1, min(int(nb_jours), predict_day))

        date_cible = pd.to_datetime(date_cible_str)
        data_split = self.test2 if self.test2 is not None else self.df_final
        ligne_J0 = data_split[data_split['date'] == date_cible]
        if ligne_J0.empty:
            return None

        targets_Y = [f"cible_debit_J+{i}" for i in range(h_max + 1)]
        contexte = self.df_final[(self.df_final['date'] >= date_cible - pd.Timedelta(days=15)) &
                                  (self.df_final['date'] <= date_cible)]
        X_input = self._preparer_X_input(ligne_J0[self.features_X2], nom_modele)

        Y_vrai = ligne_J0[targets_Y].values[0]
        Y_pred = model.predict(X_input)[0][:h_max + 1]
        debit_actuel = ligne_J0['debit_L_s'].values[0]
        dates_futures = [date_cible] + list(pd.date_range(start=date_cible + pd.Timedelta(days=1), periods=h_max + 1))
        Y_vrai_continu = [debit_actuel] + list(Y_vrai)
        Y_pred_continu = [debit_actuel] + list(Y_pred)

        est_probabiliste = hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'pred_dist')
        intervalles = {}
        if est_probabiliste:
            mus, sigmas = [], []
            for i in range(h_max + 1):
                dist = model.estimators_[i].pred_dist(X_input)
                mus.append(float(dist.params['loc'][0]))
                sigmas.append(float(dist.params['scale'][0]))
            sigmas = pd.Series(sigmas).ffill().bfill().fillna(0).tolist()
            for niveau, z in [("50", 0.674), ("95", 1.960), ("99", 2.576)]:
                intervalles[niveau] = (
                    [debit_actuel] + [max(0, m - z * s) for m, s in zip(mus, sigmas)],
                    [debit_actuel] + [m + z * s for m, s in zip(mus, sigmas)],
                )

        D = L_PAR_S_VERS_M3_PAR_S
        observe = [{"date": d.strftime('%Y-%m-%d'), "debit": float(v) / D}
                   for d, v in zip(contexte['date'], contexte['debit_L_s'])]
        points = []
        for i, dt in enumerate(dates_futures):
            p = {"date": dt.strftime('%Y-%m-%d'),
                 "prev": float(Y_pred_continu[i]) / D, "reel": float(Y_vrai_continu[i]) / D}
            for niveau in ("50", "95", "99"):
                if intervalles:
                    p[f"ic{niveau}_bas"] = float(intervalles[niveau][0][i]) / D
                    p[f"ic{niveau}_haut"] = float(intervalles[niveau][1][i]) / D
            points.append(p)
        return {"pivot": date_cible.strftime('%Y-%m-%d'), "hybride": bool(intervalles),
                "observe": observe, "points": points}

    # ==========================================================================
    # 9bis. PRÉVISION RÉELLE (météo FORECAST, pas archive)
    # ==========================================================================
    def predire_futur(self, nom_modele="gradient_boosting", params_hourly=None, params_daily=None,
                       nb_jours=None, utiliser_hybride=True, afficher_graphique=True, titre=None):
        """
        Prévision en conditions réelles, AFFICHÉE À PARTIR D'AUJOURD'HUI (le
        débit moyen du jour n'étant pas encore publié, la première valeur
        prédite est celle d'aujourd'hui, raccordée visuellement au dernier
        débit connu).

        nb_jours : nombre de jours prédits à partir d'aujourd'hui (défaut 15,
        maximum predict_day2 défini à l'entraînement). La météo prévisionnelle
        d'Open-Meteo s'arrête à ~15 jours ; au-delà, les features météo
        manquantes sont remplacées par la moyenne de la fenêtre téléchargée
        (~40 jours autour d'aujourd'hui, donc saisonnière). Pour les horizons
        lointains, privilégier 'ridge_causal' (n'utilise structurellement que
        la météo <= J+15) ou 'gradient_boosting'.

        Retourne un DataFrame (date, debit_predit_L_s[, ic_bas/haut_50/95/99
        si un modèle hybride est disponible]), une ligne par jour à partir
        d'aujourd'hui.
        """
        if self.coords_finales is None:
            raise Exception("Aucun point optimal défini -- lance lancer_selection_genetique() ou definir_points_optimaux() avant.")
        if nom_modele not in self.modeles:
            raise Exception(f"Modèle '{nom_modele}' introuvable. Entraîne-le d'abord.")
        if self.past_day2 is None or self.predict_day2 is None or self.features_X2 is None:
            raise Exception("past_day2/predict_day2/features_X2 inconnus -- appelle construire_donnees_completes() ou charger_donnees_completes() avant.")
        if params_hourly is None or params_daily is None:
            raise Exception("params_hourly/params_daily requis (les mêmes variables que lors de l'entraînement).")

        past_day = self.past_day2
        predict_day = self.predict_day2
        titre = titre or nom_modele

        aujourdhui = pd.Timestamp.now().normalize()
        start_fetch = (aujourdhui - pd.Timedelta(days=past_day + 7)).strftime('%Y-%m-%d')

        self._log(f"💧 Téléchargement du débit récent Hub'Eau ({start_fetch} -> aujourd'hui)...")
        df_eau_recent = self._data.fetch_river_data(self.code_station, start_fetch, aujourdhui.strftime('%Y-%m-%d'))
        df_eau_recent = df_eau_recent.dropna(subset=['debit_L_s']).sort_values('date').reset_index(drop=True)
        if df_eau_recent.empty:
            raise Exception("Aucun débit récent exploitable sur Hub'Eau pour cette station.")

        # Série de débit indexée par DATE (les données récentes de Hub'Eau ont
        # parfois des jours manquants : un accès positionnel décalerait les
        # retards, et un NaN rendrait le réseau de variance muet -> sigma nul).
        serie_eau = df_eau_recent.set_index('date')['debit_L_s'].asfreq('D')
        nb_trous = int(serie_eau.isna().sum())
        if nb_trous:
            self._log(f"{nb_trous} jour(s) manquant(s) dans le débit récent -- comblés par interpolation.", "warning")
            serie_eau = serie_eau.interpolate(limit_direction='both')

        date_J0 = df_eau_recent['date'].iloc[-1]
        debit_actuel = float(df_eau_recent['debit_L_s'].iloc[-1])
        decalage = int((aujourdhui - date_J0).days)
        self._log(f"📅 Dernier débit connu (J0) : {date_J0.strftime('%Y-%m-%d')} (il y a {decalage} jour(s))")

        if decalage > predict_day:
            raise Exception(f"Le dernier débit publié date de {decalage} jours -- au-delà de l'horizon du modèle ({predict_day} j).")

        # Horizons (comptés depuis J0) correspondant aux jours demandés depuis aujourd'hui.
        nb_jours_max = predict_day - decalage + 1
        if nb_jours is None:
            nb_jours = min(15, nb_jours_max)
        if nb_jours > nb_jours_max:
            self._log(f"nb_jours réduit de {nb_jours} à {nb_jours_max} : le modèle a été entraîné avec predict_day={predict_day} "
                      f"(relance la construction des données avec un predict_day plus grand pour aller plus loin).", "warning")
            nb_jours = nb_jours_max
        horizons = list(range(decalage, decalage + nb_jours))

        # L'horizon exact de l'API forecast Open-Meteo VARIE d'un jour à l'autre
        # (≈ +14 à +16 j) : on demande +15 et on réduit si la plage est refusée
        # (400). Les jours au-delà du forecast sont de toute façon comblés par la
        # moyenne saisonnière plus bas -- un forecast plus court n'empêche rien.
        # MÊME répartition horaire/journalier qu'à l'entraînement (sinon les
        # colonnes de features ne correspondraient pas au modèle).
        hourly_fetch, daily_fetch, renommage = self._data.repartir_variables_meteo(
            params_hourly, params_daily, self.agregations_horaires)
        resultats = None
        for cap in (15, 13, 11, 9, 7):
            fin_meteo = aujourdhui + pd.Timedelta(days=cap)
            self._log(f"🌤 Téléchargement de la PRÉVISION météo ({start_fetch} -> {fin_meteo.strftime('%Y-%m-%d')}) "
                      f"sur {len(self.coords_finales)} points...")
            try:
                resultats = self._data.fetch_weather_data_multi(
                    self.coords_finales, start_fetch, fin_meteo.strftime('%Y-%m-%d'),
                    daily_vars=daily_fetch, hourly_vars=hourly_fetch
                )
                break
            except Exception as e:
                if "400" in str(e) and cap > 7:
                    self._log(f"Horizon météo +{cap} j hors de la plage forecast Open-Meteo, réduction...", "warning")
                    continue
                raise
        if resultats is None:
            raise Exception("Impossible de récupérer la météo prévisionnelle (plage forecast Open-Meteo).")

        list_df_meteo = []
        for df_hourly_pt, df_daily_pt in resultats:
            df_daily_pt = df_daily_pt.rename(columns=renommage)
            if not df_hourly_pt.empty:
                df_hourly_wide_pt = self._data.hourly_to_daily_spec(df_hourly_pt, self.agregations_horaires)
                df_pt = pd.merge(df_daily_pt, df_hourly_wide_pt, on='date', how='outer')
            else:
                df_pt = df_daily_pt.copy()
            list_df_meteo.append(df_pt)

        df_meteo_moy = pd.concat(list_df_meteo).groupby('date').mean()
        colonnes_meteo = list(df_meteo_moy.columns)
        moyennes_fenetre = df_meteo_moy.mean()

        dict_features = {}
        for i in range(1, past_day + 1):
            date_i = date_J0 - pd.Timedelta(days=i)
            if date_i not in serie_eau.index:
                raise Exception(f"Pas assez d'historique de débit récent pour construire debit_J-{i}.")
            dict_features[f"debit_J-{i}"] = float(serie_eau.loc[date_i])

        nb_jours_extrapoles = 0
        for prefixe, plage in [("J-", range(1, past_day + 1)), ("J+", range(0, predict_day + 1))]:
            for i in plage:
                date_i = date_J0 - pd.Timedelta(days=i) if prefixe == "J-" else date_J0 + pd.Timedelta(days=i)
                dispo = date_i in df_meteo_moy.index
                if prefixe == "J+" and not dispo:
                    nb_jours_extrapoles += 1
                for var in colonnes_meteo:
                    if dispo:
                        dict_features[f"{var}_{prefixe}{i}"] = df_meteo_moy.loc[date_i, var]
                    else:
                        # Au-delà de l'horizon météo : moyenne de la fenêtre
                        # téléchargée (saisonnier local), plutôt que NaN.
                        dict_features[f"{var}_{prefixe}{i}"] = moyennes_fenetre[var]

        if nb_jours_extrapoles:
            self._log(f"Météo prévisionnelle indisponible sur {nb_jours_extrapoles} jour(s) lointain(s) : "
                      f"features remplacées par la moyenne saisonnière locale ('ridge_causal' recommandé à ces horizons).", "warning")

        dict_features['sin_date'] = np.sin(2 * np.pi * date_J0.dayofyear / 365.25)
        dict_features['cos_date'] = np.cos(2 * np.pi * date_J0.dayofyear / 365.25)

        X_input_brut = pd.DataFrame([dict_features])[self.features_X2]

        # Dernier filet de sécurité : le réseau de variance (Keras) renvoie NaN
        # dès qu'UNE feature est NaN -> sigma nul -> bandes de confiance
        # invisibles. On comble par la moyenne de la fenêtre météo (ou 0).
        colonnes_nan = X_input_brut.columns[X_input_brut.isna().any()].tolist()
        if colonnes_nan:
            self._log(f"{len(colonnes_nan)} feature(s) sans valeur (ex. {colonnes_nan[:3]}) -- comblées par la moyenne de la fenêtre.", "warning")
            for col in colonnes_nan:
                base = re.sub(r'_J[+-]\d+$', '', col)
                valeur = moyennes_fenetre.get(base, 0.0)
                X_input_brut[col] = 0.0 if pd.isna(valeur) else float(valeur)

        info = self.modeles[nom_modele]
        model = info.get("hybride") if (utiliser_hybride and "hybride" in info) else info["model"]
        X_input = self._preparer_X_input(X_input_brut, nom_modele)

        # Borné à 0 : un débit ne peut pas être négatif (les modèles linéaires
        # peuvent extrapoler sous zéro sur les décrues aux horizons lointains).
        prediction_complete = np.maximum(model.predict(X_input)[0], 0)
        dates_affichees = [date_J0 + pd.Timedelta(days=k) for k in horizons]
        valeurs_affichees = [float(prediction_complete[k]) for k in horizons]

        resultat = pd.DataFrame({"date": dates_affichees, "debit_predit_L_s": valeurs_affichees})

        est_probabiliste = hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'pred_dist')
        intervalles = {}
        if est_probabiliste:
            mus, sigmas = [], []
            for k in horizons:
                dist = model.estimators_[k].pred_dist(X_input)
                mus.append(max(0.0, float(dist.params['loc'][0])))
                sigmas.append(float(dist.params['scale'][0]))
            sigmas = pd.Series(sigmas).ffill().bfill().fillna(0).tolist()

            for niveau, z in [("50", 0.674), ("95", 1.960), ("99", 2.576)]:
                intervalles[niveau] = (
                    [max(0, m - z * s) for m, s in zip(mus, sigmas)],
                    [m + z * s for m, s in zip(mus, sigmas)],
                )
                resultat[f'ic_bas_{niveau}'] = intervalles[niveau][0]
                resultat[f'ic_haut_{niveau}'] = intervalles[niveau][1]

        # Données prêtes pour un graphique interactif (m³/s), stockées pour l'API.
        D = L_PAR_S_VERS_M3_PAR_S
        ctx = df_eau_recent[df_eau_recent['date'] >= date_J0 - pd.Timedelta(days=15)]
        pts = [{"date": date_J0.strftime('%Y-%m-%d'), "prev": float(debit_actuel) / D}]
        if intervalles:
            for niv in ("50", "95", "99"):
                pts[0][f"ic{niv}_bas"] = float(debit_actuel) / D
                pts[0][f"ic{niv}_haut"] = float(debit_actuel) / D
        for i, dt in enumerate(dates_affichees):
            p = {"date": dt.strftime('%Y-%m-%d'), "prev": float(valeurs_affichees[i]) / D}
            if intervalles:
                for niv in ("50", "95", "99"):
                    p[f"ic{niv}_bas"] = float(intervalles[niv][0][i]) / D
                    p[f"ic{niv}_haut"] = float(intervalles[niv][1][i]) / D
            pts.append(p)
        self.derniere_prevision = {
            "pivot": date_J0.strftime('%Y-%m-%d'),
            "aujourdhui": aujourdhui.strftime('%Y-%m-%d'),
            "hybride": bool(intervalles),
            "observe": [{"date": d.strftime('%Y-%m-%d'), "debit": float(v) / D}
                        for d, v in zip(ctx['date'], ctx['debit_L_s'])],
            "points": pts,
        }

        self._log(f"🔮 PRÉVISION RÉELLE ({titre}) à partir d'aujourd'hui ({aujourdhui.strftime(FORMAT_DATE_FR)}) :")
        for _, row in resultat.iterrows():
            self._log(f"  {row['date'].strftime(FORMAT_DATE_FR)} : {row['debit_predit_L_s'] / L_PAR_S_VERS_M3_PAR_S:.3f} m³/s")

        if afficher_graphique:
            # Contexte : le débit OBSERVÉ des 15 derniers jours (téléchargé à
            # l'instant), toujours disponible contrairement à df_final.
            contexte = df_eau_recent[df_eau_recent['date'] >= date_J0 - pd.Timedelta(days=15)]

            # Le point d'ancrage (dernier débit connu) est ajouté en tête pour
            # relier visuellement le passé à la prévision -- exactement comme
            # dans les tests sur le passé (tracer_prediction).
            intervalles_ancres = {
                niveau: ([debit_actuel] + bas, [debit_actuel] + haut)
                for niveau, (bas, haut) in intervalles.items()
            }
            titre_final = f"{titre} (Probabilités & Intervalles)" if intervalles else titre
            self._tracer_fan_chart(
                contexte=contexte,
                dates_futures=[date_J0] + dates_affichees,
                valeurs_pred=[debit_actuel] + valeurs_affichees,
                intervalles=intervalles_ancres,
                date_pivot=aujourdhui,
                titre=f"{titre_final} -- Prévision réelle depuis aujourd'hui ({aujourdhui.strftime(FORMAT_DATE_FR)})",
            )

        return resultat

    def predire_15_jours_reels(self, *args, **kwargs):
        """Ancien nom conservé pour compatibilité -- voir predire_futur()."""
        return self.predire_futur(*args, **kwargs)

    # ==========================================================================
    # 10. RÉSUMÉ
    # ==========================================================================
    def afficher_resume(self):
        """Affiche un résumé de l'état actuel de l'objet (station, zones,
        capteurs retenus, scores des modèles, erreurs réseau rencontrées)."""
        print("=" * 50)
        print(f"RIVIÈRE : {self.nom_station} (code {self.code_station})")
        print(f"Position : ({self.lat_station}, {self.lon_station})")
        if self.listes_de_points:
            nb_points_total = sum(len(z) for z in self.listes_de_points)
            print(f"Zones hydrographiques : {len(self.listes_de_points)} ({nb_points_total} points candidats)")
        if self.meilleur_individu is not None:
            print(f"Capteurs retenus par le GA : {len(self.meilleur_individu)}")
        if self.X_train2 is not None:
            print(f"Jeu d'entraînement : {self.X_train2.shape} | Jeu de test : {self.X_test2.shape}")
        if self.n_composantes_pca is not None:
            print(f"Composantes PCA retenues : {self.n_composantes_pca}")
        if self.modeles:
            print("\nModèles entraînés :")
            for nom, info in sorted(self.modeles.items(), key=lambda kv: kv[1]["score"], reverse=True):
                marqueur_hybride = " (+ incertitude)" if "hybride" in info else ""
                print(f"  - {nom:<22} R² = {info['score'] * 100:>6.2f} %{marqueur_hybride}")
        if self._data.erreurs_reseau:
            print(f"\n⚠️ {len(self._data.erreurs_reseau)} erreur(s) réseau rencontrée(s) durant les téléchargements.")
        print("=" * 50)


# ==============================================================================
# EXEMPLE D'UTILISATION (à coller/adapter dans une cellule de notebook)
# ==============================================================================
"""
riviere = River("W231401001")

riviere.previsualiser_bassin()                       # aperçu rapide
riviere.determiner_bassins_versants()                # grille complète

riviere.preparer_donnees_legeres(
    params_hourly=["snow_depth", "temperature_2m"],
    params_daily=["rain_sum"],
    start_fetch="2016-01-01", end_fetch="2020-12-31"
)

riviere.lancer_selection_genetique(
    taille_population=20, nombre_generations=50,
    params_hourly=["snow_depth", "temperature_2m"],
    params_daily=["rain_sum"],
)   # télécharge automatiquement les données complètes à la fin

riviere.appliquer_pca(seuil_energie=95)
riviere.entrainer_tous_les_modeles()

riviere.afficher_resume()
riviere.tracer_prediction("gradient_boosting", "2022-03-15")
"""

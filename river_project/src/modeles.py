"""
================================================================================
MODULE modeles.py
================================================================================
Modèles de prévision du débit, entraînés sur les X, Y de `features.Features`.

- `Modele` : un modèle PAR COLONNE de Y (une par échéance), chacun sur ses colonnes de X
  (`colonnes_par_cible`) : gradient boosting (scikit-learn, entraînés en parallèle, ici) ou
  réseaux de neurones (`"keras"` = TensorFlow, `"torch"` = PyTorch, module `reseaux`, entraînés
  l'un après l'autre ; sans `colonnes`, un seul réseau sort toutes les échéances).
  Hyperparamètres par défaut = ceux de river_level (réseaux : dropout et L2 relevés), tous
  modifiables ;
- crues : `ponderation_crue` (poids à l'entraînement), `seuils_crue`, `scores_crue` ;
- `choisir_parametres` : essaie une grille d'hyperparamètres sur des plis
  (X_train, Y_train, X_val, Y_val) et garde la meilleure combinaison. Une simple validation
  est un seul pli ; la validation croisée en donne plusieurs : même fonction pour les deux.
- CACHE DES MODÈLES (`entrainer`, `charger`, `resultats`, `contenu_cache`, `vider_cache`) :
  chaque modèle entraîné est enregistré dans data/cache/modeles/<station>/<clé>/ (dossier de la
  station : ``<code>_<identifiant RiverApp>``) avec sa fiche (recette de
  X et Y, test, réglages, scores) et ses prévisions du test ; même recette + même test + mêmes
  réglages = même clé = jamais réentraîné.
================================================================================
"""

import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from tqdm import tqdm

from . import config
from .reseaux import ENTRAINEURS, RESEAU, RESEAUX

CLASSES = {"gradient_boosting": HistGradientBoostingRegressor}
"""dict of str to type : modèles « un par échéance » (scikit-learn), par nom."""

PAR_DEFAUT = {
    "gradient_boosting": dict(random_state=42, max_depth=4, max_leaf_nodes=31, learning_rate=0.1,
                              max_iter=300, l2_regularization=10, min_samples_leaf=7,
                              early_stopping=True, validation_fraction=0.1, n_iter_no_change=15),
    "keras": RESEAU,
    "torch": RESEAU,
}
"""dict of str to dict : hyperparamètres par défaut de chaque modèle (ceux de river_level)."""


def _limite(duree_max):
    """
    Heure (secondes depuis 1970) à laquelle s'arrêter, ou None.

    Parameters
    ----------
    duree_max : str, float or None
        Durée au format pandas (``"30min"``, ``"2h"``) ou en secondes ; None = pas de limite.

    Returns
    -------
    float or None
    """
    if duree_max is None:
        return None
    secondes = pd.Timedelta(duree_max).total_seconds() if isinstance(duree_max, str) else float(duree_max)
    return time.time() + secondes


def _entrainer_un(estimateur, X, y, idx, limite=None, tranche=10, poids=None):
    """
    Entraîne un régresseur sur une colonne de Y (appelée en parallèle par `Modele.fit`).

    Avec une `limite`, le gradient boosting avance par tranches de `tranche` itérations
    (``warm_start``) et s'arrête à la première tranche finie après l'heure limite : on garde
    les arbres déjà construits.

    Parameters
    ----------
    estimateur : sklearn estimator
        Régresseur neuf (copie du modèle de base).
    X : numpy.ndarray
        Toutes les entrées (partagées entre les tâches, copiées une seule fois).
    y : numpy.ndarray
        Une colonne de Y.
    idx : numpy.ndarray of int
        Colonnes de `X` qui servent cette échéance.
    limite : float, optional
        Heure d'arrêt (cf. `_limite`) ; None = entraînement complet.
    tranche : int, default 10
        Itérations ajoutées entre deux regards sur l'heure.
    poids : numpy.ndarray, optional
        Poids de chaque ligne (cf. `ponderation_crue`) ; None = toutes pareilles.

    Returns
    -------
    estimateur : sklearn estimator
        Le régresseur entraîné.
    coupe : bool
        True si l'heure limite l'a arrêté avant la fin.
    """
    if limite is None:
        return estimateur.fit(X[:, idx], y, sample_weight=poids), False
    total, n = estimateur.max_iter, 0
    estimateur.set_params(warm_start=True)
    while True:
        n = min(total, n + tranche)
        estimateur.set_params(max_iter=n).fit(X[:, idx], y, sample_weight=poids)
        if estimateur.n_iter_ < n or n >= total:                   # arrêt anticipé ou fini
            return estimateur.set_params(max_iter=total, warm_start=False), False
        if time.time() > limite:
            return estimateur.set_params(max_iter=total, warm_start=False), True



class Modele:
    """
    Un régresseur par échéance de Y, avec des hyperparamètres modifiables.

    Parameters
    ----------
    nom : {'gradient_boosting', 'keras', 'torch'}, default 'gradient_boosting'
        Modèle à utiliser (clé de `CLASSES` ou de `ENTRAINEURS`).
    **params
        Hyperparamètres qui remplacent ceux de `PAR_DEFAUT` (ex. ``max_depth=4``).

    Attributes
    ----------
    nom : str
        Nom du modèle.
    params : dict
        Hyperparamètres utilisés (défauts + changements).
    estimateurs : list
        Gradient boosting : un régresseur entraîné par colonne de Y ; réseau : une seule
        fonction de prévision pour toutes les colonnes (vide avant `fit`).
    colonnes : list of str
        Noms des colonnes de Y vues à l'entraînement.
    entrees : list of list of str
        Pour chaque échéance, les colonnes de X qui lui servent.
    coupe : bool
        True si l'entraînement a été arrêté par `duree_max` avant la fin.

    See Also
    --------
    choisir_parametres : choisit les hyperparamètres sur une validation.
    features.Features : fournit X et Y.

    Notes
    -----
    Gradient boosting : 31 feuilles, 300 itérations au plus, régularisation L2 10, 7 exemples
    par feuille au moins (river_level), profondeur 4 et pas d'apprentissage 0,1 (choisis par
    validation croisée sur La Clapière), arrêt anticipé sur 10 % de l'entraînement après 15
    itérations sans progrès. Ces 10 % sont tirés
    au hasard parmi les heures : des heures voisines se ressemblent, l'arrêt anticipé est donc
    un peu tardif (il ne touche pas la validation ni le test).

    Examples
    --------
    >>> X_train, Y_train, X_test, Y_test = F.decouper(test=[2023, 2024])
    >>> m = Modele(max_depth=6).fit(X_train, Y_train)
    >>> m.score(X_test, Y_test)                 # R² par échéance
    >>> m.predict(X_test)                       # même colonnes que Y
    >>> Modele("torch", couches=(128, 128)).fit(X_train, Y_train).score(X_test, Y_test)
    """

    def __init__(self, nom="gradient_boosting", **params):
        self.nom = nom
        self.params = {**PAR_DEFAUT[nom], **params}
        self.estimateurs = []
        self.colonnes = []
        self.entrees = []
        self.coupe = False

    def fit(self, X, Y, barre=True, colonnes=None, duree_max=None, poids=None):
        """
        Entraîne le modèle : gradient boosting = un régresseur par colonne de Y, en parallèle
        sur tous les cœurs ; réseau = un seul réseau pour toutes les colonnes.

        Parameters
        ----------
        X : pandas.DataFrame
            Entrées d'entraînement.
        Y : pandas.DataFrame
            Cibles d'entraînement, une colonne par échéance.
        barre : bool, default True
            Afficher une barre d'avancement (une case par échéance terminée).
        colonnes : dict of str to list of str, optional
            Pour chaque colonne de Y, les colonnes de X qui lui servent
            (``features.Features.colonnes_par_cible``). Par défaut, tout X pour chaque échéance.
            Réseaux : avec `colonnes`, UN RÉSEAU PAR ÉCHÉANCE sur ses colonnes (comme le gradient
            boosting : la première échéance ne voit pas la météo de la dernière), entraînés l'un
            après l'autre ; sans `colonnes`, un seul réseau qui sort toutes les échéances.
        duree_max : str or float, optional
            Temps maximal d'entraînement (``"30min"``, ``"2h"``, ou secondes). À l'heure dite on
            s'arrête et on garde ce qu'on a : les arbres déjà construits (gradient boosting), les
            meilleurs poids vus en validation (réseaux). `coupe` dit si c'est arrivé. Pour le
            gradient boosting, l'heure vaut pour toutes les échéances (celles qui attendent un
            cœur libre ont moins de temps). Réseaux par échéance : le temps restant est partagé
            équitablement entre les échéances qui restent.
        poids : numpy.ndarray, optional
            Poids des lignes, une colonne par échéance (cf. `ponderation_crue`) ; un réseau unique pour
            toutes les échéances prend la moyenne des colonnes. None = toutes les lignes pareilles.

        Returns
        -------
        Modele
            Lui-même, entraîné (permet ``Modele().fit(X, Y).score(...)``).
        """
        limite = _limite(duree_max)
        x, y = X.to_numpy(), Y.to_numpy()
        self.colonnes = list(Y.columns)
        if self.nom in ENTRAINEURS and colonnes is None:            # un réseau, toutes les échéances
            self.entrees = [list(X.columns)] * y.shape[1]
            w = None if poids is None else poids.mean(axis=1)
            self.estimateurs = [ENTRAINEURS[self.nom](x, y, barre=barre, limite=limite, poids=w, **self.params)]
            self.coupe = self.estimateurs[0].coupe
            return self
        if self.nom in ENTRAINEURS:                                 # un réseau par échéance
            self.entrees = [list(colonnes[c]) for c in Y.columns]
            self.estimateurs, n = [], y.shape[1]
            for k in tqdm(range(n), desc=f"{self.nom} : échéances", disable=not barre):
                lim = None if limite is None else time.time() + max(0.0, limite - time.time()) / (n - k)
                self.estimateurs.append(ENTRAINEURS[self.nom](X[self.entrees[k]].to_numpy(), y[:, [k]], barre=False,
                                                              limite=lim, poids=None if poids is None else poids[:, k],
                                                              **self.params))
            self.coupe = any(e.coupe for e in self.estimateurs)
            return self
        base = CLASSES[self.nom](**self.params)
        self.entrees = [list(X.columns) if colonnes is None else list(colonnes[c]) for c in Y.columns]
        position = {c: i for i, c in enumerate(X.columns)}
        idx = [np.array([position[c] for c in cols]) for cols in self.entrees]
        travaux = Parallel(n_jobs=-1, return_as="generator")(
            delayed(_entrainer_un)(clone(base), x, y[:, k], idx[k], limite, 10, None if poids is None else poids[:, k])
            for k in range(y.shape[1]))
        sortie = list(tqdm(travaux, total=y.shape[1], desc=f"{self.nom} : échéances", disable=not barre))
        self.estimateurs = [e for e, _ in sortie]
        self.coupe = any(c for _, c in sortie)
        return self

    def predict(self, X):
        """
        Prévision de chaque échéance.

        Parameters
        ----------
        X : pandas.DataFrame
            Entrées, mêmes colonnes qu'à l'entraînement.

        Returns
        -------
        pandas.DataFrame
            Même index que `X`, une colonne par échéance (noms de Y).
        """
        if self.nom in ENTRAINEURS and len(self.estimateurs) == 1:
            valeurs = self.estimateurs[0].predict(X[self.entrees[0]].to_numpy())
        elif self.nom in ENTRAINEURS:
            valeurs = np.column_stack([e.predict(X[cols].to_numpy()).ravel() for e, cols in zip(self.estimateurs, self.entrees)])
        else:
            valeurs = np.column_stack([e.predict(X[cols].to_numpy()) for e, cols in zip(self.estimateurs, self.entrees)])
        return pd.DataFrame(valeurs, index=X.index, columns=self.colonnes)

    def score(self, X, Y):
        """
        R² de chaque échéance.

        Parameters
        ----------
        X : pandas.DataFrame
            Entrées.
        Y : pandas.DataFrame
            Vrais débits, mêmes colonnes qu'à l'entraînement.

        Returns
        -------
        pandas.Series
            R² par échéance (1 = parfait, 0 = pas mieux que la moyenne), index = colonnes de Y.
        """
        return pd.Series(r2_score(Y.to_numpy(float), self.predict(X).to_numpy(float), multioutput="raw_values"),
                         index=self.colonnes, name="r2")


    def sauver(self, dossier):
        """
        Écrit le modèle entraîné dans `dossier` (créé s'il manque).

        ``modele.json`` (nom, réglages, colonnes de Y, colonnes de X par échéance) + le modèle :
        ``estimateurs.joblib`` (gradient boosting), ``reseau_000/``, ``reseau_001/``... (un dossier
        par réseau : ``reseau.keras`` ou ``reseau.pt`` + ``echelles.joblib``).

        Parameters
        ----------
        dossier : str or Path
            Dossier de destination.
        """
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        if self.nom in RESEAUX:
            for k, e in enumerate(self.estimateurs):
                (dossier / f"reseau_{k:03d}").mkdir(exist_ok=True)
                e.sauver(dossier / f"reseau_{k:03d}")
        else:
            joblib.dump(self.estimateurs, dossier / "estimateurs.joblib")
        (dossier / "modele.json").write_text(json.dumps({"nom": self.nom, "params": self.params,
                                                         "colonnes": self.colonnes, "entrees": self.entrees}))

    @classmethod
    def charger(cls, dossier):
        """
        Relit un modèle écrit par `sauver`.

        Parameters
        ----------
        dossier : str or Path
            Dossier du modèle.

        Returns
        -------
        Modele
            Prêt à prévoir (``predict``, ``score``).
        """
        dossier = Path(dossier)
        d = json.loads((dossier / "modele.json").read_text())
        m = cls(d["nom"], **d["params"])
        m.colonnes, m.entrees = d["colonnes"], d["entrees"]
        if m.nom not in RESEAUX:
            m.estimateurs = joblib.load(dossier / "estimateurs.joblib")
        elif (dossier / "echelles.joblib").exists():                 # ancien format : un seul réseau, à la racine
            m.estimateurs = [RESEAUX[m.nom].charger(dossier)]
        else:
            m.estimateurs = [RESEAUX[m.nom].charger(d) for d in sorted(dossier.glob("reseau_*"))]
        return m


# ==============================================================================
# CRUES : POIDS À L'ENTRAÎNEMENT ET SCORES DÉDIÉS
# ==============================================================================

def ponderation_crue(Y, actuel, regle):
    """
    Poids de chaque heure d'entraînement, pour insister sur les crues (une colonne par échéance).

    Parameters
    ----------
    Y : pandas.DataFrame
        Débits à prévoir (une colonne par échéance).
    actuel : pandas.Series
        Débit à l'instant t, même index (``Features.debit_actuel``).
    regle : dict
        - ``{"type": "montee", "facteur": 1}`` : poids = 1 + facteur × (montée du débit entre t
          et l'échéance, si elle monte) / débit médian. Insiste sur ce qu'il faut ANTICIPER :
          rien de plus quand le débit reste stable ou baisse (étiage, plateau de fonte, décrue).
        - ``{"type": "niveau", "seuil": 0.9, "facteur": 5}`` : × facteur quand le débit à
          prévoir dépasse son quantile `seuil` (0.9 = les 10 % les plus forts).
        - ``{"type": "niveau", "alpha": 1}`` : poids = (débit à prévoir / débit médian) ^ alpha.

    Returns
    -------
    numpy.ndarray
        Lignes × échéances, moyenne 1 par colonne (l'échelle de l'apprentissage ne change pas).

    Raises
    ------
    ValueError
        Si la règle est inconnue.
    """
    y, q = Y.to_numpy(float), actuel.to_numpy(float)[:, None]
    mediane = np.median(q)
    if regle.get("type") == "montee":
        w = 1 + regle.get("facteur", 1) * np.maximum(y - q, 0) / mediane
    elif regle.get("type") == "niveau" and "alpha" in regle:
        w = (np.maximum(y, 0) / np.median(y, axis=0)) ** regle["alpha"]
    elif regle.get("type") == "niveau":
        w = np.where(y > np.quantile(y, regle.get("seuil", 0.9), axis=0), regle.get("facteur", 5), 1.0)
    else:
        raise ValueError(f"poids_crue inconnu : {regle}")
    return (w / w.mean(axis=0)).astype("float32")


def seuils_crue(Y, actuel, quantile=0.9):
    """
    Seuils qui définissent les crues et les montées, appris sur l'ENTRAÎNEMENT.

    Parameters
    ----------
    Y : pandas.DataFrame
        Débits à prévoir de l'entraînement.
    actuel : pandas.Series
        Débit à t, même index.
    quantile : float, default 0.9
        0.9 = les 10 % les plus forts.

    Returns
    -------
    dict
        ``crue`` : débit (m³/s) au-dessus duquel une échéance est « en crue » ; ``montee`` :
        montée entre t et l'échéance (m³/s) au-dessus de laquelle c'est « une montée » ; par
        colonne de Y.
    """
    montee = Y.sub(actuel, axis=0)
    return {"crue": Y.quantile(quantile).astype(float).to_dict(), "montee": montee.quantile(quantile).astype(float).to_dict()}


def scores_crue(reel, prevu, actuel, seuils, ecart_pics="72h", tolerance="12h"):
    """
    Scores réservés aux crues, échéance par échéance.

    Parameters
    ----------
    reel, prevu : pandas.DataFrame
        Débits réels et prévus du test (une colonne par échéance).
    actuel : pandas.Series
        Débit à t, même index.
    seuils : dict
        Sortie de `seuils_crue` (calculés sur l'entraînement).
    ecart_pics : str, default "72h"
        Deux pics de crue sont distincts s'ils sont séparés d'au moins cette durée.
    tolerance : str, default "12h"
        Pour un pic réel, on prend le maximum prévu à ± cette durée (un pic prévu un peu en
        avance ou en retard compte quand même).

    Returns
    -------
    pandas.DataFrame
        Une ligne par échéance : ``r2_crue``, ``erreur_crue`` (erreur absolue moyenne, m³/s) sur
        les heures de crue ; ``r2_montee``, ``erreur_montee`` sur les heures de montée ;
        ``pics`` (nombre), ``erreur_pic_pct`` (écart moyen, en % du pic réel) et ``biais_pic_pct``
        (négatif = pics sous-estimés).
    """
    from scipy.signal import find_peaks
    lignes = {}
    pas = pd.Timedelta("1h")
    for col in reel.columns:
        r, p = reel[col].to_numpy(float), prevu[col].to_numpy(float)
        en_crue = r > seuils["crue"][col]
        en_montee = (r - actuel.to_numpy(float)) > seuils["montee"][col]
        d = {}
        for nom, m in (("crue", en_crue), ("montee", en_montee)):
            d[f"r2_{nom}"] = float(r2_score(r[m], p[m])) if m.sum() > 1 else np.nan
            d[f"erreur_{nom}"] = float(np.abs(r[m] - p[m]).mean()) if m.any() else np.nan
        serie = reel[col].asfreq("1h")                               # trous comptés comme des heures vides
        pics, _ = find_peaks(serie.fillna(-np.inf).to_numpy(), height=seuils["crue"][col],
                             distance=int(pd.Timedelta(ecart_pics) / pas))
        ecarts = []
        for i in pics:
            t0 = serie.index[i]
            fenetre = prevu[col].loc[t0 - pd.Timedelta(tolerance):t0 + pd.Timedelta(tolerance)]
            if len(fenetre):
                ecarts.append((fenetre.max() - serie.iloc[i]) / serie.iloc[i])
        d["pics"] = len(ecarts)
        d["erreur_pic_pct"] = float(100 * np.mean(np.abs(ecarts))) if ecarts else np.nan
        d["biais_pic_pct"] = float(100 * np.mean(ecarts)) if ecarts else np.nan
        lignes[col] = d
    return pd.DataFrame(lignes).T


# ==============================================================================
# CACHE DES MODÈLES ENTRAÎNÉS
# ==============================================================================

def dossier_station(id_station):
    """
    Dossier des modèles d'une station : ``<code>_<identifiant RiverApp>`` (lisible dans le Finder).

    Le code chez le fournisseur est lu dans les listes de stations en cache (``config.CACHE_HYDRO``),
    sans appel à RiverApp ; sans code, le dossier porte le seul identifiant.

    Parameters
    ----------
    id_station : str
        Identifiant RiverApp.

    Returns
    -------
    Path
    """
    for f in sorted(config.CACHE_HYDRO.glob("stations_*.parquet")):
        codes = pd.read_parquet(f, columns=["code"])["code"]
        if id_station in codes.index and isinstance(codes[id_station], str) and codes[id_station]:
            return config.CACHE_MODELES / f"{codes[id_station]}_{id_station}"
    return config.CACHE_MODELES / id_station


def _dossier(cle):
    """
    Dossier d'un modèle enregistré, quelle que soit sa station.

    Raises
    ------
    KeyError
        Si aucun modèle n'a cette clé.
    """
    for d in config.CACHE_MODELES.glob(f"*/{cle}"):
        if (d / "fiche.json").exists():
            return d
    raise KeyError(f"aucun modèle de clé {cle} dans {config.CACHE_MODELES}")


def cle_modele(recette, test, nom, params):
    """
    Identité d'un modèle : mêmes données, même test, même modèle, mêmes réglages = même clé.

    Parameters
    ----------
    recette : dict
        ``Features.recette()``.
    test : list of int
        Années de test.
    nom : str
        Nom du modèle.
    params : dict
        Réglages COMPLETS (défauts compris).

    Returns
    -------
    str
        12 caractères (empreinte SHA-1), nom du dossier du modèle (dans celui de sa station).
    """
    texte = json.dumps({"recette": recette, "test": sorted(int(a) for a in test), "nom": nom, "params": params},
                       sort_keys=True, default=str)
    return hashlib.sha1(texte.encode()).hexdigest()[:12]


def entrainer(F, test, nom="gradient_boosting", etiquette=None, barre=True, duree_max=None, poids_crue=None,
              un_reseau_par_echeance=False, **params):
    """
    Entraîne, note et enregistre un modèle — ou le relit s'il a déjà été entraîné.

    Même recette de X et Y, mêmes années de test, même modèle et mêmes réglages qu'un modèle
    du cache : il est relu, rien n'est réentraîné (`F` n'est alors même pas construit). Sinon :
    `F.construire` si besoin, entraînement sur toutes les années hors test, note sur le test,
    enregistrement dans ``dossier_station(station) / <clé>``.

    Parameters
    ----------
    F : features.Features
        Recette de X et Y (construite ou non).
    test : list of int
        Années de test.
    nom : {'gradient_boosting', 'keras', 'torch'}, default 'gradient_boosting'
        Modèle.
    etiquette : str, optional
        Nom lisible, pour s'y retrouver dans `contenu_cache` (ne compte pas dans l'identité).
    barre : bool, default True
        Barres d'avancement.
    duree_max : str or float, optional
        Temps maximal d'entraînement (``"30min"``, ``"2h"``, ou secondes), cf. `Modele.fit` : à
        l'heure dite on garde ce qu'on a. Fait partie de l'identité du modèle (un modèle coupé
        n'est pas le même qu'un modèle complet) ; ``coupe`` dans la fiche dit s'il a été coupé.
    poids_crue : dict, optional
        Insister sur les crues à l'entraînement (cf. `ponderation_crue`), ex.
        ``{"type": "montee", "facteur": 1}``. Fait partie de l'identité du modèle. Les scores de
        crue (`scores_crue`) sont calculés pour TOUS les modèles, avec ou sans poids.
    un_reseau_par_echeance : bool, default False
        Réseaux seulement. False : un seul réseau qui sort toutes les échéances, sur tout X.
        True : un réseau par échéance, sur ses colonnes (`F.colonnes_par_cible`). Mesuré le
        2026-10-13 à La Clapière : même R² moyen (à 0,01 près), 12 à 15 fois plus long ; d'où le
        défaut. Fait partie de l'identité du modèle.
    **params
        Réglages qui remplacent ceux de `PAR_DEFAUT`.

    Returns
    -------
    Modele
        Avec en plus ``cle`` et ``fiche`` (cf. `resultats`).

    See Also
    --------
    resultats : relire seulement les scores et les prévisions du test, sans le modèle.
    contenu_cache, vider_cache : lister / supprimer les modèles enregistrés.

    Examples
    --------
    >>> m = entrainer(F, test=[2023, 2024], nom="torch", etiquette="torch 7 carreaux")
    >>> m = entrainer(F, test=[2023, 2024], nom="keras", duree_max="20min")     # au plus 20 minutes
    >>> m = entrainer(F, test=[2023, 2024], nom="keras", un_reseau_par_echeance=True)    # un réseau par échéance
    >>> r = resultats(m.cle)                 # plus tard, même après redémarrage du noyau
    """
    params = {**PAR_DEFAUT[nom], **params}
    recette = F.recette()
    identite = {**params, **({"duree_max": duree_max} if duree_max is not None else {}),
                **({"poids_crue": poids_crue} if poids_crue is not None else {}),
                **({"un_reseau_par_echeance": True} if nom in ENTRAINEURS and un_reseau_par_echeance else {})}
    cle = cle_modele(recette, test, nom, identite)
    dossier = dossier_station(recette["id_station"]) / cle
    if (dossier / "fiche.json").exists():
        print(f"{nom} déjà entraîné (clé {cle}) : relu depuis le cache")
        return charger(cle)
    if F.X.shape[0] == 0:
        F.construire()
    X_train, Y_train, X_test, Y_test = F.decouper(test)
    actuel_train, actuel_test = F.debit_actuel.loc[X_train.index], F.debit_actuel.loc[X_test.index]
    w = None if poids_crue is None else ponderation_crue(Y_train, actuel_train, poids_crue)
    debut = time.time()
    colonnes = None if nom in ENTRAINEURS and not un_reseau_par_echeance else F.colonnes_par_cible
    m = Modele(nom, **params).fit(X_train, Y_train, barre=barre, colonnes=colonnes, duree_max=duree_max, poids=w)
    duree = time.time() - debut
    prevu = m.predict(X_test)
    r2 = r2_score(Y_test.to_numpy(float), prevu.to_numpy(float), multioutput="raw_values")
    seuils = seuils_crue(Y_train, actuel_train)
    crue = scores_crue(Y_test, prevu, actuel_test, seuils)

    provisoire = dossier.parent / f"{cle}.en_cours"                 # rien n'est « en cache » avant la fiche
    shutil.rmtree(provisoire, ignore_errors=True)
    m.sauver(provisoire)
    Y_test.astype("float32").to_parquet(provisoire / "test_reel.parquet")
    prevu.astype("float32").to_parquet(provisoire / "test_prevu.parquet")
    actuel_test.to_frame().to_parquet(provisoire / "test_debit_actuel.parquet")
    fiche = {"cle": cle, "etiquette": etiquette, "nom": nom, "params": params, "recette": recette,
             "test": sorted(int(a) for a in test), "date": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
             "duree_s": round(duree, 1), "duree_max": duree_max, "coupe": bool(m.coupe),
             "lignes_train": len(X_train), "colonnes_x": X_train.shape[1],
             "poids_crue": poids_crue, "r2_moyen": float(np.mean(r2)), "r2": dict(zip(Y_test.columns, map(float, r2))),
             "r2_crue_moyen": float(crue["r2_crue"].mean()), "erreur_pic_pct_moyen": float(crue["erreur_pic_pct"].mean()),
             "seuils_crue": seuils, "crue": crue.to_dict(orient="index")}
    (provisoire / "fiche.json").write_text(json.dumps(fiche, ensure_ascii=False, indent=1, default=str))
    shutil.rmtree(dossier, ignore_errors=True)
    provisoire.rename(dossier)
    m.cle, m.fiche = cle, fiche
    if m.coupe:
        print(f"{nom} : arrêté par duree_max={duree_max} (meilleur état gardé), R² moyen {fiche['r2_moyen']:.3f}")
    return m


def charger(cle):
    """
    Relit un modèle enregistré (avec sa fiche).

    Parameters
    ----------
    cle : str
        Clé du modèle (colonne ``cle`` de `contenu_cache`).

    Returns
    -------
    Modele
        Avec ``cle`` et ``fiche``.
    """
    dossier = _dossier(cle)
    m = Modele.charger(dossier)
    m.cle, m.fiche = cle, json.loads((dossier / "fiche.json").read_text())
    return m


def resultats(cle):
    """
    Scores et prévisions du test d'un modèle enregistré, SANS charger le modèle.

    Léger et rapide (ni X, ni TensorFlow, ni PyTorch) : de quoi tracer après un redémarrage.

    Parameters
    ----------
    cle : str
        Clé du modèle.

    Returns
    -------
    dict
        ``fiche`` (dict), ``r2`` (Series par échéance), ``crue`` (scores de crue par échéance,
        cf. `scores_crue` ; None pour les anciens modèles), ``reel`` et ``prevu`` (DataFrame :
        index = heure t du test, une colonne par échéance).

    Examples
    --------
    >>> r = resultats("3f2a9c1b7d4e")
    >>> r["r2"].plot()
    >>> r["reel"]["debit_8h_t+24h"].plot(); r["prevu"]["debit_8h_t+24h"].plot()
    """
    dossier = _dossier(cle)
    fiche = json.loads((dossier / "fiche.json").read_text())
    return {"fiche": fiche, "r2": pd.Series(fiche["r2"], name="r2"),
            "crue": pd.DataFrame(fiche["crue"]).T if "crue" in fiche else None,
            "reel": pd.read_parquet(dossier / "test_reel.parquet"),
            "prevu": pd.read_parquet(dossier / "test_prevu.parquet")}


def contenu_cache(station=None):
    """
    Les modèles enregistrés, du plus récent au plus ancien.

    Parameters
    ----------
    station : str, optional
        Identifiant RiverApp : seulement les modèles de cette station. Par défaut, tous.

    Returns
    -------
    pandas.DataFrame
        Une ligne par modèle : ``cle``, ``etiquette``, ``nom``, ``coupe`` (arrêté par
        `duree_max`), ``station`` (identifiant RiverApp), ``dossier`` (celui de la station), ``periode``,
        ``test``, ``colonnes_x``, ``echeances``, ``r2_moyen``, ``r2_crue`` et ``erreur_pic_pct``
        (moyennes sur les échéances, cf. `scores_crue`), ``poids_crue``, ``duree_min``, ``date``, ``mo``
        (place sur le disque).

    See Also
    --------
    vider_cache : supprimer des modèles.
    """
    lignes = []
    for f in config.CACHE_MODELES.glob("*/*/fiche.json"):
        d = json.loads(f.read_text())
        if station is not None and d["recette"]["id_station"] != station:
            continue
        taille = sum(x.stat().st_size for x in f.parent.iterdir())
        lignes.append({"cle": d["cle"], "etiquette": d["etiquette"], "nom": d["nom"], "coupe": d.get("coupe", False),
                       "station": d["recette"]["id_station"], "dossier": f.parent.parent.name,
                       "periode": f"{d['recette']['debut']} → {d['recette']['fin']}", "test": d["test"],
                       "colonnes_x": d["colonnes_x"], "echeances": len(d["r2"]), "r2_moyen": round(d["r2_moyen"], 4),
                       "r2_crue": round(d.get("r2_crue_moyen", np.nan), 4),
                       "erreur_pic_pct": round(d.get("erreur_pic_pct_moyen", np.nan), 1),
                       "poids_crue": d.get("poids_crue"),
                       "duree_min": round(d["duree_s"] / 60, 1), "date": d["date"], "mo": round(taille / 1e6, 1)})
    colonnes = ["cle", "etiquette", "nom", "coupe", "station", "dossier", "periode", "test", "colonnes_x", "echeances", "r2_moyen",
                "r2_crue", "erreur_pic_pct", "poids_crue",
                "duree_min", "date", "mo"]
    return pd.DataFrame(lignes, columns=colonnes).sort_values("date", ascending=False).reset_index(drop=True)


def vider_cache(cles=None):
    """
    Supprime des modèles enregistrés (tous si `cles` est None).

    Parameters
    ----------
    cles : str or list of str, optional
        Clés à supprimer. Par défaut, TOUS les modèles.

    Returns
    -------
    int
        Nombre de modèles supprimés.

    Examples
    --------
    >>> c = contenu_cache()
    >>> vider_cache(c.loc[c["r2_moyen"] < 0.8, "cle"])       # les moins bons
    >>> vider_cache(c["cle"][10:])                           # garder les 10 plus récents
    >>> vider_cache("3f2a9c1b7d4e")                           # un seul
    """
    if cles is None:
        cles = [f.parent.name for f in config.CACHE_MODELES.glob("*/*/fiche.json")]
    cles = [cles] if isinstance(cles, str) else list(cles)
    n = 0
    for cle in cles:
        try:
            shutil.rmtree(_dossier(cle))
            n += 1
        except KeyError:
            pass
    return n


def choisir_parametres(plis, grille, nom="gradient_boosting", colonnes=None, **fixes):
    """
    Choisit les hyperparamètres : essaie toute la grille sur chaque pli, garde le meilleur.

    Chaque combinaison est entraînée sur ``X_train`` et notée sur ``X_val`` de chaque pli ;
    la note est le R² moyen sur toutes les échéances, puis sur tous les plis.

    Parameters
    ----------
    plis : iterable of tuple
        Plis ``(X_train, Y_train, X_val, Y_val)``. Une simple validation :
        ``[F.decouper(test, val)[:4]]`` ; une validation croisée :
        ``F.validation_croisee(test, n_plis)``.
    grille : dict of str to list
        Valeurs à essayer par hyperparamètre, ex. ``{"max_depth": [4, 8], "learning_rate": [0.05, 0.1]}``
        (toutes les combinaisons sont essayées).
    nom : {'gradient_boosting', 'keras', 'torch'}, default 'gradient_boosting'
        Modèle.
    colonnes : dict of str to list of str, optional
        Colonnes de X par échéance (``F.colonnes_par_cible``), transmis à `Modele.fit`.
    **fixes
        Hyperparamètres fixés pour tous les essais (ex. ``max_iter=100`` pour aller plus vite).

    Returns
    -------
    meilleurs : dict
        Hyperparamètres gagnants (+ `fixes`), à passer à ``Modele(nom, **meilleurs)``.
    tableau : pandas.DataFrame
        Une ligne par combinaison : ses valeurs, ``r2_moyen`` et ``r2_par_pli`` ; triée du
        meilleur au moins bon.

    See Also
    --------
    features.Features.validation_croisee : fournit les plis.
    Modele : le modèle à réentraîner ensuite avec les meilleurs hyperparamètres.

    Notes
    -----
    Coût = nombre de combinaisons × nombre de plis × nombre d'échéances entraînements :
    commencer petit. Une barre d'avancement compte les essais (combinaison × pli) ; son total
    n'est connu que si `plis` est une liste (``list(F.validation_croisee(...))``), au prix de
    garder tous les plis en mémoire. Les plis sont parcourus une seule fois (un générateur suffit, rien n'est
    gardé en mémoire d'un pli à l'autre).

    Après le choix, réentraîner sur TOUTES les années hors test (validation comprise), puis
    noter une seule fois sur le test.

    Examples
    --------
    >>> test = [2023, 2024]
    >>> meilleurs, tableau = choisir_parametres(F.validation_croisee(test, n_plis=5),
    ...                                         {"max_depth": [4, 8], "learning_rate": [0.05, 0.1]})
    >>> X_train, Y_train, X_test, Y_test = F.decouper(test)
    >>> Modele(**meilleurs).fit(X_train, Y_train).score(X_test, Y_test)
    """
    combinaisons = [dict(zip(grille, valeurs)) for valeurs in itertools.product(*grille.values())]
    notes = [[] for _ in combinaisons]
    total = len(combinaisons) * len(plis) if hasattr(plis, "__len__") else None
    with tqdm(total=total, desc="essais (combinaison × pli)") as avance:
        for X_train, Y_train, X_val, Y_val in plis:
            for i, c in enumerate(combinaisons):
                m = Modele(nom, **fixes, **c).fit(X_train, Y_train, barre=False, colonnes=colonnes)
                notes[i].append(float(m.score(X_val, Y_val).mean()))
                avance.update()
    tableau = pd.DataFrame(combinaisons).assign(r2_moyen=[np.mean(n) for n in notes],
                                                r2_par_pli=[np.round(n, 4).tolist() for n in notes])
    tableau = tableau.sort_values("r2_moyen", ascending=False)
    meilleurs = {**fixes, **combinaisons[tableau.index[0]]}       # valeurs d'origine (None reste None)
    tableau = tableau.reset_index(drop=True)
    return meilleurs, tableau

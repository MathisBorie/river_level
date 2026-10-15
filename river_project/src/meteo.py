"""
================================================================================
MODULE meteo.py
================================================================================
Accès à Open-Meteo (archive = données passées), PAR CARREAU de la grille O1280
(cf. bassins.carreaux_amont). Toute requête passe par la classe `Meteo` :

- compteur : coût de chaque requête calculé AVANT l'envoi (`cout_appel`, règle mesurée),
  journal de chaque requête envoyée (data/cache/meteo/journal_appels.csv) -> le compte
  survit aux redémarrages ; attente si la limite minute / heure serait dépassée, arrêt
  (`QuotaEpuise`) pour le jour / le mois ;
- cache par carreau (data/cache/meteo/carreaux/) + table de couverture : seul ce qui
  manque est téléchargé, un carreau partagé par deux stations n'est payé qu'une fois ;
- requêtes groupées (plusieurs carreaux, périodes découpées par an), centre du carreau
  demandé en mode « nearest » et vérifié au retour ;
- deux API : `"archive"` (météo observée, « best_match » d'Open-Meteo : ERA5 / ERA5-Land avant 2017, ECMWF IFS 9 km depuis — voir doc/open_meteo) et `"previous_runs"` (VRAIES
  prévisions passées : `<variable>_previous_dayN` = valeur prévue N jours avant, N = 1..7), chacune
  avec son cache ; UN SEUL compteur pour les deux (constaté le 2026-10-10 : Open-Meteo refuse les
  prévisions passées quand la limite du jour est atteinte par l'archive).
Heures en UTC.
================================================================================
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from . import config

ABONNEMENTS = {
    # limites par fenêtre glissante (None = aucune) ; historique = accès à l'archive
    "gratuit": dict(minute=600, heure=5_000, jour=10_000, mois=300_000, historique=True,
                    url="https://archive-api.open-meteo.com/v1/archive"),
    "standard": dict(minute=None, heure=None, jour=None, mois=1_000_000, historique=False, url=None),
    # adresse de l'archive abonnés : donnée par Open-Meteo avec la clé (non publiée)
    "professional": dict(minute=None, heure=None, jour=None, mois=5_000_000, historique=True, url=None),
}
"""dict : limites de chaque abonnement Open-Meteo, en appels par fenêtre glissante.

Clés ``minute``, ``heure``, ``jour``, ``mois`` (None = pas de limite), ``historique``
(accès à l'archive ou non) et ``url`` (adresse de l'archive ; None = inconnue tant
qu'Open-Meteo ne l'a pas donnée avec la clé).
"""
FENETRES = {"minute": 60, "heure": 3600, "jour": 86400, "mois": 30 * 86400}
"""dict of str to int : durée de chaque fenêtre de quota, en secondes (mois = 30 jours)."""
MAX_POINTS = 20       # par requête : 20 × 26 = 520 appels pour 10 variables sur 1 an, réponse de ~15 Mo max
"""int : nombre maximal de carreaux par requête (20 × 26 = 520 appels pour 10 variables sur 1 an)."""
MAX_JOURS = 366       # périodes découpées par an : une coupure ne fait perdre qu'un morceau
"""int : durée maximale d'une requête en jours ; une coupure ne fait perdre qu'un an au plus."""
RESOLUTIONS = ("horaire", "journalier")
"""tuple of str : les deux résolutions du cache (variables horaires et journalières)."""
URL_PREVIOUS_RUNS = "https://previous-runs-api.open-meteo.com/v1/forecast"
"""str : API des prévisions passées (même quota gratuit que l'archive, partagé avec elle)."""
MEME_GRILLE = (None, "ecmwf_ifs")
"""tuple : modèles servis sur la grille O1280 de nos carreaux (centre vérifié au retour)."""


def cout_appel(nb_points, nb_variables, nb_jours):
    """
    Coût en appels Open-Meteo d'une requête archive, calculé avant de l'envoyer.

    Règle MESURÉE le 2026-10-06 sur la limite par minute (la formule de la doc d'Open-Meteo
    est fausse sur les longues périodes) : un appel = 10 variables pendant 14 jours pour un
    point, avec au minimum 1 appel par point. Variables horaires et journalières coûtent
    pareil.

    Parameters
    ----------
    nb_points : int
        Nombre de carreaux (points) demandés dans la requête.
    nb_variables : int
        Nombre total de variables, horaires et journalières confondues.
    nb_jours : int
        Nombre de jours de la période, bornes incluses.

    Returns
    -------
    float
        ``nb_points × max(1, nb_variables / 10, nb_variables × nb_jours / 140)``.

    See Also
    --------
    Meteo.cout_prevu : coût total d'une demande, en ne comptant que ce qui manque au cache.

    Notes
    -----
    Mesures qui ont servi à trouver la règle : 1 variable 1 an = 2,6 ; 20 variables 1 an = 52 ;
    5 points 1 variable 1 an = 13 ; 9 horaires + 1 journalière 1 an = 26 ;
    20 variables 1 jour = 2. Détails dans ``doc/open_meteo/open_meteo.md``.

    Examples
    --------
    >>> cout_appel(1, 10, 365)        # 10 variables, 1 an, 1 carreau
    26.071428571428573
    >>> cout_appel(20, 10, 366)       # une requête pleine
    522.8571428571429
    >>> cout_appel(5, 1, 1)           # requête courte : 1 appel minimum par point
    5.0
    """
    return nb_points * max(1.0, nb_variables / 10, nb_variables * nb_jours / 140)


def _fusionner(periodes):
    """
    Fusionne des périodes qui se chevauchent ou se touchent.

    Les périodes sont en jours entiers, bornes incluses : ``(1er janv., 31 déc.)`` et
    ``(1er janv. suivant, ...)`` se touchent et ne forment qu'une période. Deux périodes
    séparées par au moins un jour restent séparées (un trou reste un trou).

    Parameters
    ----------
    periodes : list of tuple of pandas.Timestamp
        Périodes ``(debut, fin)``, dans n'importe quel ordre.

    Returns
    -------
    list of tuple of pandas.Timestamp
        Périodes disjointes, triées par début.

    See Also
    --------
    _manques : morceaux d'une période que ces périodes ne couvrent pas.

    Examples
    --------
    >>> T = pd.Timestamp
    >>> _fusionner([(T("2001-01-01"), T("2001-12-31")), (T("2000-01-01"), T("2000-12-31")),
    ...             (T("2005-01-01"), T("2005-12-31"))])
    [(Timestamp('2000-01-01 00:00:00'), Timestamp('2001-12-31 00:00:00')),
     (Timestamp('2005-01-01 00:00:00'), Timestamp('2005-12-31 00:00:00'))]
    """
    out = []
    for d0, d1 in sorted(periodes):
        if out and d0 <= out[-1][1] + pd.Timedelta(days=1):
            out[-1] = (out[-1][0], max(out[-1][1], d1))
        else:
            out.append((d0, d1))
    return out


def _manques(periodes, debut, fin):
    """
    Morceaux de ``[debut, fin]`` que les `periodes` ne couvrent pas.

    Parameters
    ----------
    periodes : list of tuple of pandas.Timestamp
        Périodes déjà couvertes, en jours inclus, fusionnées et triées (sortie de `_fusionner`).
    debut, fin : pandas.Timestamp
        Période demandée, bornes incluses.

    Returns
    -------
    list of tuple of pandas.Timestamp
        Morceaux manquants ``(debut, fin)``, triés ; liste vide si tout est couvert.

    See Also
    --------
    _fusionner : prépare `periodes`.
    Meteo._plan : se sert des morceaux manquants pour bâtir les requêtes.

    Examples
    --------
    >>> T = pd.Timestamp
    >>> _manques([(T("2000-01-01"), T("2024-12-31"))], T("1990-01-01"), T("2026-10-03"))
    [(Timestamp('1990-01-01 00:00:00'), Timestamp('1999-12-31 00:00:00')),
     (Timestamp('2025-01-01 00:00:00'), Timestamp('2026-10-03 00:00:00'))]
    """
    out, a = [], debut
    for d0, d1 in periodes:
        if d1 < a:
            continue
        if d0 > fin:
            break
        if d0 > a:
            out.append((a, min(fin, d0 - pd.Timedelta(days=1))))
        a = max(a, d1 + pd.Timedelta(days=1))
    if a <= fin:
        out.append((a, fin))
    return out


class QuotaEpuise(Exception):
    """
    Limite d'appels du jour ou du mois atteinte.

    Attendre quelques minutes n'y changera rien : il faut relancer plus tard. Le cache
    garde tout ce qui a déjà été téléchargé, rien n'est perdu.

    See Also
    --------
    Meteo.etat_quota : appels consommés et restants dans chaque fenêtre.
    """


class Meteo:
    """
    Accès à l'archive Open-Meteo avec compteur d'appels et cache par carreau.

    Toute requête météo passe par cette classe : elle calcule le coût avant l'envoi, attend
    si la limite par minute ou par heure serait dépassée, et ne télécharge que ce qui manque
    au cache.

    Parameters
    ----------
    abonnement : {'gratuit', 'standard', 'professional'}, default config.ABONNEMENT_OPEN_METEO
        Abonnement Open-Meteo, qui fixe les limites (cf. `ABONNEMENTS`).
    dossier : str or Path, default config.CACHE_METEO
        Dossier du cache (sous-dossier ``carreaux/`` + ``couverture.csv``). Le journal des
        appels, lui, est toujours ``config.CACHE_METEO / "journal_appels.csv"`` : un seul
        compteur, quel que soit le cache.
    cle : str, optional
        Clé des abonnements payants (par défaut lue dans ``OPEN_METEO_CLE``).
    api : {'archive', 'previous_runs'}, default 'archive'
        ``archive`` : météo observée (best_match : ERA5 / ERA5-Land avant 2017, IFS 9 km depuis). ``previous_runs`` : prévisions passées ;
        les variables s'écrivent ``<variable>_previous_dayN`` (prévue N jours avant, N = 1..7).
        Même journal que l'archive : Open-Meteo compte les deux ensemble.
    modele : str, optional
        Modèle Open-Meteo (paramètre ``models``), ex. ``"ecmwf_ifs"`` (9 km, même grille que
        nos carreaux, archivé depuis le 2025-10-01) ou ``"ecmwf_ifs025"`` (0,25°, depuis le
        2024-02-04 : le centre servi n'est pas celui du carreau, la valeur est celle du point
        0,25° le plus proche). Par défaut, aucun (best_match).

    Attributes
    ----------
    abonnement : str
        Nom de l'abonnement.
    limites : dict
        Ligne de `ABONNEMENTS` correspondante.
    cle : str or None
        Clé Open-Meteo.
    dossier : Path
        Dossier du cache.
    api : str
        API utilisée.
    modele : str or None
        Modèle Open-Meteo.

    See Also
    --------
    bassins.carreaux_amont : fournit la table `carreaux` attendue par les méthodes.
    cout_appel : règle de coût d'une requête.

    Notes
    -----
    Partout, `carreaux` est une table avec au moins les colonnes ``carreau`` (numéro dans
    la grille mondiale O1280), ``lon_centre`` et ``lat_centre`` : la sortie de
    `bassins.carreaux_amont` convient telle quelle.

    Fichiers du cache :

    - ``carreaux/<numero>_<resolution>.parquet`` : une colonne par variable, index = date-heure
      UTC, valeurs en float32 ;
    - ``couverture.csv`` : une ligne par (carreau, variable, résolution, période couverte) ;
    - ``journal_appels.csv`` : une ligne par requête envoyée (heure, points, variables, jours, coût).

    Examples
    --------
    >>> from src import bassins, meteo
    >>> R = bassins.Reseau()
    >>> zone = bassins.localiser([6.498010], [44.555356])[0]       # La Clapière
    >>> carreaux = bassins.carreaux_amont(zone, R.foret)            # 41 carreaux
    >>> m = meteo.Meteo()
    >>> m.cout_prevu(carreaux, ["temperature_2m", "precipitation"], [], "2000-01-01", "2024-12-31")
    >>> donnees = m.telecharger(carreaux, ["temperature_2m", "precipitation"], [],
    ...                         "2000-01-01", "2024-12-31")
    >>> donnees["horaire"][(int(carreaux["carreau"][0]), "temperature_2m")]
    """

    def __init__(self, abonnement=config.ABONNEMENT_OPEN_METEO, dossier=config.CACHE_METEO,
                 cle=config.CLE_OPEN_METEO, api="archive", modele=None):
        if api not in ("archive", "previous_runs"):
            raise ValueError(f"api inconnue : {api}")
        self.abonnement = abonnement
        self.limites = ABONNEMENTS[abonnement]
        self.cle = cle
        self.api, self.modele = api, modele
        self._url = self.limites["url"] if api == "archive" else URL_PREVIOUS_RUNS
        self.dossier = Path(dossier)
        (self.dossier / "carreaux").mkdir(parents=True, exist_ok=True)
        config.CACHE_METEO.mkdir(parents=True, exist_ok=True)
        # un seul compteur, quels que soient le cache et l'API (Open-Meteo compte archive et prévisions ensemble)
        self._journal = config.CACHE_METEO / "journal_appels.csv"
        self._couverture = self.dossier / "couverture.csv"

    # ---------------------------------------------------------------- compteur
    def _lire_journal(self):
        """
        Lit le journal des appels envoyés.

        Returns
        -------
        pandas.DataFrame
            Une ligne par requête : ``horodatage`` (UTC), ``points``, ``variables``, ``jours``,
            ``cout``. Table vide (colonnes ``horodatage`` et ``cout``) s'il n'y a pas encore
            de journal.
        """
        if not self._journal.exists():
            return pd.DataFrame({"horodatage": pd.to_datetime([], utc=True), "cout": []})
        j = pd.read_csv(self._journal)
        j["horodatage"] = pd.to_datetime(j["horodatage"], utc=True)
        return j

    def _noter(self, nb_points, nb_variables, nb_jours, cout):
        """
        Ajoute une requête réussie à la fin du journal des appels.

        Parameters
        ----------
        nb_points : int
            Nombre de carreaux de la requête.
        nb_variables : int
            Nombre de variables (horaires + journalières).
        nb_jours : int
            Nombre de jours de la période.
        cout : float
            Coût en appels (cf. `cout_appel`).
        """
        ligne = pd.DataFrame([{"horodatage": pd.Timestamp.now(tz="UTC").isoformat(), "points": nb_points,
                               "variables": nb_variables, "jours": nb_jours, "cout": round(cout, 3)}])
        ligne.to_csv(self._journal, mode="a", header=not self._journal.exists(), index=False)

    def etat_quota(self):
        """
        Appels consommés et restants dans chaque fenêtre de quota.

        D'après le journal : les appels faits ailleurs (autre ordinateur, navigateur) ne sont
        pas comptés.

        Returns
        -------
        pandas.DataFrame
            Index ``fenetre`` (minute, heure, jour, mois = 30 jours glissants) ; colonnes
            ``utilise``, ``limite`` et ``reste`` (None si la fenêtre n'a pas de limite).

        See Also
        --------
        QuotaEpuise : levée quand la limite du jour ou du mois est atteinte.

        Examples
        --------
        >>> Meteo().etat_quota().loc["jour", "reste"]
        """
        j, maintenant = self._lire_journal(), pd.Timestamp.now(tz="UTC")
        lignes = []
        for nom, secondes in FENETRES.items():
            utilise = j.loc[j["horodatage"] > maintenant - pd.Timedelta(seconds=secondes), "cout"].sum()
            limite = self.limites[nom]
            lignes.append({"fenetre": nom, "utilise": round(utilise, 1), "limite": limite,
                           "reste": None if limite is None else round(limite - utilise, 1)})
        return pd.DataFrame(lignes).set_index("fenetre")

    def _reserver(self, cout):
        """
        Attend que la requête tienne dans les limites, avant de l'envoyer.

        Pour la minute et l'heure : dort le temps qu'il faut pour que les appels les plus
        anciens sortent de la fenêtre. Pour le jour et le mois : attendre serait trop long,
        on s'arrête.

        Parameters
        ----------
        cout : float
            Coût en appels de la requête à envoyer.

        Raises
        ------
        ValueError
            Si la requête seule dépasse une limite (elle ne passera jamais).
        QuotaEpuise
            Si la limite du jour ou du mois serait dépassée.
        """
        while True:
            j, maintenant = self._lire_journal(), pd.Timestamp.now(tz="UTC")
            attente = 0.0
            for nom, secondes in FENETRES.items():
                limite = self.limites[nom]
                if limite is None:
                    continue
                if cout > limite:
                    raise ValueError(f"une seule requête ({cout:.0f} appels) dépasse la limite {nom} ({limite})")
                dedans = j[j["horodatage"] > maintenant - pd.Timedelta(seconds=secondes)]
                if dedans["cout"].sum() + cout <= limite:
                    continue
                if nom in ("jour", "mois"):
                    raise QuotaEpuise(f"limite {nom} atteinte ({dedans['cout'].sum():.0f}/{limite}) : "
                                      f"relancer plus tard, le cache garde ce qui est déjà téléchargé")
                plus_ancien = dedans["horodatage"].min()
                attente = max(attente, (plus_ancien + pd.Timedelta(seconds=secondes) - maintenant).total_seconds() + 1)
            if attente <= 0:
                return
            time.sleep(attente)

    # ---------------------------------------------------------------- cache
    def _lire_couverture(self):
        """
        Lit la table de couverture du cache.

        Returns
        -------
        pandas.DataFrame
            Une ligne par période couverte : ``carreau``, ``variable``, ``resolution``,
            ``debut``, ``fin`` (jours inclus). Plusieurs lignes pour une même variable d'un
            même carreau s'il y a des trous. Table vide s'il n'y a pas encore de cache.
        """
        if not self._couverture.exists():
            return pd.DataFrame(columns=["carreau", "variable", "resolution", "debut", "fin"])
        return pd.read_csv(self._couverture, parse_dates=["debut", "fin"])

    def contenu_cache(self):
        """
        Ce qui est en cache, avec la place prise sur le disque.

        Returns
        -------
        pandas.DataFrame
            Une ligne par carreau, variable, résolution et période couverte (plusieurs
            périodes possibles s'il y a des trous) : ``carreau``, ``variable``,
            ``resolution``, ``debut``, ``fin``, ``mo_carreau`` (Mo de ses fichiers, toutes
            variables confondues). Triée par carreau, résolution, variable.

        See Also
        --------
        vider_cache : retire tout ou partie du cache.

        Examples
        --------
        >>> c = Meteo().contenu_cache()
        >>> c.groupby("carreau")["mo_carreau"].first().sum()       # Mo au total
        """
        c = self._lire_couverture()
        if c.empty:
            return c
        taille = {}
        for f in (self.dossier / "carreaux").glob("*.parquet"):
            num = int(f.stem.split("_")[0])
            taille[num] = taille.get(num, 0) + f.stat().st_size
        c["mo_carreau"] = c["carreau"].map(lambda n: round(taille.get(n, 0) / 1e6, 3))
        return c.sort_values(["carreau", "resolution", "variable"]).reset_index(drop=True)

    def vider_cache(self, carreaux=None, variables=None):
        """
        Retire tout le cache, ou seulement certains carreaux et/ou variables.

        Le journal des appels n'est jamais effacé : c'est le compteur du quota.

        Parameters
        ----------
        carreaux : list of int, optional
            Numéros des carreaux à retirer. Par défaut, tous.
        variables : list of str, optional
            Variables à retirer (dans les deux résolutions). Par défaut, toutes.

        See Also
        --------
        contenu_cache : ce qui est en cache.

        Examples
        --------
        >>> m = Meteo()
        >>> m.vider_cache(variables=["snowfall"])          # une variable, partout
        >>> m.vider_cache(carreaux=[3_401_234])            # un carreau, toutes variables
        """
        c = self._lire_couverture()
        if c.empty:
            return
        cible = c["carreau"].isin(carreaux) if carreaux is not None else pd.Series(True, index=c.index)
        if variables is not None:
            cible &= c["variable"].isin(variables)
        for (num, res), groupe in c[cible].groupby(["carreau", "resolution"]):
            chemin = self.dossier / "carreaux" / f"{num}_{res}.parquet"
            if not chemin.exists():
                continue
            df = pd.read_parquet(chemin).drop(columns=list(groupe["variable"]), errors="ignore")
            if df.shape[1]:
                df.to_parquet(chemin)
            else:
                chemin.unlink()
        c[~cible].to_csv(self._couverture, index=False)

    # ---------------------------------------------------------------- plan des requêtes
    def _plan(self, carreaux, horaires, journalieres, debut, fin):
        """
        Liste des requêtes à envoyer pour compléter le cache.

        Pour chaque carreau et chaque variable, cherche les morceaux de ``[debut, fin]``
        absents du cache ; regroupe dans une même requête les carreaux à qui manquent la
        même période et les mêmes variables ; découpe en requêtes d'au plus `MAX_JOURS`
        jours et `MAX_POINTS` carreaux (moins si la limite par minute l'exige).

        Parameters
        ----------
        carreaux : pandas.DataFrame
            Colonnes ``carreau``, ``lon_centre``, ``lat_centre``.
        horaires, journalieres : list of str
            Noms Open-Meteo des variables horaires et journalières.
        debut, fin : str or pandas.Timestamp
            Période demandée, jours inclus.

        Returns
        -------
        list of dict
            Une requête par élément : ``carreaux`` (lignes de la table), ``debut``, ``fin``,
            ``horaires``, ``journalieres`` (tuples), ``cout`` (en appels). Vide si tout est
            en cache.
        """
        debut, fin = pd.Timestamp(debut), pd.Timestamp(fin)
        couv = {cle: _fusionner(list(zip(g["debut"], g["fin"])))
                for cle, g in self._lire_couverture().groupby(["carreau", "variable", "resolution"])}
        groupes = {}
        for _, ligne in carreaux.iterrows():
            manque = {}
            for res, noms in (("horaire", horaires), ("journalier", journalieres)):
                for v in noms:
                    morceaux = _manques(couv.get((int(ligne["carreau"]), v, res), []), debut, fin)
                    for m in morceaux:
                        manque.setdefault(m, {"horaire": [], "journalier": []})[res].append(v)
            for m, vars_ in manque.items():
                cle = (m, tuple(vars_["horaire"]), tuple(vars_["journalier"]))
                groupes.setdefault(cle, []).append(ligne)
        plan = []
        for ((d0, d1), h, jr), lignes in groupes.items():
            pts = pd.DataFrame(lignes)
            while d0 <= d1:
                f = min(d1, d0 + pd.Timedelta(days=MAX_JOURS - 1))
                jours = (f - d0).days + 1
                par_point = cout_appel(1, len(h) + len(jr), jours)
                n = MAX_POINTS if self.limites["minute"] is None else \
                    max(1, min(MAX_POINTS, int(self.limites["minute"] // par_point)))
                for i in range(0, len(pts), n):
                    lot = pts.iloc[i:i + n]
                    plan.append({"carreaux": lot, "debut": d0, "fin": f, "horaires": h, "journalieres": jr,
                                 "cout": cout_appel(len(lot), len(h) + len(jr), jours)})
                d0 = f + pd.Timedelta(days=1)
        return plan

    def cout_prevu(self, carreaux, horaires=(), journalieres=(), debut="2000-01-01", fin="2024-12-31"):
        """
        Coût d'une demande en appels, SANS rien télécharger.

        Seul ce qui manque au cache est compté : c'est exactement ce que coûterait
        `telecharger` avec les mêmes arguments.

        Parameters
        ----------
        carreaux : pandas.DataFrame
            Colonnes ``carreau``, ``lon_centre``, ``lat_centre`` (sortie de
            `bassins.carreaux_amont`).
        horaires : list of str, default ()
            Variables horaires (noms Open-Meteo, ex. ``"temperature_2m"``).
        journalieres : list of str, default ()
            Variables journalières (ex. ``"precipitation_sum"``).
        debut : str or pandas.Timestamp, default "2000-01-01"
            Premier jour, inclus.
        fin : str or pandas.Timestamp, default "2024-12-31"
            Dernier jour, inclus.

        Returns
        -------
        dict
            ``appels`` (coût total), ``requetes`` (nombre de requêtes),
            ``carreaux_a_telecharger`` et ``carreaux_complets_en_cache``.

        See Also
        --------
        telecharger : fait réellement les requêtes.
        etat_quota : ce qui reste dans chaque fenêtre.

        Notes
        -----
        Sur l'offre gratuite (10 000 appels par jour), diviser ``appels`` par 10 000 donne
        le nombre de jours de téléchargement.

        Examples
        --------
        >>> Meteo().cout_prevu(carreaux, ["temperature_2m"], [], "1980-01-01", "2026-10-03")
        """
        plan = self._plan(carreaux, list(horaires), list(journalieres), debut, fin)
        a_telecharger = {int(n) for r in plan for n in r["carreaux"]["carreau"]}
        return {"appels": round(sum(r["cout"] for r in plan), 1), "requetes": len(plan),
                "carreaux_a_telecharger": len(a_telecharger),
                "carreaux_complets_en_cache": len(set(carreaux["carreau"].astype(int)) - a_telecharger)}

    # ---------------------------------------------------------------- téléchargement
    def _envoyer(self, r):
        """
        Envoie une requête à Open-Meteo, après avoir réservé sa place dans le quota.

        Réessaie jusqu'à 6 fois : après 61 s si Open-Meteo signale la limite par minute
        (appels faits ailleurs que notre journal ne voit pas), après quelques secondes si
        le serveur est en panne passagère (codes 500 à 504).

        Parameters
        ----------
        r : dict
            Une requête du plan (cf. `_plan`).

        Returns
        -------
        list of dict
            Une réponse JSON par carreau, dans l'ordre de ``r["carreaux"]``.

        Raises
        ------
        ValueError
            Si l'abonnement n'a pas d'adresse d'archive.
        QuotaEpuise
            Si Open-Meteo refuse pour la limite de l'heure, du jour ou du mois.
        RuntimeError
            Pour toute autre erreur, ou après 6 échecs.
        """
        url = self._url
        if url is None:
            raise ValueError(f"abonnement {self.abonnement} : pas d'adresse d'archive (historique absent, ou "
                             f"adresse abonnés à renseigner dans meteo.ABONNEMENTS)")
        params = {"latitude": list(r["carreaux"]["lat_centre"]), "longitude": list(r["carreaux"]["lon_centre"]),
                  "start_date": r["debut"].strftime("%Y-%m-%d"), "end_date": r["fin"].strftime("%Y-%m-%d"),
                  "cell_selection": "nearest", "timezone": "GMT"}
        if r["horaires"]:
            params["hourly"] = ",".join(r["horaires"])
        if r["journalieres"]:
            params["daily"] = ",".join(r["journalieres"])
        if self.modele:
            params["models"] = self.modele
        if self.cle:
            params["apikey"] = self.cle
        nb_var, jours = len(r["horaires"]) + len(r["journalieres"]), (r["fin"] - r["debut"]).days + 1
        return self._poster(url, params, r["cout"], len(r["carreaux"]), nb_var, jours)

    def _poster(self, url, params, cout, nb_points, nb_var, jours):
        """
        Envoie une requête POST après avoir réservé sa place dans le quota, et la note au journal.

        Parameters
        ----------
        url : str
            Adresse de l'API.
        params : dict
            Paramètres (listes de latitudes / longitudes acceptées).
        cout : float
            Coût en appels (cf. `cout_appel`).
        nb_points, nb_var, jours : int
            Pour le journal.

        Returns
        -------
        list of dict
            Une réponse JSON par point.

        Raises
        ------
        QuotaEpuise
            Si Open-Meteo refuse pour la limite de l'heure, du jour ou du mois.
        RuntimeError
            Pour toute autre erreur (le texte d'Open-Meteo est dans le message), ou après 6 échecs.
        """
        for tentative in range(6):
            self._reserver(cout)
            rep = requests.post(url, data=params, timeout=300)
            if rep.status_code == 200:
                self._noter(nb_points, nb_var, jours, cout)
                data = rep.json()
                return data if isinstance(data, list) else [data]
            if rep.status_code == 429 and "Minutely" in rep.text:
                time.sleep(61)          # notre journal sous-estimait (appels faits ailleurs) : on se recale
            elif rep.status_code == 429:
                raise QuotaEpuise(f"Open-Meteo refuse : {rep.text[:200]}")
            elif rep.status_code in (500, 502, 503, 504):
                time.sleep(min(60, 5 * (tentative + 1)))
            else:
                raise RuntimeError(f"Open-Meteo {rep.status_code} : {rep.text[:300]}")
        raise RuntimeError("Open-Meteo : échec après 6 tentatives")

    def _ranger(self, r, reponses):
        """
        Range les réponses dans le cache et met à jour la couverture.

        Un fichier par carreau et par résolution ; les nouvelles valeurs complètent les
        anciennes. Valeurs gardées en float32 : deux fois moins de mémoire une fois chargées,
        même taille sur le disque, erreur d'environ 3e-6 en relatif.

        Parameters
        ----------
        r : dict
            La requête envoyée (cf. `_plan`).
        reponses : list of dict
            Les réponses JSON, une par carreau (sortie de `_envoyer`).

        Raises
        ------
        ValueError
            Si Open-Meteo a servi un autre point que le centre du carreau demandé (écart
            de plus de 0,001°) : la grille calculée ne serait plus la bonne. Vérifié seulement
            pour les modèles de `MEME_GRILLE`.
        """
        couv = self._lire_couverture()
        nouvelles = []
        for (_, ligne), d in zip(r["carreaux"].iterrows(), reponses):
            decale = abs(d["latitude"] - ligne["lat_centre"]) > 1e-3 or abs(d["longitude"] - ligne["lon_centre"]) > 1e-3
            if decale and self.modele in MEME_GRILLE:
                raise ValueError(f"carreau {ligne['carreau']} : Open-Meteo a servi le centre "
                                 f"({d['latitude']}, {d['longitude']}), pas celui de la grille")
            num = int(ligne["carreau"])
            for res, cle, noms in (("horaire", "hourly", r["horaires"]), ("journalier", "daily", r["journalieres"])):
                if not noms:
                    continue
                df = pd.DataFrame(d[cle])
                df.index = pd.to_datetime(df.pop("time"), utc=True)
                chemin = self.dossier / "carreaux" / f"{num}_{res}.parquet"
                if chemin.exists():
                    df = df.combine_first(pd.read_parquet(chemin))
                # 4 octets : 2× moins de RAM une fois chargé, même taille sur disque, erreur ~3e-6
                df.astype("float32").sort_index().to_parquet(chemin)
                for v in noms:
                    nouvelles.append({"carreau": num, "variable": v, "resolution": res,
                                      "debut": r["debut"], "fin": r["fin"]})
        tout = pd.DataFrame(nouvelles) if couv.empty else pd.concat([couv, pd.DataFrame(nouvelles)], ignore_index=True)
        lignes = [{"carreau": k[0], "variable": k[1], "resolution": k[2], "debut": d0, "fin": d1}
                  for k, g in tout.groupby(["carreau", "variable", "resolution"])
                  for d0, d1 in _fusionner(list(zip(g["debut"], g["fin"])))]     # un trou reste un trou
        pd.DataFrame(lignes).to_csv(self._couverture, index=False)

    def telecharger(self, carreaux, horaires=(), journalieres=(), debut="2000-01-01", fin="2024-12-31"):
        """
        Complète le cache (seulement ce qui manque), puis renvoie les données demandées.

        Peut être long : sur l'offre gratuite, attend toute seule quand la limite par minute
        ou par heure est atteinte. Si elle s'arrête (`QuotaEpuise`, coupure réseau), relancer
        la même commande reprend là où elle en était.

        Parameters
        ----------
        carreaux : pandas.DataFrame
            Colonnes ``carreau``, ``lon_centre``, ``lat_centre`` (sortie de
            `bassins.carreaux_amont`).
        horaires : list of str, default ()
            Variables horaires (noms Open-Meteo, ex. ``"temperature_2m"``, ``"precipitation"``).
        journalieres : list of str, default ()
            Variables journalières (ex. ``"precipitation_sum"``).
        debut : str or pandas.Timestamp, default "2000-01-01"
            Premier jour, inclus.
        fin : str or pandas.Timestamp, default "2024-12-31"
            Dernier jour, inclus (jusqu'à 23 h UTC).

        Returns
        -------
        dict of str to pandas.DataFrame
            ``"horaire"`` et/ou ``"journalier"`` (seulement les résolutions demandées) : un
            tableau par résolution, index = date-heure UTC, colonnes à deux niveaux
            ``(carreau, variable)``, valeurs en float32.

        Raises
        ------
        QuotaEpuise
            Si la limite du jour ou du mois est atteinte avant la fin.

        See Also
        --------
        cout_prevu : le même calcul, sans rien télécharger.
        contenu_cache : ce qui est déjà là.

        Examples
        --------
        >>> d = Meteo().telecharger(carreaux, ["temperature_2m", "precipitation"], [],
        ...                         "2000-01-01", "2026-10-03")
        >>> h = d["horaire"]
        >>> h.xs("temperature_2m", axis=1, level="variable")    # une colonne par carreau
        >>> h[(int(carreaux["carreau"][0]), "precipitation")]    # une série
        """
        horaires, journalieres = list(horaires), list(journalieres)
        for r in self._plan(carreaux, horaires, journalieres, debut, fin):
            self._ranger(r, self._envoyer(r))
        sortie = {}
        for res, noms in (("horaire", horaires), ("journalier", journalieres)):
            if not noms:
                continue
            morceaux = {}
            for num in carreaux["carreau"].astype(int):
                df = pd.read_parquet(self.dossier / "carreaux" / f"{num}_{res}.parquet")[noms]
                morceaux[num] = df.loc[pd.Timestamp(debut, tz="UTC"):pd.Timestamp(fin, tz="UTC") + pd.Timedelta(hours=23)]
            sortie[res] = pd.concat(morceaux, axis=1, names=["carreau", "variable"])
        return sortie


# ==============================================================================
# PRÉVISIONS PASSÉES COMPLÈTES (API Single Runs)
# ==============================================================================

URL_SINGLE_RUNS = "https://single-runs-api.open-meteo.com/v1/forecast"
"""str : API des prévisions archivées une par une (même quota que l'archive)."""


class PrevisionsArchivees:
    """
    Prévisions météo passées COMPLÈTES, telles qu'émises (API Single Runs), avec cache.

    Une prévision (un « run ») = ce que le modèle annonçait, émis à une heure donnée, pour
    les heures suivantes (10 jours pour ECMWF IFS 9 km). Contrairement à `Meteo`, chaque heure
    a donc plusieurs valeurs : une par prévision qui la couvre.

    Parameters
    ----------
    modele : str, default "ecmwf_ifs"
        Modèle Open-Meteo. ``ecmwf_ifs`` (9 km) : même grille que nos carreaux, archivé depuis le
        2024-03-14 à 0 h UTC. ``ecmwf_ifs_europe_ensemble`` : les 51 scénarios ECMWF à 9 km (même
        grille), runs de 0 h UTC (15 jours) et 6 h UTC (6 jours), disponibles depuis le 2026-06-24
        (vérifié le 2026-10-15).
    membres : int, default 1
        Séries par variable : 1, ou 51 pour un ensemble (contrôle ``<variable>`` + scénarios
        ``<variable>_member01`` … ``_member50``). Le coût compte chaque série comme une variable
        (règle d'Open-Meteo pour les ensembles).
    dossier : str or Path, optional
        Dossier du cache. Par défaut ``config.CACHE_PREVISION / "single_runs" / modele``.
    meteo : Meteo, optional
        Fournit le compteur d'appels (journal commun à toutes les API). Par défaut `Meteo()`.

    Attributes
    ----------
    modele : str
        Modèle.
    membres : int
        Séries par variable.
    dossier : Path
        Dossier du cache : ``runs/<run>__<n>.parquet`` (une requête = un fichier : colonnes
        ``carreau``, ``run``, ``time`` + une par variable) et ``couverture.csv`` (``carreau``,
        ``run``, ``etat`` = ok ou absent).

    See Also
    --------
    Meteo : météo observée (best_match) et compteur d'appels.

    Notes
    -----
    Coût (règle de `cout_appel`) : jusqu'à 10 variables sur 10 jours = 1 appel par carreau et
    par prévision. Une prévision qu'Open-Meteo n'a pas (« run not available ») est notée
    ``absent`` et n'est plus redemandée.

    Examples
    --------
    >>> pa = PrevisionsArchivees()
    >>> pa.cout_prevu(carreaux, "2024-03-14", "2026-10-10")
    >>> pa.telecharger(carreaux, VARIABLES, "2024-03-14", "2026-10-10")
    >>> pa.lire(835931)                  # index (run, time), une colonne par variable
    """

    def __init__(self, modele="ecmwf_ifs", membres=1, dossier=None, meteo=None):
        self.modele, self.membres = modele, membres
        self.dossier = Path(dossier) if dossier else config.CACHE_PREVISION / "single_runs" / modele
        (self.dossier / "runs").mkdir(parents=True, exist_ok=True)
        self._couverture = self.dossier / "couverture.csv"
        self._meteo = meteo or Meteo()

    def _lire_couverture(self):
        """Table ``carreau``, ``run``, ``etat`` des prévisions déjà demandées (vide au départ)."""
        if not self._couverture.exists():
            return pd.DataFrame(columns=["carreau", "run", "etat"])
        return pd.read_csv(self._couverture)

    def _runs(self, debut, fin, heures):
        """Heures d'émission voulues : chaque jour de `debut` à `fin`, aux `heures` (UTC)."""
        jours = pd.date_range(debut, fin, freq="D")
        return [(j + pd.Timedelta(f"{h}:00")).strftime("%Y-%m-%dT%H:%M") for j in jours for h in heures]

    def manquants(self, carreaux, debut, fin, heures=("00:00",)):
        """
        Prévisions encore à télécharger, carreau par carreau.

        Parameters
        ----------
        carreaux : pandas.DataFrame
            Colonnes ``carreau``, ``lon_centre``, ``lat_centre``.
        debut, fin : str
            Premier et dernier jour d'émission.
        heures : tuple of str, default ("00:00",)
            Heures d'émission UTC (ECMWF : 00:00 et 12:00 vont à 10 jours, 06:00 et 18:00 à 90 h).

        Returns
        -------
        dict of str to list of int
            Pour chaque prévision (``run``), les carreaux qui ne l'ont pas encore.
        """
        faits = set(map(tuple, self._lire_couverture()[["carreau", "run"]].astype({"carreau": int}).to_numpy()))
        numeros = [int(c) for c in carreaux["carreau"]]
        sortie = {}
        for run in self._runs(debut, fin, heures):
            reste = [c for c in numeros if (c, run) not in faits]
            if reste:
                sortie[run] = reste
        return sortie

    def cout_prevu(self, carreaux, debut, fin, heures=("00:00",), nb_variables=9, jours=10):
        """
        Coût en appels de ce qui manque, SANS rien télécharger.

        Returns
        -------
        dict
            ``appels``, ``requetes`` (une par prévision), ``jours_de_quota_gratuit`` (10 000 / jour).
        """
        m = self.manquants(carreaux, debut, fin, heures)
        appels = sum(cout_appel(len(c), nb_variables * self.membres, jours) for c in m.values())
        return {"appels": round(appels, 1), "requetes": len(m), "jours_de_quota_gratuit": round(appels / 10_000, 2)}

    def telecharger(self, carreaux, variables, debut, fin, heures=("00:00",), jours=10, barre=True):
        """
        Télécharge les prévisions qui manquent (une requête par prévision, tous ses carreaux).

        Peut être long : attend toute seule quand la limite par minute ou par heure est atteinte ;
        relancer reprend là où elle en était.

        Parameters
        ----------
        carreaux : pandas.DataFrame
            Colonnes ``carreau``, ``lon_centre``, ``lat_centre``.
        variables : list of str
            Variables horaires Open-Meteo (≤ 10 pour 1 appel par carreau et par prévision).
        debut, fin : str
            Premier et dernier jour d'émission.
        heures : tuple of str, default ("00:00",)
            Heures d'émission UTC.
        jours : int, default 10
            Durée demandée de chaque prévision.
        barre : bool, default True
            Barre d'avancement (une case par prévision).

        Raises
        ------
        QuotaEpuise
            Si la limite du jour ou du mois est atteinte avant la fin.
        """
        centres = carreaux.set_index(carreaux["carreau"].astype(int))[["lat_centre", "lon_centre"]]
        manque = self.manquants(carreaux, debut, fin, heures)
        for run, numeros in tqdm(manque.items(), desc="prévisions", disable=not barre):
            params = {"latitude": list(centres.loc[numeros, "lat_centre"]),
                      "longitude": list(centres.loc[numeros, "lon_centre"]),
                      "models": self.modele, "run": run, "hourly": ",".join(variables),
                      "forecast_days": jours, "cell_selection": "nearest", "timezone": "GMT"}
            series = len(variables) * self.membres
            cout = cout_appel(len(numeros), series, jours)
            try:
                reponses = self._meteo._poster(URL_SINGLE_RUNS, params, cout, len(numeros), series, jours)
            except RuntimeError as e:
                if "not available" not in str(e):
                    raise
                self._noter_couverture(numeros, run, "absent")
                continue
            morceaux = []
            for num, d in zip(numeros, reponses):
                if abs(d["latitude"] - centres.loc[num, "lat_centre"]) > 1e-3 or \
                        abs(d["longitude"] - centres.loc[num, "lon_centre"]) > 1e-3:
                    raise ValueError(f"carreau {num} : Open-Meteo a servi ({d['latitude']}, {d['longitude']})")
                h = pd.DataFrame(d["hourly"])
                h.insert(0, "run", pd.Timestamp(run, tz="UTC"))
                h.insert(0, "carreau", num)
                morceaux.append(h)
            table = pd.concat(morceaux, ignore_index=True)
            table["time"] = pd.to_datetime(table["time"], utc=True)
            table = table.astype({c: "float32" for c in table.columns if c not in ("carreau", "run", "time")})
            n = len(list((self.dossier / "runs").glob(f"{run.replace(':', '')}__*.parquet")))
            table.to_parquet(self.dossier / "runs" / f"{run.replace(':', '')}__{n}.parquet", index=False)
            self._noter_couverture(numeros, run, "ok")

    def _noter_couverture(self, numeros, run, etat):
        """Ajoute ``(carreau, run, etat)`` à la couverture."""
        ligne = pd.DataFrame({"carreau": numeros, "run": run, "etat": etat})
        ligne.to_csv(self._couverture, mode="a", header=not self._couverture.exists(), index=False)

    def lire(self, carreaux=None):
        """
        Prévisions en cache.

        Parameters
        ----------
        carreaux : int or list of int, optional
            Carreaux voulus. Par défaut, tous.

        Returns
        -------
        pandas.DataFrame
            Index (``carreau``, ``run``, ``time``) : heure d'émission de la prévision et heure
            prévue (UTC) ; une colonne par variable, et par scénario pour un ensemble (float32).
        """
        filtre = None if carreaux is None else [("carreau", "in", [int(c) for c in np.atleast_1d(carreaux)])]
        table = pd.read_parquet(self.dossier / "runs", filters=filtre)
        return table.set_index(["carreau", "run", "time"]).sort_index()

    def contenu_cache(self):
        """
        Ce qui est en cache : par carreau, nombre de prévisions, première et dernière, absentes.

        Returns
        -------
        pandas.DataFrame
            Index ``carreau`` ; colonnes ``ok``, ``absent``, ``premiere``, ``derniere``.
        """
        c = self._lire_couverture()
        if c.empty:
            return c
        ok = c[c["etat"] == "ok"].groupby("carreau")["run"]
        return pd.DataFrame({"ok": ok.size(), "absent": c[c["etat"] == "absent"].groupby("carreau").size(),
                             "premiere": ok.min(), "derniere": ok.max()}).fillna({"absent": 0}).astype({"absent": int})

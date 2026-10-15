"""
================================================================================
MODULE station.py
================================================================================
Une station de mesure, point d'entrée unique : son identité (RiverApp), son bassin et les
carreaux météo de son amont (`bassins`), ses mesures (`hydro.RiverApp`, avec son cache), de
nouveaux `features.Features` pour composer X et Y, et ses modèles enregistrés (`modeles`).

`Station` ne refait rien elle-même : elle passe par les classes qui gèrent déjà les API et les
caches. Le réseau des bassins (≈ 1 million de lignes) n'est chargé qu'une fois et partagé.
Pour remonter ou descendre les rivières (chaînes de stations), partir de ``st.bassin`` dans
l'arbre ``st.reseau.foret``.
================================================================================
"""

import functools

import pandas as pd

from .bassins import Reseau, carreaux_amont, localiser
from .features import Features
from .hydro import RiverApp
from .modeles import charger, contenu_cache, resultats, dossier_station


@functools.lru_cache(maxsize=1)
def _reseau():
    """Le réseau des bassins du monde, chargé une seule fois et partagé par toutes les stations."""
    return Reseau()


class Station:
    """
    Une station : identité, bassin, carreaux de l'amont, mesures, features et modèles.

    Parameters
    ----------
    station : str
        Identifiant RiverApp (ex. ``"52a0c9cee4b03f58295e6f8a"``) OU code chez le fournisseur
        (ex. Hub'Eau ``"X031001001"``).
    fournisseur : str, default "hubeau"
        Nom court du fournisseur dont la liste de stations sert à la reconnaître.
    reseau : bassins.Reseau, optional
        Réseau des bassins. Par défaut, celui partagé par toutes les stations (chargé au premier
        besoin).
    riverapp : hydro.RiverApp, optional
        Accès RiverApp. Par défaut, `RiverApp(fournisseur)`.

    Attributes
    ----------
    id : str
        Identifiant RiverApp (clé partout dans le projet).
    code : str or None
        Code chez le fournisseur.
    nom, cours_eau : str
        Nom de la station et de la rivière.
    lon, lat : float
        Position, en degrés.
    debut_donnees : pandas.Timestamp
        Date de la plus ancienne mesure (UTC).
    bassin : int
        HYBAS_ID du bassin où se trouve la station (entrée dans l'arbre ``reseau.foret``).

    Raises
    ------
    KeyError
        Si la station n'existe pas chez ce fournisseur.
    ValueError
        Si le code correspond à plusieurs identifiants RiverApp (ex. Langon O945001001) :
        donner alors l'identifiant.

    See Also
    --------
    features.Features : la recette de X et Y.
    modeles.entrainer : entraîner un modèle (il est rangé avec la station).

    Examples
    --------
    >>> from src.station import Station
    >>> st = Station("X031001001")                       # La Clapière
    >>> st.nom, st.debut_donnees, st.bassin
    >>> st.mesures("2024-01-01", "2024-12-31", pas="1h")
    >>> F = st.features("2010-01-01", "2025-12-31")      # nouveau, vide : un par recette
    >>> st.modeles()                                      # ses modèles, meilleur R² en premier
    """

    def __init__(self, station, fournisseur="hubeau", reseau=None, riverapp=None):
        self._riverapp = riverapp or RiverApp(fournisseur)
        self._reseau_donne = reseau
        liste = self._riverapp.stations()
        if station in liste.index:
            ligne = liste.loc[station]
            self.id = station
        else:
            trouves = liste[liste["code"] == station]
            if trouves.empty:
                raise KeyError(f"station {station} inconnue chez le fournisseur {fournisseur}")
            if len(trouves) > 1:
                raise ValueError(f"le code {station} correspond à plusieurs stations RiverApp : {list(trouves.index)}")
            self.id, ligne = trouves.index[0], trouves.iloc[0]
        self.code = ligne["code"] if isinstance(ligne["code"], str) and ligne["code"] else None
        self.nom, self.cours_eau = ligne["nom"], ligne["cours_eau"]
        self.lon, self.lat = float(ligne["lon"]), float(ligne["lat"])
        self.debut_donnees = ligne["plus_ancienne_mesure"]
        self.bassin = int(localiser([self.lon], [self.lat])[0])

    def __repr__(self):
        return f"Station({self.code or self.id} : {self.nom}, {self.cours_eau})"

    @property
    def reseau(self):
        """bassins.Reseau : le réseau des bassins (chargé au premier besoin, puis partagé)."""
        return self._reseau_donne if self._reseau_donne is not None else _reseau()

    @functools.cached_property
    def carreaux(self):
        """pandas.DataFrame : carreaux météo de tout l'amont (cf. `bassins.carreaux_amont`)."""
        return carreaux_amont(self.bassin, self.reseau.foret)

    @property
    def dossier_modeles(self):
        """Path : dossier où sont rangés les modèles de la station."""
        return dossier_station(self.id)

    def mesures(self, debut=None, fin=None, pas=None, trou_max="3h", filtre=False):
        """
        Débits et hauteurs de la station (cf. `hydro.RiverApp.mesures`, avec son cache).

        Parameters
        ----------
        debut, fin : str, optional
            Période (UTC). Par défaut, de la plus ancienne mesure à maintenant.
        pas : str, optional
            Pas régulier (``"1h"``, ``"1D"``…) ; par défaut, les mesures brutes.
        trou_max : str, default "3h"
            Avec `pas` : pas d'interpolation à travers un trou plus long.
        filtre : bool, default False
            Retirer les pics isolés.

        Returns
        -------
        pandas.DataFrame
            Colonnes ``flow`` (m³/s) et ``level`` (m).
        """
        return self._riverapp.mesures(self.id, debut, fin, pas=pas, trou_max=trou_max, filtre=filtre)

    def features(self, debut, fin, filtre=False):
        """
        Un NOUVEAU `features.Features` de la station, vide : à remplir avec sa propre recette.

        Chaque appel en donne un neuf, indépendant : autant de recettes (et de modèles) qu'on veut.

        Parameters
        ----------
        debut, fin : str
            Période, jours inclus.
        filtre : bool, default False
            Retirer les pics isolés de débit.

        Returns
        -------
        features.Features
        """
        return Features(self.id, debut, fin, filtre=filtre, reseau=self.reseau, riverapp=self._riverapp)

    def modeles(self):
        """
        Les modèles enregistrés de la station, du meilleur R² moyen au moins bon.

        Returns
        -------
        pandas.DataFrame
            Comme `modeles.contenu_cache` (clé, étiquette, type, R², scores de crue, durée…).
        """
        return contenu_cache(station=self.id).sort_values("r2_moyen", ascending=False).reset_index(drop=True)

    def charger(self, cle):
        """Un modèle enregistré (`modeles.charger`), prêt à prévoir."""
        return charger(cle)

    def resultats(self, cle):
        """Scores et prévisions du test d'un modèle, sans le charger (`modeles.resultats`)."""
        return resultats(cle)

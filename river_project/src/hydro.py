"""
================================================================================
MODULE hydro.py
================================================================================
Débits et hauteurs des stations, via l'API partenaire RiverApp. Toute lecture passe par
la classe `RiverApp` :

- liste des stations d'un fournisseur (Hub'Eau par défaut), mise en cache ;
- mesures d'une station par tranches de 32 jours (limite de l'API, bornes incluses),
  mises en cache : seul ce qui manque est téléchargé ;
- le téléchargement lui-même vit dans `riverapp_acces.py`, un fichier PRIVÉ (non publié). Sans
  lui, la classe ne lit que le cache ;
- une station se désigne par son IDENTIFIANT RIVERAPP (unique, valable pour tous les fournisseurs et
  tous les pays) ; le code chez le fournisseur (ex. Hub'Eau X031001001),
  facultatif selon les fournisseurs, reste une colonne de `stations()` pour retrouver l'identifiant ;
- au choix, mesures brutes ou ramenées à un pas régulier (`pas`, ex. "4h") : grille calée sur
  minuit UTC, interpolation linéaire dans le temps, jamais à travers un trou (`trou_max`) ;
- au choix (`filtre=True`), retrait des pics isolés (erreurs ponctuelles du capteur) ; le cache
  garde toujours les mesures brutes.
Dates en UTC ; débit en m³/s, hauteur en m ; une valeur absente n'est pas renvoyée (pas de 0).
Constaté sur La Clapière : historique = débit horaire seul ; période récente = débit +
hauteur toutes les 15 min (cf. notebook 04).
================================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

try:
    from . import riverapp_acces as _acces     # privé : absent d'une copie publique du projet
except ImportError:
    _acces = None

TRANCHE = pd.Timedelta(days=32)     # limite de l'API, bornes incluses
"""pandas.Timedelta : durée maximale d'une requête de mesures (limite de l'API, bornes incluses)."""


def _filtrer(df, fenetre="90min", seuil=0.5):
    """
    Vide (NaN) les pics isolés de débit, erreurs ponctuelles du capteur.

    Une mesure est un pic si elle s'écarte de plus de `seuil` (en relatif), dans le même sens,
    à la fois de la médiane des mesures de la `fenetre` d'avant ET de celle de la `fenetre`
    d'après. Le bord d'une vraie montée (lâcher de barrage, crue) ne ressemble qu'à un seul
    des deux côtés : il est gardé.

    Parameters
    ----------
    df : pandas.DataFrame
        Mesures brutes : index = date UTC (trié), colonnes ``flow`` et ``level``.
    fenetre : str, default "90min"
        Durée regardée de chaque côté de la mesure (format pandas : ``"90min"``, ``"3h"``...).
    seuil : float, default 0.5
        Écart relatif à partir duquel une mesure est retirée (0.5 = 50 %).

    Returns
    -------
    pandas.DataFrame
        Copie de `df` où les pics sont vides (NaN), débit ET hauteur (le débit est calculé à
        partir de la hauteur). `df` tel quel si la station n'a aucun débit.

    See Also
    --------
    RiverApp.mesures : l'applique avec ``filtre=True`` ; le cache, lui, reste brut.

    Notes
    -----
    Calé sur La Clapière (notebook 04) : 17 mesures retirées sur 362 169, toutes des erreurs
    visibles (ex. 46 → 0,2 → 46 m³/s en 30 min). Un seuil de 30 % retirait aussi les lâchers
    d'hiver (13 → 27 m³/s).

    Examples
    --------
    >>> brut = RiverApp().mesures(ID)
    >>> propre = _filtrer(brut)
    >>> brut["flow"].notna().sum() - propre["flow"].notna().sum()     # mesures retirées
    """
    q = df["flow"].dropna()
    if q.empty:
        return df
    avant = q.shift(1).rolling(fenetre, min_periods=1).median()
    apres = q[::-1].shift(1).rolling(fenetre, min_periods=1).median()[::-1]
    ra, rp = q / avant - 1, q / apres - 1
    pic = ((ra < -seuil) & (rp < -seuil)) | ((ra > seuil) & (rp > seuil))
    out = df.copy()
    out.loc[pic[pic].index] = np.nan
    return out


def _au_pas(df, debut, fin, pas, trou_max):
    """
    Ramène des mesures irrégulières à un pas de temps régulier.

    Les instants sont les multiples de `pas` comptés depuis minuit UTC : la même grille pour
    toutes les stations et pour la météo. Chaque valeur est interpolée linéairement dans le
    temps entre la mesure juste avant et la mesure juste après.

    Parameters
    ----------
    df : pandas.DataFrame
        Mesures : index = date UTC trié, colonnes ``flow``, ``level`` (ou autres).
    debut, fin : pandas.Timestamp
        Période voulue (avec fuseau UTC) ; la grille va du premier au dernier multiple de
        `pas` compris dedans.
    pas : str
        Pas de temps, format pandas (``"15min"``, ``"1h"``, ``"4h"``, ``"1D"``).
    trou_max : str or pandas.Timedelta
        Écart maximal entre les deux mesures qui encadrent un instant : au-delà, NaN (on
        n'invente pas de valeur à travers un trou).

    Returns
    -------
    pandas.DataFrame
        Index ``date`` = la grille, mêmes colonnes que `df`, valeurs en float32. NaN si
        l'instant est dans un trou trop long ou hors des mesures.

    See Also
    --------
    RiverApp.mesures : l'applique avec ``pas=...``.

    Notes
    -----
    Chaque colonne est traitée séparément : débit et hauteur n'ont pas les mêmes trous.
    Une mesure qui tombe pile sur un instant de la grille est reprise telle quelle.
    Pour que les bords de la période soient bien calculés, `df` doit contenir quelques
    mesures juste avant `debut` et juste après `fin` (``mesures`` prend une marge de
    `trou_max`).

    Examples
    --------
    >>> brut = RiverApp().mesures(ID, "2020-01-01", "2020-12-31")
    >>> _au_pas(brut, pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-12-31", tz="UTC"),
    ...         "1h", "3h")
    """
    grille = pd.date_range(debut.ceil(pas), fin.floor(pas), freq=pas)
    g = grille.asi8
    sortie = {}
    for col in df.columns:
        s = df[col].dropna()
        t, v = s.index.asi8, s.to_numpy(dtype="float64")
        if len(t) == 0:
            sortie[col] = np.full(len(g), np.nan)
            continue
        i = np.searchsorted(t, g)                  # t[i-1] < g <= t[i]
        ok = (i > 0) & (i < len(t))
        avant, apres = np.clip(i - 1, 0, len(t) - 1), np.clip(i, 0, len(t) - 1)
        exact = (i < len(t)) & (t[apres] == g)
        dt = (t[apres] - t[avant]).astype("float64")
        poids = np.divide(g - t[avant], dt, out=np.zeros(len(g)), where=dt > 0)
        val = v[avant] + poids * (v[apres] - v[avant])
        val[~ok | (t[apres] - t[avant] > pd.Timedelta(trou_max).value)] = np.nan
        val[exact] = v[apres][exact]
        sortie[col] = val
    return pd.DataFrame(sortie, index=grille.rename("date")).astype("float32")


class RiverApp:
    """
    Accès aux débits et hauteurs des stations par l'API partenaire RiverApp, avec cache.

    Parameters
    ----------
    fournisseur : str, default "hubeau"
        Nom court du fournisseur de données (Hub'Eau par défaut).
    dossier : str or Path, default config.CACHE_HYDRO
        Dossier du cache.
    utilisateur, mot_de_passe : str, optional
        Identifiants RiverApp, nécessaires seulement pour télécharger. Par défaut lus dans
        ``.env`` (``RIVERAPP_UTILISATEUR``, ``RIVERAPP_MOT_DE_PASSE``) : ne jamais les écrire
        dans le code ni dans un notebook.

    Attributes
    ----------
    fournisseur : str
        Nom court du fournisseur.
    dossier : Path
        Dossier du cache.

    Raises
    ------
    RuntimeError
        À la lecture de ce qui n'est pas dans le cache, si `riverapp_acces.py` (privé) est
        absent ou si les identifiants le sont.

    See Also
    --------
    meteo.Meteo : l'équivalent pour la météo.

    Notes
    -----
    Une station se désigne par son identifiant RiverApp (unique pour tous les fournisseurs
    et tous les pays). Le code chez le fournisseur (ex. Hub'Eau ``X031001001``) est une
    colonne de `stations` qui permet de retrouver l'identifiant.

    Fichiers du cache :

    - ``stations_<fournisseur>.parquet`` : la liste des stations ;
    - ``stations/<id>.parquet`` : les mesures brutes d'une station (``flow``, ``level`` en float32) ;
    - ``couverture.csv`` : une ligne par station, première et dernière date demandées.

    Examples
    --------
    >>> from src.hydro import RiverApp
    >>> ra = RiverApp()
    >>> st = ra.stations()
    >>> ID = st.query("code == 'X031001001'").index[0]          # La Clapière
    >>> ra.mesures(ID, "2020-01-01", "2020-12-31", pas="1h", filtre=True)
    """

    def __init__(self, fournisseur="hubeau", dossier=config.CACHE_HYDRO,
                 utilisateur=config.RIVERAPP_UTILISATEUR, mot_de_passe=config.RIVERAPP_MOT_DE_PASSE):
        self.fournisseur = fournisseur
        self._auth = (utilisateur, mot_de_passe)
        self.dossier = Path(dossier)
        (self.dossier / "stations").mkdir(parents=True, exist_ok=True)
        self._couverture = self.dossier / "couverture.csv"
        self._liste = self.dossier / f"stations_{fournisseur}.parquet"

    def _telechargeur(self):
        """
        Module privé qui parle à l'API, après avoir vérifié qu'on peut télécharger.

        Returns
        -------
        module
            `riverapp_acces`.

        Raises
        ------
        RuntimeError
            Si le fichier privé `riverapp_acces.py` est absent (copie publique du projet : seul le
            cache est lisible) ou si les identifiants RiverApp sont absents de ``.env``.
        """
        if _acces is None:
            raise RuntimeError("source des débits privée : src/riverapp_acces.py absent, seul le cache "
                               f"({self.dossier}) est lisible")
        if not all(self._auth):
            raise RuntimeError("identifiants RiverApp absents : RIVERAPP_UTILISATEUR et RIVERAPP_MOT_DE_PASSE dans .env")
        return _acces

    # ---------------------------------------------------------------- stations
    def stations(self, rafraichir=False):
        """
        Liste des stations du fournisseur, mise en cache.

        Parameters
        ----------
        rafraichir : bool, default False
            Relire la liste depuis l'API au lieu du cache (nouvelles stations).

        Returns
        -------
        pandas.DataFrame
            Index ``id`` (identifiant RiverApp) ; colonnes ``code`` (chez le fournisseur,
            peut être vide), ``lat``, ``lon``, ``nom``, ``cours_eau``,
            ``plus_ancienne_mesure`` (UTC), et les autres champs de l'API.

        Examples
        --------
        >>> st = RiverApp().stations()
        >>> st.query("code == 'X031001001'")          # La Clapière
        """
        if self._liste.exists() and not rafraichir:
            return pd.read_parquet(self._liste)
        st = self._telechargeur().stations(self.fournisseur, self._auth)
        st.to_parquet(self._liste)
        return st

    # ---------------------------------------------------------------- cache
    def _lire_couverture(self):
        """
        Lit la table de couverture du cache.

        Returns
        -------
        pandas.DataFrame
            Index ``id`` (identifiant RiverApp) ; colonnes ``debut`` et ``fin`` (UTC) :
            période déjà demandée à l'API. Table vide s'il n'y a pas encore de cache.
        """
        if not self._couverture.exists():
            return pd.DataFrame(columns=["id", "debut", "fin"]).set_index("id")
        c = pd.read_csv(self._couverture, index_col="id")
        return c.assign(debut=pd.to_datetime(c["debut"], utc=True), fin=pd.to_datetime(c["fin"], utc=True))

    def contenu_cache(self):
        """
        Stations en cache, avec la place prise sur le disque.

        Returns
        -------
        pandas.DataFrame
            Une ligne par station : ``id``, ``code`` (chez le fournisseur), ``debut``, ``fin``
            (période couverte), ``mesures`` (nombre de lignes), ``mo`` (Mo sur le disque).

        See Also
        --------
        vider_cache : retire tout ou partie du cache.
        """
        c = self._lire_couverture()
        codes = self.stations()["code"]
        lignes = []
        for id_, (debut, fin) in c[["debut", "fin"]].iterrows():
            f = self.dossier / "stations" / f"{id_}.parquet"
            lignes.append({"id": id_, "code": codes.get(id_), "debut": debut, "fin": fin,
                           "mesures": len(pd.read_parquet(f)) if f.exists() else 0,
                           "mo": round(f.stat().st_size / 1e6, 3) if f.exists() else 0})
        return pd.DataFrame(lignes)

    def vider_cache(self, ids=None):
        """
        Retire les mesures en cache de toutes les stations, ou seulement de certaines.

        La liste des stations est gardée.

        Parameters
        ----------
        ids : list of str, optional
            Identifiants RiverApp des stations à retirer. Par défaut, toutes.

        See Also
        --------
        contenu_cache : ce qui est en cache.
        """
        c = self._lire_couverture()
        cibles = list(c.index) if ids is None else list(ids)
        for id_ in cibles:
            (self.dossier / "stations" / f"{id_}.parquet").unlink(missing_ok=True)
        c.drop(index=[k for k in cibles if k in c.index]).to_csv(self._couverture)

    # ---------------------------------------------------------------- mesures
    def mesures(self, id_, debut=None, fin=None, pas=None, trou_max="3h", filtre=False):
        """
        Mesures d'une station, brutes ou ramenées à un pas régulier.

        Ne télécharge que ce qui manque au cache, avant et après la période déjà en cache,
        par tranches de 32 jours (~0,6 s par tranche).

        Parameters
        ----------
        id_ : str
            Identifiant RiverApp de la station (index de `stations`).
        debut : str or pandas.Timestamp, optional
            Début (UTC). Par défaut, la plus ancienne mesure de la station.
        fin : str or pandas.Timestamp, optional
            Fin (UTC). Par défaut, maintenant.
        pas : str, optional
            Pas de temps régulier, format pandas (``"1h"``, ``"4h"``, ``"1D"``) : valeurs
            interpolées sur une grille calée sur minuit UTC (cf. `_au_pas`). Par défaut,
            les mesures brutes, à leurs instants d'origine.
        trou_max : str, default "3h"
            Avec `pas` seulement : au-delà de cet écart entre deux mesures, pas d'interpolation
            (NaN).
        filtre : bool, default False
            Retirer d'abord les pics isolés (cf. `_filtrer`). Le cache garde les mesures brutes,
            on peut donc comparer avec et sans.

        Returns
        -------
        pandas.DataFrame
            Index = date UTC ; colonnes ``flow`` (débit, m³/s) et ``level`` (hauteur, m),
            float32, NaN si absent.

        Raises
        ------
        KeyError
            Si `id_` n'est pas une station du fournisseur.

        See Also
        --------
        stations : pour trouver l'identifiant à partir du code de la station.

        Examples
        --------
        >>> ra = RiverApp()
        >>> ID = ra.stations().query("code == 'X031001001'").index[0]      # La Clapière
        >>> ra.mesures(ID)                                    # tout, brut
        >>> ra.mesures(ID, "2017-01-01", "2024-12-31", pas="1h", filtre=True)
        """
        st = self.stations().loc[id_]
        debut = pd.Timestamp(debut, tz="UTC") if debut else st["plus_ancienne_mesure"]
        fin = pd.Timestamp(fin, tz="UTC") if fin else pd.Timestamp.now(tz="UTC").floor("min")
        couv = self._lire_couverture()
        if id_ in couv.index:
            d0, d1 = couv.loc[id_, "debut"], couv.loc[id_, "fin"]
            manques = ([(debut, d0)] if debut < d0 else []) + ([(d1, fin)] if fin > d1 else [])
        else:
            manques = [(debut, fin)]

        chemin = self.dossier / "stations" / f"{id_}.parquet"
        nouveaux = []
        for a, b in manques:
            while a < b:
                c = min(b, a + TRANCHE)
                lot = self._telechargeur().mesures(id_, a, c, self._auth)
                if len(lot):
                    nouveaux.append(lot)
                a = c
        if manques:
            df = pd.concat(nouveaux) if nouveaux else pd.DataFrame(
                columns=["flow", "level"], index=pd.DatetimeIndex([], tz="UTC"), dtype="float32")
            if chemin.exists():
                df = pd.concat([pd.read_parquet(chemin), df])
            df = df[~df.index.duplicated(keep="last")].sort_index()     # bornes incluses : doublons aux raccords
            df.to_parquet(chemin)
            nouveau_debut = min(debut, couv.loc[id_, "debut"]) if id_ in couv.index else debut
            nouvelle_fin = max(fin, couv.loc[id_, "fin"]) if id_ in couv.index else fin
            couv.loc[id_, ["debut", "fin"]] = [nouveau_debut, nouvelle_fin]
            couv.to_csv(self._couverture)
        tout = pd.read_parquet(chemin)
        if filtre:
            tout = _filtrer(tout)
        if pas is None:
            return tout.loc[debut:fin]
        marge = pd.Timedelta(trou_max)       # les mesures voisines juste hors de la période servent aux bords
        return _au_pas(tout.loc[debut - marge:fin + marge], debut, fin, pas, trou_max)

"""
================================================================================
MODULE bassins.py
================================================================================
Forêt d'anti-arborescences des bassins versants (BasinATLAS / HydroBASINS, niveau 12).

- Un NŒUD = un bassin ; son arc sort vers le bassin AVAL (colonne NEXT_DOWN).
  Un bassin a au plus un aval, mais plusieurs affluents : c'est une anti-arborescence
  (tout converge vers une racine = exutoire vers la mer ou un bassin fermé).
  L'ensemble des exutoires donne une FORÊT.

- `ForetBassins` : la structure elle-même, sur des tableaux numpy (1 million de nœuds
  sans souci). Remonter vers l'amont = lire une plage contiguë d'un tableau.

- `localiser` / `bassins_amont_du_point` : dans quel bassin tombe un point, et tout son amont.

- `recouvrement` / `carreaux_amont` : quels carreaux météo Open-Meteo couvrent un bassin (ou tout
  son amont), et quelle part ; `voronoi_meteo_amont` : leurs cellules, pour les dessiner.

- `Reseau` : la table des bassins + la forêt + les stations.

Les identifiants HYBAS_ID sont manipulés en entiers 64 bits (jamais en flottants).
================================================================================
"""

import functools

import numpy as np
import pandas as pd
import pyogrio
import geopandas as gpd
import shapely

from . import config


# ==============================================================================
# 1. LECTURE DES BASSINS
# ==============================================================================

def _en_entier64(serie, nom):
    """
    Convertit une colonne d'identifiants en entiers 64 bits, sans perte.

    Les identifiants HYBAS_ID ont 10 chiffres ; lus par GDAL, ils arrivent parfois en
    flottants. On vérifie qu'ils sont bien entiers avant de convertir.

    Parameters
    ----------
    serie : pandas.Series
        Colonne d'identifiants (entiers ou flottants).
    nom : str
        Nom de la colonne, pour le message d'erreur.

    Returns
    -------
    numpy.ndarray of int64
        Les identifiants.

    Raises
    ------
    ValueError
        Si une valeur manque ou n'est pas entière.
    """
    v = serie.to_numpy()
    if np.issubdtype(v.dtype, np.floating):
        if np.any(np.isnan(v)) or not np.array_equal(v, np.round(v)):
            raise ValueError(f"colonne {nom} : valeurs non entières ou manquantes")
    return v.astype(np.int64)


def charger_bassins():
    """
    Table des bassins du monde entier, niveau 12, sans géométrie.

    Topologie (`config.COLONNES_TOPOLOGIE`) + attributs de `config.ATTRIBUTS`. Mise en cache
    dans ``data/cache/bassins_lev12.parquet`` : la première lecture (~1 million de bassins)
    est lente, les suivantes quasi instantanées.

    Returns
    -------
    pandas.DataFrame
        Index ``HYBAS_ID`` (int64) ; une colonne par champ. ``NEXT_DOWN`` = bassin aval
        (0 = exutoire), ``UP_AREA`` = surface amont totale en km².

    See Also
    --------
    ForetBassins.depuis_table : construit la forêt à partir de cette table.
    Reseau : la charge automatiquement.

    Examples
    --------
    >>> b = charger_bassins()
    >>> b.loc[2121189970, ["NEXT_DOWN", "UP_AREA"]]       # bassin de La Clapière
    """
    chemin = config.CACHE / "bassins_lev12.parquet"
    if chemin.exists():
        return pd.read_parquet(chemin)
    colonnes = list(dict.fromkeys(config.COLONNES_TOPOLOGIE + config.attributs_par_defaut()))
    df = pyogrio.read_dataframe(config.GDB_BASSINS, layer=config.COUCHE_NIVEAU_12, columns=colonnes,
                                read_geometry=False, use_arrow=True)
    for c in ("HYBAS_ID", "NEXT_DOWN", "NEXT_SINK", "MAIN_BAS", "PFAF_ID"):
        df[c] = _en_entier64(df[c], c)
    df = df.set_index("HYBAS_ID")
    config.CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(chemin)
    return df


# ==============================================================================
# 2. LA FORÊT D'ANTI-ARBORESCENCES
# ==============================================================================

class ForetBassins:
    """
    Réseau des rivières : chaque bassin pointe vers son bassin aval.

    Un bassin a au plus un aval mais peut avoir plusieurs affluents ; tout converge vers une
    racine (exutoire vers la mer ou zone fermée). Stockée sur des tableaux numpy : un million
    de bassins sans souci, et tout l'amont d'un bassin se lit en une seule tranche de tableau.

    Parameters
    ----------
    ids : array-like of int
        Identifiants des bassins (HYBAS_ID), sans doublon.
    next_down : array-like of int
        Pour chaque bassin, l'identifiant de son aval. 0, ou un aval absent de `ids`,
        fait du bassin une racine.

    Attributes
    ----------
    n : int
        Nombre de bassins.
    ids : numpy.ndarray of int64
        Identifiants, dans l'ordre donné.
    aval_idx : numpy.ndarray of int64
        ``aval_idx[i]`` = position du bassin aval de i (-1 pour une racine).
    racines : numpy.ndarray of int64
        Positions des racines.
    profondeur : numpy.ndarray of int32
        Nombre de bassins entre i et sa racine (0 pour une racine).
    taille : numpy.ndarray of int64
        Nombre de bassins du sous-arbre amont de i, i compris.
    tin : numpy.ndarray of int64
        Rang de i dans un parcours « parent avant enfants ».
    ordre : numpy.ndarray of int64
        ``ordre[k]`` = position du bassin de rang k (l'inverse de `tin`).

    Raises
    ------
    ValueError
        Si un identifiant est en double, si un bassin pointe vers lui-même, ou s'il y a un
        cycle dans `next_down`.

    See Also
    --------
    depuis_table : la construit directement depuis la table des bassins.
    Reseau : la contient, avec la table des bassins et les stations.

    Notes
    -----
    Propriété clé : le sous-arbre amont de i occupe exactement les rangs
    ``tin[i] .. tin[i] + taille[i] - 1`` de `ordre`. Les affluents directs sont rangés au
    format CSR (``_debut``, ``_enfants``). Tous les pré-calculs sont faits une fois, niveau de
    profondeur par niveau, sans boucle sur les bassins.

    Examples
    --------
    >>> f = ForetBassins([1, 2, 3, 4], [0, 1, 1, 2])     # 4 -> 2 -> 1 <- 3
    >>> f.amont(1)
    array([1, 2, 4, 3])
    >>> f.aval(4)
    array([4, 2, 1])
    """

    def __init__(self, ids, next_down):
        ids = np.asarray(ids, dtype=np.int64)
        nd = np.asarray(next_down, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("HYBAS_ID en double")
        self.n = len(ids)
        self.ids = ids
        self._idx = pd.Index(ids)
        # -1 : pas d'aval dans la forêt (NEXT_DOWN = 0, ou aval hors du périmètre lu)
        self.aval_idx = self._idx.get_indexer(nd).astype(np.int64)
        if np.any(self.aval_idx == np.arange(self.n)):
            raise ValueError("un bassin pointe vers lui-même")
        self.racines = np.flatnonzero(self.aval_idx < 0)
        self._construire()

    @classmethod
    def depuis_table(cls, bassins, couper_endoreiques=True):
        """
        Construit la forêt depuis la table des bassins.

        Parameters
        ----------
        bassins : pandas.DataFrame
            Index HYBAS_ID ; colonnes ``NEXT_DOWN`` et ``ENDO`` (sortie de `charger_bassins`,
            éventuellement réduite à une région).
        couper_endoreiques : bool, default True
            Une zone fermée (ENDO > 0, sans écoulement vers la mer) reliée par NEXT_DOWN à un
            bassin ouvert ne lui envoie pas réellement d'eau : on coupe ce lien, le bassin
            fermé devient une racine.

        Returns
        -------
        ForetBassins

        Notes
        -----
        Avec la coupe, la surface amont recalculée retombe sur UP_AREA : France + bordures,
        0 bassin discordant (contre 3,3 % sans coupe).

        Examples
        --------
        >>> f = ForetBassins.depuis_table(charger_bassins())
        """
        nd = bassins["NEXT_DOWN"].to_numpy().copy()
        if couper_endoreiques:
            endo = bassins["ENDO"].to_numpy()
            par = bassins.index.get_indexer(nd)
            ok = par >= 0
            coupe = np.zeros(len(nd), dtype=bool)
            coupe[ok] = (endo[ok] > 0) & (endo[par[ok]] == 0)
            nd[coupe] = 0
        return cls(bassins.index.to_numpy(), nd)

    def _construire(self):
        """
        Pré-calcule affluents, profondeur, taille des sous-arbres, `tin` et `ordre`.

        Parcourt la forêt niveau de profondeur par niveau (des racines vers l'amont), avec
        des opérations vectorisées : aucune boucle sur les bassins.

        Raises
        ------
        ValueError
            Si des bassins ne sont reliés à aucune racine (cycle dans NEXT_DOWN).
        """
        n, down = self.n, self.aval_idx
        # --- affluents directs (CSR) : enfants triés par parent
        enfants = np.flatnonzero(down >= 0)
        enfants = enfants[np.argsort(down[enfants], kind="stable")]
        nb_enf = np.bincount(down[down >= 0], minlength=n)
        debut = np.concatenate(([0], np.cumsum(nb_enf)))
        self._enfants, self._debut = enfants, debut

        # --- niveaux de profondeur, en partant des racines (les enfants d'un même
        #     parent restent contigus dans chaque niveau)
        niveaux = [self.racines]
        front = self.racines
        while True:
            c = nb_enf[front]
            total = int(c.sum())
            if total == 0:
                break
            excl = np.cumsum(c) - c
            idx = np.repeat(debut[front] - excl, c) + np.arange(total)
            front = enfants[idx]
            niveaux.append(front)
        vus = sum(len(x) for x in niveaux)
        if vus != n:
            raise ValueError(f"{n - vus} bassins inaccessibles depuis une racine : "
                             f"cycle dans NEXT_DOWN")
        self.profondeur = np.empty(n, dtype=np.int32)
        for k, niv in enumerate(niveaux):
            self.profondeur[niv] = k

        def groupes(niv):
            """Début de chaque groupe de frères dans un niveau + leur longueur."""
            par = down[niv]
            deb = np.flatnonzero(np.r_[True, par[1:] != par[:-1]])
            return par, deb, np.diff(np.r_[deb, len(niv)])

        # --- taille des sous-arbres : du bas vers le haut
        taille = np.ones(n, dtype=np.int64)
        for niv in reversed(niveaux[1:]):
            par, deb, _ = groupes(niv)
            taille[par[deb]] += np.add.reduceat(taille[niv], deb)

        # --- rang de parcours : du haut vers le bas
        tin = np.empty(n, dtype=np.int64)
        tr = taille[self.racines]
        tin[self.racines] = np.cumsum(tr) - tr
        for niv in niveaux[1:]:
            par, deb, lon = groupes(niv)
            cs = np.cumsum(taille[niv])
            excl = cs - taille[niv]
            tin[niv] = tin[par] + 1 + (excl - np.repeat(excl[deb], lon))
        ordre = np.empty(n, dtype=np.int64)
        ordre[tin] = np.arange(n)
        self.taille, self.tin, self.ordre = taille, tin, ordre

    # ---------------------------------------------------------------- accès
    def indice(self, hybas_id):
        """
        Position d'un bassin dans les tableaux de la forêt.

        Parameters
        ----------
        hybas_id : int
            Identifiant du bassin.

        Returns
        -------
        int
            Position i, telle que ``ids[i] == hybas_id``.

        Raises
        ------
        KeyError
            Si le bassin n'est pas dans la forêt.
        """
        i = int(self._idx.get_indexer([hybas_id])[0])
        if i < 0:
            raise KeyError(hybas_id)
        return i

    def amont(self, hybas_id, inclure=True):
        """
        Tous les bassins situés en amont d'un bassin.

        Parameters
        ----------
        hybas_id : int
            Identifiant du bassin.
        inclure : bool, default True
            Compter le bassin lui-même.

        Returns
        -------
        numpy.ndarray of int64
            Identifiants HYBAS_ID ; le bassin lui-même en premier si `inclure`.

        Raises
        ------
        KeyError
            Si le bassin n'est pas dans la forêt.

        See Also
        --------
        affluents : seulement les bassins qui se jettent directement dedans.
        aval : le chemin dans l'autre sens.

        Examples
        --------
        >>> R = Reseau()
        >>> len(R.foret.amont(2121189970))          # amont de La Clapière
        """
        i = self.indice(hybas_id)
        a = self.tin[i] + (0 if inclure else 1)
        return self.ids[self.ordre[a:self.tin[i] + self.taille[i]]]

    def affluents(self, hybas_id):
        """
        Bassins qui se jettent DIRECTEMENT dans un bassin.

        Parameters
        ----------
        hybas_id : int
            Identifiant du bassin.

        Returns
        -------
        numpy.ndarray of int64
            Identifiants HYBAS_ID ; vide pour un bassin de tête.

        See Also
        --------
        amont : tout l'amont, pas seulement le premier rang.
        """
        i = self.indice(hybas_id)
        return self.ids[self._enfants[self._debut[i]:self._debut[i + 1]]]

    def aval(self, hybas_id, inclure=True):
        """
        Chemin d'un bassin jusqu'à son exutoire.

        Parameters
        ----------
        hybas_id : int
            Identifiant du bassin.
        inclure : bool, default True
            Commencer par le bassin lui-même.

        Returns
        -------
        numpy.ndarray of int64
            Le bassin, son aval, l'aval de l'aval, ..., jusqu'à la racine (mer ou zone fermée).

        See Also
        --------
        amont : dans l'autre sens.
        """
        i = self.indice(hybas_id)
        chemin = [i] if inclure else []
        while self.aval_idx[i] >= 0:
            i = int(self.aval_idx[i])
            chemin.append(i)
        return self.ids[chemin]


# ==============================================================================
# 3. LOCALISER DES POINTS DANS LES BASSINS
# ==============================================================================

def localiser(lon, lat, gdb=config.GDB_BASSINS, couche=config.COUCHE_NIVEAU_12, tuile=2.0):
    """
    Bassin qui contient chaque point.

    Les points sont regroupés par tuiles de `tuile` degrés : pour chaque tuile, seuls les
    polygones de bassins utiles sont lus dans la base.

    Parameters
    ----------
    lon, lat : array-like of float
        Coordonnées des points, en degrés (WGS84).
    gdb : str or Path, default config.GDB_BASSINS
        Base BasinATLAS.
    couche : str, default config.COUCHE_NIVEAU_12
        Couche de bassins à utiliser.
    tuile : float, default 2.0
        Côté des tuiles de regroupement, en degrés.

    Returns
    -------
    pandas.arrays.IntegerArray
        HYBAS_ID du bassin de chaque point, dans l'ordre des points ; <NA> si le point
        n'est dans aucun bassin (en mer).

    See Also
    --------
    bassins_amont_du_point : localise un point puis remonte tout son amont.
    Reseau.ajouter_stations : localise des stations et contrôle le résultat.

    Notes
    -----
    Les coordonnées d'une station peuvent être décalées de quelques centaines de mètres :
    près d'une limite de bassin, comparer la surface amont du bassin trouvé avec la surface
    officielle de la station (``ecart_surface`` dans `Reseau.ajouter_stations`).

    Examples
    --------
    >>> localiser([6.498010], [44.555356])          # La Clapière
    <IntegerArray>
    [2121189970]
    Length: 1, dtype: Int64
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    pts = gpd.GeoDataFrame({"i": np.arange(len(lon))},
                           geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")
    tuiles = pd.DataFrame({"tx": np.floor(lon / tuile), "ty": np.floor(lat / tuile)})
    res = np.full(len(lon), -1, dtype=np.int64)
    eps = 1e-6
    for _, idx in tuiles.groupby(["tx", "ty"]).indices.items():
        sub = pts.iloc[idx]
        x0, y0, x1, y1 = sub.total_bounds
        polys = pyogrio.read_dataframe(gdb, layer=couche, columns=["HYBAS_ID"],
                                       bbox=(x0 - eps, y0 - eps, x1 + eps, y1 + eps))
        if polys.empty:
            continue
        j = gpd.sjoin(sub, polys, how="left", predicate="intersects")
        j = j[~j.index.duplicated(keep="first")]
        ok = j["HYBAS_ID"].notna().to_numpy()
        res[j["i"].to_numpy()[ok]] = j["HYBAS_ID"].to_numpy()[ok].astype(np.int64)
    return pd.arrays.IntegerArray(res, mask=res < 0)


def bassins_amont_du_point(lon, lat, foret, bassins=None):
    """
    Tous les bassins versants en amont d'un point quelconque.

    LA fonction de base : le point n'a pas besoin d'être une station enregistrée. Revient à
    `localiser` puis `ForetBassins.amont`.

    Parameters
    ----------
    lon, lat : float
        Coordonnées du point, en degrés.
    foret : ForetBassins
        La forêt (ex. ``Reseau().foret``).
    bassins : pandas.DataFrame, optional
        Table des bassins : si donnée, renvoie leurs lignes (attributs ATLAS) plutôt que les
        seuls identifiants.

    Returns
    -------
    numpy.ndarray of int64 or pandas.DataFrame
        Identifiants HYBAS_ID de l'amont (bassin du point compris), ou leurs lignes de
        `bassins`.

    Raises
    ------
    ValueError
        Si aucun bassin ne contient le point.

    Examples
    --------
    >>> R = Reseau()
    >>> amont = bassins_amont_du_point(6.498010, 44.555356, R.foret, R.bassins)   # La Clapière
    >>> amont["SUB_AREA"].sum()                     # surface amont, km²
    """
    bassin = localiser([lon], [lat])[0]
    if bassin is pd.NA:
        raise ValueError(f"aucun bassin ne contient le point ({lon}, {lat})")
    amont = foret.amont(int(bassin))
    return bassins.loc[amont] if bassins is not None else amont


# ==============================================================================
# 4. CARREAUX MÉTÉO (grille Open-Meteo) ET RECOUVREMENT DES BASSINS
# ==============================================================================

def _grille(x0, y0, x1, y1):
    """
    Centres des carreaux Open-Meteo sur un rectangle, calculés sans aucun appel.

    Ajoute 2 rangées et 2 points de chaque côté, pour que les cellules du bord soient fermées
    par leurs vraies voisines (un nombre de points et pas des degrés : les carreaux
    s'élargissent en longitude vers les pôles).

    Parameters
    ----------
    x0, y0, x1, y1 : float
        Rectangle (longitude min, latitude min, longitude max, latitude max), en degrés.

    Returns
    -------
    numero : numpy.ndarray of int64
        Numéro de chaque carreau dans la grille mondiale (0 = premier point de la rangée la
        plus au nord).
    lon : numpy.ndarray of float
        Longitude du centre, en degrés (peut sortir de [-180, 180] près de l'antiméridien).
    lat : numpy.ndarray of float
        Latitude du centre, en degrés.

    See Also
    --------
    recouvrement, voronoi_meteo_amont : s'en servent pour le pavage de Voronoï.

    Notes
    -----
    Les centres que renvoie l'archive Open-Meteo sont exactement les points de la grille
    gaussienne réduite octaédrique O1280 de l'ECMWF (celle d'ERA5-Land) -- vérifié sur des
    centres réels de 10 régions du monde : latitudes = les 2560 latitudes gaussiennes (à 3e-6°
    près), rangée i (1 = la plus proche du pôle) de 4i + 16 points, longitudes k × 360 / (4i + 16).
    """
    lat_g = np.sort(np.degrees(np.arcsin(np.polynomial.legendre.leggauss(2560)[0])))[::-1]
    nb_lon = 4 * np.minimum(np.arange(1, 2561), np.arange(2560, 0, -1)) + 16
    debut_rangee = np.concatenate(([0], np.cumsum(nb_lon)))
    j_nord, j_sud = np.searchsorted(-lat_g, [-y1, -y0])
    numeros, lons, lats = [], [], []
    for j in range(max(j_nord - 2, 0), min(j_sud + 2, 2560)):
        n = nb_lon[j]
        k = np.arange(np.floor(x0 * n / 360) - 2, np.ceil(x1 * n / 360) + 3).astype(np.int64)
        numeros.append(debut_rangee[j] + k % n)
        lons.append(k * 360 / n)
        lats.append(np.full(len(k), lat_g[j]))
    return np.concatenate(numeros), np.concatenate(lons), np.concatenate(lats)


@functools.lru_cache(maxsize=None)
def recouvrement(continent, gdb=config.GDB_BASSINS, couche=config.COUCHE_NIVEAU_12):
    """
    Table de recouvrement bassins × carreaux météo d'un continent.

    Une ligne par morceau commun à un bassin et un carreau. Calculée la première fois (long),
    puis relue depuis ``data/cache/recouvrement_<continent>.parquet`` et gardée en mémoire.

    Parameters
    ----------
    continent : int
        Premier chiffre du HYBAS_ID : 1 Afrique, 2 Europe, 3 Sibérie, 4 Asie, 5 Australie,
        6 Amérique du Sud, 7 Amérique du Nord, 8 Arctique, 9 Groenland.
    gdb : str or Path, default config.GDB_BASSINS
        Base BasinATLAS.
    couche : str, default config.COUCHE_NIVEAU_12
        Couche de bassins.

    Returns
    -------
    pandas.DataFrame
        Index ``HYBAS_ID`` (les carreaux d'un bassin se lisent directement) ; colonnes
        ``carreau`` (numéro dans la grille mondiale), ``lon_centre``, ``lat_centre``,
        ``surface_km2`` (surface du morceau). Triée par bassin puis surface décroissante.

    See Also
    --------
    carreaux_amont : additionne ces lignes sur tout l'amont d'un bassin.

    Notes
    -----
    Aucune rivière ne passe d'un continent à l'autre (vérifié) : une table par continent
    suffit. Le pavage de Voronoï est fait en degrés bruts, comme Open-Meteo choisit son point
    en mode ``cell_selection=nearest`` ; les surfaces sont calculées dans une projection qui
    conserve les surfaces (Lambert azimutale centrée sur le continent).

    Examples
    --------
    >>> t = recouvrement(2)                     # Europe
    >>> t.loc[[2121189970]]                     # carreaux du bassin de La Clapière
    """
    chemin = config.CACHE / f"recouvrement_{continent}.parquet"
    if chemin.exists():
        return pd.read_parquet(chemin)
    polys = pyogrio.read_dataframe(gdb, layer=couche, columns=["HYBAS_ID"],
                                   where=f"HYBAS_ID >= {continent}000000000 AND HYBAS_ID < {continent + 1}000000000")
    numero, lon, lat = _grille(*polys.total_bounds)
    cellules = np.array(shapely.voronoi_polygons(shapely.multipoints(shapely.points(lon, lat)), ordered=True).geoms)
    i_b, i_c = shapely.STRtree(cellules).query(polys.geometry.values, predicate="intersects")
    morceaux = gpd.GeoSeries(shapely.intersection(polys.geometry.values[i_b], cellules[i_c]), crs="EPSG:4326")
    c = shapely.box(*polys.total_bounds).centroid
    surface = morceaux.to_crs(f"+proj=laea +lat_0={c.y} +lon_0={c.x} +units=m").area.to_numpy() / 1e6
    garde = surface > 0
    table = pd.DataFrame({
        "carreau": numero[i_c][garde], "lon_centre": lon[i_c][garde], "lat_centre": lat[i_c][garde],
        "surface_km2": surface[garde],
    }, index=pd.Index(_en_entier64(polys["HYBAS_ID"], "HYBAS_ID")[i_b][garde], name="HYBAS_ID"))
    table = table.sort_values(["HYBAS_ID", "surface_km2"], ascending=[True, False])
    config.CACHE.mkdir(parents=True, exist_ok=True)
    table.to_parquet(chemin)
    return table


def carreaux_amont(zone, foret, amont=True):
    """
    Carreaux météo qui couvrent un bassin, ou tout son amont, avec la part de chacun.

    Lit les lignes de ces bassins dans la table de recouvrement et les additionne carreau par
    carreau. Aucun contour, aucun appel à Open-Meteo : instantané une fois la table en cache.

    Parameters
    ----------
    zone : int
        HYBAS_ID du bassin (ex. celui d'une station, trouvé par `localiser`).
    foret : ForetBassins
        La forêt (ex. ``Reseau().foret``).
    amont : bool, default True
        Prendre tout l'amont de `zone` (zone comprise) ; sinon `zone` seule.

    Returns
    -------
    pandas.DataFrame
        Une ligne par carreau : ``carreau``, ``lon_centre``, ``lat_centre``, ``surface_km2``
        (surface couverte), ``pourcentage`` (part de la surface totale). Triée par
        pourcentage décroissant. Se passe telle quelle à `meteo.Meteo.telecharger`.

    See Also
    --------
    recouvrement : la table lue.
    voronoi_meteo_amont : les mêmes carreaux, avec leurs contours pour une carte.

    Examples
    --------
    >>> R = Reseau()
    >>> zone = localiser([6.498010], [44.555356])[0]        # La Clapière
    >>> c = carreaux_amont(zone, R.foret)
    >>> len(c)
    41
    """
    ids = foret.amont(zone) if amont else [zone]
    lignes = recouvrement(int(zone) // 10**9).loc[ids]
    t = lignes.groupby(["carreau", "lon_centre", "lat_centre"], as_index=False)["surface_km2"].sum()
    t["pourcentage"] = 100 * t["surface_km2"] / t["surface_km2"].sum()
    return t.sort_values("pourcentage", ascending=False).reset_index(drop=True)


def voronoi_meteo_amont(zone, foret, gdb=config.GDB_BASSINS, couche=config.COUCHE_NIVEAU_12):
    """
    Cellules des carreaux météo découpées par le contour de l'amont, pour DESSINER une carte.

    Les chiffres seuls (surfaces, pourcentages) s'obtiennent plus vite, sans contour, par
    `carreaux_amont`.

    Parameters
    ----------
    zone : int
        HYBAS_ID du bassin ; tout son amont est pris (zone comprise).
    foret : ForetBassins
        La forêt.
    gdb : str or Path, default config.GDB_BASSINS
        Base BasinATLAS.
    couche : str, default config.COUCHE_NIVEAU_12
        Couche de bassins.

    Returns
    -------
    geopandas.GeoDataFrame
        Une ligne par carreau : ``carreau``, ``lon_centre``, ``lat_centre``, ``surface_km2``,
        ``pourcentage``, et la géométrie de la cellule découpée (EPSG:4326). Triée par
        pourcentage décroissant.

    See Also
    --------
    carreaux_amont : les mêmes chiffres, sans géométrie.

    Notes
    -----
    Voronoï en degrés bruts, comme Open-Meteo en mode ``cell_selection=nearest`` : 400 points
    sur 400 servis par le centre de leur cellule, contre 385 avec des distances en km.
    Près des côtes, à vérifier : ERA5-Land n'a pas de points en mer.

    Examples
    --------
    >>> R = Reseau()
    >>> v = voronoi_meteo_amont(2121189970, R.foret)        # La Clapière
    >>> v.plot(column="pourcentage", legend=True)
    """
    ids = ",".join(str(int(h)) for h in foret.amont(zone))
    polys = pyogrio.read_dataframe(gdb, layer=couche, where=f"HYBAS_ID IN ({ids})", columns=["HYBAS_ID"])
    contour = polys.geometry.union_all()
    numero, lon, lat = _grille(*polys.total_bounds)
    cellules = shapely.voronoi_polygons(shapely.multipoints(shapely.points(lon, lat)), ordered=True)
    decoupe = gpd.GeoSeries(shapely.intersection(np.array(cellules.geoms), contour), crs="EPSG:4326")
    garde = ~decoupe.is_empty.to_numpy()
    decoupe = decoupe[garde]
    c = contour.centroid
    surface = decoupe.to_crs(f"+proj=laea +lat_0={c.y} +lon_0={c.x} +units=m").area.to_numpy() / 1e6

    return gpd.GeoDataFrame({
        "carreau": numero[garde],
        "lon_centre": lon[garde],
        "lat_centre": lat[garde],
        "surface_km2": surface,
        "pourcentage": 100 * surface / surface.sum(),
    }, geometry=decoupe.to_numpy(), crs="EPSG:4326").sort_values(
        "pourcentage", ascending=False).reset_index(drop=True)


# ==============================================================================
# 5. LE RÉSEAU : bassins + forêt + stations
# ==============================================================================

_COLS_STATIONS = ["lon", "lat", "bassin", "surface_officielle_km2", "surface_amont_bassin_km2", "ecart_surface"]


class Reseau:
    """
    La table des bassins, la forêt et les stations, réunies.

    Parameters
    ----------
    bassins : pandas.DataFrame, optional
        Table des bassins (index HYBAS_ID). Par défaut, le monde entier (`charger_bassins`).

    Attributes
    ----------
    bassins : pandas.DataFrame
        Index HYBAS_ID : topologie + attributs ATLAS.
    foret : ForetBassins
        Pour remonter ou descendre les rivières.
    stations : pandas.DataFrame
        Index ``code`` ; colonnes ``lon``, ``lat``, ``bassin`` (HYBAS_ID où elle se trouve),
        ``surface_officielle_km2``, ``surface_amont_bassin_km2``, ``ecart_surface``. Vide au
        départ, remplie par `ajouter_stations`.

    Examples
    --------
    >>> R = Reseau()
    >>> R.foret.amont(2121189970)                    # amont de La Clapière
    """

    def __init__(self, bassins=None):
        self.bassins = charger_bassins() if bassins is None else bassins
        self.foret = ForetBassins.depuis_table(self.bassins)
        self.stations = pd.DataFrame(columns=_COLS_STATIONS).rename_axis("code")

    def ajouter_stations(self, df, code="code", lon="lon", lat="lat", surface_km2=None):
        """
        Place des stations dans leur bassin et contrôle le résultat.

        Une station déjà présente (même code) est remplacée.

        Parameters
        ----------
        df : pandas.DataFrame
            Table des stations.
        code : str, default "code"
            Nom de la colonne du code de station (devient l'index de `stations`).
        lon, lat : str, default "lon", "lat"
            Noms des colonnes des coordonnées, en degrés.
        surface_km2 : str, optional
            Nom de la colonne de la surface de bassin OFFICIELLE (ex. Hub'Eau). Sert à
            calculer ``ecart_surface``.

        Returns
        -------
        int
            Nombre de stations hors de tout bassin.

        Notes
        -----
        ``ecart_surface`` = (surface amont du bassin trouvé - officielle) / officielle. Un
        grand écart signale un mauvais bassin (station près d'une limite) ou un très petit
        bassin jaugé (plus petit qu'un bassin de niveau 12).

        Examples
        --------
        >>> R = Reseau()
        >>> st = pd.DataFrame({"code": ["X031001001"], "lon": [6.498010], "lat": [44.555356]})
        >>> R.ajouter_stations(st)
        >>> R.stations
        """
        d = pd.DataFrame({"lon": df[lon].to_numpy(float), "lat": df[lat].to_numpy(float)},
                         index=pd.Index(df[code].to_numpy(), name="code"))
        d["bassin"] = localiser(d["lon"], d["lat"])
        d["surface_officielle_km2"] = df[surface_km2].to_numpy(float) if surface_km2 else np.nan
        up = self.bassins["UP_AREA"].reindex(d["bassin"].astype("float64").to_numpy())
        d["surface_amont_bassin_km2"] = up.to_numpy()
        d["ecart_surface"] = (d["surface_amont_bassin_km2"] - d["surface_officielle_km2"]) / d["surface_officielle_km2"]
        tout = pd.concat([self.stations, d[_COLS_STATIONS]]) if len(self.stations) else d[_COLS_STATIONS]
        self.stations = tout[~tout.index.duplicated(keep="last")]
        return d["bassin"].isna().sum()

"""
================================================================================
MODULE features.py
================================================================================
Entrées (X) et cible (Y) du modèle pour une station, puis découpage en ensembles
d'entraînement / validation / test. Relie les trois autres modules :

- `bassins` : bassin de la station et carreaux météo de tout son amont ;
- `meteo` : météo horaire de ces carreaux (cache Open-Meteo) ;
- `hydro` : débit de la station au même pas (cache RiverApp).

On compose X et Y PIÈCE PAR PIÈCE, sans rien charger :

1. ``F = Features(id_station, debut, fin)`` : X et Y vides, sans colonne ;
2. ``ajouter_meteo``, ``ajouter_debit``, ``ajouter_date``, ``ajouter_cible`` ; ``retirer``,
   ``retirer_colonnes``, ``retirer_cibles`` pour enlever ce qu'on veut. Après chaque
   changement, X et Y montrent leurs colonnes (encore sans ligne) ;
3. ``construire()`` : lit les caches et remplit X et Y.

Tout est compté depuis t, l'instant où l'on prévoit : `depuis` remonte dans le passé, `jusqu_a`
va dans le futur (météo réellement observée, qui joue le rôle des prévisions météo), `pas` =
taille des morceaux. Météo : sommes sur le pas ; débit : moyennes sur le pas (rien n'est sauté).

UN X PAR ÉCHÉANCE : X contient toutes les colonnes, et `colonnes_par_cible` dit lesquelles sert
chaque échéance de Y. Pour la météo, une fenêtre attachée à l'échéance t + h : on garde les
morceaux qui touchent [t + h − `avant_echeance`, t + h + `apres_echeance`] (réglable par
entrée). Débit passé et date servent à toutes les échéances.
Les découpages se font PAR ANNÉES ENTIÈRES : chaque ensemble est une suite de morceaux continus,
jamais des heures tirées au hasard.
================================================================================
"""

import numpy as np
import pandas as pd

from .bassins import Reseau, localiser, carreaux_amont, recouvrement
from .hydro import RiverApp
from .meteo import Meteo

_COLS_ENTREES = ["type", "carreau", "variable", "depuis", "jusqu_a", "pas", "avant_echeance", "apres_echeance"]
_COLONNES_DATE = ["jour_cos", "jour_sin", "heure_cos", "heure_sin"]


def _heures(duree):
    """
    Nombre entier d'heures d'une durée.

    Parameters
    ----------
    duree : str or pandas.Timedelta
        Durée au format pandas (``"0h"``, ``"24h"``, ``"2D"``).

    Returns
    -------
    int
        Nombre d'heures.

    Raises
    ------
    ValueError
        Si la durée n'est pas un nombre entier d'heures (0 compris).
    """
    h = pd.Timedelta(duree) / pd.Timedelta("1h")
    if h < 0 or h != int(h):
        raise ValueError(f"{duree} : il faut un nombre entier d'heures")
    return int(h)


def _bloc(depuis, jusqu_a, pas, nom):
    """
    Décalages (en heures, depuis t) des colonnes d'une entrée.

    Parameters
    ----------
    depuis, jusqu_a : str
        Durées couvertes avant et après t (``"0h"`` = rien de ce côté).
    pas : str
        Pas, au moins 1 h ; `depuis` et `jusqu_a` doivent en être des multiples.
    nom : str
        Nom de l'entrée, pour le message d'erreur.

    Returns
    -------
    p : int
        Pas en heures.
    decalages : list of int
        Passé : 0, p, ..., depuis − p (morceau qui finit à t − k) ;
        futur : −p, −2p, ..., −jusqu_a (morceau qui finit à t + k).

    Raises
    ------
    ValueError
        Si le pas est nul ou si `depuis` / `jusqu_a` n'en sont pas des multiples.
    """
    n, m, p = _heures(depuis), _heures(jusqu_a), _heures(pas)
    if p == 0 or n % p or m % p:
        raise ValueError(f"{nom} : depuis={depuis} et jusqu_a={jusqu_a} doivent être des multiples de pas={pas} (> 0)")
    return p, list(range(0, n, p)) + [-k for k in range(p, m + 1, p)]


def _nom_decalage(k):
    """Suffixe de colonne : ``t-3h`` pour le morceau qui finit 3 h avant t, ``t+6h`` 6 h après."""
    return f"t-{k}h" if k >= 0 else f"t+{-k}h"


def _dans_fenetre(p, k, h, avant, apres):
    """
    Le morceau météo (p heures qui finissent à t − k) sert-il l'échéance t + h ?

    Oui s'il touche la fenêtre [t + h − `avant`, t + h + `apres`].

    Parameters
    ----------
    p : int
        Taille du morceau, en heures.
    k : int
        Décalage de la fin du morceau (positif = avant t, négatif = après t).
    h : int
        Échéance, en heures après t.
    avant, apres : str or None
        Durées de la fenêtre avant et après l'échéance ; None = pas de limite de ce côté.

    Returns
    -------
    bool
    """
    fin_morceau = -k
    if not pd.isna(avant) and fin_morceau <= h - _heures(avant):
        return False
    if not pd.isna(apres) and fin_morceau - p >= h + _heures(apres):
        return False
    return True


class Features:
    """
    X (entrées) et Y (débits à prévoir) d'une station, composés pièce par pièce.

    À la création, X et Y sont vides et sans colonne : rien n'est lu. On ajoute ou retire des
    entrées et des échéances, X et Y montrent alors leurs colonnes (sans ligne) ; `construire`
    lit les caches et les remplit.

    Parameters
    ----------
    id_station : str
        Identifiant RiverApp de la station (index de `hydro.RiverApp.stations`).
    debut : str
        Premier jour, inclus.
    fin : str
        Dernier jour, inclus.
    filtre : bool, default False
        Retirer les pics isolés de débit (cf. `hydro._filtrer`), dans X comme dans Y.
    reseau : bassins.Reseau, optional
        Réseau déjà chargé, pour ne pas le relire. Par défaut, un nouveau `Reseau()`.
    meteo : meteo.Meteo, optional
        Accès météo. Par défaut, `Meteo()`.
    riverapp : hydro.RiverApp, optional
        Accès débits. Par défaut, `RiverApp()`.

    Attributes
    ----------
    id_station : str
        Identifiant RiverApp de la station.
    debut, fin : str
        Période.
    filtre : bool
        Filtre des pics de débit.
    zone : int
        HYBAS_ID du bassin où se trouve la station.
    carreaux : pandas.DataFrame
        Carreaux météo de l'amont (sortie de `bassins.carreaux_amont`) : ceux qu'on peut ajouter.
    entrees : pandas.DataFrame
        La recette de X, une ligne par entrée : ``type`` (meteo, debit, date), ``carreau``,
        ``variable``, ``depuis``, ``jusqu_a``, ``pas``.
    cibles : pandas.DataFrame
        La recette de Y : blocs ``duree``, ``pas`` mis bout à bout après t.
    colonnes_retirees : set of str
        Colonnes de X retirées une à une (`retirer_colonnes`).
    colonnes_par_cible : dict of str to list of str
        Pour chaque colonne de Y, les colonnes de X qui lui servent (à passer à
        ``modeles.Modele.fit(..., colonnes=F.colonnes_par_cible)``).
    X : pandas.DataFrame
        Index = heure UTC t. Colonnes (dans l'ordre des entrées) :
        ``<variable>_<carreau>_<pas>_t-<k>h`` / ``_t+<k>h`` = somme de la variable sur les
        `pas` heures qui finissent à t − k / t + k ; ``debit_passe_<pas>_t-<k>h`` = débit
        moyen sur les `pas` heures qui finissent à t − k ; ``jour_cos``, ``jour_sin``,
        ``heure_cos``, ``heure_sin``. Sans ligne tant que `construire` n'a pas tourné.
    Y : pandas.DataFrame
        Même index ; ``debit_<pas>_t+<h>h`` = débit moyen (m³/s) sur les `pas` heures qui
        finissent à t + h. Les heures où il manque une valeur de X ou de Y sont retirées.
    debit_actuel : pandas.Series
        Même index ; débit à l'instant t (m³/s), après `construire` : sert à repérer les montées
        (poids et scores de crue de `modeles.entrainer`).

    See Also
    --------
    choisir_carreaux : choisit les carreaux qui représentent le mieux l'amont.
    construire : remplit X et Y.
    decouper : X_train, Y_train, (X_val, Y_val,) X_test, Y_test par années.
    validation_croisee : plis de validation par blocs d'années.

    Notes
    -----
    Météo : toujours des SOMMES sur le pas (avec ``pas="2h"``, la valeur vaut l'heure t +
    l'heure t − 1h : la pluie de l'heure du milieu n'est pas sautée ; pour une température,
    la somme est le double de la moyenne, même information pour le modèle). Débit : moyennes.

    La météo future est la météo RÉELLEMENT observée : en vrai on n'aura que des prévisions,
    moins justes. Les scores sont donc un peu trop optimistes aux longues échéances.

    Examples
    --------
    >>> F = Features("52a0c9cee4b03f58295e6f8a", "2010-01-01", "2024-12-31")    # La Clapière
    >>> F.ajouter_meteo(["precipitation", "temperature_2m"], depuis="24h", pas="6h", jusqu_a="72h")
    >>> F.ajouter_meteo("et0_fao_evapotranspiration", carreaux=830767, depuis="48h", pas="2h")
    >>> F.ajouter_meteo("snow_depth", depuis="360h", pas="24h", avant_echeance="360h")    # neige : longue mémoire
    >>> F.ajouter_debit(depuis="24h", pas="1h")
    >>> F.ajouter_date()
    >>> F.ajouter_cible("5D", "4h").ajouter_cible("5D", "8h").ajouter_cible("5D", "1D")
    >>> F.retirer(variable="temperature_2m", carreau=830767)
    >>> F.X.columns          # colonnes prévues, encore sans ligne
    >>> F.construire()
    >>> X_train, Y_train, X_test, Y_test = F.decouper(test=[2023, 2024])
    """

    def __init__(self, id_station, debut, fin, filtre=False, reseau=None, meteo=None, riverapp=None):
        self.id_station, self.debut, self.fin, self.filtre = id_station, debut, fin, filtre
        self._riverapp = riverapp or RiverApp()
        self._meteo = meteo or Meteo()
        self._reseau = reseau or Reseau()
        st = self._riverapp.stations().loc[id_station]
        self.zone = int(localiser([st["lon"]], [st["lat"]])[0])
        self.carreaux = carreaux_amont(self.zone, self._reseau.foret)
        self.entrees = pd.DataFrame(columns=_COLS_ENTREES)
        self.cibles = pd.DataFrame(columns=["duree", "pas"])
        self.colonnes_retirees = set()
        self._colonnes_vides()

    # ---------------------------------------------------------------- recette
    def recette(self):
        """
        Tout ce qu'il faut pour reconstruire X et Y à l'identique (sans les données).

        Returns
        -------
        dict
            ``id_station``, ``debut``, ``fin``, ``filtre``, ``entrees`` (liste des lignes de
            `entrees`), ``cibles`` (liste des blocs), ``colonnes_retirees`` (liste triée).
            Simple à écrire en JSON ; c'est l'identité des données d'un modèle enregistré.

        See Also
        --------
        depuis_recette : l'inverse.

        Examples
        --------
        >>> r = F.recette()
        >>> F2 = Features.depuis_recette(r).construire()      # mêmes X et Y
        """
        def lignes(df):
            return [{k: (None if pd.isna(v) else v.item() if hasattr(v, "item") else v) for k, v in d.items()}
                    for d in df.astype(object).to_dict("records")]
        return {"id_station": self.id_station, "debut": self.debut, "fin": self.fin, "filtre": bool(self.filtre),
                "entrees": lignes(self.entrees), "cibles": lignes(self.cibles),
                "colonnes_retirees": sorted(self.colonnes_retirees)}

    @classmethod
    def depuis_recette(cls, recette, reseau=None, meteo=None, riverapp=None):
        """
        Recrée un `Features` (X et Y encore vides) à partir d'une `recette`.

        Parameters
        ----------
        recette : dict
            Sortie de `recette` (par exemple lue dans la fiche d'un modèle enregistré).
        reseau, meteo, riverapp : optional
            Comme pour `Features`.

        Returns
        -------
        Features
            Mêmes entrées, échéances et colonnes retirées ; appeler `construire` pour remplir.
        """
        F = cls(recette["id_station"], recette["debut"], recette["fin"], filtre=recette["filtre"],
                reseau=reseau, meteo=meteo, riverapp=riverapp)
        F.entrees = pd.DataFrame(recette["entrees"], columns=_COLS_ENTREES, dtype=object)   # carreaux restent entiers
        F.cibles = pd.DataFrame(recette["cibles"], columns=["duree", "pas"])
        F.colonnes_retirees = set(recette["colonnes_retirees"])
        F._colonnes_vides()
        return F

    # ---------------------------------------------------------------- choix des carreaux
    def choisir_carreaux(self, k_max=10, seuil=0.02, criteres=None, poids="run_mm_syr", dedans_min=0.5, largeur=1.5):
        """
        Choisit, un par un, les carreaux qui représentent le mieux tout l'amont (glouton).

        Chaque morceau de l'amont (un petit bassin BasinATLAS coupé par un carreau) est
        « représenté » par le carreau choisi qui lui ressemble le plus selon les `criteres` ; on
        ajoute à chaque tour le carreau qui améliore le plus la représentation totale, chaque
        morceau comptant selon `poids`. Aucun appel à Open-Meteo. Tout ce qui entre dans le
        score est une option : on peut tester avec et sans chaque critère.

        Parameters
        ----------
        k_max : int, default 10
            Nombre maximal de carreaux (garde-fou) ; la courbe va jusque-là.
        seuil : float, default 0.02
            On s'arrête quand un carreau de plus fait gagner moins que ce score (0.02 = 2 points).
        criteres : dict of str to float, list of str, or None
            Ce qui définit la ressemblance, avec son importance. Par défaut
            ``{"snw_pc_syr": 1, "pre_mm_syr": 1, "position": 1}`` : part de neige (%), pluie
            annuelle (mm) et position. ``"position"`` = où est le carreau : un morceau est mieux
            représenté par un carreau proche, le score monte donc quand les carreaux choisis
            sont étalés sur tout l'amont. Tout autre attribut de `config.ATTRIBUTS` est permis.
            Une liste = importance 1 pour chacun. Importance 2 = un écart compte comme √2 fois plus.
        poids : str or None, default "run_mm_syr"
            Attribut qui, multiplié par la surface, donne le poids d'un morceau : par défaut le
            ruissellement annuel (mm/an), donc le volume d'eau envoyé à la rivière. None = la
            surface seule.
        dedans_min : float, default 0.5
            Seuls les carreaux dont au moins cette part est dans l'amont peuvent être choisis
            (part mesurée par rapport au plus grand carreau de l'amont, qui y est entier) : la
            météo d'un carreau du bord vaut surtout pour l'extérieur. 0 = tous. Tous les
            morceaux restent à représenter.
        largeur : float, default 1.5
            Écart (en écarts-types) pour lequel la ressemblance tombe à 0,61. Plus grand =
            plus tolérant (un carreau « représente » des morceaux plus différents de lui).

        Returns
        -------
        pandas.DataFrame
            Une ligne par tour du glouton, jusqu'à `k_max` : ``rang``, ``carreau``, ``gain`` (score
            apporté par ce carreau), ``score`` (représentation totale après ce tour, de 0 à 1),
            ``retenu`` (True avant l'arrêt), ``part_eau`` (part du poids total de l'amont qui
            tombe dans ce carreau), ``lon_centre``, ``lat_centre`` et la valeur des critères
            BasinATLAS sur ce carreau.

        See Also
        --------
        ajouter_meteo : y passer ``carreaux=choix.loc[choix["retenu"], "carreau"].tolist()``.

        Notes
        -----
        Écart entre un morceau et un carreau : d² = Σ importance × (écart du critère, mesuré en
        écarts-types de l'amont)² ; pour la position, écart en km (le morceau est placé au
        centre de son carreau) rapporté à l'étalement de l'amont. Ressemblance = exp(−d² / (2 ×
        largeur²)) : 1 si identiques, 0,61 à une largeur, 0,14 à deux. Un carreau a pour
        critères la moyenne de ceux de ses morceaux, pondérée par le poids.

        Score = moyenne, pondérée par le poids, de la ressemblance de chaque morceau à son
        meilleur représentant parmi les carreaux choisis : 1 si chaque morceau a un carreau
        identique à lui. Chaque tour rapporte de moins en moins (problème « de couverture ») :
        le glouton est garanti à au moins 63 % (1 − 1/e) du meilleur choix possible. À gain
        égal, le carreau le plus lourd passe devant.

        Les attributs BasinATLAS sont connus par petit bassin (≈ 130 km² à La Clapière) :
        sans la position, les carreaux d'un même petit bassin se ressemblent tous.

        Examples
        --------
        >>> choix = F.choisir_carreaux()
        >>> choix.plot(x="rang", y="score", marker="o")                        # la courbe
        >>> F.choisir_carreaux(criteres={"snw_pc_syr": 1, "pre_mm_syr": 1})    # sans la position
        >>> F.choisir_carreaux(criteres={"snw_pc_syr": 2, "position": 1}, poids=None)
        >>> F.ajouter_meteo("precipitation", carreaux=choix.loc[choix["retenu"], "carreau"].tolist())
        """
        if criteres is None:
            criteres = {"snw_pc_syr": 1, "pre_mm_syr": 1, "position": 1}
        if not isinstance(criteres, dict):
            criteres = {c: 1 for c in criteres}
        atlas = [c for c in criteres if c != "position"]
        morceaux = recouvrement(self.zone // 10**9).loc[self._reseau.foret.amont(self.zone)]
        attributs = self._reseau.bassins.loc[morceaux.index]
        w = morceaux["surface_km2"].to_numpy() * (attributs[poids].to_numpy(float) if poids else 1.0)

        def reduire(valeurs):
            """Écarts à la moyenne de l'amont, en écarts-types (pondérés) ; critère constant → 0."""
            m = np.average(valeurs, axis=0, weights=w)
            e = np.sqrt(np.average((valeurs - m) ** 2, axis=0, weights=w))
            return (valeurs - m) / np.where(e > 0, e, 1.0)

        blocs, importances = [], []
        if atlas:
            blocs.append(reduire(attributs[atlas].to_numpy(float)))
            importances += [criteres[c] for c in atlas]
        if "position" in criteres:
            lat0 = np.radians(morceaux["lat_centre"].mean())
            km = np.column_stack([morceaux["lon_centre"].to_numpy() * 111.32 * np.cos(lat0),
                                  morceaux["lat_centre"].to_numpy() * 110.57])
            km = km - np.average(km, axis=0, weights=w)
            etalement = np.sqrt(np.average((km ** 2).sum(axis=1), weights=w) / 2)   # même échelle en x et en y
            blocs.append(km / etalement)
            importances += [criteres["position"]] * 2
        z = np.column_stack(blocs) * np.sqrt(importances)
        noms = [f"z{i}" for i in range(z.shape[1])]

        table = pd.DataFrame(z, columns=noms).assign(carreau=morceaux["carreau"].to_numpy(), w=w,
                                                     surface=morceaux["surface_km2"].to_numpy())
        cand = (table.groupby("carreau").apply(lambda g: pd.Series({**{c: np.average(g[c], weights=g["w"]) for c in noms},
                                                                     "eau": g["w"].sum(), "surface": g["surface"].sum()}),
                                               include_groups=False)
                .sort_values("eau", ascending=False))                   # à gain égal, le plus lourd d'abord
        cand = cand[cand["surface"] >= dedans_min * cand["surface"].max()]
        d2 = ((z[:, None, :] - cand[noms].to_numpy()[None, :, :]) ** 2).sum(axis=2)
        ressemblance = np.exp(-d2 / (2 * largeur ** 2))                  # morceaux × carreaux
        w_norm = w / w.sum()

        meilleure, choisis, lignes = np.zeros(len(w)), [], []
        for rang in range(1, min(k_max, len(cand)) + 1):
            gains = (w_norm[:, None] * np.maximum(ressemblance - meilleure[:, None], 0)).sum(axis=0)
            gains[choisis] = -1
            j = int(np.argmax(gains))
            choisis.append(j)
            meilleure = np.maximum(meilleure, ressemblance[:, j])
            lignes.append({"rang": rang, "carreau": int(cand.index[j]), "gain": float(gains[j]),
                           "score": float((w_norm * meilleure).sum())})
        sortie = pd.DataFrame(lignes)
        sortie["retenu"] = (sortie["gain"] >= seuil).cummin()
        sortie.loc[0, "retenu"] = True                                   # au moins un carreau
        sortie["part_eau"] = (cand["eau"].to_numpy()[choisis] / w.sum()).round(4)
        centres = self.carreaux.set_index("carreau")[["lon_centre", "lat_centre"]]
        sortie = sortie.join(centres, on="carreau")
        reel = attributs[atlas].assign(carreau=morceaux["carreau"].to_numpy(), w=w)
        for c in atlas:
            vals = reel.groupby("carreau").apply(lambda g: np.average(g[c], weights=g["w"]), include_groups=False)
            sortie[c] = sortie["carreau"].map(vals).round(1)
        return sortie

    # ---------------------------------------------------------------- recette de X
    def _ajouter(self, lignes):
        """
        Ajoute des entrées à la recette (sans doublon), puis met à jour les colonnes de X.

        Parameters
        ----------
        lignes : list of dict
            Entrées à ajouter (clés de ``entrees``).
        """
        self.entrees = pd.concat([self.entrees, pd.DataFrame(lignes, columns=_COLS_ENTREES)], ignore_index=True)
        self.entrees = self.entrees.astype(object).drop_duplicates(ignore_index=True)
        self._colonnes_vides()

    def ajouter_meteo(self, variables, carreaux="tous", depuis="1h", pas="1h", jusqu_a="0h",
                      avant_echeance=None, apres_echeance="0h"):
        """
        Ajoute de la météo à X.

        Parameters
        ----------
        variables : str or list of str
            Variables horaires Open-Meteo (ex. ``"precipitation"``, ``"temperature_2m"``).
        carreaux : 'tous', int or list of int, default 'tous'
            Carreaux concernés, parmi ceux de l'amont (`carreaux`).
        depuis : str, default "1h"
            Jusqu'où remonter avant t.
        pas : str, default "1h"
            Taille des morceaux ; chaque valeur est la somme des heures du morceau.
        jusqu_a : str, default "0h"
            Jusqu'où aller après t (météo future ; ``"0h"`` = aucune).
        avant_echeance : str, optional
            Fenêtre AVANT chaque échéance : pour prévoir t + h, ne garder que les morceaux qui
            finissent après t + h − `avant_echeance`. Par défaut aucune limite (tout le passé
            sert à toutes les échéances). Court pour une réponse rapide (pluie, ex. ``"72h"``),
            long pour une longue mémoire (neige, ex. ``"360h"``).
        apres_echeance : str or None, default "0h"
            Marge APRÈS chaque échéance : par défaut, la météo qui tombe après t + h ne sert pas
            à prévoir t + h. Une marge (ex. ``"12h"``) garde un peu de météo prévue plus tard,
            utile quand les prévisions sont décalées dans le temps ; None = aucune coupure.

        Returns
        -------
        Features
            Lui-même (pour enchaîner les appels).

        Raises
        ------
        ValueError
            Si un carreau n'est pas dans l'amont, ou si `depuis` / `jusqu_a` ne sont pas des
            multiples de `pas`.

        Examples
        --------
        >>> F.ajouter_meteo(["precipitation", "snowfall"], depuis="24h", pas="6h", jusqu_a="120h")
        >>> F.ajouter_meteo("temperature_2m", carreaux=[830767, 833347], depuis="12h")
        """
        variables = [variables] if isinstance(variables, str) else list(variables)
        if isinstance(carreaux, str) and carreaux == "tous":
            numeros = [int(c) for c in self.carreaux["carreau"]]
        else:
            numeros = [int(c) for c in np.atleast_1d(carreaux)]
        inconnus = set(numeros) - set(self.carreaux["carreau"].astype(int))
        if inconnus:
            raise ValueError(f"carreaux hors de l'amont de la station : {sorted(inconnus)}")
        _bloc(depuis, jusqu_a, pas, "ajouter_meteo")
        for duree in (avant_echeance, apres_echeance):
            if duree is not None:
                _heures(duree)
        self._ajouter([dict(type="meteo", carreau=c, variable=v, depuis=depuis, jusqu_a=jusqu_a, pas=pas,
                            avant_echeance=avant_echeance, apres_echeance=apres_echeance)
                       for c in numeros for v in variables])
        return self

    def ajouter_debit(self, depuis="24h", pas="1h"):
        """
        Ajoute le débit passé de la station à X.

        Parameters
        ----------
        depuis : str, default "24h"
            Jusqu'où remonter avant t.
        pas : str, default "1h"
            Taille des morceaux ; chaque valeur est le débit moyen du morceau.

        Returns
        -------
        Features
            Lui-même.
        """
        _bloc(depuis, "0h", pas, "ajouter_debit")
        self._ajouter([dict(type="debit", carreau=None, variable="debit", depuis=depuis, jusqu_a="0h", pas=pas)])
        return self

    def ajouter_date(self):
        """
        Ajoute à X la date et l'heure en cos / sin.

        ``jour_cos``, ``jour_sin`` : place dans l'année ; ``heure_cos``, ``heure_sin`` : heure UTC
        dans la journée. Le 31 décembre et le 1er janvier, 23 h et 0 h restent voisins.

        Returns
        -------
        Features
            Lui-même.
        """
        self._ajouter([dict(type="date", carreau=None, variable="date", depuis="0h", jusqu_a="0h", pas="1h")])
        return self

    def retirer(self, type=None, carreau=None, variable=None, pas=None):
        """
        Retire de X toutes les entrées qui correspondent à TOUS les critères donnés.

        Parameters
        ----------
        type : {'meteo', 'debit', 'date'}, optional
            Nature de l'entrée.
        carreau : int or list of int, optional
            Carreau(x).
        variable : str or list of str, optional
            Variable(s).
        pas : str, optional
            Pas.

        Returns
        -------
        Features
            Lui-même.

        Examples
        --------
        >>> F.retirer(carreau=830767)                                # tout ce carreau
        >>> F.retirer(variable="temperature_2m", carreau=[830767, 833347])
        >>> F.retirer(type="debit")                                  # plus de débit passé
        """
        garde = pd.Series(True, index=self.entrees.index)
        for nom, valeur in (("type", type), ("carreau", carreau), ("variable", variable), ("pas", pas)):
            if valeur is not None:
                garde &= self.entrees[nom].isin(list(np.atleast_1d(valeur)))
        self.entrees = self.entrees[~garde].reset_index(drop=True)
        self._colonnes_vides()
        return self

    def retirer_colonnes(self, noms):
        """
        Retire de X des colonnes précises, par leur nom.

        Pour enlever un seul décalage (ex. ``"temperature_2m_830767_1h_t-3h"``) sans toucher au
        reste de l'entrée. Marche avant comme après `construire`.

        Parameters
        ----------
        noms : str or list of str
            Noms de colonnes de X.

        Returns
        -------
        Features
            Lui-même.
        """
        self.colonnes_retirees |= {noms} if isinstance(noms, str) else set(noms)
        self.X = self.X.drop(columns=list(self.colonnes_retirees), errors="ignore")
        self.colonnes_par_cible = {y: [c for c in cols if c not in self.colonnes_retirees]
                                   for y, cols in self.colonnes_par_cible.items()}
        return self

    # ---------------------------------------------------------------- recette de Y
    def ajouter_cible(self, duree, pas):
        """
        Ajoute un bloc d'échéances à Y, à la suite des précédents.

        Parameters
        ----------
        duree : str
            Durée couverte par le bloc (ex. ``"5D"``).
        pas : str
            Écart entre deux échéances ; chaque valeur est le débit moyen sur le pas.

        Returns
        -------
        Features
            Lui-même.

        Examples
        --------
        >>> F.ajouter_cible("5D", "4h").ajouter_cible("5D", "8h").ajouter_cible("5D", "1D")
        >>> F.Y.columns[[0, -1]]       # debit_4h_t+4h ... debit_24h_t+360h (50 échéances)
        """
        n, p = _heures(duree), _heures(pas)
        if p == 0 or n % p:
            raise ValueError(f"ajouter_cible : durée {duree} n'est pas un multiple du pas {pas} (> 0)")
        self.cibles = pd.concat([self.cibles, pd.DataFrame([{"duree": duree, "pas": pas}])], ignore_index=True)
        self._colonnes_vides()
        return self

    def retirer_cibles(self, numeros=None):
        """
        Retire des blocs d'échéances de Y.

        Parameters
        ----------
        numeros : int or list of int, optional
            Positions des blocs dans `cibles` (0 = le premier). Par défaut, tous.

        Returns
        -------
        Features
            Lui-même.
        """
        cible = self.cibles.index if numeros is None else list(np.atleast_1d(numeros))
        self.cibles = self.cibles.drop(index=cible).reset_index(drop=True)
        self._colonnes_vides()
        return self

    # ---------------------------------------------------------------- colonnes et construction
    def _decalages(self, ligne):
        """
        Colonnes que donne une entrée de la recette.

        Parameters
        ----------
        ligne : pandas.Series
            Une ligne de `entrees`.

        Returns
        -------
        list of tuple
            ``(nom de colonne, pas en heures, décalage en heures)`` ; pas et décalage valent None
            pour la date.
        """
        if ligne["type"] == "date":
            return [(nom, None, None) for nom in _COLONNES_DATE]
        p, decalages = _bloc(ligne["depuis"], ligne["jusqu_a"], ligne["pas"], ligne["variable"])
        debut_nom = "debit_passe" if ligne["type"] == "debit" else f"{ligne['variable']}_{ligne['carreau']}"
        return [(f"{debut_nom}_{p}h_{_nom_decalage(k)}", p, k) for k in decalages]

    def _echeances(self):
        """
        Échéances de Y, dans l'ordre.

        Returns
        -------
        list of tuple
            ``(nom de colonne, pas en heures, échéance en heures après t)``.
        """
        out, h = [], 0
        for duree, pas in self.cibles[["duree", "pas"]].itertuples(index=False):
            p = _heures(pas)
            for _ in range(_heures(duree) // p):
                h += p
                out.append((f"debit_{p}h_t+{h}h", p, h))
        return out

    def _colonnes_vides(self):
        """
        Remet X et Y à vide, avec seulement les colonnes prévues par les recettes, et calcule
        `colonnes_par_cible` (fenêtre de chaque entrée météo autour de chaque échéance).
        """
        echeances = self._echeances()
        noms, par_cible = [], {y: [] for y, _, _ in echeances}
        for _, l in self.entrees.iterrows():
            for nom, p, k in self._decalages(l):
                if nom in self.colonnes_retirees or nom in noms:
                    continue
                noms.append(nom)
                for y, _, h in echeances:
                    if l["type"] != "meteo" or _dans_fenetre(p, k, h, l["avant_echeance"], l["apres_echeance"]):
                        par_cible[y].append(nom)
        self.X = pd.DataFrame(columns=noms)
        self.Y = pd.DataFrame(columns=[y for y, _, _ in echeances])
        self.colonnes_par_cible = par_cible

    def construire(self):
        """
        Lit les caches et remplit X et Y selon les recettes.

        La météo manquante au cache est téléchargée (quota Open-Meteo) : vérifier avant avec
        `meteo.Meteo.cout_prevu` sur une nouvelle station. Les heures où il manque une valeur
        (début de période sans tout le passé, trous de débit) sont retirées.

        Returns
        -------
        Features
            Lui-même, avec X et Y remplis.

        Raises
        ------
        ValueError
            Si X ou Y n'a aucune colonne.
        """
        if self.X.shape[1] == 0 or self.Y.shape[1] == 0:
            raise ValueError("ajouter d'abord des entrées (X) et des échéances (Y)")
        derniere_heure = pd.Timestamp(self.fin, tz="UTC") + pd.Timedelta(hours=23)
        index = pd.date_range(pd.Timestamp(self.debut, tz="UTC"), derniere_heure, freq="1h", name="time")
        echeances = self._echeances()
        fin_debit = derniere_heure.tz_localize(None) + pd.Timedelta(hours=echeances[-1][2])
        q = self._riverapp.mesures(self.id_station, self.debut, fin_debit, pas="1h", filtre=self.filtre)["flow"].asfreq("1h")

        series = {}                                       # (carreau, variable) -> série horaire
        meteo = self.entrees[self.entrees["type"] == "meteo"]
        if not meteo.empty:
            futur = max(_heures(j) for j in meteo["jusqu_a"])
            fin_meteo = (pd.Timestamp(self.fin) + pd.Timedelta(hours=futur + 24)).strftime("%Y-%m-%d")
            for c, lignes in meteo.groupby("carreau", sort=False):
                ligne_carreau = self.carreaux[self.carreaux["carreau"] == c]
                h = self._meteo.telecharger(ligne_carreau, list(lignes["variable"].unique()), (),
                                            self.debut, fin_meteo)["horaire"][int(c)].asfreq("1h")
                for v in lignes["variable"].unique():
                    series[(c, v)] = h[v]

        colonnes = {}                                     # en float32 dès le départ : X de milliers de colonnes
        for _, l in self.entrees.iterrows():
            for nom, p, k in self._decalages(l):
                if nom in self.colonnes_retirees:
                    continue
                if l["type"] == "date":
                    angle = (2 * np.pi * (index.dayofyear - 1) / 365.25) if nom.startswith("jour") \
                        else (2 * np.pi * index.hour / 24)
                    colonnes[nom] = np.asarray(np.cos(angle) if nom.endswith("cos") else np.sin(angle), dtype="float32")
                elif l["type"] == "debit":
                    colonnes[nom] = q.rolling(p, min_periods=p).mean().reindex(index - pd.Timedelta(hours=k)).to_numpy("float32")
                else:
                    s = series[(l["carreau"], l["variable"])].rolling(p, min_periods=p).sum()
                    colonnes[nom] = s.reindex(index - pd.Timedelta(hours=k)).to_numpy("float32")
        X = pd.DataFrame(colonnes, index=index).astype("float32")
        Y = pd.DataFrame({nom: q.rolling(p, min_periods=p).mean().reindex(index + pd.Timedelta(hours=h)).to_numpy()
                          for nom, p, h in echeances}, index=index).astype("float32")
        garde = X.notna().all(axis=1) & Y.notna().all(axis=1)
        self.X, self.Y = X[garde], Y[garde]
        self.debit_actuel = pd.Series(q.reindex(index).to_numpy(), index=index, name="debit_t")[garde].astype("float32")
        return self

    # ---------------------------------------------------------------- découpages
    def annees(self):
        """
        Années présentes dans X et Y (après `construire`).

        Returns
        -------
        list of int
            Années triées.
        """
        return sorted(self.X.index.year.unique())

    def decouper(self, test, val=()):
        """
        Sépare X et Y en entraînement, (validation,) test, par années entières.

        Toutes les années qui ne sont ni dans `test` ni dans `val` vont à l'entraînement.

        Parameters
        ----------
        test : list of int
            Années de test.
        val : list of int, default ()
            Années de validation (pour choisir les hyperparamètres). Sans elles, pas
            d'ensemble de validation.

        Returns
        -------
        tuple of pandas.DataFrame
            Sans `val` : ``X_train, Y_train, X_test, Y_test``.
            Avec `val` : ``X_train, Y_train, X_val, Y_val, X_test, Y_test``.

        Raises
        ------
        ValueError
            Si une année est à la fois dans `test` et dans `val`.

        Notes
        -----
        L'année d'une ligne est celle de l'instant des entrées t ; la cible des derniers
        jours d'une année tombe donc au début de l'année suivante.

        Examples
        --------
        >>> X_train, Y_train, X_test, Y_test = F.decouper(test=[2023, 2024])
        >>> X_train, Y_train, X_val, Y_val, X_test, Y_test = F.decouper(test=[2024], val=[2022, 2023])
        """
        if set(test) & set(val):
            raise ValueError(f"années à la fois en test et en validation : {sorted(set(test) & set(val))}")
        an = self.X.index.year
        m_test, m_val = an.isin(list(test)), an.isin(list(val))
        m_train = ~(m_test | m_val)
        sortie = (self.X[m_train], self.Y[m_train])
        if len(val):
            sortie += (self.X[m_val], self.Y[m_val])
        return sortie + (self.X[m_test], self.Y[m_test])

    def validation_croisee(self, test, n_plis=5):
        """
        Plis de validation croisée par blocs d'années consécutives, test mis de côté.

        Les années hors `test` sont coupées en `n_plis` blocs d'années qui se suivent ;
        chaque bloc sert une fois de validation, les autres d'entraînement. Le test n'est
        jamais utilisé.

        Parameters
        ----------
        test : list of int
            Années de test, exclues de tous les plis.
        n_plis : int, default 5
            Nombre de plis.

        Yields
        ------
        tuple
            ``X_train, Y_train, X_val, Y_val`` pour chaque pli.

        See Also
        --------
        decouper : un seul découpage.

        Examples
        --------
        >>> for X_train, Y_train, X_val, Y_val in F.validation_croisee(test=[2023, 2024], n_plis=5):
        ...     print(sorted(X_val.index.year.unique()))
        """
        reste = [a for a in self.annees() if a not in set(test)]
        for bloc in np.array_split(reste, n_plis):
            yield self.decouper(test, val=[int(a) for a in bloc])[:4]

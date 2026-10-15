"""
================================================================================
MODULE reseaux.py
================================================================================
Les deux réseaux de neurones, même architecture dans deux bibliothèques :

- `entrainer_keras` → `ReseauKeras` (TensorFlow / Keras, sur le processeur) ;
- `entrainer_torch` → `ReseauTorch` (PyTorch, sur la carte graphique du Mac si disponible).
Réseau de river_level : couches de neurones (swish) → normalisation des lots → dropout, une
connexion résiduelle entre deux couches de même taille, AdamW, arrêt anticipé sur les 15 % les
plus récents de l'entraînement, meilleurs poids gardés. Entrées et sorties centrées réduites.
Utilisés par `modeles.Modele` (``Modele("keras")``, ``Modele("torch")``).
================================================================================
"""

import time
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

RESEAU = dict(couches=(256, 256, 128), dropout=(0.3, 0.3, 0.2), l2=3e-3, taux=3e-3, decroissance=1e-4,
              epoques=300, lot=64, validation=0.15, patience=20, graine=42)
"""dict : hyperparamètres par défaut des réseaux : neurones par couche (``couches=(64, 32)`` = deux
couches de 64 puis 32), dropout par couche (ou un seul nombre pour toutes), pénalité
L2 des poids, pas d'apprentissage, décroissance AdamW, époques au plus, taille des lots, part de
validation (les lignes les plus récentes), patience de l'arrêt anticipé, graine. Ceux de river_level,
sauf dropout (0,2 / 0,2 / 0,1 → 0,3 / 0,3 / 0,2) et L2 (1e-3 → 3e-3), relevés le 2026-10-11 contre le
sur-apprentissage constaté avec 532 colonnes."""


def _couches(couches, dropout):
    """
    Couples (neurones, dropout) de chaque couche, en vérifiant les longueurs.

    Parameters
    ----------
    couches : int or tuple of int
        Neurones par couche (un entier = une seule couche).
    dropout : float or tuple of float
        Dropout par couche, ou un seul nombre pour toutes.

    Returns
    -------
    list of tuple

    Raises
    ------
    ValueError
        Si `dropout` est une liste d'une autre longueur que `couches`.
    """
    couches = (couches,) if isinstance(couches, int) else tuple(couches)
    dropout = (dropout,) * len(couches) if isinstance(dropout, (int, float)) else tuple(dropout)
    if len(dropout) != len(couches):
        raise ValueError(f"{len(couches)} couches mais {len(dropout)} valeurs de dropout : en donner autant, ou un seul nombre")
    return list(zip(couches, dropout))


class _Echelle:
    """
    Centre et réduit des colonnes (moyenne 0, écart-type 1) et fait l'inverse.

    Parameters
    ----------
    a : numpy.ndarray
        Tableau d'apprentissage (lignes × colonnes) d'où viennent moyenne et écart-type.
    """

    def __init__(self, a):
        self.moyenne = a.mean(axis=0)
        self.ecart = np.where(a.std(axis=0) > 0, a.std(axis=0), 1.0)

    def aller(self, a):
        """Valeurs centrées réduites, en float32."""
        return ((a - self.moyenne) / self.ecart).astype("float32")

    def retour(self, a):
        """Valeurs dans l'unité d'origine."""
        return a * self.ecart + self.moyenne


class ReseauKeras:
    """
    Réseau TensorFlow / Keras entraîné, avec ses mises à l'échelle (sortie de `entrainer_keras`).

    Parameters
    ----------
    reseau : tensorflow.keras.Model
        Le réseau.
    ex, ey : _Echelle
        Mises à l'échelle des entrées et des sorties.
    """

    def __init__(self, reseau, ex, ey):
        self.reseau, self.ex, self.ey = reseau, ex, ey

    def predict(self, x):
        """Prévisions (numpy, unité d'origine) pour les entrées `x` (numpy)."""
        return self.ey.retour(self.reseau.predict(self.ex.aller(x), batch_size=4096, verbose=0))

    def sauver(self, dossier):
        """Écrit ``reseau.keras`` et ``echelles.joblib`` dans `dossier`."""
        self.reseau.save(Path(dossier) / "reseau.keras")
        joblib.dump((self.ex, self.ey), Path(dossier) / "echelles.joblib")

    @classmethod
    def charger(cls, dossier):
        """Relit un réseau écrit par `sauver`."""
        import tensorflow as tf
        ex, ey = joblib.load(Path(dossier) / "echelles.joblib")
        return cls(tf.keras.models.load_model(Path(dossier) / "reseau.keras", compile=False), ex, ey)


def entrainer_keras(x, y, barre=True, limite=None, poids=None, couches=(256, 256, 128), dropout=(0.2, 0.2, 0.1), l2=1e-3,
                    taux=3e-3, decroissance=1e-4, epoques=300, lot=64, validation=0.15, patience=20, graine=42):
    """
    Entraîne le réseau de river_level avec TensorFlow / Keras (sur le processeur).

    Chaque couche : neurones (swish) → normalisation des lots → dropout ; une couche de même
    taille que la précédente lui ajoute sa sortie (connexion résiduelle). Entrées ET sorties
    centrées réduites (l'erreur pèse alors autant pour chaque échéance).

    Parameters
    ----------
    x : numpy.ndarray
        Entrées, lignes dans l'ordre du temps.
    y : numpy.ndarray
        Cibles, une colonne par échéance.
    barre : bool, default True
        Barre d'avancement par époque.
    limite : float, optional
        Heure d'arrêt (cf. `_limite`) : l'entraînement s'arrête au lot en cours et garde les
        meilleurs poids vus en validation. ``coupe`` du résultat dit si c'est arrivé.
    poids : numpy.ndarray, optional
        Poids de chaque ligne (cf. `ponderation_crue`), aussi en validation ; None = toutes pareilles.
    couches, dropout, l2, taux, decroissance, epoques, lot, validation, patience, graine
        Voir `RESEAU`.

    Returns
    -------
    ReseauKeras
        Le réseau entraîné (``.predict(x)``, ``.sauver(dossier)``).

    Notes
    -----
    Arrêt anticipé sur les `validation` × 100 % de lignes LES PLUS RÉCENTES (pas tirées au
    hasard : des heures voisines se ressemblent), meilleurs poids gardés ; pas d'apprentissage
    divisé par 2 après 6 époques sans progrès ; gradient de chaque poids limité à une norme de 1.
    """
    import tensorflow as tf
    from tqdm.keras import TqdmCallback

    tf.get_logger().setLevel("ERROR")                               # avertissements « retracing » sans gravité

    tf.keras.utils.set_random_seed(graine)
    ex, ey = _Echelle(x), _Echelle(y)
    reg = tf.keras.regularizers.l2(l2)
    entree = tf.keras.Input(shape=(x.shape[1],))
    h, avant = entree, None
    for n, d in _couches(couches, dropout):
        z = tf.keras.layers.Dense(n, activation="swish", kernel_regularizer=reg)(h)
        z = tf.keras.layers.Dropout(d)(tf.keras.layers.BatchNormalization()(z))
        h = tf.keras.layers.Add()([h, z]) if avant == n else z
        avant = n
    reseau = tf.keras.Model(entree, tf.keras.layers.Dense(y.shape[1])(h))
    reseau.compile(loss="mse", optimizer=tf.keras.optimizers.AdamW(learning_rate=taux, weight_decay=decroissance,
                                                                   clipnorm=1.0))
    class Chrono(tf.keras.callbacks.Callback):
        """Arrête l'entraînement dès que l'heure limite est passée."""
        coupe = False

        def on_train_batch_end(self, batch, logs=None):
            if limite is not None and time.time() > limite:
                self.model.stop_training, Chrono.coupe = True, True

    arret = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    rappels = [arret, tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6),
               Chrono()]
    if barre:
        rappels.append(TqdmCallback(verbose=0, desc="keras : époques"))
    reseau.fit(ex.aller(x), ey.aller(y), sample_weight=poids, epochs=epoques, batch_size=lot,
               validation_split=validation, callbacks=rappels, verbose=0)
    if arret.best_weights is not None:                              # meilleurs poids, même si coupé par l'heure
        reseau.set_weights(arret.best_weights)
    r = ReseauKeras(reseau, ex, ey)
    r.coupe = Chrono.coupe
    return r


def _appareil_torch():
    """``mps`` (carte graphique Apple) si disponible, sinon ``cuda``, sinon ``cpu``."""
    import torch
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


def _module_torch(n_entrees, n_sorties, couches, dropout):
    """
    Le réseau PyTorch (même architecture que `entrainer_keras`), non entraîné.

    Parameters
    ----------
    n_entrees, n_sorties : int
        Colonnes de X et de Y.
    couches, dropout : tuple
        Neurones et dropout par couche (cf. `RESEAU`).

    Returns
    -------
    torch.nn.Module
    """
    import torch

    class Reseau(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocs, self.residuel, avant = torch.nn.ModuleList(), [], n_entrees
            for n, d in _couches(couches, dropout):
                self.blocs.append(torch.nn.Sequential(torch.nn.Linear(avant, n), torch.nn.SiLU(),
                                                      torch.nn.BatchNorm1d(n), torch.nn.Dropout(d)))
                self.residuel.append(avant == n)
                avant = n
            self.sortie = torch.nn.Linear(avant, n_sorties)

        def forward(self, h):
            for bloc, res in zip(self.blocs, self.residuel):
                h = h + bloc(h) if res else bloc(h)
            return self.sortie(h)

    return Reseau()


class ReseauTorch:
    """
    Réseau PyTorch entraîné, avec ses mises à l'échelle (sortie de `entrainer_torch`).

    Parameters
    ----------
    reseau : torch.nn.Module
        Le réseau (sur l'appareil de calcul).
    ex, ey : _Echelle
        Mises à l'échelle des entrées et des sorties.
    forme : dict
        ``n_entrees``, ``n_sorties``, ``couches``, ``dropout`` : de quoi le recréer.
    """

    def __init__(self, reseau, ex, ey, forme):
        self.reseau, self.ex, self.ey, self.forme = reseau, ex, ey, forme

    def predict(self, x):
        """Prévisions (numpy, unité d'origine) pour les entrées `x` (numpy), par lots de 4096."""
        import torch
        appareil = next(self.reseau.parameters()).device
        self.reseau.eval()
        with torch.no_grad():
            lots = np.array_split(self.ex.aller(x), max(1, len(x) // 4096))
            return self.ey.retour(torch.cat([self.reseau(torch.from_numpy(b).to(appareil)).cpu() for b in lots]).numpy())

    def sauver(self, dossier):
        """Écrit ``reseau.pt`` (poids) et ``echelles.joblib`` (échelles + forme) dans `dossier`."""
        import torch
        torch.save(self.reseau.state_dict(), Path(dossier) / "reseau.pt")
        joblib.dump((self.ex, self.ey, self.forme), Path(dossier) / "echelles.joblib")

    @classmethod
    def charger(cls, dossier):
        """Relit un réseau écrit par `sauver`, sur l'appareil disponible."""
        import torch
        ex, ey, forme = joblib.load(Path(dossier) / "echelles.joblib")
        reseau = _module_torch(**forme).to(_appareil_torch())
        reseau.load_state_dict(torch.load(Path(dossier) / "reseau.pt", map_location=_appareil_torch()))
        reseau.eval()
        return cls(reseau, ex, ey, forme)


def entrainer_torch(x, y, barre=True, limite=None, poids=None, couches=(256, 256, 128), dropout=(0.2, 0.2, 0.1), l2=1e-3,
                    taux=3e-3, decroissance=1e-4, epoques=300, lot=64, validation=0.15, patience=20, graine=42):
    """
    Entraîne le même réseau que `entrainer_keras`, avec PyTorch (carte graphique du Mac si dispo).

    Mêmes couches, même pénalité L2, même AdamW, même arrêt anticipé, même réduction du pas et
    même limite des gradients que la version Keras : les deux doivent donner des scores proches.

    Parameters
    ----------
    x : numpy.ndarray
        Entrées, lignes dans l'ordre du temps.
    y : numpy.ndarray
        Cibles, une colonne par échéance.
    barre : bool, default True
        Barre d'avancement par époque (avec l'erreur de validation).
    limite : float, optional
        Heure d'arrêt (cf. `_limite`) : l'entraînement s'arrête au lot en cours et garde les
        meilleurs poids vus en validation. ``coupe`` du résultat dit si c'est arrivé.
    poids : numpy.ndarray, optional
        Poids de chaque ligne (cf. `ponderation_crue`), aussi en validation ; None = toutes pareilles.
    couches, dropout, l2, taux, decroissance, epoques, lot, validation, patience, graine
        Voir `RESEAU`.

    Returns
    -------
    ReseauTorch
        Le réseau entraîné (``.predict(x)``, ``.sauver(dossier)``).

    Notes
    -----
    Appareil : ``mps`` (carte graphique Apple) si disponible, sinon ``cuda``, sinon processeur.
    """
    import torch

    torch.manual_seed(graine)
    appareil = _appareil_torch()
    ex, ey = _Echelle(x), _Echelle(y)
    coupe = int(len(x) * (1 - validation))
    vers = lambda a: torch.from_numpy(a).to(appareil)
    xa, ya = vers(ex.aller(x[:coupe])), vers(ey.aller(y[:coupe]))
    xv, yv = vers(ex.aller(x[coupe:])), vers(ey.aller(y[coupe:]))
    w = np.ones(len(x), dtype="float32") if poids is None else np.asarray(poids, dtype="float32").ravel()
    wa, wv = vers(w[:coupe]), vers(w[coupe:])
    erreur = lambda p_, y_, w_: (w_[:, None] * (p_ - y_) ** 2).mean()   # erreur au carré pondérée
    forme = dict(n_entrees=x.shape[1], n_sorties=y.shape[1], couches=couches, dropout=dropout)
    reseau = _module_torch(**forme).to(appareil)
    poids_l2 = [b[0].weight for b in reseau.blocs]                  # pénalité L2 comme Keras (pas la sortie)
    opt = torch.optim.AdamW(reseau.parameters(), lr=taux, weight_decay=decroissance)
    pas = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6, min_lr=1e-6)
    meilleur, attente, etat, coupe = float("inf"), 0, None, False
    for _ in (avance := tqdm(range(epoques), desc="torch : époques", disable=not barre)):
        reseau.train()
        for i in torch.randperm(len(xa), device=appareil).split(lot):
            if limite is not None and time.time() > limite:
                coupe = True
                break
            opt.zero_grad()
            perte = erreur(reseau(xa[i]), ya[i], wa[i]) + l2 * sum((p_ ** 2).sum() for p_ in poids_l2)
            perte.backward()
            for p_ in reseau.parameters():                          # comme clipnorm=1 de Keras : poids par poids
                torch.nn.utils.clip_grad_norm_([p_], 1.0)
            opt.step()
        if coupe and etat is not None:                              # époque inachevée : on garde le meilleur
            break
        reseau.eval()
        with torch.no_grad():
            val = float(erreur(reseau(xv), yv, wv))
        pas.step(val)
        avance.set_postfix(validation=f"{val:.4f}")
        if val < meilleur:
            meilleur, attente = val, 0
            etat = {k: v.detach().clone() for k, v in reseau.state_dict().items()}
        else:
            attente += 1
            if attente >= patience:
                break
        if coupe:
            break
    reseau.load_state_dict(etat)
    reseau.eval()
    r = ReseauTorch(reseau, ex, ey, forme)
    r.coupe = coupe
    return r


ENTRAINEURS = {"keras": entrainer_keras, "torch": entrainer_torch}
"""dict of str to callable : réseaux de neurones (un réseau pour toutes les échéances), par nom."""

RESEAUX = {"keras": ReseauKeras, "torch": ReseauTorch}
"""dict of str to type : classe de chaque réseau entraîné (pour le relire)."""

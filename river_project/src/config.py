"""
Réglages du projet : chemins, abonnement Open-Meteo, accès RiverApp, colonnes BasinATLAS.

Les secrets (mots de passe, clés) ne sont jamais écrits dans le code : ils sont lus dans le
fichier ``RACINE/.env`` (lignes ``NOM=valeur``) ou dans les variables d'environnement.
"""

import os
from pathlib import Path

RACINE = Path(__file__).resolve().parents[1]
"""Path : dossier racine du projet (celui qui contient ``src/`` et ``data/``)."""
DATA = RACINE / "data"
"""Path : données du projet (BasinATLAS, caches)."""
CACHE = DATA / "cache"
"""Path : caches calculés (table des bassins, tables de recouvrement, météo, hydro)."""
CACHE_METEO = CACHE / "meteo"
"""Path : cache Open-Meteo (carreaux, couverture) et journal unique des appels."""
CACHE_HYDRO = CACHE / "hydro"
"""Path : cache RiverApp (liste des stations, mesures, couverture)."""
CACHE_PREVISION = CACHE / "prevision"
"""Path : cache des prévisions météo passées (API Previous Runs), un sous-dossier par modèle."""
CACHE_MODELES = CACHE / "modeles"
"""Path : modèles entraînés, un dossier par modèle (fiche, modèle, prévisions du test)."""

# Secrets (mots de passe, clés) : jamais dans le code, dans RACINE/.env (lignes NOM=valeur) ou l'environnement.
_ENV = RACINE / ".env"
if _ENV.exists():
    for _ligne in _ENV.read_text().splitlines():
        if "=" in _ligne and not _ligne.lstrip().startswith("#"):
            _nom, _valeur = _ligne.split("=", 1)
            os.environ.setdefault(_nom.strip(), _valeur.strip().strip('"'))

# Abonnement Open-Meteo en cours : "gratuit", "standard" ou "professional" (cf. meteo.ABONNEMENTS).
ABONNEMENT_OPEN_METEO = "gratuit"
"""str : abonnement Open-Meteo en cours, ``"gratuit"``, ``"standard"`` ou ``"professional"``
(limites dans ``meteo.ABONNEMENTS``)."""
CLE_OPEN_METEO = os.environ.get("OPEN_METEO_CLE")
"""str or None : clé Open-Meteo des abonnements payants, lue dans ``OPEN_METEO_CLE`` (.env)."""

# API partenaire RiverApp : débits et hauteurs des stations. L'adresse et les identifiants des
# fournisseurs sont dans src/riverapp_acces.py (privé, non publié) ; ici, seulement la connexion.
RIVERAPP_UTILISATEUR = os.environ.get("RIVERAPP_UTILISATEUR")
"""str or None : identifiant RiverApp, lu dans ``RIVERAPP_UTILISATEUR`` (.env)."""
RIVERAPP_MOT_DE_PASSE = os.environ.get("RIVERAPP_MOT_DE_PASSE")
"""str or None : mot de passe RiverApp, lu dans ``RIVERAPP_MOT_DE_PASSE`` (.env)."""

GDB_BASSINS = DATA / "hydroatlas" / "BasinATLAS_Data_v10" / "BasinATLAS_v10.gdb"
"""Path : base BasinATLAS v10 (polygones et attributs des bassins)."""
COUCHE_NIVEAU_12 = "BasinATLAS_v10_lev12"
"""str : couche des bassins de niveau 12 (le plus fin, ~1 million de bassins)."""

# Champs de topologie (hérités de HydroBASINS) : toujours lus.
# NEXT_DOWN = bassin aval (0 = exutoire), UP_AREA = surface amont totale (km²).
COLONNES_TOPOLOGIE = [
    "HYBAS_ID", "NEXT_DOWN", "NEXT_SINK", "MAIN_BAS", "DIST_SINK", "DIST_MAIN",
    "SUB_AREA", "UP_AREA", "PFAF_ID", "ENDO", "COAST", "ORDER_", "SORT",
]
"""list of str : champs de topologie toujours lus (NEXT_DOWN = bassin aval, 0 = exutoire ;
UP_AREA = surface amont totale en km² ; ENDO > 0 = zone fermée sans sortie vers la mer)."""

_MOIS = [f"{m:02d}" for m in range(1, 13)]

# Attributs utiles pour décider où placer les points météo.
# Suffixes (doc HydroATLAS) : s = dans le bassin, u = tout l'amont, p = à l'exutoire ;
# av = moyenne, mn = min, mx = max, yr = moyenne annuelle, se = part de surface (%),
# su = somme, mj = classe majoritaire, 01..12 = mois.
# Les valeurs sont BRUTES (cf. catalogue) ; la température est stockée en dixièmes de degré
# (doc HydroATLAS, table 2) : diviser par 10 pour des °C.
ATTRIBUTS = {
    "hydrologie": [
        "dis_m3_pyr", "dis_m3_pmn", "dis_m3_pmx",   # débit naturel à l'exutoire (m3/s)
        "run_mm_syr",                               # ruissellement (mm/an)
        "lka_pc_sse", "lkv_mc_usu", "rev_mc_usu", "dor_pc_pva",  # lacs, réservoirs, régulation
        "gwt_cm_sav",                               # profondeur de la nappe
    ],
    "relief": [
        "ele_mt_smn", "ele_mt_sav", "ele_mt_smx", "ele_mt_uav",  # altitude min/moy/max, moy. amont
        "slp_dg_sav", "slp_dg_uav",
    ],
    "climat": (
        ["tmp_dc_syr", "tmp_dc_smn", "tmp_dc_smx", "pre_mm_syr", "pet_mm_syr", "aet_mm_syr",
         "ari_ix_sav", "cmi_ix_syr", "snw_pc_syr", "snw_pc_smx", "clz_cl_smj"]
        + [f"tmp_dc_s{m}" for m in _MOIS]
        + [f"pre_mm_s{m}" for m in _MOIS]
        + [f"snw_pc_s{m}" for m in _MOIS]
    ),
    "occupation": [
        "gla_pc_sse", "prm_pc_sse", "for_pc_sse", "crp_pc_sse", "pst_pc_sse",
        "ire_pc_sse", "urb_pc_sse", "ppd_pk_sav", "hft_ix_s09",
    ],
    "sols": [
        "kar_pc_sse", "lit_cl_smj", "swc_pc_syr", "cly_pc_sav", "slt_pc_sav", "snd_pc_sav",
    ],
}
"""dict of str to list of str : attributs BasinATLAS gardés, rangés par thème.

Suffixes (doc HydroATLAS) : s = dans le bassin, u = tout l'amont, p = à l'exutoire ;
av = moyenne, mn = min, mx = max, yr = moyenne annuelle, se = part de surface (%),
su = somme, mj = classe majoritaire, 01..12 = mois. Valeurs brutes : la température
est en dixièmes de degré (diviser par 10 pour des °C).
"""


def attributs_par_defaut():
    """
    Liste à plat des attributs thématiques de `ATTRIBUTS`, sans la topologie.

    Returns
    -------
    list of str
        Les noms de colonnes de tous les thèmes, dans l'ordre de `ATTRIBUTS`
        (hydrologie, relief, climat, occupation, sols).

    See Also
    --------
    COLONNES_TOPOLOGIE : champs de topologie, lus en plus de ceux-ci.
    bassins.charger_bassins : lit topologie + ces attributs.

    Examples
    --------
    >>> attributs_par_defaut()[:3]
    ['dis_m3_pyr', 'dis_m3_pmn', 'dis_m3_pmx']
    """
    return [c for groupe in ATTRIBUTS.values() for c in groupe]

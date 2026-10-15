#!/usr/bin/env python3
"""
Prépare le jeu de rivières LIVRÉ sur la démo publique.

Copie les stations choisies de webapp/data/ vers webapp/data_public/ en :
  - ne gardant que meta.json, zones.geojson, modeles/, evaluation/ (le jeu de
    test, indispensable aux « tests sur le passé ») ;
  - RECOMPRESSANT les modèles .joblib (le Gradient Boosting passe de ~130 Mo à
    ~30 Mo) ;
  - en jetant tous les fichiers de travail (caches, X_train/Y_train...).

Usage :
    python3 preparer_donnees_publiques.py W231401001 X045401001 [...]
    python3 preparer_donnees_publiques.py --sans-gb W231401001   # sans le gros GB
Sans argument, prend une petite sélection par défaut.
"""
import os
import sys
import shutil
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import river  # noqa: nécessaire pour dé-sérialiser les modèles maison

SRC = os.path.join("webapp", "data")
DST = os.path.join("webapp", "data_public")
DEFAUT = ["W231401001", "X045401001"]


def octets(chemin):
    return os.path.getsize(chemin) if os.path.isfile(chemin) else 0


def preparer_station(code, sans_gb=False):
    src = os.path.join(SRC, code)
    dst = os.path.join(DST, code)
    if not os.path.isdir(src):
        print(f"  ⚠️  {code} : introuvable dans {SRC}, ignorée.")
        return 0
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(os.path.join(dst, "modeles"), exist_ok=True)

    total = 0
    # Fichiers racine légers.
    for f in ("meta.json", "zones.geojson"):
        p = os.path.join(src, f)
        if os.path.isfile(p):
            shutil.copy2(p, os.path.join(dst, f)); total += octets(os.path.join(dst, f))

    # evaluation/ (jeu de test).
    d_eval = os.path.join(src, "evaluation")
    if os.path.isdir(d_eval):
        shutil.copytree(d_eval, os.path.join(dst, "evaluation"))
        for r, _, fs in os.walk(os.path.join(dst, "evaluation")):
            total += sum(octets(os.path.join(r, f)) for f in fs)

    # modeles/ : recompression des .joblib, copie des .keras.
    d_mod = os.path.join(src, "modeles")
    if os.path.isdir(d_mod):
        for f in sorted(os.listdir(d_mod)):
            if sans_gb and f.startswith("gradient_boosting"):
                print(f"     · {f} ignoré (--sans-gb)"); continue
            src_f = os.path.join(d_mod, f)
            dst_f = os.path.join(dst, "modeles", f)
            if f.endswith(".joblib"):
                modele = joblib.load(src_f)
                joblib.dump(modele, dst_f, compress=3)
                print(f"     · {f} : {octets(src_f)//10**6} → {octets(dst_f)//10**6} Mo")
            else:
                shutil.copy2(src_f, dst_f)
            total += octets(dst_f)
    print(f"  ✅ {code} : {total//10**6} Mo")
    return total


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sans_gb = "--sans-gb" in sys.argv
    codes = args or DEFAUT
    os.makedirs(DST, exist_ok=True)
    print(f"Préparation de {len(codes)} station(s) dans {DST}/ (--sans-gb={sans_gb}) :")
    total = sum(preparer_station(c, sans_gb) for c in codes)
    print(f"\nTOTAL livré : {total//10**6} Mo dans {DST}/")
    print("Pense à : git add -f webapp/data_public/  (le dossier data/ est gitignoré)")


if __name__ == "__main__":
    main()

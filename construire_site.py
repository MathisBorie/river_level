#!/usr/bin/env python3
"""Construit le site STATIQUE (100% navigateur) dans docs/, prêt pour GitHub Pages.

Met à jour la version en ligne = relancer ce script puis `git add docs && git commit && git push`.
Le site n'a AUCUN serveur : tout (téléchargements API, sélection des points,
entraînement, prévision) tourne dans le navigateur via Pyodide (river_web.py)."""
import shutil
from pathlib import Path

RACINE = Path(__file__).resolve().parent
STATIC = RACINE / "webapp" / "static"
DOCS = RACINE / "docs"
DOCS.mkdir(exist_ok=True)

OVERLAY = '''<body>
<div id="overlay-chargement">
  <div class="spin"></div>
  <div class="titre">🌊 River Lab</div>
  <div class="msg">Démarrage du moteur dans ton navigateur (Python + scikit-learn)…<br>Le 1er chargement récupère ~15 Mo (mis en cache ensuite).</div>
</div>'''

# 1) index.html : ajoute l'overlay + le boot client, et rend les chemins RELATIFS
html = (STATIC / "index.html").read_text(encoding="utf-8")
html = html.replace("<body>", OVERLAY, 1)
html = html.replace('<script src="/static/app.js"></script>',
                    '<script src="/static/client_boot.js"></script>\n<script src="/static/app.js"></script>', 1)
html = html.replace("<title>River Lab — Prévision de débit</title>", "<title>River Lab — prévision de débit</title>", 1)
html = html.replace('href="/static/', 'href="').replace('src="/static/', 'src="')
(DOCS / "index.html").write_text(html, encoding="utf-8")

# 2) fichiers moteur (déjà en URLs relatives entre eux)
for f in ["style.css", "app.js", "client_boot.js", "pyodide_worker.js", "river_web.py"]:
    shutil.copy(STATIC / f, DOCS / f)

# 3) empêche GitHub Pages de passer les fichiers dans Jekyll (sinon river_web.py ignoré)
(DOCS / ".nojekyll").write_text("", encoding="utf-8")

restants = [x for x in ['href="/static', 'src="/static'] if x in html]
print("Site construit dans", DOCS)
print("Fichiers :", sorted(p.name for p in DOCS.iterdir()))
print("Chemins absolus restants dans index.html :", restants or "aucun ✅")

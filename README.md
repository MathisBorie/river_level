# river_level

Prévision du débit des rivières à partir de la météo. Un projet qui lie mes études en
mathématiques / science des données et ma pratique du kayak de rivière.

**Démo en ligne : [River Lab](https://mathisborie.github.io/river_level/)** — tout tourne dans le
navigateur (Pyodide), sans serveur.

## Organisation du dépôt

```
river_level/
├── river_project/   projet actuel : pipeline météo → bassins amont → modèles
├── brouillon/       première version (une station, La Bonne) et code du site River Lab
└── docs/            site River Lab publié (GitHub Pages), généré depuis brouillon/
```

## river_project — le projet actuel

Pour une station de mesure, on cherche à prévoir le débit à quelques jours.

1. **Bassin amont** : à partir de BasinATLAS (niveau 12), on remonte l'arbre des bassins pour
   obtenir la zone qui alimente la station, avec la surface, l'altitude et les caractéristiques
   physiques de chaque sous-bassin.
2. **Météo** : on choisit des points de la grille native du modèle météo dans cette zone, puis on
   télécharge la météo observée et les prévisions archivées (Open-Meteo), en comptant précisément
   le quota d'appels.
3. **Débits** : mesures des stations, ramenées à un pas régulier, avec retrait des pics isolés
   du capteur.
4. **Modèles** : réseaux de neurones (Keras / PyTorch), évalués par station et par échéance.

Le guide de lecture (notebooks dans l'ordre, rôle de chaque module) est dans
[`river_project/README.md`](river_project/README.md) ; le diagramme de classes dans
[`river_project/UML.md`](river_project/UML.md).

### Lancer

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r river_project/requirements.txt
jupyter lab river_project/notebooks
```

## Ce qui n'est pas dans le dépôt

- **Les données** : BasinATLAS (5,6 Go, à télécharger sur [hydrosheds.org](https://www.hydrosheds.org)),
  les caches météo et les sorties. Elles sont recréées par le code.
- **L'accès à la source des débits** : API partenaire privée. Sans elle, `src/hydro.py` ne lit que
  le cache ; pour rejouer le pipeline, il faut fournir ses propres débits au même format.
- **Les identifiants** : dans un fichier `river_project/.env` qui n'est jamais publié.

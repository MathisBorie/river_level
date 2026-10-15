# River Lab — site web d'interaction avec `river.py`

Interface web locale pour piloter tout le pipeline de prévision de débit :
carte des ~6 400 stations Hub'Eau, pipeline automatique (bassins versants →
sélection génétique des points météo → téléchargement 2000-2026 → modèles),
choix manuel des points sur carte, évaluation sur le passé (backtest avec
intervalles de confiance) et prévision réelle des 15 prochains jours.

## Lancer

```bash
cd webapp
pip install -r requirements.txt   # flask + joblib (le reste vient de river.py)
python3 app.py
```

Puis ouvrir **http://localhost:5050**.

## Gestion des appels API (Open-Meteo)

- **Fenêtre d'apprentissage** : deux sélecteurs de date (carte « Pipeline »)
  choisissent sur quelle période on télécharge/apprend, au lieu de prendre
  toute l'histoire. C'est le vrai levier pour économiser le quota (le coût
  Open-Meteo ≈ variables × jours / 14, indépendant du nombre de points groupés
  dans une requête batchée).
- **Agrégation météo par variable** (options avancées) : la neige est prise
  comme un STOCK (photo à minuit, 1 colonne), la température au choix en
  moyenne journalière (léger, ~130 dimensions) ou horaire (24 valeurs, ~1786
  dimensions). Divise la taille du problème sans perte utile.
- **Carte « Quotas API »** : tableau minute/heure/jour/mois (utilisé / limite /
  rechargement), partagé par toutes les stations (le quota est lié à l'IP),
  persisté dans `data/quota_openmeteo.json`. Avant chaque appel, l'appli patiente
  automatiquement pour rester sous les limites minute/heure (anti-429).

## Organisation du dépôt

`river.py` (bibliothèque) et `webapp/` (ce site) à la racine ; les notebooks
sont dans `notebooks/`, les cartes Folium générées dans `cartes_html/`, les
anciens CSV de travail dans `donnees_travail/`.

## Parcours utilisateur

1. **Carte des stations** : ~6 400 stations hydrométriques (clusters, recherche
   par nom/cours d'eau/code, filtre « en service »). Clic sur une station →
   « Ouvrir cette station ».
2. **Pipeline automatique** : un bouton fait tout — bassins versants (Sandre),
   cache météo léger 2016-2020 (réutilisé s'il existe déjà), sélection
   génétique DEAP des meilleurs points, téléchargement complet 2000-2026 sur
   les points retenus, entraînement du Gradient Boosting + réseau d'incertitude,
   sauvegarde automatique. Le log défile en direct dans la barre du bas.
3. **Ou choix manuel des points** : bouton « Choisir mes points à la main » →
   clic sur la carte (les zones vertes Sandre servent de guide), puis « Lancer
   avec ces points » : mêmes données complètes + Gradient Boosting, sans GA.
4. **PCA & les 4 modèles** : boutons « Appliquer la PCA » (graphique d'énergie)
   et « Entraîner les 4 modèles » (Linéaire-PCA, Ridge, Gradient Boosting,
   Keras + incertitude pour chacun). Tableau des R² mis à jour.
5. **Backtest** : choisir modèle + date (bornée au jeu de test 2021-2026, la
   date est ajustée automatiquement à la plus proche disponible) → fan chart
   IC 50/95/99 % + R² par horizon J+0…J+15.
6. **Prévision réelle** : choisir un modèle → débit récent Hub'Eau + météo
   **prévisionnelle** Open-Meteo sur les points retenus → courbe + tableau des
   15 prochains jours avec IC 95 %.
7. **Stockage** : voir plus bas.

## Architecture

```
webapp/
  app.py        Flask : API JSON + sert le front + les cartes Folium
  jobs.py       jobs de fond (1 à la fois — contrainte DEAP), log streamé
  stockage.py   persistance modèles + points, nettoyage des gros fichiers
  static/       index.html, app.js (vanilla + Leaflet), style.css
  data/
    stations.json          cache du référentiel Hub'Eau
    {code_station}/        un dossier par rivière (cwd des jobs de cette station)
      meta.json            points retenus, features, horizons, scores, zones…
      zones.geojson        polygones Sandre
      modeles/             *.joblib (sklearn), *.keras (TF), variance_*.keras,
                           scaler_pca.joblib, pca_model.joblib
      evaluation/          X_test.csv, Y_test.csv, dates_test.csv (~25 Mo)
      *.csv, *.html        fichiers de travail river.py (supprimables)
```

Les opérations longues tournent en **job de fond unique** ; le front les suit en
polling (`/api/jobs/<id>`). Les opérations rapides (backtest, prévision) sont
synchrones.

## Stockage : ce qu'on garde, ce qu'on jette

Une rivière entraînée ≈ 190 Mo de fichiers de travail (X_train ~100 Mo, cache
météo ~45 Mo…). Comme convenu, on ne garde que :

| Conservé | Rôle | Poids |
|---|---|---|
| `meta.json` + `zones.geojson` | points météo retenus, features, scores, zones | < 1 Mo |
| `modeles/` | les 4 modèles + réseaux de variance + scaler/PCA | ~15 Mo |
| `evaluation/` | jeu de test → backtests + audit sans retélécharger | ~25 Mo |

Le bouton **« Supprimer les gros fichiers »** efface le reste (refusé tant que
la sauvegarde n'existe pas). Après nettoyage : backtest, prévision réelle et
cartes fonctionnent toujours (rechargés depuis la sauvegarde au démarrage) ;
seul un **ré-entraînement** demanderait de relancer le pipeline (les données
d'entraînement ne sont pas conservées).

La sauvegarde est automatique après chaque entraînement (pipeline, points
manuels, « Entraîner les 4 modèles ») ; le bouton « Sauvegarder » force une
sauvegarde manuelle.

## Notes

- Une seule tâche lourde à la fois (contrainte du registre global DEAP + quotas
  API) : lancer un 2e job renvoie 409 avec le job en cours.
- Quota Open-Meteo gratuit : le cache météo léger d'une station est réutilisé
  d'un run à l'autre ; en cas de 429 persistant, attendre le début d'heure
  suivant et relancer.
- Le serveur est prévu pour un usage local (pas d'authentification).

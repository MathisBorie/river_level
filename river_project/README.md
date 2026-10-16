# river_project

Prévision du débit d'une rivière à quelques jours, à partir de la météo. Station d'étude :
**La Durance à Embrun (La Clapière)**.

## Par où commencer

Les notebooks se lisent dans l'ordre : chacun correspond à une étape du pipeline et appelle le
module de `src/` du même nom.

| Notebook | Étape |
|---|---|
| `01_exploration_donnees` | Les données de la station : débits, saisons, trous |
| `02_selection_points` | L'arbre des bassins (BasinATLAS) : quelle zone alimente la station, quels points météo choisir |
| `03_meteo` | Téléchargement de la météo par carreau (Open-Meteo), avec compteur d'appels et cache |
| `04_hydro` | Les débits mesurés, ramenés à un pas régulier (source privée, voir plus bas) |
| `05_modeles` | Construction de X et Y, trois modèles (gradient boosting, réseau Keras, réseau PyTorch), validation croisée |
| `06_erreurs_previsions` | Erreur des vraies prévisions météo passées face à la météo observée *(brouillon)* |
| `08_glofas` | Comparaison avec GloFAS, le modèle hydrologique de Copernicus *(brouillon)* |
| `00_brouillon` | Essais |

## Le code (`src/`)

| Module | Rôle |
|---|---|
| `bassins.py` | Forêt des bassins versants (BasinATLAS niveau 12) : amont d'une station, carreaux météo |
| `meteo.py` | Accès à Open-Meteo par carreau, cache et décompte du quota |
| `hydro.py` | Débits et hauteurs des stations, avec cache |
| `features.py` | Entrées (X) et cible (Y), découpage entraînement / validation / test |
| `modeles.py` | Modèles de prévision, un par échéance |
| `reseaux.py` | Les deux réseaux de neurones (Keras et PyTorch) |
| `station.py` | Point d'entrée : une station, son bassin, ses données, ses modèles |
| `graphes.py` | Graphiques interactifs des résultats |
| `config.py` | Chemins et réglages (les secrets sont lus dans `.env`, jamais écrits) |

Diagramme de classes : [`UML.md`](UML.md).

## Autres dossiers

- `scripts/` : téléchargements longs (météo observée, prévisions archivées, ensembles).
- `doc/` : notes sur les quotas Open-Meteo et schémas.
- `data/` : vide sur GitHub ; reçoit les bassins (BasinATLAS), les caches et les sorties.

## Source des débits

Les débits viennent d'une API partenaire privée, dont l'accès n'est pas publié. Sans lui,
`hydro.py` ne lit que le cache : pour rejouer le pipeline, il faut fournir ses propres débits au
même format (voir `src/hydro.py`).

## Installer

```bash
pip install -r requirements.txt
```

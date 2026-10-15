# Open-Meteo — ce qu'on sait et ce qu'on a mesuré

Synthèse des recherches et mesures faites pour river_project (septembre – octobre 2026).
Chaque chiffre est **mesuré** (avec la date) ou **tiré d'une source** (citée) ; ce qui n'est pas
vérifié est marqué **[à vérifier]**.

---

## 1. Deux API : le passé et la prévision

| | Archive (passé) | Prévision (futur) |
|---|---|---|
| Adresse (gratuit) | `archive-api.open-meteo.com/v1/archive` | `api.open-meteo.com/v1/forecast` |
| Données | réanalyses ERA5 / ERA5-Land, depuis 1940 | modèles de prévision (« best_match » : le meilleur modèle selon l'endroit) |
| Usage chez nous | entraîner les modèles | faire tourner les prévisions |
| Grille | **O1280 de l'ECMWF, vérifiée** (section 2) | **[à vérifier]** : d'autres modèles, donc a priori d'autres grilles (ex. AROME de Météo-France à 1,25 km en France, échéances 1 à 42 h — PDF Météo-France) |
| Horizon | — | variable d'un jour à l'autre, ~14 à 16 jours (observé sur river_level : demander +15 j en dur donne parfois une erreur 400) |
| Compteur de quota | séparé de celui des prévisions (observé sur river_level, 2026-07) | séparé |

Le code (`src/meteo.py`) ne gère pour l'instant **que l'archive**. Avant de brancher les prévisions,
il faudra vérifier leur grille : le pavage et la table de recouvrement ne valent que pour l'archive.

---

## 2. Répartition spatiale : la grille des carreaux

### Ce qu'Open-Meteo renvoie
Chaque réponse contient `latitude` / `longitude` = « WGS84 of the center of the weather grid-cell which was
used to generate this forecast » (doc Open-Meteo), même **sans aucune variable demandée**, et identique quelles
que soient la date et les variables.

### La grille est calculable
Les centres renvoyés par l'archive sont **exactement** les points de la grille gaussienne réduite octaédrique
**O1280** de l'ECMWF (celle d'ERA5-Land, « TCo1279 ») :
- 2560 latitudes = les latitudes gaussiennes (racines du polynôme de Legendre de degré 2560), à 3·10⁻⁶ ° près ;
- la rangée i (1 = la plus proche du pôle) a **4i + 16** points (doc ECMWF : « 20 longitude points at the latitude
  circle closest to the poles; the number of points increases by 4 at each latitude line ») ;
- longitudes = k × 360 / (4i + 16), en partant de 0° ;
- total mondial : 4N(N+9) = **6 599 680 points**.

**Vérifié** (2026-09-23) : 160 centres sur 160 dans 10 régions du monde (Amazonie, Mississippi, Congo, Mékong,
Murray, Ienisseï, Mackenzie, Groenland, Gange, Nil) sont sur la grille calculée, les deux hémisphères et les
longitudes négatives compris ; **mêmes centres en 1985 et en 2023**.

→ **Les centres se calculent, sans aucun appel** (`bassins._grille`).

### Taille des carreaux
- Nord-sud : **7,8 km** partout (rangées tous les ~0,0703°).
- Est-ouest : 7,8 km à l'équateur → 11,8 km vers 68° (moins de points par rangée vers les pôles).
- Surface : **61 km² à l'équateur → 93 km² vers 68°** (mesuré sur les 10 régions). Plus homogène qu'une grille en
  degrés, dont les carreaux rétréciraient vers les pôles : c'est la raison d'être de cette grille.

### Quel carreau sert un point donné : `cell_selection`
- `land` (**défaut**) : carreau voisin d'**altitude proche** du point (modèle de terrain à 90 m). En montagne, ce
  n'est souvent pas le plus proche : 11 points sur 61 changent de carreau sur une ligne test dans l'Ubaye.
- `nearest` : le centre **le plus proche en degrés bruts** (longitude, latitude telles quelles, pas en km) —
  vérifié 400/400 (contre 385/400 avec des distances en km), puis 500/500 et 200/200 ; dans le monde 158/160, les
  2 écarts étant des points pile sur une frontière (égalité ou quasi-égalité de distance).
- `sea` : préfère les carreaux en mer.

### Pavage de Voronoï
La zone servie par chaque carreau (en mode `nearest`) = sa cellule de Voronoï **en degrés bruts**. Forme : les
rangées n'ont pas toutes le même nombre de points, donc elles sont décalées (~0,01°, soit ~800 m par rangée dans
les Alpes) → frontières entre rangées qui penchent (vers le sud-est dans les Alpes) et croisements de 3 cellules
(normal pour tout Voronoï dès que les points ne forment pas des rectangles parfaits). Sans effet pour nous.

### Combien de carreaux (table de recouvrement, 2026-10-06)
| Zone | Carreaux |
|---|---|
| Amont d'Espinasses (Durance, 3 583 km²) | 61 |
| France métropolitaine (contour IGN, 549 167 km²) | 7 042 |
| France + amont étranger de ses rivières (619 178 km²) | **7 621** |
| Europe (région HydroBASINS 2) | 221 784 |
| Monde (tous les bassins BasinATLAS) | ~1,79 million |

Côtes **[à vérifier]** : ERA5-Land n'a pas de points en mer ; un pavage calculé sur un bassin côtier peut contenir
des carreaux qu'Open-Meteo ne sert jamais.

---

## 3. Ce que contient un carreau : la correction d'altitude

Mesuré (2026-09-23) sur 74 points d'un même carreau au-dessus de Barcelonnette (1 125 → 2 452 m, 15/01/2023) :
- pluie, neige, vent, rayonnement : **identiques** pour tous les points du carreau ;
- température : **ramenée à l'altitude du point** demandé, ~0,65 °C / 100 m (+3,1 °C à 1 125 m, −5,5 °C à 2 452 m) ;
- humidité : suit un peu la température.

Paramètre `elevation` (doc) : « The elevation used for statistical downscaling » ; `elevation=nan` (une valeur par
point dans une requête groupée, sinon erreur 400) donne la valeur brute du carreau, à **son altitude moyenne**
(1 995 m pour ce carreau, alors que la vallée est à 1 130 m).

Conséquences :
- « même carreau = même prévision » est faux pour la température, vrai pour le reste ;
- plusieurs altitudes dans un même carreau n'apportent **aucune information nouvelle** au modèle (décalage fixe) :
  ce qu'on choisit, ce sont des **carreaux**, pas des points.

---

## 4. Le coût des requêtes

### La règle (mesurée le 2026-10-06)
```
coût (en appels) = nombre de points × le plus grand de : 1 ; variables / 10 ; variables × jours / 140
```
Autrement dit : **1 appel = 10 variables pendant 14 jours pour 1 point**, au minimum 1 par point.

| Requête (1 point sauf indication) | Coût mesuré | Règle |
|---|---|---|
| 10 points, 1 jour, 0 variable | 10 | 10 |
| 0, 1 ou 10 variables, 1 jour | 1 | 1 |
| 11 variables, 1 jour | ~1,05–1,1 | 1,1 |
| 20 variables, 1 jour | 2 | 2 |
| 1 variable, 14 / 15 / 28 jours | 1 | 1 |
| 0 variable, 1 an | 1 | 1 |
| 1 variable horaire, 1 an | **2,6** | 2,61 |
| 1 variable journalière, 1 an | 2–3 | 2,61 |
| 5 points, 1 variable, 1 an | 13 | 13,0 |
| 9 horaires + 1 journalière, 1 an | 26 | 26,1 |
| 20 variables, 1 an | 52 | 52,1 |

À retenir :
- **chaque point d'une requête groupée compte pour un appel** (aussi mesuré le 2026-09-23 : 620 points acceptés dans
  la minute, puis refus) ;
- **la doc est trompeuse sur les longues périodes** : elle laisse croire « plus de 2 semaines = plusieurs appels »
  (1 variable sur 1 an = 26), alors que c'est le **volume** qui compte (1 variable sur 1 an = 2,6) ;
- **sur un long historique, chaque variable se paie** (25 ans, 1 carreau : 65 appels par variable, 652 pour 10) ;
  sous 10 variables et 2 semaines, en revanche, le nombre de variables ne change rien ;
- **horaire et journalier coûtent pareil** : c'est compté en variables × jours, pas en valeurs ; une variable
  journalière compte comme une horaire. Mieux vaut donc télécharger l'horaire (on peut en tirer sommes, max,
  moyennes ; l'inverse est impossible), sauf pour les variables qui n'existent qu'en journalier (ex. température max).

### Comment on l'a mesuré (aucune API ne donne le compteur)
Open-Meteo ne renvoie jamais le quota restant ; il refuse seulement (code 429, « Minutely API request limit
exceeded ») au-delà de 600 appels dans la minute. On s'en sert comme d'une **balance** : en début de minute, on
remplit le compteur avec une requête de coût connu (600 − m points, 1 jour, 0 variable), on envoie r fois la
requête à peser, puis des requêtes d'un point (coût 1) jusqu'au refus ; si c passent, la requête pesait
(m − c) / r. Calibrage : 10 points pèsent exactement 10. Script : `doc/open_meteo/pesee_openmeteo.py` (modifier la liste `TESTS` ; chaque mesure consomme ~600 appels, compter 1 h de pause toutes les 7 mesures à cause de la limite horaire).

### Limites de l'accès gratuit
600 appels / minute, 5 000 / heure, 10 000 / jour, 300 000 / mois (page Pricing).
Le refus précise la fenêtre (« Minutely », « Hourly », « Daily ») : la minute se recharge en ~60 s, le jour non.
Aucun en-tête de quota n'est renvoyé.

### Détails pratiques
- Au-delà d'environ 150 points, une requête GET dépasse la taille d'adresse acceptée (erreur 414) : utiliser **POST**
  (testé jusqu'à 500 points).
- Plusieurs coordonnées : `latitude=a,b,c&longitude=x,y,z` ; la réponse devient une liste.

### Ordres de grandeur (10 variables, 25 ans)
| Zone | Appels | Avec le quota gratuit |
|---|---|---|
| 1 carreau | 652 | — |
| Amont d'Espinasses (61) | ~40 000 | ~4 jours |
| France + amont étranger (7 621) | ~5 millions | ~16 mois |
| Europe (221 784) | ~145 millions | ~40 ans |

Pistes pour réduire : moins de carreaux par bassin (les voisins sont très corrélés ; à mesurer : précision en
fonction du nombre de carreaux gardés, tirés au hasard ou stratifiés par surface / altitude), moins de variables
(proportionnel), moins d'années (mais la précision baisse : R² J+3 de 0,866 avec 10 ans à 0,886 avec 42 ans).

---

## 5. Les abonnements

| Forfait | Prix | Quota | Historique | Usage commercial |
|---|---|---|---|---|
| Gratuit | 0 € | 10 000 / jour, 300 000 / mois | oui | **non** |
| Standard | 29 € / mois | 1 million / mois, pas de limite minute ni jour | **non** | oui |
| Professional | 99 € / mois | 5 millions / mois | **oui** | oui |
| Enterprise | sur devis | 50 millions + | oui | oui |

(Page Pricing et annonce Open-Meteo ; prix relevés le 2026-10-06.)

Stratégie retenue :
1. **Développement** : le gratuit suffit.
2. **Entraînement** : **un mois de Professional (99 €)** — seul forfait avec historique ET usage commercial —
   pour remplir le cache (toute la France sur 25 ans ≈ 5 millions d'appels, soit un mois).
3. **Production** : **Standard (29 € / mois)** pour les prévisions. Ordre de grandeur : une prévision à 16 jours
   avec 10 variables ≈ 1,1 appel par carreau ; toute la France rafraîchie 4 fois par jour ≈ 1 million / mois ;
   une seule rivière comme Espinasses ≈ 8 000 / mois.

**[à vérifier]** :
- l'adresse de l'archive pour les abonnés (la doc publique ne donne que `customer-api.open-meteo.com` pour la
  prévision, avec `&apikey=...`) ;
- le droit de continuer à utiliser les données téléchargées après l'arrêt de l'abonnement (les données Open-Meteo
  sont sous licence CC-BY 4.0, donc a priori oui) ;
- le coût exact des prévisions (la règle de la section 4 a été mesurée sur l'archive seulement).

---

## 5 bis. Prévisions archivées et dispersion des ensembles (lu dans la doc le 2026-10-10)

Pour apprendre / tester avec de VRAIES prévisions (et pas la météo observée) :

| API | Ce qu'elle donne | Archivé depuis |
|---|---|---|
| Previous Runs (`previous-runs-api`) | la valeur prévue 1 à 7 jours avant chaque heure : variables `temperature_2m_previous_day1` … `_previous_day7` | la plupart des modèles **janvier 2024** (GFS température mars 2021, JMA 2018) |
| Single Runs (`single-runs-api`, `&run=2024-06-01T00:00`) | une prévision complète telle qu'émise à un instant | ECMWF IFS HRES 9 km **14 mars 2024** (10 jours) ; autres modèles 2 avril 2026 |
| Historical Forecast | les premières heures de chaque prévision, mises bout à bout | 2021-2022 ; **pas** les longues échéances |
| Ensemble (`ensemble-api`) | les membres (ex. ECMWF IFS 51 membres, 15 jours) | **3 jours seulement** |
| Ensemble Mean | moyenne + **dispersion = écart-type** (`temperature_2m_spread`) | **mars 2026** pour la plupart |

Modèles d'ensemble ECMWF de l'API Ensemble (lu dans la doc le 2026-10-13) : IFS 0,25° (51 membres, 15 j, toutes les 6 h),
IFS 9 km O1280 **Europe** (51 membres, 15 j, runs de 0 h et 6 h seulement, moins de variables), AIFS 0,25° et AIFS 31 km Europe.
Depuis juin 2023 (cycle 48r1), l'ensemble ECMWF tourne à 9 km comme HRES, et son membre de contrôle est « météorologiquement
équivalent » à HRES (ECMWF). Coût : selon Open-Meteo, le nombre de variables « peut être multiplié par le nombre de membres »
(1 variable × 51 membres ≈ 51 variables, soit ≈ 5 à 6 appels par point pour 15 jours avec la règle de la section 4) ; **pas encore mesuré**.
Bassin de La Clapière : 41 carreaux à 9 km, 11 points à 0,25°.
**Les 51 scénarios complets sont archivés (vérifié le 2026-10-15)** : par l'API **Single Runs**, `models=ecmwf_ifs_europe_ensemble`
(9 km, grille O1280 = nos carreaux) donne chaque run COMPLET avec ses 51 séries (`precipitation`, `precipitation_member01` … `_member50`),
runs de **0 h UTC (15 jours)** et **6 h UTC (6 jours)** seulement, **depuis le 2026-06-24** (à revérifier : fenêtre glissante ou non).
Variables : précipitations, température, neige qui tombe, soleil ; pas de neige au sol ni d'humidité du sol. Les « 3 jours » ne
concernent que l'API Ensemble (série continue). `ecmwf_ifs025_ensemble` = la version 0,25°. Téléchargement :
scripts/telecharger_ensemble_clapiere.py (précipitations + température, 41 carreaux, ≈ 10,9 appels par carreau et par run estimés).
**Pas de choix de membres** : l'API Ensemble n'a aucun paramètre pour ne demander que quelques membres (doc lue le 2026-10-13).
L'API Ensemble Mean donne, à l'heure, la moyenne et la **dispersion = écart-type** des membres (pluie, neige, température,
neige au sol, sol…) pour ECMWF IFS 0,25° et IFS 9 km Europe ; archivée depuis mars 2026 environ ; jusqu'à 35 jours.
**D'où vient notre météo « observée » (mesuré le 2026-10-13, La Clapière)** : sans `models`, l'archive sert « best_match »,
qui CHANGE DE SOURCE LE 1er JANVIER 2017. Comparaison exacte du cache avec `models=era5`, `era5_land`, `ecmwf_ifs` :

| variable | avant 2017 | depuis 2017 |
|---|---|---|
| pluie, rayonnement | ERA5 (0,25°, point le plus proche : plusieurs carreaux ont la MÊME série) | ECMWF IFS 9 km |
| température, humidité du sol | ERA5-Land (9 km) | ECMWF IFS 9 km |
| neige au sol | ERA5-Land | ERA5-Land (IFS n'en a pas dans l'archive) |
| neige qui tombe, ET0 | calculées par Open-Meteo (aucun modèle identique) | idem |

Degrés de liberté de la pluie sur les 41 carreaux (composantes pour 99,99 % de la variance) : **11 avant 2017** (= les 11 points
0,25° du bassin), **41 depuis**. Notre historique est donc fait de deux jeux de données différents recollés.
(ERA5-Land n'a pas de pluie chez Open-Meteo ; d'après l'ECMWF, ses champs de forçage sont « purely interpolated » depuis ERA5.)

Conséquences :
- Previous Runs = la façon la moins chère d'avoir des prévisions passées : une échéance = une variable de plus
  (×7 variables, pas ×500) ; mais seulement 1 à 7 jours d'avance et environ 2,8 ans d'archive.
- La dispersion d'ensemble existe mais n'est archivée que depuis mars 2026 : utilisable pour l'incertitude en
  temps réel, trop courte pour apprendre.
- **Mesuré le 2026-10-10 (requêtes d'essai, La Clapière)** :
  - `models=ecmwf_ifs` (9 km) sert **exactement le centre de nos carreaux** (même grille O1280 qu'ERA5-Land),
    mais Previous Runs ne l'a que depuis le **2025-10-01** ; variables présentes : precipitation, temperature_2m,
    rain, snowfall, shortwave_radiation, et0_fao_evapotranspiration ;
  - `models=ecmwf_ifs025` (0,25°) depuis le **2024-02-04** ; seulement precipitation, temperature_2m,
    shortwave_radiation, et0_fao_evapotranspiration ; le centre servi est le point 0,25° le plus proche ;
  - **ni snow_depth ni soil_moisture** en prévision passée (état du sol et de la neige : à prendre au temps t) ;
  - requêtes groupées en POST acceptées ; **même compteur que l'archive** (refus « Daily API request limit exceeded » à 10 483 appels cumulés archive + prévisions) ; ECMWF 0,25° écarté : on reste sur nos carreaux.
- **Single Runs, mesuré le 2026-10-11** : `models=ecmwf_ifs` + `&run=2024-06-01T00:00` renvoie la prévision COMPLÈTE
  de 10 jours, **sur le centre exact de nos carreaux**, avec precipitation, snowfall, snow_depth,
  soil_moisture_7_to_28cm et 28_to_100cm (runs testés : 2024-06-01, 2025-11-01, 2026-09-01). C'est mieux que
  Previous Runs pour nous : archivé depuis 2024 (au lieu d'octobre 2025), 10 jours d'avance (au lieu de 7), et
  l'humidité du sol et la neige au sol y sont. Coût estimé (règle de la section 4) : ≈ 1 appel par run et par
  carreau pour ≤ 10 variables, soit ~950 appels par carreau pour un run par jour depuis 2024.
- La prévision en temps réel (`api.open-meteo.com`, `models=ecmwf_ifs`) donne aussi l'humidité du sol sur 10 jours.
- **[à vérifier]** : le coût réel de ces API (règle de la section 4 mesurée sur l'archive seulement).

---

## 5 ter. Débits GloFAS (Flood API, vérifié le 2026-10-13)

`https://flood-api.open-meteo.com/v1/flood`, `daily=river_discharge` (m³/s, moyenne du jour), grille 0,05° ; la rivière servie
est la plus grosse dans un rayon de 5 km (à La Clapière, le carreau par défaut 44,575 / 6,475 est bien la Durance). GloFAS
(modèle LISFLOOD) n'utilise pas les débits mesurés. Versions (`models=`) :

| `models` | contenu | dates (La Clapière) |
|---|---|---|
| `consolidated_v4` | reconstitution avec la météo observée | 1997-01-01 → 2025-05-31 (la doc dit « 1984 → juillet 2022 ») |
| `forecast_v4` | prévision quotidienne sur 30 jours ; chaque run écrase le précédent (code Open-Meteo : `storePreviousForecast = false`), donc pour un jour passé il reste la prévision émise juste avant (≈ 1 jour d'avance) | 2023-06-01 → aujourd'hui + 30 j |
| `seamless_v4` (défaut) | `consolidated_v4` jusqu'au 2023-05-31, puis `forecast_v4` (vérifié valeur par valeur) | 1997 → aujourd'hui + 30 j |

`ensemble=true` : `river_discharge` (contrôle, membre 0) + `river_discharge_member01` … `member50`. Statistiques
(`_mean`, `_median`, `_p25`…) seulement pour les prévisions. Les anciennes prévisions ne sont pas gardées : pas de score à
5 ou 10 jours d'avance possible avec Open-Meteo (archives Copernicus EWDS : `cems-glofas-forecast`, `cems-glofas-reforecast`).
Analyse : notebooks/08_glofas.ipynb.

## 6. Stockage

- 10 variables (9 horaires + 1 journalière) sur 25 ans ≈ **8 Mo bruts par carreau** (valeurs sur 4 octets).
- Compression mesurée (Parquet, sur 9,3 millions de vraies valeurs) : **×5,8** → Espinasses ≈ 85 Mo,
  France ≈ 10 Go, Europe ≈ 300 Go.
- 8, 4 ou 2 octets par valeur donnent **la même taille sur disque** (6,4 / 6,4 / 6,3 Mo) : le format range la liste
  des valeurs différentes (les valeurs Open-Meteo sont arrondies, donc peu nombreuses) puis seulement leurs numéros.
  En mémoire, en revanche : 75 / 37 / 19 Mo.
- **Choix : 4 octets partout** (cache et mémoire). Précision perdue ~3·10⁻⁶ ; aucun effet mesuré sur les modèles
  (gradient boosting R² J+3 0,88954 → 0,88939 ; Ridge identique) ; mémoire divisée par 2.

---

## 7. Dans le code

- `src/config.py` : `ABONNEMENT_OPEN_METEO` (« gratuit », « standard », « professional ») ; clé API dans la variable
  d'environnement `OPEN_METEO_CLE` (jamais dans le code).
- `src/bassins.py` : `_grille` (centres O1280 calculés), `recouvrement(continent)` (table bassin × carreau × surface),
  `carreaux_amont(zone, foret)` (carreaux d'un amont et leurs pourcentages), `voronoi_meteo_amont` (pour dessiner).
- `src/meteo.py` : classe `Meteo` — `cout_prevu`, `telecharger`, `etat_quota`, `contenu_cache`, `vider_cache` (`api="archive"` ou `"previous_runs"`) ; classe `PrevisionsArchivees` (API Single Runs : prévisions complètes,
  cache `data/cache/prevision/single_runs/<modele>/`, même compteur) ;
  `cout_appel` (règle de la section 4) ; journal unique `data/cache/meteo/journal_appels.csv` ; cache par carreau
  `data/cache/meteo/carreaux/` + `couverture.csv`.
- `notebooks/03_meteo.ipynb` : tests de tout ça.
- `UML.md` : diagramme de classes.

---

## Sources

- [Open-Meteo — Docs (prévision)](https://open-meteo.com/en/docs)
- [Open-Meteo — Historical Weather API](https://open-meteo.com/en/docs/historical-weather-api)
- [Open-Meteo — Previous Runs API](https://open-meteo.com/en/docs/previous-runs-api)
- [Open-Meteo — Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
- [Open-Meteo — Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
- [Open-Meteo — Ensemble API](https://open-meteo.com/en/docs/ensemble-api)
- [Open-Meteo — Ensemble Mean API](https://open-meteo.com/en/docs/ensemble-mean-api)
- [Open-Meteo — Pricing](https://open-meteo.com/en/pricing)
- [Open-Meteo — Terms](https://open-meteo.com/en/terms)
- [Open-Meteo — API Subscriptions for Commercial Use](https://openmeteo.substack.com/p/api-subscriptions-for-commercial)
- [ECMWF — ERA5-Land data documentation](https://confluence.ecmwf.int/display/CKB/ERA5-Land:+data+documentation)
- [ECMWF — Octahedral grid (OpenIFS)](https://confluence.ecmwf.int/display/OIFS/2+Octahedral+grid)
- [Météo-France — documentation AROME (PDF)](https://donneespubliques.meteofrance.fr/client/document/doc_arome_pour-portail_20250806_410.pdf)
- [france-geojson (contour France métropolitaine)](https://github.com/gregoiredavid/france-geojson)

# Sélection des points météo — méthode en deux temps

On choisit ~5 points météo (entrées du modèle de débit) en deux étapes : d'abord
des signaux **gratuits** (altitude, géométrie), puis un petit téléchargement météo
sur les survivants. Coût Open-Meteo $\approx \text{points} \times \lceil \text{jours}/14 \rceil$ : on réduit donc points **et** jours.

## Étape 1 — gratuite (altitude + couverture + zones)

Parmi $N$ candidats d'une grille dense (altitude $h_i$ via l'endpoint élévation,
zone $z_i$), on en garde $k_1\approx30$ en maximisant, glouton :

$$F(S)=\underbrace{\sum_{j}\max_{i\in S} e^{-d_{ij}^2/\sigma^2}}_{\text{couverture}}
\;+\;\lambda_h\sum_{i\in S}\widehat h_i
\;+\;\lambda_z\sum_{z}\sqrt{n_z(S)}$$

- **couverture** (facility location) : $d_{ij}$ = distance dans $(x,y,\text{altitude})$ → étale les points, altitude comprise ;
- **altitude** : $\widehat h_i\in[0,1]$ (rang) → favorise les points hauts (neige, pluie orographique) ;
- **zones** : $\sqrt{n_z}$ **concave** → le énième point d'une zone rapporte moins → répartition entre zones.

$F$ est **sous-modulaire croissante** (rendements décroissants), donc le glouton
atteint $\ge (1-1/e)\approx 63\%$ de l'optimum (**Nemhauser–Wolsey–Fisher, 1978**) —
mieux, et bien plus rapide, qu'un algorithme génétique.

## Étape 2 — API légère (corrélation pluie→débit + neige)

On télécharge **seulement** ces $k_1$ points sur ~2 ans (pluie quotidienne, neige)
et on en garde $k_2\approx5$. Pour chaque point $P$ :

$$\text{pluie}(P)=\sum_{k=0}^{20}\max\bigl(0,\;r_k(P)\bigr),\qquad
r_k(P)=\operatorname{corr}\bigl(\text{pluie}_P(t),\,\text{débit}(t{+}k)\bigr)$$

$$\text{neige}(P)=\overline{\text{snow\_depth}}(P)$$

On **somme** les corrélations décalées (positives) pour capter une réponse étalée
dans le temps ; le $k$ du pic donne le **temps de réponse du bassin**. La neige est
un **stock** (la fonte, c'est l'affaire du modèle).

Les deux scores, normalisés **par rang** (sans unité, robuste), donnent
$\text{pertinence}(P)=w_p\,\widehat{\text{pluie}}(P)+w_n\,\widehat{\text{neige}}(P)$.
Sélection gloutonne avec **pénalité de redondance** :

$$\text{score}(P)=\text{pertinence}(P)\,\Bigl(1-\max_{Q\in S}e^{-d_{PQ}^2/\sigma_r^2}\Bigr)$$

→ des points **pertinents et étalés**.

## En bref

$$\text{grille dense}\ \xrightarrow[\text{gratuit}]{\text{étape 1}}\ \sim\!30\ \text{points}\ \xrightarrow[\text{API légère}]{\text{étape 2}}\ \sim\!5\ \text{points}\ \rightarrow\ \text{modèle}$$

Coût sélection $\approx 30\times\lceil730/14\rceil\approx1500$ appels (vs ~11 700 pour l'ancien
cache de 150 points × 3 ans). Le seul gros téléchargement restant : l'historique
complet, sur les 5 points finaux.

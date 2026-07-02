# river_level

Ce projet a pour objectif de prédire le niveau d'eau des rivières. Un projet qui lie mes études en mathématiques/science de la donnée et ma pratique de kayakiste de rivière à très bon niveau.



Nous allons, au moins pour l'instant, nous intéresser aux rivières à débit naturel, c'est à dire celles qui ne sont pas régulées par des barrages ou autres aménagements.

Et dans un premier temps, nous prendrons comme premier cas d'étude la rivière "La Bonne" dans le département de l'Isère (38).

Nous allons tenter de construire des modèles mathématiques prenant en compte des données météorologiques (pluie, température, etc.), des données hydrologiques (débit) ainsi que la date pour prédire le niveau d'eau de la rivière sur les prochains jours (typiquement 15 jours qui la durée de la prévision météo).

# Les données

## Les données hydrologiques


https://hydro.eaufrance.fr met a disposition les niveaux d'eau d'un très grand nombre de rivière.

![Niveau d'eau de la rivière](figure\niveau.png)

## Données météorologiques


https://archive-api.open-meteo.com/v1/archive et https://api.open-meteo.com/v1/forecast mettent à disposition des données météorologiques historiques et prévisionnelles.


![Niveau d'eau de la rivière](figure\meteo.png)

## La date

Nous allons encoder la date de manière cyclique pour que le modèle puisse comprendre la saisonnalité des données (ex: éviter la discontinuité entre le 31/12 et le 1/01).


![Niveau d'eau de la rivière](figure\date.png)



# Les premiers modèles

Tout d'abord les données d'eau heures par heures ne sont gardées que pour les 30 derniers jours, Ce qui est insuffisant pour entraîner un modèle de machine learning. Nous allons donc nous intéresser aux données journalières, qui elles sont conservées depuis plusieurs dizaines d'années (dépandant des rivières).

Nos premiers modèles prennent donc en entrée les données météorologiques et hydroliques des 20 derniers jours pour prédire le niveau d'eau de la rivière sur les 15 prochains jours.

ce qui donne un problème de régression multivariée avec 67 variables d'entrée et 16 variables de sortie (jour J et les 15 jours suivants).
Le résultat des différents modeles est évalué avec le coefficient de détermination R² donné jour par jour.


## Premiers résultats

un exemple de prédiction sur 15 jours avec le modèle de réseau de neurones est donné ci-dessous :

![Niveau d'eau de la rivière](figure\visualisation1.png)

Les performances des différents modèles sont données ci-dessous :

![Niveau d'eau de la rivière](figure\resultats_premiers_modeles.png)

C'est premiers résultats sont déjà satisfaisants (Les modèles polynomiaux était assez mauvais donc je ne les ai pas mis sur le graphique).
J'ai toutefois l'impression d'avoir heurté un plafond de performance. Je vais donc maintenant essayer d'augmenter et d'améliorer les variables d'entrées.

PS : J'ai pour l'instant choisi les hyperparamètres des modèles sur les données test pour aller plus vite mais pour les modèles finaux je ferai un vrai split train/test/validation voir de la cross-validation.


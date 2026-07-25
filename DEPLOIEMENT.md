# Mettre River Lab en ligne

Objectif : un site **accessible à tout le monde**, en ligne, gratuit, qui marche
aussi sur téléphone, **démarré à vide** — chaque visiteur analyse et gère **ses
propres** rivières (aucune rivière n'est pré-livrée).

Hébergeur retenu : **Hugging Face Spaces** (seul gratuit avec assez de RAM pour
TensorFlow : 2 vCPU / 16 Go).

## Ce qui est déjà prêt
- `Dockerfile` — image prête pour HF Spaces (port 7860, utilisateur non-root, app complète).
- `requirements.txt` — dépendances figées (`tensorflow-cpu`).
- `.dockerignore` — exclut les 1,5 Go de données de dev (`webapp/data/`).
- `HOST`/`PORT` par variables d'environnement ; endpoint `/api/config`.

## Étapes
1. Compte gratuit sur https://huggingface.co → **New Space** → SDK **Docker**, visibilité **Public**.
2. Dans le `README.md` du Space, mettre en tête :
   ```
   ---
   title: River Lab
   emoji: 🌊
   colorFrom: green
   colorTo: blue
   sdk: docker
   app_port: 7860
   ---
   ```
3. Pousser le code (le dépôt part **vide de données** ; les modèles seront créés par les visiteurs) :
   ```bash
   git clone https://huggingface.co/spaces/<toi>/river-lab && cd river-lab
   cp -r /chemin/river_level/{Dockerfile,requirements.txt,river.py,webapp,.dockerignore} .
   # (ajouter l'entête HF dans README.md)
   git add . && git commit -m "River Lab" && git push
   ```
4. HF construit et lance (~5-10 min). URL publique = celle du Space, utilisable sur ordi **et téléphone**.

**Mises à jour** : `git push` → reconstruction automatique.

## Ce qu'il faut savoir (hébergement gratuit partagé)
- **Quota Open-Meteo partagé** (lié à l'IP du Space) : si beaucoup de gens
  entraînent en même temps, ça peut être limité (l'app patiente / réessaie).
- **1 entraînement à la fois** (verrou global) : un 2ᵉ visiteur qui lance une
  analyse pendant qu'une autre tourne reçoit « occupé ».
- **Stockage éphémère** : les modèles entraînés en ligne sont perdus au
  redémarrage/reconstruction du Space. Pour une persistance durable il faudra un
  stockage persistant (option payante) ; à voir plus tard.
- Un **mode lecture seule** existe (env `RIVER_PUBLIC=1`) si un jour tu veux
  figer une démo avec quelques rivières et bloquer l'entraînement public.

## Voir l'utilisation (compteur de visites — GoatCounter, gratuit)
1. Crée un compte gratuit sur https://www.goatcounter.com → choisis un **code**
   (ex. `riverlab`) → ton tableau de bord est `https://riverlab.goatcounter.com`.
2. Sur le Space HF : **Settings → Variables and secrets → New variable**
   `GOATCOUNTER = riverlab` (juste le code, pas l'URL).
3. C'est tout : le site charge automatiquement le compteur et enregistre chaque
   visite **et chaque station ouverte** (`station/<code>`). Statistiques sur ton
   tableau de bord GoatCounter (visiteurs, pages, pays, provenance).

En local, laisse `GOATCOUNTER` non défini → aucun compteur (rien n'est envoyé).

## Notes
- Si le build casse sur `tensorflow-cpu==2.20.0`, remplace par `tensorflow==2.20.0`.

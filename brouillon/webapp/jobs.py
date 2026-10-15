"""
Gestionnaire de tâches de fond pour le site.

Les opérations longues de river.py (pipeline complet, entraînement...) tournent
dans un thread, UNE SEULE à la fois :
- c'est une contrainte réelle de DEAP (registre global creator.create, cf.
  docstring de river.py) ;
- ça évite aussi de saturer les quotas API et le CPU.

Chaque job accumule les messages de log de River (via un branchement sur
riviere._log) que le front récupère en polling.
"""

import os
import threading
import traceback
from datetime import datetime

from river import ArretDemande


class JobOccupe(Exception):
    pass


class Job:
    def __init__(self, job_id, type_job, code_station):
        self.id = job_id
        self.type = type_job
        self.code_station = code_station
        self.statut = "en_attente"       # en_attente | en_cours | termine | erreur | arrete
        self.log = []
        self.resultat = None
        self.erreur = None
        self.debut = datetime.now().isoformat(timespec="seconds")
        self.fin = None
        self.demande_arret = threading.Event()   # levée par le bouton « Arrêter »
        self.progression = None                  # {"phase", "courant", "total", "pct"}

    def ajouter_log(self, message):
        self.log.append(message)

    def definir_progres(self, phase, courant, total):
        if total:
            pct = max(0, min(100, round(100 * courant / total)))
            self.progression = {"phase": phase, "courant": int(courant), "total": int(total), "pct": pct}
        else:
            # Étape à durée non quantifiable : barre indéterminée.
            self.progression = {"phase": phase, "courant": None, "total": None, "pct": None}

    def vers_dict(self, depuis_ligne=0):
        return {
            "id": self.id,
            "type": self.type,
            "code_station": self.code_station,
            "statut": self.statut,
            "log": self.log[depuis_ligne:],
            "nb_lignes_log": len(self.log),
            "resultat": self.resultat,
            "erreur": self.erreur,
            "debut": self.debut,
            "fin": self.fin,
            "progression": self.progression,
        }


class GestionnaireJobs:
    def __init__(self):
        self._verrou = threading.Lock()
        self._jobs = {}
        self._compteur = 0
        self._job_actif = None

    def job(self, job_id):
        return self._jobs.get(job_id)

    def liste(self):
        return [j.vers_dict() for j in self._jobs.values()]

    def job_en_cours(self):
        with self._verrou:
            if self._job_actif and self._job_actif.statut in ("en_attente", "en_cours"):
                return self._job_actif
        return None

    def lancer(self, type_job, code_station, dossier_travail, fonction):
        """fonction(job) -> dict résultat. Exécutée dans `dossier_travail`
        (les fichiers relatifs écrits par river.py y atterrissent)."""
        with self._verrou:
            if self._job_actif and self._job_actif.statut in ("en_attente", "en_cours"):
                raise JobOccupe(
                    f"Un job est déjà en cours ({self._job_actif.type} sur "
                    f"{self._job_actif.code_station}, id {self._job_actif.id})."
                )
            self._compteur += 1
            job = Job(self._compteur, type_job, code_station)
            self._jobs[job.id] = job
            self._job_actif = job

        thread = threading.Thread(target=self._executer, args=(job, dossier_travail, fonction), daemon=True)
        thread.start()
        return job

    def demander_arret(self, job_id):
        """Demande l'arrêt coopératif d'un job : il s'interrompra au prochain
        point de contrôle (entre deux lots de téléchargement, deux générations
        du GA, deux modèles, fin d'epoch Keras). Retourne le job ou None."""
        job = self._jobs.get(job_id)
        if job and job.statut in ("en_attente", "en_cours"):
            job.demande_arret.set()
        return job

    def _executer(self, job, dossier_travail, fonction):
        job.statut = "en_cours"
        cwd_initial = os.getcwd()
        try:
            os.makedirs(dossier_travail, exist_ok=True)
            os.chdir(dossier_travail)
            job.resultat = fonction(job) or {}
            job.statut = "termine"
        except ArretDemande:
            job.ajouter_log("⏹️ Calcul arrêté à ta demande.")
            job.statut = "arrete"
        except Exception as e:
            job.erreur = str(e)
            job.ajouter_log("❌ " + str(e))
            job.ajouter_log(traceback.format_exc())
            job.statut = "erreur"
        finally:
            os.chdir(cwd_initial)
            job.fin = datetime.now().isoformat(timespec="seconds")


def brancher_log_sur_job(riviere, job):
    """Duplique les messages _log de la River (et de son GestionnaireDonnees)
    dans le log du job, sans toucher à river.py."""
    prefixes = {"info": "ℹ️ ", "warning": "⚠️ ", "erreur": "❌ ", "succes": "✅ "}

    def fabriquer_hook(log_original):
        def hook(message, niveau="info"):
            job.ajouter_log(f"{prefixes.get(niveau, '')}{message}")
            log_original(message, niveau)
        return hook

    riviere._log = fabriquer_hook(type(riviere)._log.__get__(riviere))
    riviere._data._log = fabriquer_hook(type(riviere._data)._log.__get__(riviere._data))

    # Branche l'arrêt coopératif : river.py vérifiera ce drapeau à ses points
    # de contrôle et lèvera ArretDemande.
    riviere._doit_arreter = job.demande_arret.is_set
    riviere._data._doit_arreter = job.demande_arret.is_set

    # Branche la barre de progression : river.py pousse (phase, courant, total).
    riviere._hook_progres = job.definir_progres
    riviere._data._hook_progres = job.definir_progres

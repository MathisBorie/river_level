# Les 51 scénarios ECMWF à 9 km (ecmwf_ifs_europe_ensemble, API Single Runs), runs de 0 h UTC (15 jours) :
# précipitations + température, La Clapière, depuis le 2026-06-24 (plus ancien run disponible le 2026-10-15).
# Ordre, repris à chaque relance (donc chaque jour) :
#   1. les 2 plus anciens jours encore incomplets, sur les 41 carreaux (au cas où les vieux runs disparaissent) ;
#   2. les 7 carreaux du modèle (F.choisir_carreaux(), notebook 05), tous les runs ;
#   3. les 41 carreaux, tous les runs, les plus anciens d'abord.
# Puis chaque nouveau run du jour. Relance toute seule après QuotaEpuise (1 h) ou une erreur (10 min).
import sys, time, datetime as dt
sys.path.insert(0, "/Users/mathis/ponts/projet/river_project")
import pandas as pd
from src.bassins import charger_bassins, ForetBassins, carreaux_amont
from src.meteo import PrevisionsArchivees, QuotaEpuise

DEBUT, V, JOURS = "2026-06-24", ["precipitation", "temperature_2m"], 15
PREMIERS = [835931, 841112, 838522, 830767, 846307, 841113, 833346]
carr = carreaux_amont(2121189970, ForetBassins.depuis_table(charger_bassins()))
pa = PrevisionsArchivees("ecmwf_ifs_europe_ensemble", membres=51)

def maintenant():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M")

while True:
    fin = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    try:
        manque = pa.manquants(carr, DEBUT, fin)
        anciens = sorted(manque)[:2]
        etapes = [(f"plus anciens ({', '.join(r[:10] for r in anciens)})", carr, anciens[0][:10] if anciens else fin,
                   anciens[-1][:10] if anciens else fin),
                  ("7 carreaux du modèle", carr[carr.carreau.isin(PREMIERS)], DEBUT, fin),
                  ("41 carreaux", carr, DEBUT, fin)]
        for nom, lot, debut, fin_lot in etapes:
            reste = pa.cout_prevu(lot, debut, fin_lot, nb_variables=len(V), jours=JOURS)
            print(maintenant(), nom, "reste :", reste, flush=True)
            pa.telecharger(lot, V, debut, fin_lot, jours=JOURS, barre=False)
        print(maintenant(), "à jour jusqu'au run du", fin, "-> prochain essai dans 6 h", flush=True)
        time.sleep(6 * 3600)
    except QuotaEpuise as e:
        print(maintenant(), "quota :", e, "-> nouvel essai dans 1 h", flush=True)
        time.sleep(3600)
    except Exception as e:
        print(maintenant(), "erreur :", repr(e)[:300], "-> nouvel essai dans 10 min", flush=True)
        time.sleep(600)

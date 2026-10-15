# Téléchargement météo de La Clapière (41 carreaux, 10 variables, 1980-01-01 → 2026-10-03) ; relance toute seule après QuotaEpuise.
import sys, time, datetime as dt
sys.path.insert(0, "/Users/mathis/ponts/projet/river_project")
from src.bassins import charger_bassins, ForetBassins, carreaux_amont
from src.meteo import Meteo, QuotaEpuise
H = ["precipitation", "temperature_2m", "snow_depth", "shortwave_radiation", "rain", "soil_moisture_7_to_28cm",
     "snowfall", "et0_fao_evapotranspiration", "soil_moisture_28_to_100cm"]
J = ["temperature_2m_max"]
carr = carreaux_amont(2121189970, ForetBassins.depuis_table(charger_bassins()))
m = Meteo()
while True:
    reste = m.cout_prevu(carr, H, J, "1980-01-01", "2026-10-03")
    print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "reste :", reste, flush=True)
    if reste["requetes"] == 0:
        print("TERMINÉ", flush=True); break
    try:
        m.telecharger(carr, H, J, "1980-01-01", "2026-10-03")
    except QuotaEpuise as e:
        print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "quota :", e, "-> nouvel essai dans 1 h", flush=True)
        time.sleep(3600)
    except Exception as e:
        print(dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "erreur :", repr(e)[:300], "-> nouvel essai dans 10 min", flush=True)
        time.sleep(600)

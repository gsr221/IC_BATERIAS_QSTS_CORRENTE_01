import time as t

#==diretorios dos arquivos==#
# pasta = "13Bus"
# arquivo = "IEEE13Nodeckt.dss"

#==Dados da bateria==#
pMax = 1000  # kW
Ebat = 1000  # kWh
iMax = 850  # A
SOCmin = 0.10
SOCmax = 0.90
eficiencia = 0.95

#==Dados do sistema==#
# barra = 671
NG = 150
NP = 600
NREP = 1
dT = 1  # Intervalo de tempo em horas

cc = [
    0.40,  # 00h
    0.35,  # 01h
    0.30,  # 02h
    0.28,  # 03h
    0.30,  # 04h
    0.40,  # 05h
    0.55,  # 06h
    0.70,  # 07h
    0.75,  # 08h
    0.65,  # 09h
    0.60,  # 10h
    0.68,  # 11h
    0.80,  # 12h
    0.75,  # 13h
    0.70,  # 14h
    0.68,  # 15h
    0.72,  # 16h
    0.78,  # 17h
    0.90,  # 18h
    1.00,  # 19h
    0.95,  # 20h
    0.85,  # 21h
    0.70,  # 22h
    0.55   # 23h
]

sistemas = {
    'IEEE 4 bus':{'pasta': 'sistemas/4Bus-DY-Bal', 'arquivo': '4Bus-DY-Bal.DSS', 'baseKVmediaTensao': 4.16},
    'IEEE 13 bus':{'pasta': 'sistemas/13Bus', 'arquivo': 'IEEE13Nodeckt.dss', 'baseKVmediaTensao': 4.16},
    'IEEE 34 bus':{'pasta': 'sistemas/34Bus', 'arquivo': 'RUN_IEEE34Mod1.DSS', 'baseKVmediaTensao': 24.9},
    'IEEE 123 bus':{'pasta': 'sistemas/123Bus', 'arquivo': 'RUN_IEEE123Bus.DSS', 'baseKVmediaTensao': 4.16}
}


def converte_tempo(t0):
    # Fim do tempo
    end_time = t.time()

    # Tempo total em segundos
    elapsed_time = end_time - t0
    # Converte para h:m:s
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"

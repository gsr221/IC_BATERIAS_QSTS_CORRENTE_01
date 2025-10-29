import time as t

linkFile = "'D:/IC/Segundo_ano/4Bus-DY-Bal/4Bus-DY-Bal.DSS'"
seqVoltageDir = "D:/IC/Segundo_ano/4Bus-DY-Bal/4busDYBal_EXP_SEQVOLTAGES.CSV"

# barra = 671
SOCmin = 0.20
SOCmax = 0.80

eficiencia = 0.95

dT = 1  # Intervalo de tempo em horas

baseKVmediaTensao = 4.16  # kV media tensão base do sistema

cc1 = [
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
    0.60
]

def converte_tempo(t0):
    # Fim do tempo
    end_time = t.time()

    # Tempo total em segundos
    elapsed_time = end_time - t0
    # Converte para h:m:s
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
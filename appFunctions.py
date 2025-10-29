from consts import *
import numpy as np
import time as t
import tkinter as tk
from funAG import FunAG as AG
import pandas as pd
from funODSS import DSS


#Limpa os dados da TreeView:
def clearData(tv):
    tv.delete(*tv.get_children())


#Função que plota a curva de carga:
def FunBotaoPlotar(ax, canva):
    ax.clear()
    ax.plot(cc, color='red', label='Curva de carga', linewidth=4)
    ax.set_ylabel('Porcentagem de carga')
    # ax.set_xticks(range(24))
    ax.set_yticks(np.arange(0,1.25,0.25))
    ax.grid(True)
    canva.draw()
    
    
#Função que roda o algoritmo genético: 
def FunBotaoRoda(tv, pma, pmc, pmb, ax, canva):
    #Contador de tempo:
    t1 = t.time()
    #Lista com os valores máximos de potência:
    pms = [pma.get(), pmb.get(), pmc.get()]
    
    #Verifica se algum dos valores de potência está vazio:
    if '' in pms:
        pms = [1000, 1000, 1000]
    
    #Limpa os dados da TreeView:
    clearData(tv)
    
    #Passa os valores de potência para inteiros:
    pms = [int(pm) for pm in pms]
    
    #Cria o dicionario com as potencias em cada fase, barramento e valor da fob:
    dicResultadoAg = {'Hora':[], 'I A':[], 'I B':[], 'I C':[], 'Barra':[], 'FOB':[], 'Deseq Depois':[], 'Deseq Antes':[]}
    ag = AG()
    dss = DSS()

    #Pega os valores de desequilíbrio a cada tempo antes da alocação das baterias:
    deseqMax = []
    for valCC in cc:
        #Roda o DSS
        dss.clearAll()
        dss.compileFile(linkFile)
        dss.solve(valCC)
        
        #Pega o DataFrame com as tensões de sequência:
        df = dss.dfSeqVolt()
        dicSecVoltages = df.to_dict(orient = 'list')
        
        #Pega os valores de desequilíbrio:
        
        deseq = dicSecVoltages[' %V2/V1']
        #Adiciona o valor máximo de desequilíbrio na lista:
        deseqMax.append(max(deseq))
    
    
    #Chama o método de execução do algoritmo genético:
    results, log, dicMelhoresIndiv, bestFobs, listaBarras = ag.execAg(pms=pms, numGen=200)
    
    #print(dicMelhoresIndiv)
    print("melhores fobs:", bestFobs)
    
    #Adiciona os valores no dicionário:
    listaCrom = dicMelhoresIndiv['cromossomos']
    listaFobs = dicMelhoresIndiv['fobs']
    
    n = len(cc)
        
    melhorBarra = listaBarras[listaCrom[0][3*n]]
    
    listaIA = listaCrom[0][:n]
    listaIB = listaCrom[0][n:2*n]
    listaIC = listaCrom[0][2*n:3*n]

    i = [listaIA, listaIB, listaIC]
    i = np.array(i)
    #==Calcula a potência de cada fase==#
    pots = i * (baseKVmediaTensao/1.732050807)

    #==Calcula a energia máxima de cada fase==#
    #Pmax * dT, onde Pmax é a potência máxima de cada fase e dT é o intervalo de tempo em horas
    # deltaE = [[],[],[]]
    # E_bat = [pms[0] * dT, pms[1] * dT, pms[2] * dT]
    
    #==Calcula a variação de SOC para cada fase==#
    # for fase in range(3):    
    #     for i in range(n):
    #         if i == 0:
    #             deltaE[fase].append((soc[fase][i]) * E_bat[fase])
    #         else:
    #             deltaE[fase].append((soc[fase][i] - soc[fase][i-1]) * E_bat[fase])
            
    #==Calcula a potência de cada fase==#
    # pots = [[],[],[]]
    # for fase in range(3):
    #     for i in range(n):
    #         if deltaE[fase][i] > 0:
    #             pots[fase].append(deltaE[fase][i] / (1 * eficiencia))
    #         else:
    #             pots[fase].append(deltaE[fase][i] / (1 * 1/eficiencia))
        
    deseqs_max = []
    
    #========ALOCA A BATERIA========#        
    for i in range(n):
        potsBat = [pots[0][i], pots[1][i], pots[2][i]]
        # print(f"Potências: {potsBat}")
        # print(f"Barramento: {barra}")
        # print(f"cc: {cc[i]}")
        
        #==Aloca as potências no barramento e os bancos de capacitores e resolve o sistema==#
        dss.alocaPot(barramento=melhorBarra, listaPoten=potsBat)
        dss.solve(cc[i])
    
        #==Recebe as tensões de sequência e as coloca em um dicionário==#
        dfSeqVoltages = dss.dfSeqVolt()
        dicSecVoltages = dfSeqVoltages.to_dict(orient = 'list')
        deseq = dicSecVoltages[' %V2/V1']
        
        deseqs_max.append(max(deseq))
    
    #Coluna de horas:
    dicResultadoAg['Hora'] = [i for i in range(n)]
    #Colunas de potências:
    dicResultadoAg['I A'] = listaIA
    dicResultadoAg['I B'] = listaIB
    dicResultadoAg['I C'] = listaIC
    #Coluna de barramento:
    dicResultadoAg['Barra'].append(melhorBarra)
    #Coluna de FOB e Desequilíbrio máximo:
    dicResultadoAg['FOB'].append(listaFobs[0])
    dicResultadoAg['Deseq Antes'] = deseqMax
    dicResultadoAg['Deseq Depois'] = deseqs_max

    
    #Printa o dicionário com os resultados:  
    # print(dicResultadoAg)
    
    #Ajusta o dicionário para que as colunas de 'Barra', 'FOB' e 'Deseq Antes' tenham o mesmo tamanho:
    num_linhas = len(dicResultadoAg['Hora'])
    dicResultadoAg['Barra'] = dicResultadoAg['Barra'] * num_linhas
    dicResultadoAg['FOB'] = dicResultadoAg['FOB'] * num_linhas
    
    
    #Cria um DataFrame com os resultados:
    dfResultadoAg = pd.DataFrame(dicResultadoAg)
    
    #Cria uma TreeView com os melhores indivíduos:
    tv["column"] = list(dfResultadoAg)
    tv["show"] = "headings"
    for column in tv["columns"]:
        tv.heading(column, text=column) 
    df_rows = dfResultadoAg.to_numpy().tolist()
    for row in df_rows:
        tv.insert("", "end", values=row)

    print(converte_tempo(t1))
    
    deseqMed = sum(dicResultadoAg['Deseq Depois']) / len(dicResultadoAg['Deseq Depois'])
    
    ax.clear()
    ax.plot(dicResultadoAg['Deseq Depois'], color='green', label='Desequilíbrios Max', linewidth=3)
    ax.plot(dicResultadoAg['Deseq Antes'], color='red', label='Desequilíbrios Antes', linewidth=3)
    ax.axhline(y=deseqMed, color='green', linestyle='--', label='Desequilíbrio Médio')
    ax.set_ylabel('Porcentagem de carga')
    # ax.set_xticks(range(24))
    ax.set_yticks(np.arange(0,3.75,0.25))
    ax.grid(True)
    canva.draw()
    
    return None
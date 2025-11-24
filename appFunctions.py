from consts import *
import numpy as np
import time as t
import pandas as pd
import tkinter as tk
from funAG import FunAG as AG
from funODSS import DSS
import threading as th
import matplotlib.pyplot as plt
import csv

class AppFunctions:
    def __init__(self):
        self.bestFobs = []
        self.dicMelhoresIndiv = {}
    
    #Limpa os dados da TreeView:
    def clearData(self, tv):
        tv.delete(*tv.get_children())


    #Função que plota a curva de carga:
    def FunBotaoPlotar(self):
        fig, ax = plt.subplots()
        ax.clear()
        ax.set_ylabel('Porcentagem de carga')
        ax.set_xticks(range(24))
        ax.set_yticks(np.arange(0,1.25,0.25))
        ax.grid(True)
        ax.plot(cc, color='red', label='Curva de carga', linewidth=4)
        fig.show()
         
           
    #Função que roda o algoritmo genético: 
    def FunBotaoRoda(self, tv, ax, canva, sist):
        self.sistema = sist.get()
        
        #Inicializa os objetos necessários:
        ag = AG(self.sistema)
        dss = DSS()
        
        #Contador de tempo:
        t1 = t.time()
        
        #Verifica se algum dos valores de potência está vazio:
        pms = [pMax for _ in range(3)] #kW
        
        #Limpa os dados da TreeView:
        self.clearData(tv)
        
        #Cria o dicionario com as potencias em cada fase, barramento e valor da fob:
        dicResultadoAg = {'Hora':[], 'I A':[], 'I B':[], 'I C':[], 'Barra':[], 'FOB':[], 'Deseq Depois':[], 'Deseq Antes':[]}

        #Pega os valores de desequilíbrio a cada tempo antes da alocação das baterias:
        deseqMax = []
        for valCC in cc:
            #Roda o DSS
            dss.clearAll()
            dss.compileFile(sistemas[self.sistema]['pasta'], sistemas[self.sistema]['arquivo'])
            dss.solve(valCC)
            deseq = dss.deseq()
            deseqMax.append(max(deseq))
           
        #Chama o método de execução do algoritmo genético:
        _, _, self.dicMelhoresIndiv, self.bestFobs, listaBarras = ag.execAg(pms=pms, numGen=NG, numPop=NP, numRep=NREP)
        self.dicMelhoresIndiv['cromossomos'] = [list(map(float, crom)) for crom in self.dicMelhoresIndiv['cromossomos']]
        # print('\n',self.dicMelhoresIndiv)
        # print("melhores fobs:", self.bestFobs)
        
        #Adiciona os valores no dicionário:
        listaCrom = self.dicMelhoresIndiv['cromossomos']
        listaFobs = self.dicMelhoresIndiv['fobs']
        
        n = len(cc)
            
        melhorBarra = listaBarras[int(listaCrom[0][-1])]
        
        listaIA = listaCrom[0][:n]
        listaIB = listaCrom[0][n:2*n]
        listaIC = listaCrom[0][2*n:3*n]

        i = np.array([listaIA, listaIB, listaIC])
        #==Calcula a potência de cada fase==#
        pots = i * (baseKVmediaTensao/1.732050807)
            
        deseqs_max = []
        
        #========ALOCA A BATERIA========#        
        for i in range(n):
            potsBat = [pots[0][i], pots[1][i], pots[2][i]]
            
            #==Aloca as potências no barramento e os bancos de capacitores e resolve o sistema==#
            dss.alocaPot(barra=melhorBarra, listaPot=potsBat)
            dss.solve(cc[i])
        
            #==Recebe as tensões de sequência e as coloca em um dicionário==#
            deseq = dss.deseq()
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


    #Função que executa o AG em uma thread separada:
    def run_ag_in_thread(self, tv, ax, canva, sistema):
        #Executa o AG em uma thread separada para não travar a GUI.
        thread = th.Thread(target=self.FunBotaoRoda, args=(tv, ax, canva, sistema))
        thread.daemon = True
        thread.start()


    def FunBotaoPlotarFOB(self):
        if self.bestFobs:
            fig, ax = plt.subplots()
            ax.clear()
            ax.set_ylabel('FOB')
            ax.set_xticks(range(len(self.bestFobs[0])))
            ax.grid(True)
            ax.plot(np.arange(1,len(self.bestFobs[0])+1,1), self.bestFobs[0], color='blue', label='FOB', linewidth=4)
            fig.show()
        else:
            print("Nenhum FOB encontrado. Execute o algoritmo genético primeiro.")
            
    def FunBotaoCSVFOBS(self):
        if not self.dicMelhoresIndiv:
            print("Nenhum indivíduo encontrado. Execute o algoritmo genético primeiro.")
            return
        
        import os
        df = pd.DataFrame(self.dicMelhoresIndiv)
        # escolhe uma etiqueta segura para o sistema
        sistema_key = getattr(self, 'sistema', None)
        if sistema_key is None:
            try:
                sistema_key = next(iter(sistemas))
            except Exception:
                sistema_key = 'sistema'
        safe_name = str(sistema_key).replace(os.sep, '_').replace(' ', '_')

        timestamp = t.strftime('%Y%m%d_%H%M%S')
        folder = 'csvs'
        os.makedirs(folder, exist_ok=True)
        nome_arquivo = os.path.join(folder, f"melhores_results_{safe_name}_{timestamp}.csv")
        try:
            df.to_csv(nome_arquivo, index=False)
            print(f"Salvando melhores indivíduos em {nome_arquivo}...")
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")

import win32com.client
import pandas as pd
from consts import *


class DSS():
    def __init__(self):
        #==Objetos do openDSS==#
        self.dssObj = win32com.client.Dispatch("OpenDSSEngine.DSS")
        
        #==Incializa o openDSS no código e seus objetos==#
        if self.dssObj.Start(0) == False:
            print("Inicialização do DSS falhou")
        else:
            self.dssTxt = self.dssObj.Text
            self.dssCircuit = self.dssObj.ActiveCircuit
            self.dssSolution = self.dssCircuit.Solution
            self.dssBus = self.dssCircuit.ActiveBus
      
            
    #==Limpa a memoria do openDSS==#       
    def clearAll(self):
        self.dssTxt.Command = "ClearAll"
        
        
    #==Compila o arquivo desejado==#
    def compileFile(self, dssFileName):
        self.dssTxt.Command = "Compile " + dssFileName
        
        
    #==Soluciona o circuito do arquivo especificado com o loadMult desejado==#
    def solve(self, loadMult):
        self.dssSolution.LoadMult = loadMult
        self.dssSolution.Solve()
        
        
    #==Retorna o nome de todos os barramentos trifásicos==#
    def BusNames(self):
        bussesNames = self.dssCircuit.AllBusNames
        tPBusses = []

        #==Verifica o número de fases do barramento, se for >= 3, adiciona na lista==#
        for busses in bussesNames:
            self.dssCircuit.SetActiveBus(busses)
            if self.dssBus.NumNodes >= 3:
                tPBusses.append(busses)

        #==Retorna a lista com os nomes do barramentos==#
        return tPBusses


    #==Exporta as tensões de sequência para um arquivo CSV==#
    def exportSeqVoltages(self):
        self.dssTxt.Command = "Export seqVoltages"
        
        
    #==Retorna um DataFrame com as tensões de sequência==#
    def dfSeqVolt(self):
        #==Exporta as tensões de sequência==#
        self.exportSeqVoltages()
        
        #==Tenta ler o arquivo CSV com as tensões de sequência==#
        try:
            dfSeqVoltages = pd.read_csv(seqVoltageDir)
            
        except FileNotFoundError:
            return pd.DataFrame()
        
        return dfSeqVoltages
    
    #==Aloca as potências no barramento==#
    def alocaPot(self, barramento, listaPoten):
        #==Limpa a memória do openDSS e compila o arquivo original novamente==#
        self.clearAll()
        self.compileFile(linkFile)
        
        #==Ativa o barramento desejado==#
        # print(f"Alocando potências no barramento: {barramento}")
        self.dssCircuit.SetActiveBus(barramento)
        # print(f"Barramento ativo: {self.dssBus.Name}")
        #==Recebe a tensão base do barramento==#
        kVBaseBarra = self.dssBus.kVBase
        #==Aloca as potências no barramento==#
        for fase in range(3):
            comando = "New Load.NEW"+str(fase+1)+" Bus1="+str(barramento)+"."+str(fase+1)+" Phases=1 Conn=Wye Model=1 kV="+str(round(kVBaseBarra, 2))+" kW="+str(listaPoten[fase])+" kvar=0"
            self.dssTxt.Command = comando
    
import py_dss_interface as pdss
from py_dss_toolkit import dss_tools
from consts import *
import numpy as np
import os
import pathlib

class DSS():
    def __init__(self):
        # Cria objetos do openDSS
        self.dss = pdss.DSS()
        self.dssTools = dss_tools
        self.dssTools.update_dss(self.dss)
        self._pasta = None
        self._arquivo = None
          
    def compileFile(self, paste, file):
        self._pasta = paste
        self._arquivo = file
        script_path = os.path.dirname(os.path.abspath(__file__))
        dss_file = pathlib.Path(script_path).joinpath(paste, file)
        self.dss.text(f"compile [{dss_file}]")
        
    def get_pasta(self):
        return self._pasta

    def get_arquivo(self):
        return self._arquivo

    def clearAll(self):
        self.dss.text("ClearAll")
        
    def solve(self, loadMult):
        self.dss.solution.load_mult = loadMult
        self.dss.solution.solve()
        
    def BusNames(self):
        busesNames = self.dss.circuit.buses_names
        tPBuses = []
        
        for bus in busesNames:
            self.dss.circuit.set_active_bus(bus)
            if self.dss.bus.num_nodes >= 3:
                tPBuses.append(bus)
        
        return tPBuses, busesNames
    
    def retornaLoadsDF(self):
        return self.dssTools.model.loads_df
    
    def iniciaBESS(self):
        barra = self.BusNames()[0][0]  # Seleciona a primeira barra de 3 fases para alocar a bateria
        self.dss.circuit.set_active_bus(barra)
        # Indentifica a tensão nominal do barramento
        kVBaseBarra = self.dss.bus.kv_base 
        # Aloca a bateria no barramento
        for fase in range(3):
            self.dssTools.model.add_element("load", f"newload{fase+1}", dict(phases=1, bus1=f"{barra}.{fase+1}", kv={round(kVBaseBarra, 2)}, kw=0, pf=1))
        
    def alocaPot(self, barra, listaPot):
        # Ativa o barramento em que será aloacada a potência
        self.dss.circuit.set_active_bus(barra)
        # Indentifica a tensão nominal do barramento
        kVBaseBarra = self.dss.bus.kv_base
        # Modifica a potência da bateria em cada fase
        for fase in range(3):
            self.dssTools.model.edit_element("load", f"newload{fase+1}", {"bus1":f"{barra}.{fase+1}", "kV":f"{round(kVBaseBarra, 2)}", "kW":f"{listaPot[fase]}"})
         
    def deseq(self):
        busesNames = self.BusNames()[0]
        v1 = []
        v2 = []
        for bus in busesNames:
            self.dss.circuit.set_active_bus(bus)
            v1.append(self.dss.bus.seq_voltages[1])
            v2.append(self.dss.bus.seq_voltages[2])
            
        v1 = np.array(v1)
        v2 = np.array(v2)

        deseq = (v2 / v1) * 100

        return deseq
    
    def retornaTensoes(self):  
        return self.dss.circuit.buses_vmag, self.dss.circuit.buses_vmag_pu
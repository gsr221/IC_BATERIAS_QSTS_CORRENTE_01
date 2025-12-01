from consts import *
import numpy as np
import time as t
import pandas as pd
import tkinter as tk
from funAG import FunAG as AG
from funODSS import DSS
import threading as th
import matplotlib.pyplot as plt
import os
from typing import Optional, Dict, List, Any


class AppFunctions:
    """Classe responsável pelas funções da aplicação relacionadas ao algoritmo genético."""
    def __init__(self):
        self.bestFobs: List[List[float]] = []
        self.dicMelhoresIndiv: Dict[str, Any] = {}
        self.sistema: Optional[str] = None
        self._cache_deseq: Dict[str, List[float]] = {}  # Cache para desequilíbrios
    
    #Limpa os dados da TreeView:
    def clearData(self, tv):
        tv.delete(*tv.get_children())


    def FunBotaoPlotar(self) -> None:
        """Plota a curva de carga do sistema."""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.clear()
            ax.set_ylabel('Porcentagem de carga')
            ax.set_xlabel('Hora do dia')
            ax.set_title('Curva de Carga Diária')
            ax.set_xticks(range(len(cc)))
            ax.set_yticks(np.arange(0, 1.25, 0.25))
            ax.grid(True, alpha=0.7)
            ax.plot(cc, color='red', label='Curva de carga', linewidth=3, marker='o', markersize=4)
            ax.legend()
            plt.tight_layout()
            fig.show()
        except Exception as e:
            print(f"Erro ao plotar curva de carga: {e}")
         
           
    #Função que roda o algoritmo genético: 
    def FunBotaoRoda(self, tv, ax, canva, sist: str) -> None:
        """Executa o algoritmo genético para o sistema especificado."""
        try:
            self.sistema = sist
            print(f"Iniciando execução para o sistema: {self.sistema}")
            
            # Validação do sistema
            if self.sistema not in sistemas:
                raise ValueError(f"Sistema '{self.sistema}' não encontrado")
            
            # Inicializa os objetos necessários
            ag = AG(self.sistema)
            dss = DSS()
            
            # Contador de tempo
            t1 = t.time()
            
            # Parâmetros da bateria
            pms = [pMax] * 3  # kW
            
            # Limpa os dados da TreeView
            self.clearData(tv)
            
            # Estrutura de dados otimizada para resultados
            dicResultadoAg = {
                'Hora': [],
                'I A': [], 'I B': [], 'I C': [],
                'Barra': [], 'FOB': [],
                'Deseq Depois': [], 'Deseq Antes': []
            }

            # Cache para evitar recálculos
            cache_key = f"{self.sistema}_{hash(tuple(cc))}"
            if cache_key in self._cache_deseq:
                deseqMax = self._cache_deseq[cache_key]
                print("Usando desequilíbrios em cache")
            else:
                print("Calculando desequilíbrios iniciais...")
                deseqMax = self._calcular_desequilibrios_iniciais(dss)
                self._cache_deseq[cache_key] = deseqMax
           
            # Executa o algoritmo genético com multiprocessamento
            print("Executando algoritmo genético...")
            _, _, self.dicMelhoresIndiv, self.bestFobs, listaBarras = ag.execAg(
                pms=pms, num_gen=NG, num_pop=NP, num_rep=NREP
            )
            # Processa resultados do AG
            if not self.dicMelhoresIndiv or not self.dicMelhoresIndiv.get('cromossomos'):
                raise ValueError("Algoritmo genético não retornou resultados válidos")
            
            self.dicMelhoresIndiv['cromossomos'] = [
                list(map(float, crom)) for crom in self.dicMelhoresIndiv['cromossomos']
            ]
            
            listaCrom = self.dicMelhoresIndiv['cromossomos']
            listaFobs = self.dicMelhoresIndiv['fobs']
            n = len(cc)
            
            # Valida se há cromossomos válidos
            if not listaCrom or len(listaCrom[0]) < 3*n + 1:
                raise ValueError("Cromossomo inválido retornado pelo AG")
            
            melhorBarra = listaBarras[int(listaCrom[0][-1])]
            
            # Extrai correntes por fase de forma mais clara
            cromossomo = listaCrom[0]
            listaIA = cromossomo[:n]
            listaIB = cromossomo[n:2*n]
            listaIC = cromossomo[2*n:3*n]

            # Calcula potências com melhor legibilidade
            correntes = np.array([listaIA, listaIB, listaIC])
            tensao_fase = sistemas[self.sistema]['baseKVmediaTensao'] / np.sqrt(3)
            pots = correntes * tensao_fase
            
            # Calcula desequilíbrios após alocação da bateria
            print("Calculando desequilíbrios finais...")
            deseqs_max = self._calcular_desequilibrios_finais(
                dss, melhorBarra, pots, n
            )
            
            # Popula dicionário de resultados de forma mais eficiente
            dicResultadoAg.update({
                'Hora': list(range(n)),
                'I A': listaIA,
                'I B': listaIB, 
                'I C': listaIC,
                'Barra': [melhorBarra] * n,
                'FOB': [listaFobs[0]] * n,
                'Deseq Antes': deseqMax,
                'Deseq Depois': deseqs_max
            })
            
            # Cria DataFrame com tratamento de erro
            try:
                dfResultadoAg = pd.DataFrame(dicResultadoAg)
                self._atualizar_treeview(tv, dfResultadoAg)
            except Exception as e:
                print(f"Erro ao criar DataFrame: {e}")
                return

            # Log do tempo de execução
            tempo_execucao = converte_tempo(t1)
            print(f"Execução concluída: {tempo_execucao}")
            
            # Atualiza gráficos
            self._atualizar_graficos(ax, canva, dicResultadoAg)
            
            print(f"Sistema {self.sistema} processado com sucesso!")
            
        except Exception as e:
            print(f"Erro durante execução do AG para {sist}: {e}")
            import traceback
            traceback.print_exc()


    def _calcular_desequilibrios_iniciais(self, dss: DSS) -> List[float]:
        """Calcula desequilíbrios iniciais do sistema."""
        deseqMax = []
        for valCC in cc:
            dss.clearAll()
            dss.compileFile(sistemas[self.sistema]['pasta'], sistemas[self.sistema]['arquivo'])
            dss.solve(valCC)
            deseq = dss.deseq()
            deseqMax.append(max(deseq))
        return deseqMax
    
    def _calcular_desequilibrios_finais(self, dss: DSS, barra: str, pots: np.ndarray, n: int) -> List[float]:
        """Calcula desequilíbrios após alocação da bateria."""
        deseqs_max = []
        for i in range(n):
            potsBat = [pots[0][i], pots[1][i], pots[2][i]]
            dss.alocaPot(barra=barra, listaPot=potsBat)
            dss.solve(cc[i])
            deseq = dss.deseq()
            deseqs_max.append(max(deseq))
        return deseqs_max
    
    def _atualizar_treeview(self, tv, df: pd.DataFrame) -> None:
        """Atualiza a TreeView com os dados do DataFrame."""
        tv["column"] = list(df.columns)
        tv["show"] = "headings"
        
        for column in tv["columns"]:
            tv.heading(column, text=column)
            # Ajusta largura da coluna baseado no conteúdo
            tv.column(column, width=80, anchor='center')
        
        df_rows = df.round(4).to_numpy().tolist()  # Arredonda valores
        for row in df_rows:
            tv.insert("", "end", values=row)
    
    def _atualizar_graficos(self, ax, canva, dados: Dict) -> None:
        """Atualiza os gráficos com os resultados."""
        try:
            deseqMed = np.mean(dados['Deseq Depois'])
            
            ax.clear()
            ax.plot(dados['Deseq Depois'], color='green', label='Desequilíbrios Máx', 
                   linewidth=2, marker='o', markersize=3)
            ax.plot(dados['Deseq Antes'], color='red', label='Desequilíbrios Antes', 
                   linewidth=2, marker='s', markersize=3)
            ax.axhline(y=deseqMed, color='green', linestyle='--', 
                      label=f'Desequilíbrio Médio ({deseqMed:.3f})', alpha=0.7)
            
            ax.set_ylabel('Desequilíbrio (%)')
            ax.set_xlabel('Hora')
            ax.set_title(f'Desequilíbrios - Sistema {self.sistema}')
            ax.set_yticks(np.arange(0, max(max(dados['Deseq Depois']), max(dados['Deseq Antes'])) + 0.5, 0.25))
            ax.grid(True, alpha=0.7)
            ax.legend()
            canva.draw()
        except Exception as e:
            print(f"Erro ao atualizar gráficos: {e}")


    #Função que executa o AG em uma thread separada:
    def run_ag_in_thread(self, tv, ax, canva, sistema):
        #Executa o AG em uma thread separada para não travar a GUI.
        thread = th.Thread(target=self.FunBotaoRoda, args=(tv, ax, canva, sistema))
        thread.daemon = True
        thread.start()


    def FunBotaoPlotarFOB(self) -> None:
        """Plota o comportamento da FOB ao longo das gerações."""
        if not self.bestFobs or not self.bestFobs[0]:
            print("Nenhum FOB encontrado. Execute o algoritmo genético primeiro.")
            return
            
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.clear()
            
            fobs = self.bestFobs[0]
            geracoes = np.arange(1, len(fobs) + 1)
            
            ax.plot(geracoes, fobs, color='blue', label='Melhor FOB', 
                   linewidth=2, marker='o', markersize=3)
            ax.set_ylabel('FOB (Função Objetivo)')
            ax.set_xlabel('Geração')
            ax.set_title('Convergência do Algoritmo Genético')
            ax.grid(True, alpha=0.7)
            ax.legend()
            
            # Adiciona estatísticas
            melhoria = ((fobs[0] - fobs[-1]) / fobs[0]) * 100 if fobs[0] != 0 else 0
            ax.text(0.02, 0.98, f'Melhoria: {melhoria:.2f}%\nFOB Final: {fobs[-1]:.4f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            fig.show()
        except Exception as e:
            print(f"Erro ao plotar FOB: {e}")
            
    def FunBotaoCSVFOBS(self) -> None:
        """Salva os melhores indivíduos de cada repetição em um arquivo CSV."""
        if not self.dicMelhoresIndiv:
            print("Nenhum indivíduo encontrado. Execute o algoritmo genético primeiro.")
            return
            
        try:
            df = pd.DataFrame(self.dicMelhoresIndiv)
            
            # Nome seguro para o arquivo
            sistema_key = getattr(self, 'sistema', 'sistema_desconhecido')
            safe_name = str(sistema_key).replace(os.sep, '_').replace(' ', '_')
            
            timestamp = t.strftime('%Y%m%d_%H%M%S')
            folder = 'csvs'
            os.makedirs(folder, exist_ok=True)
            
            nome_arquivo = os.path.join(folder, f"melhores_results_{safe_name}_{timestamp}.csv")
            
            # Salva com metadados adicionais
            with open(nome_arquivo, 'w', newline='', encoding='utf-8') as f:
                f.write(f"# Resultados do Algoritmo Genético\n")
                f.write(f"# Sistema: {sistema_key}\n")
                f.write(f"# Data/Hora: {t.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# Parâmetros: NG={NG}, NP={NP}, NREP={NREP}\n")
                f.write("\n")
            
            # Append do DataFrame
            df.to_csv(nome_arquivo, mode='a', index=False)
            print(f"Resultados salvos em: {nome_arquivo}")
            
        except Exception as e:
            print(f"Erro ao salvar CSV: {e}")
            
            
    def FunBotao(self, tv, ax, canva):
        for sistema in list(sistemas.keys())[2:]:
            if sistema == 'IEEE 123 bus':
                for _ in range(2):
                    print("\n")
                    print('Sistema:', sistema)
                    self.FunBotaoRoda(tv, ax, canva, sistema)
                    self.FunBotaoCSVFOBS()
            else:
                print("\n")
                print('Sistema:', sistema)
                self.FunBotaoRoda(tv, ax, canva, sistema)
                self.FunBotaoCSVFOBS()
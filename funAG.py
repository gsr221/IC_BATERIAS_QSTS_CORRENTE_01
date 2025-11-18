from deap import creator, base, tools
import random
from consts import *
from funODSS import DSS
import time as t
import numpy as np


class FunAG:
    def __init__(self):
        self.dss = DSS()
        self.dss.compileFile(pasta, arquivo)
        self.barras, _ = self.dss.BusNames()
        self.pmList = []
        # Protege creator.create para evitar erro se já existir
        if not hasattr(creator, "fitnessMulti"):
            creator.create("fitnessMulti", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "estrIndiv"):
            creator.create("estrIndiv", list, fitness=creator.fitnessMulti)
        self.fobs = []
    
    
    def clone_indiv(self, ind):
        import copy
        return copy.deepcopy(ind)


    ################ Cria um cromossomo (indivíduo) com valores de Corrente e barramento aleatórios
    def criaCromBatCorr(self):
        # Gera valores aleatórios de corrente para cada fase em cada instante de tempo
        currents = np.array([random.uniform(-iMax, iMax) for _ in range(3*len(cc))], dtype=float)
        
        # Sorteia um barramento aleatório para alocação da bateria
        bus_idx = random.randint(0, len(self.barras)-1)
        
        # Concatena os valores de corrente com o barramento em um único indivíduo
        # O último gene será o índice do barramento (como float)
        indiv = np.concatenate([currents, np.array([bus_idx], dtype=float)])
        return indiv

    ################ Método de cruzamento aritmético
    def cruzamentoAritmetico(self, indiv1, indiv2):
        alfa = random.uniform(0, 1)
        indiv1 = alfa * np.array(indiv1) + (1 - alfa) * np.array(indiv2)
        indiv2 = alfa * np.array(indiv2) + (1 - alfa) * np.array(indiv1)
        return indiv1, indiv2

    ################ Método de cruzamento BLX
    def cruzamentoFunBLX(self, indiv1, indiv2):
        # Trata-se de cruzamento in-place (DEAP espera que mate altere os indivíduos)
        # Garante que operamos em listas/ndarrays mutáveis
        # alfa controlando a expansão
        alfa = random.uniform(0.2, 0.5)
        last_gene_index = len(indiv1) - 1

        for gene in range(len(indiv1)):
            if gene != last_gene_index:
                #Calcula delta para genes de corrente
                delta = abs(indiv1[gene] - indiv2[gene])
                #Pega os valores maximo e mínimo para o gene
                minGene = min(indiv1[gene], indiv2[gene]) - alfa * delta
                maxGene = max(indiv1[gene], indiv2[gene]) + alfa * delta
                #Gera novos valores para os genes sorteando dentro do intervalo
                indiv1[gene] = random.uniform(minGene, maxGene)
                indiv2[gene] = random.uniform(minGene, maxGene)
            else:
                # gene do barramento: tratar como índice inteiro
                delta = abs(indiv1[gene] - indiv2[gene])
                minGene = int(np.floor(min(indiv1[gene], indiv2[gene]) - alfa * delta))
                maxGene = int(np.ceil(max(indiv1[gene], indiv2[gene]) + alfa * delta))
                # clamp nos limites válidos dos índices dos barramentos
                min_idx = max(0, minGene)
                max_idx = min(len(self.barras) - 1, maxGene)
                if min_idx > max_idx:
                    # se por algum motivo os limites ficaram invertidos, iguala
                    min_idx = max_idx = max(0, min(len(self.barras) - 1, int(round((indiv1[gene] + indiv2[gene]) / 2))))
                indiv1[gene] = float(random.randint(min_idx, max_idx))
                indiv2[gene] = float(random.randint(min_idx, max_idx))
        # Retorna os indivíduos (modificados in-place)
        return indiv1, indiv2


    ################ Função objetivo para bateria com cromossomo de corrente
    def FOBbatCurrent(self, indiv):
        n = len(cc)
        # Array de correntes por fase e tempo [[fase A], [fase B], [fase C]]
        currents = np.array([indiv[:n], indiv[n:2*n], indiv[2*n:3*n]])
        
        # Índice do barramento (força int)
        bus_idx = int(indiv[-1])
        # Verifica se o barramento é válido
        if bus_idx < 0 or bus_idx >= len(self.barras):
            self.fobs.append(1000)
            return (1000.,)
        
        # Ativa o barramento
        barra = str(self.barras[bus_idx])
        self.dss.dss.circuit.set_active_bus(barra)
        kVBaseBarra = self.dss.dss.bus.kv_base
        # Verifica tensão base do barramento
        if round(kVBaseBarra, 2) != round(baseKVmediaTensao / 1.732050807, 2):
            self.fobs.append(1000)
            return (1000.,)
        
        # Verifica limites de corrente
        maskAcima = []
        # Cria máscaras para correntes acima do limite
        for fase in range(3):
            maskAcima.append(np.abs(currents[fase]) > iMax)
        # Se alguma corrente ultrapassa o limite, retorna penalidade
        if np.any(maskAcima):
            dists = np.array([0.0, 0.0, 0.0])
            for fase in range(3):
                dists[fase] = float(np.max(np.abs(currents[fase][maskAcima[fase]]) - iMax)) if np.any(maskAcima[fase]) else 0.0
            return (300.0 + float(np.max(dists)),)
        
        # Potência de cada fase [[kW fase A], [kW fase B], [kW fase C]] / P = I*V_fase 
        pot = currents * (baseKVmediaTensao / 1.732050807)
        
        # Energia armazenada e SOC
        e = np.zeros((3, n))
        for fase in range(3):
            for t_idx in range(n):
                # Calcula energia armazenada em cada instante
                # Considera SOC inicial de 80%
                if t_idx == 0:
                    # Carregamento
                    if pot[fase][t_idx] > 0:
                        e[fase][t_idx] = Ebat * 0.8 + pot[fase][t_idx] * dT * eficiencia
                    # Descarregamento
                    else:
                        e[fase][t_idx] = Ebat * 0.8 + pot[fase][t_idx] * dT * (1.0 / eficiencia)
                # Para os demais instantes
                else:
                    # Carregamento
                    if pot[fase][t_idx] > 0:
                        e[fase][t_idx] = e[fase][t_idx - 1] + pot[fase][t_idx] * dT * eficiencia
                    # Descarregamento
                    else:
                        e[fase][t_idx] = e[fase][t_idx - 1] + pot[fase][t_idx] * dT * (1.0 / eficiencia)
        
        soc = e * (1.0 / Ebat)
        
        # Verifica limites de SOC
        # Cria máscaras para SOC acima e abaixo dos limites
        maskAcimaSOC = soc > SOCmax
        maskAbaixoSOC = soc < SOCmin
        # Se algum SOC ultrapassa os limites, retorna penalidade
        if np.any(maskAcimaSOC) or np.any(maskAbaixoSOC):
            distAcima = soc[maskAcimaSOC] - SOCmax if np.any(maskAcimaSOC) else 0.0
            distAbaixo = SOCmin - soc[maskAbaixoSOC] if np.any(maskAbaixoSOC) else 0.0
            maiorDist = float(max(np.max(distAcima), np.max(distAbaixo)))
            return (200.0 + maiorDist,)
        
        # Aloca e resolve para cada instante de tempo
        deseqs_max = []
        for t_idx in range(n):
            potsBat = [pot[0][t_idx], pot[1][t_idx], pot[2][t_idx]]
            self.dss.alocaPot(barra=barra, listaPot=potsBat)
            self.dss.solve(cc[t_idx])
            deseq = self.dss.deseq()
            deseqs_max.append(max(deseq))
    
        
        fobVal = max(deseqs_max)
        if fobVal > 2.0:
            self.fobs.append(10 + fobVal)
            return (10.0 + float(fobVal),)
        
        self.fobs.append(float(fobVal))
        return (float(fobVal),)


    ################ Algoritmo Genético
    def execAg(self, pms, probCruz=0.9, probMut=0.1, numGen=700, numRep=1, numPop=300, numTorneio=3, eliteSize=10):
        self.pmList = pms
        
        self.dss.iniciaBESS()
        # Configuração do AG
        toolbox = base.Toolbox()
        toolbox.register("clone", self.clone_indiv)                                # registra função de clone personalizada
        toolbox.register("mate", self.cruzamentoFunBLX)                            # cruzamento BLX
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)  # mutação gaussiana
        toolbox.register("select", tools.selTournament, tournsize=numTorneio)      # seleção por torneio
        toolbox.register("evaluate", self.FOBbatCurrent)                           # função objetivo

        dicMelhoresIndiv = {"cromossomos": [], "fobs": []}
        all_best_fobs = []  # lista de listas: best_fobs por repeticao

        t0 = t.time()
        for rep in range(numRep):
            print(f"{converte_tempo(t0)} - Iniciando execução do AG... Repetição {rep + 1} de {numRep}")
            best_fobs = []  # lista de best_fobs desta repetição
            
            toolbox.register("indiv", tools.initIterate, creator.estrIndiv, self.criaCromBatCorr) # registra criação de indivíduo
            toolbox.register("pop", tools.initRepeat, list, toolbox.indiv)                        # registra criação de população                 
            populacao = toolbox.pop(n=numPop) # cria população inicial

            hof = tools.HallOfFame(1) # guarda melhor indivíduo

            # Avalia população inicial
            invalid_ind = [ind for ind in populacao if not ind.fitness.valid]
            for ind in invalid_ind:
                ind.fitness.values = toolbox.evaluate(ind)

            print("===INICIO DAS GERAÇÕES===")
            for gen in range(numGen):
                if gen % 10 == 0:
                    print(f"{converte_tempo(t0)} - Geração {gen + 1} de {numGen}... / valor FOB: {melhor_fob if gen > 0 else 'N/A'}")
                    # print(f"len pop: {len(populacao)}")

                # elitismo
                elite = tools.selBest(populacao, eliteSize)

                # seleção e clonagem
                offspring = toolbox.select(populacao, len(populacao) - eliteSize) # seleciona o restante da população
                offspring = list(map(toolbox.clone, offspring)) # clona os selecionados

                # cruzamento
                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < probCruz:
                        toolbox.mate(c1, c2)
                        del c1.fitness.values
                        del c2.fitness.values

                # mutação
                for mut in offspring:
                    if random.random() < probMut:
                        toolbox.mutate(mut)
                        del mut.fitness.values

                # avaliação dos novos
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                for ind in invalid_ind:
                    ind.fitness.values = toolbox.evaluate(ind)

                # nova população
                populacao[:] = elite + offspring
                hof.update(populacao)
                melhor_fob = hof[0].fitness.values[0]
                best_fobs.append(melhor_fob)

            # guarda melhor (clonado) e seus fobs
            dicMelhoresIndiv["cromossomos"].append(toolbox.clone(hof[0]))
            dicMelhoresIndiv["fobs"].append(hof[0].fitness.values[0])
            all_best_fobs.append(best_fobs)

        t1 = t.time()
        print(f"Tempo total: {t1 - t0:.2f} s")

        # Retorna: população final da última repetição, placeholder, dicionário, lista de best_fobs por repetição, barras
        return populacao, None, dicMelhoresIndiv, all_best_fobs, self.barras
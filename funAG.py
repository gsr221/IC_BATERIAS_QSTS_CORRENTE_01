from deap import creator, base, tools
import random
from consts import *
from funODSS import DSS
import time as t
import numpy as np


def calculaSOC(SOC_n, corrente):
    pot_n = corrente * (baseKVmediaTensao / 1.732050807)
    energia_n = SOC_n * Ebat
    if pot_n > 0:
        # Carregamento
        energia_n1 = energia_n + (pot_n * dT * eficiencia)
    else:
        # Descarregamento
        energia_n1 = energia_n + (pot_n * dT / eficiencia)
    SOC_n1 = energia_n1 / Ebat
    return SOC_n1


def calculaIupIdown(SOC_atual):
    dSOCup = SOCmax - SOC_atual
    dSOCdown = SOCmin - SOC_atual
    Pup = (dSOCup * Ebat) / (dT * eficiencia)
    Pdown = (dSOCdown * Ebat * eficiencia) / dT
    Iup = Pup / (baseKVmediaTensao / 1.732050807)
    Idown = Pdown / (baseKVmediaTensao / 1.732050807)
    return Iup, Idown


class FunAG():
    def __init__(self, sistema):
        self.dss = DSS()
        # print(sistemas[sistema]['pasta'])
        # print(sistemas[sistema]['arquivo'])
        self.dss.compileFile(sistemas[sistema]['pasta'], sistemas[sistema]['arquivo'])
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
        ia = []
        ib = []
        ic = []
        n = len(cc)  # Número de instantes de tempo
        SOCini = 0.1  # Estado de carga inicial (10%)
        for gene in range(n):
            if gene == 0:
                Iup, Idown = calculaIupIdown(SOCini)
                ia.append(random.uniform(Idown, Iup))
                ib.append(random.uniform(Idown, Iup))
                ic.append(random.uniform(Idown, Iup))
                SOCan = calculaSOC(SOCini, ia[-1])
                SOCbn = calculaSOC(SOCini, ib[-1])
                SOCcn = calculaSOC(SOCini, ic[-1])
            else:
                Iup_a, Idown_a = calculaIupIdown(SOCan)
                Iup_b, Idown_b = calculaIupIdown(SOCbn)
                Iup_c, Idown_c = calculaIupIdown(SOCcn)
                ia.append(random.uniform(Idown_a, Iup_a))
                ib.append(random.uniform(Idown_b, Iup_b))
                ic.append(random.uniform(Idown_c, Iup_c))
                SOCan = calculaSOC(SOCan, ia[-1])
                SOCbn = calculaSOC(SOCbn, ib[-1])
                SOCcn = calculaSOC(SOCcn, ic[-1])
        
        
        # Gera valores aleatórios de corrente para cada fase em cada instante de tempo
        currents = np.array(ia + ib + ic, dtype=float)
        
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
        
        # Potência de cada fase [[kW fase A], [kW fase B], [kW fase C]] / P = I*V_fase 
        pot = currents * (baseKVmediaTensao / 1.732050807)
        
        # Aloca e resolve para cada instante de tempo
        deseqs_max = []
        for t_idx in range(n):
            potsBat = [pot[0][t_idx], pot[1][t_idx], pot[2][t_idx]]
            self.dss.alocaPot(barra=barra, listaPot=potsBat)
            self.dss.solve(cc[t_idx])
            deseq = self.dss.deseq()
            deseqs_max.append(max(deseq))
    
        
        fobVal = max(deseqs_max)
        # if fobVal > 2.0:
        #     self.fobs.append(10 + fobVal)
        #     return (10.0 + float(fobVal),)
        
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
            print("\n","========================================")
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
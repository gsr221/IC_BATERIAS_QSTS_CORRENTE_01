from deap import creator, base, tools, algorithms
import random
from consts import *
from funODSS import DSS
import time as t
import numpy as np

class FunAG:
    def __init__(self):
        self.dss = DSS()
        self.dss.compileFile(linkFile)
        self.barras = self.dss.BusNames()
        #print(f"Barras trifásicas disponíveis para alocação: {self.barras}")
        self.pmList = []
        creator.create("fitnessMulti", base.Fitness, weights=(-1.0, ))
        #Criando a classe do indivíduo
        creator.create("estrIndiv", list, fitness = creator.fitnessMulti)
        self.fobs = []
    

################==Cria um cromossomo (indivíduo) com valores de Corrente e barramento aleatórios==################
    def criaCromBatCorr(self):
        i1=[]
        i2=[]
        i3=[]
        bus=[]
        
        ##  Dúvida, como calcular o valor máximo da corrente?  ##
        self.imax = [self.pmList[0]/(baseKVmediaTensao/1.732050807),
                self.pmList[1]/(baseKVmediaTensao/1.732050807),
                self.pmList[2]/(baseKVmediaTensao/1.732050807)]  #Corrente máxima em cada fase (A), I = P/(V/√3)

        for _ in range(len(cc)):
            i1.append(random.uniform(-self.imax[0], self.imax[0]))
            i2.append(random.uniform(-self.imax[1], self.imax[1]))
            i3.append(random.uniform(-self.imax[2], self.imax[2]))

        bus.append(random.randint(0, len(self.barras)-1))
        indiv = i1 + i2 + i3 + bus
        return indiv


################==Método de mutação==################
    def mutateFun(self, indiv):
        novoIndiv = indiv
        novoIndiv = self.criaCromBatSOC()
        return novoIndiv    
  

################==Método de cruzamento BLX==################
    def cruzamentoFunBLX(self, indiv1, indiv2):
        newIndiv1 = indiv1
        newIndiv2 = indiv2
        #==Recebe um valor de alfa aleatório==#
        alfa = random.uniform(0.3, 0.5)
        #==Cria um novo indivíduo==#
        for gene in range(len(indiv1)):
            #==Se não for o gene do barramento==#
            if gene != len(indiv1) - 1:
                #==calcula o delta==#
                delta = abs(indiv1[gene] - indiv2[gene])
                #==Calcula o mínimo e o máximo==#
                minGene = min(indiv1[gene], indiv2[gene]) - alfa*delta
                maxGene = max(indiv1[gene], indiv2[gene]) + alfa*delta
                #==Sorteia o novo gene entre o mínimo e o máximo==# 
                newIndiv1[gene] = random.uniform(minGene, maxGene)
                newIndiv2[gene] = random.uniform(minGene, maxGene)
            #==Se for o gene do barramento==#
            else:
                #==calcula o delta==#
                delta = abs(indiv1[gene] - indiv2[gene])
                #==Calcula o mínimo e o máximo==#
                minGene = int(min(indiv1[gene], indiv2[gene]) - alfa*delta)
                maxGene = int(max(indiv1[gene], indiv2[gene]) + alfa*delta)
                #==Sorteia o novo gene entre o mínimo e o máximo==# 
                newIndiv1[gene] = random.randint(minGene, maxGene)
                newIndiv2[gene] = random.randint(minGene, maxGene)
            
        #print(f"newIndiv1: {newIndiv1} - newIndiv2: {newIndiv2}")
        return newIndiv1, newIndiv2


################==Função objetivo para bateria com cromossomo de corrente==################
    def FOBbatCurrent(self, indiv):
        
        n = len(cc)
        #print(indiv)
        # Separa as correntes por fase
        i = [indiv[:n], indiv[n:2*n], indiv[2*n:3*n]]
        

        # Verifica se o barramento existe
        if indiv[3*n] < 0 or indiv[3*n] >= len(self.barras):
            self.fobs.append(1000)
            return 1000,
        
        barra = str(self.barras[int(indiv[3*n])])
        
        self.dss.dssCircuit.SetActiveBus(barra)
        kVBaseBarra = self.dss.dssBus.kVBase
        #print(f"Barramento ativo: {barra} - kVBase: {kVBaseBarra}")
        
        
        if round(kVBaseBarra,2) != round(baseKVmediaTensao/1.732050807,2):
            self.fobs.append(1000)
            return 1000,
        
        # Verifica se os valores de corrente estão dentro dos limites
        if any(abs(valI) > self.imax[fase] for fase in range(3) for valI in i[fase]):
            dists = [0, 0, 0]
            for fase in range(3):
                for valI in i[fase]:
                    if valI > 1000:
                        dists[fase] = max(dists[fase], abs(valI-1000))
            
            return 300 + max(dists),
        
        # Passa o cromossomo de corrente para potencia
        i = np.array(i)
        pot = i * (baseKVmediaTensao/1.732050807)
        
        Ebat = max(self.pmList) * dT
        e = np.zeros((3,n))
        
        for fase in range(3):
            for i in range(n):
                if i == 0:  #==Primeiro valor de energia, considera q no indice -1 a bateria estava completamente carregada==#
                    # Verifica se a bateria está sendo carregada ou descarregada
                    if pot[fase][i] > 0:
                        e[fase][i] = Ebat*0.8 + pot[fase][i]*dT*eficiencia
                    else:
                        e[fase][i] = Ebat*0.8 + pot[fase][i]*dT*(1/eficiencia)
                else:
                    # Verifica se a bateria está sendo carregada ou descarregada
                    if pot[fase][i] > 0:
                        e[fase][i] = e[fase][i-1] + pot[fase][i]*dT*eficiencia
                    else:
                        e[fase][i] = e[fase][i-1] + pot[fase][i]*dT*(1/eficiencia)

        soc = e * (1/Ebat)

        #print(f"Soc: {soc}")
        
        #==Verifica se os valores de SOC estão dentro dos limites se não aplica penalidade==#
        if any(valSoc < SOCmin or valSoc > SOCmax for fase in soc for valSoc in fase):
            maiorDist = 0
            for fase in soc:
                for valSoc in fase:
                    if valSoc < SOCmin:
                        dist = abs(SOCmin - valSoc)
                        maiorDist = max(maiorDist, dist)
                    elif valSoc > SOCmax:
                        dist = abs(valSoc - SOCmax)
                        maiorDist = max(maiorDist, dist)
            return 200 + maiorDist,  # Retorna um valor alto para a FOB
        
        deseqs_max = []
        
        #========SE TUDO ESTIVER DENTRO DOS LIMITES ALOCA A BATERIA========#        
        for i in range(n):
            potsBat = [pot[0][i], pot[1][i], pot[2][i]]
            # print(f"Potências: {potsBat}")
            # print(f"Barramento: {barra}")
            # print(f"cc: {cc[i]}")
            
            #==Aloca as potências no barramento e os bancos de capacitores e resolve o sistema==#
            self.dss.alocaPot(barramento=barra, listaPoten=potsBat)
            self.dss.solve(cc[i])
        
            #==Recebe as tensões de sequência e as coloca em um dicionário==#
            dfSeqVoltages = self.dss.dfSeqVolt()
            dicSecVoltages = dfSeqVoltages.to_dict(orient = 'list')
            deseq = dicSecVoltages[' %V2/V1']
            
            deseqs_max.append(max(deseq))
        
        #==Recebe o valor da função objetivo==#
        fobVal = max(deseqs_max)
        
        if fobVal > 2.0:
            #==Se o valor da FOB for maior que 2.0, retorna um valor alto para a FOB==#
            # print('FOB:', fobVal)
            self.fobs.append(10 + fobVal)
            # print(f"fob: {10 + fobVal} - Desequilíbrio máximo maior que 2.0.")
            # print("Indiv:",indiv)
            return 10 + fobVal,
        
        # print('FOB:', fobVal)
        self.fobs.append(fobVal)
        # print(f"fob: {fobVal} - Desequilíbrio máximo dentro dos limites.")
        return fobVal,



################==Algoritmo Genético==################
    def execAg(self, pms, probCruz=0.9, probMut=0.1, numGen=700, numRep=1, numPop=200, numTorneio=3, eliteSize=10):
        #==Inicio da contagem de tempo==#
        t0 = t.time()
        self.pmList = pms
        
        #==Configuração do AG==#
        toolbox = base.Toolbox()
        dicMelhoresIndiv = {"cromossomos": [], "fobs": []}
        toolbox.register("mate", self.cruzamentoFunBLX)  #Cruzamento BLX
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)  #Mutação Gaussiana
        toolbox.register("select", tools.selTournament, tournsize=numTorneio)  #Seleção por torneio
        toolbox.register("evaluate", self.FOBbatCurrent)  #Função objetivo para bateria com corrente variável

        for rep in range(numRep):
            print(f"{converte_tempo(t0)} - Iniciando execução do Algoritmo Genético... Repetição", rep + 1, "de", numRep)

            toolbox.register("indiv", tools.initIterate, creator.estrIndiv, self.criaCromBatCorr)  #Cria indivíduo com Corrente e Barramento
            toolbox.register("pop", tools.initRepeat, list, toolbox.indiv)  #Cria população
            populacao = toolbox.pop(n=numPop)  #Tamanho da população

            hof = tools.HallOfFame(1)  #Melhor indivíduo
            elite_size = eliteSize  #Tamanho do elitismo

            #==Avalia população inicial==#
            invalid_ind = [ind for ind in populacao if not ind.fitness.valid]  
            fitnesses = map(toolbox.evaluate, invalid_ind)  #Avalia os indivíduos inválidos
            for ind, fit in zip(invalid_ind, fitnesses):    #Atribui o valor da função objetivo ao indivíduo
                ind.fitness.values = fit

            best_fobs = []  #Lista para armazenar os melhores valores da FOB em cada geração

            #==Início das gerações==#
            for gen in range(numGen):
                # Log da geração
                if gen % 10 == 0:
                    print(f"{converte_tempo(t0)} - Geração {gen + 1} de {numGen}... ")
                    
                # Elitismo
                elite = tools.selBest(populacao, elite_size)  #Seleciona os melhores indivíduos

                # Seleção + clone
                offspring = toolbox.select(populacao, len(populacao) - elite_size)  #Seleciona os indivíduos para reprodução
                offspring = list(map(toolbox.clone, offspring))  #Clona os indivíduos selecionados

                # Cruzamento
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < probCruz:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutação
                for mutant in offspring:
                    if random.random() < probMut:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Avaliação dos novos
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Nova população
                populacao[:] = elite + offspring
                hof.update(populacao)

                # Log da geração
                melhor_fob = hof[0].fitness.values[0]
                best_fobs.append(melhor_fob)
                #print(f"Geração {gen + 1}: Melhor FOB = {melhor_fob:.4f}")

            dicMelhoresIndiv["cromossomos"].append(hof[0])
            dicMelhoresIndiv["fobs"].append(hof[0].fitness.values[0])

            return populacao, None, dicMelhoresIndiv, best_fobs, self.barras
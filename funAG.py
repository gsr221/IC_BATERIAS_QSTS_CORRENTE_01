from deap import creator, base, tools
import random
from consts import *
from funODSS import DSS
import time as t
import numpy as np
import math
import multiprocessing as mp
import os

# ------------------------
# Infra de multiprocessing
# ------------------------
# Cada processo terá seu próprio ambiente do OpenDSS para evitar conflitos.
_WORKER = {
    "dss": None,
    "kv_base_media_tensao": None,
    "barras": None,
}


def _init_worker(pasta: str, arquivo: str, kv_base_media_tensao: float):
    '''Cria o ambiente DSS para cada processo.
    
    Args:
        pasta: Pasta do arquivo DSS.
        arquivo: Nome do arquivo DSS.
        kv_base_media_tensao: Tensão base da média tensão do sistema em estudo.
    '''
    # Inicializa um DSS dedicado por processo
    dss_local = DSS()
    dss_local.compileFile(pasta, arquivo)
    # Aloca os elementos fixos (BESS) neste processo
    dss_local.iniciaBESS()
    barras_tp, _ = dss_local.BusNames()

    _WORKER["dss"] = dss_local
    _WORKER["kv_base_media_tensao"] = kv_base_media_tensao
    _WORKER["barras"] = barras_tp


def _evaluate_individual_parallel(indiv: list) -> tuple[float]:
    '''Função objetivo para avaliação paralela de um indivíduo.
    
    Args:
        indiv: Cromossomo representando as correntes e o barramento.
        
    Returns:
        (fob_val,): Tupla contendo o valor da função objetivo (FOB).
    '''
    # Avaliação independente do estado da classe; usa o DSS do processo
    try:
        # Importante: acessa o DSS específico do processo
        dss = _WORKER["dss"]
        kv_mt = _WORKER["kv_base_media_tensao"]
        barras = _WORKER["barras"]

        n = len(cc)
        bus_idx = int(indiv[-1])
        # Verifica se o índice do barramento é válido
        if not (0 <= bus_idx < len(barras)):
            return (2000.0,)

        barra = barras[bus_idx]
        dss.dss.circuit.set_active_bus(barra)
        kv_base_barra = dss.dss.bus.kv_base
        # Verifica se a tensão base do barramento é compatível
        if not math.isclose(kv_base_barra, kv_mt / math.sqrt(3), rel_tol=1e-2):
            return (1000.0,)
        # Separa as correntes por fase
        currents = np.array([indiv[:n], indiv[n:2*n], indiv[2*n:3*n]])
        # Potência de cada fase (P = I * V_fase)
        pot = currents * kv_base_barra

        deseqs_max = []
        # Simula para cada instante de tempo
        for t_idx in range(n):
            pots_bat = [pot[0][t_idx], pot[1][t_idx], pot[2][t_idx]]
            dss.alocaPot(barra=barra, listaPot=pots_bat)
            dss.solve(cc[t_idx])
            deseq = dss.deseq()
            deseqs_max.append(float(np.max(deseq)))
        # Calcula o valor da função objetivo (FOB)
        fob_val = float(max(deseqs_max)) if deseqs_max else 1e6
        return (fob_val,)
    except Exception:
        # Em caso de erro inesperado, retorna um valor ruim para não travar o pool
        return (1e6,)


def calculaSOC(soc_n: float, corrente: float, base_kv_media_tensao: float) -> float:
    """
    Calcula o Estado de Carga (SOC) do BESS para o próximo instante de tempo.

    Args:
        soc_n: Estado de carga n.
        corrente: Corrente de carga/descarga.
        base_kv_media_tensao: Tensão base da média tensão.

    Returns:
        soc_n1: Estado de carga (SOC) em n+1.
    """
    # Calcula a potência no instante n
    pot_n = corrente * (base_kv_media_tensao / math.sqrt(3))
    # Calcula a energia no instante n
    energia_n = soc_n * Ebat
    # Atualiza a energia e o SOC para o próximo instante
    if pot_n > 0:
        # Carregamento
        energia_n1 = energia_n + (pot_n * dT * eficiencia)
    else:
        # Descarregamento
        energia_n1 = energia_n + (pot_n * dT / eficiencia)
    soc_n1 = energia_n1 / Ebat
    return soc_n1


def calculaIupIdown(soc_atual: float, base_kv_media_tensao: float) -> tuple[float, float]:
    """
    Calcula os limites superior (Iup) e inferior (Idown) de corrente para o BESS.

    Args:
        soc_atual: Estado de carga n.
        base_kv_media_tensao: Tensão base da média tensão.

    Returns:
        (i_up, i_down): Uma tupla contendo (Iup, Idown).
    """
    # Cálculo das variações de SOC permitidas
    d_soc_up = SOCmax - soc_atual
    d_soc_down = SOCmin - soc_atual
    # Cálculo das potências máximas de carga e descarga
    p_up = (d_soc_up * Ebat) / (dT * eficiencia)
    p_down = (d_soc_down * Ebat * eficiencia) / dT
    # Cálculo das correntes máximas de carga e descarga
    i_up = p_up / (base_kv_media_tensao / math.sqrt(3))
    i_down = p_down / (base_kv_media_tensao / math.sqrt(3))
    return i_up, i_down


class FunAG():
    """
    Classe que encapsula a lógica do Algoritmo Genético para otimização de BESS.
    """
    def __init__(self, sistema: str):
        # Inicializa o ambiente DSS e configura o sistema
        self.dss = DSS()
        self.kv_base_media_tensao = sistemas[sistema]['baseKVmediaTensao']
        self.dss.compileFile(sistemas[sistema]['pasta'], sistemas[sistema]['arquivo'])
        self.barras, _ = self.dss.BusNames()
        self.pmList = []
        
        # Protege a criação dos tipos para evitar erros em re-execuções.
        if not hasattr(creator, "fitnessMulti"):
            creator.create("fitnessMulti", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "estrIndiv"):
            creator.create("estrIndiv", list, fitness=creator.fitnessMulti)
        
        self.fobs = []
    
    def _gerar_correntes_fase(self, n_instantes: int, soc_inicial: float) -> list[float]:
        """Gera a sequência de correntes para uma única fase.
        
        Args:
            n_instantes: Número de instantes de tempo.
            soc_inicial: Estado de carga inicial da BESS.
        
        Returns:
            correntes: Lista de correntes geradas para a fase.
        """
        correntes = []
        soc_atual = soc_inicial
        for _ in range(n_instantes):
            i_up, i_down = calculaIupIdown(soc_atual, self.kv_base_media_tensao)
            corrente = random.uniform(i_down, i_up)
            correntes.append(corrente)
            soc_atual = calculaSOC(soc_atual, corrente, self.kv_base_media_tensao)
        return correntes

    def cria_individuo(self) -> list:
        """Cria um cromossomo (indivíduo) com valores de Corrente e barramento aleatórios.
        
        Returns:
            indiv: Indivíduo criado.
        """
        n_instantes = len(cc)
        soc_inicial = 0.1  # Estado de carga inicial (10%)

        # Gera correntes para cada fase
        correntes_a = self._gerar_correntes_fase(n_instantes, soc_inicial)
        correntes_b = self._gerar_correntes_fase(n_instantes, soc_inicial)
        correntes_c = self._gerar_correntes_fase(n_instantes, soc_inicial)
        
        # Concatena as correntes
        correntes_todas = np.array(correntes_a + correntes_b + correntes_c, dtype=float)
        
        # Sorteia um barramento aleatório
        bus_idx = float(random.randint(0, len(self.barras) - 1))
        
        # Concatena os valores de corrente com o barramento em um único indivíduo
        indiv = np.concatenate([correntes_todas, [bus_idx]])
        return creator.estrIndiv(indiv)

    def cruzamento_blx(self, indiv1: list, indiv2: list) -> tuple[list, list]:
        """
        Realiza o cruzamento BLX-alpha.
        Este método modifica os indivíduos in-place.
        Args:
            indiv1: Primeiro indivíduo.
            indiv2: Segundo indivíduo.
        Returns:
            (indiv1, indiv2): Indivíduos resultantes do cruzamento.
        """
        alfa = random.uniform(0.2, 0.5)
        last_gene_index = len(indiv1) - 1

        for i in range(len(indiv1)):
            delta = abs(indiv1[i] - indiv2[i])
            min_gene = min(indiv1[i], indiv2[i]) - alfa * delta
            max_gene = max(indiv1[i], indiv2[i]) + alfa * delta

            if i != last_gene_index:
                # Genes de corrente
                indiv1[i] = random.uniform(min_gene, max_gene)
                indiv2[i] = random.uniform(min_gene, max_gene)
            else:
                # Gene do barramento (índice inteiro)
                min_idx = max(0, int(np.floor(min_gene)))
                max_idx = min(len(self.barras) - 1, int(np.ceil(max_gene)))
                
                if min_idx > max_idx:
                    # Garante que os limites sejam válidos
                    min_idx = max_idx = max(0, min(len(self.barras) - 1, int(round((indiv1[i] + indiv2[i]) / 2))))
                
                indiv1[i] = float(random.randint(min_idx, max_idx))
                indiv2[i] = float(random.randint(min_idx, max_idx))
        
        return indiv1, indiv2

    def avalia_individuo(self, indiv: list) -> tuple[float]:
        """Função objetivo: avalia um indivíduo calculando o desequilíbrio máximo (sem paralelismo).
        Args:
            indiv: Cromossomo representando as correntes e o barramento.
        Returns:
            (fob_val,): Tupla contendo o valor da função objetivo (FOB).
        """
        n = len(cc)
        
        # Obtém o índice do barramento
        bus_idx = int(indiv[-1])
        if not (0 <= bus_idx < len(self.barras)):
            self.fobs.append(2000.0)
            return (2000.0,)
        
        barra = self.barras[bus_idx]
        self.dss.dss.circuit.set_active_bus(barra)
        kv_base_barra = self.dss.dss.bus.kv_base
        
        # Verifica se a tensão base do barramento é compatível
        if not math.isclose(kv_base_barra, self.kv_base_media_tensao / math.sqrt(3), rel_tol=1e-2):
            self.fobs.append(1000.0)
            return (1000.0,)
        
        # Separa as correntes por fase
        currents = np.array([indiv[:n], indiv[n:2*n], indiv[2*n:3*n]])
        # Potência de cada fase (P = I * V_fase)
        pot = currents * kv_base_barra
        
        deseqs_max = []
        # Simula para cada instante de tempo
        for t_idx in range(n):
            pots_bat = [pot[0][t_idx], pot[1][t_idx], pot[2][t_idx]]
            self.dss.alocaPot(barra=barra, listaPot=pots_bat)
            self.dss.solve(cc[t_idx])
            deseq = self.dss.deseq()
            deseqs_max.append(max(deseq))

        # Calcula o valor da função objetivo (FOB)
        fob_val = max(deseqs_max)
        self.fobs.append(float(fob_val))
        return (float(fob_val),)

    def execAg(self, pms, prob_cruz: float =0.9, prob_mut: float =0.1, 
               num_gen: int = 700, num_rep: int = 1, num_pop: int = 300, num_torneio: int = 3, 
               elite_size: int = 10, n_jobs: int | None = None, parallel: bool = True):
        
        """
        Args:
            pms: Parâmetros do modelo.
            prob_cruz: Probabilidade de cruzamento.
            prob_mut: Probabilidade de mutação.
            num_gen: Número de gerações.
            num_rep: Número de repetições.
            num_pop: Tamanho da população.
            num_torneio: Tamanho do torneio para seleção.
            elite_size: Tamanho da elite.
            n_jobs: Número de trabalhos para paralelismo.
            parallel: Indica se a avaliação será paralela.
        Returns:
            populacao: População final.
            None: Placeholder para compatibilidade.
            dic_melhores_indiv: Dicionário com os melhores indivíduos e seus FOBs
            all_best_fobs: Lista com os melhores FOBs de todas as repetições.
            self.barras: Lista de barramentos de 3 fases.
        """

        self.pmList = pms
        self.dss.iniciaBESS()

        # --- Configuração do Algoritmo Genético com DEAP ---
        toolbox = base.Toolbox()
        toolbox.register("indiv", self.cria_individuo)
        toolbox.register("pop", tools.initRepeat, list, toolbox.indiv)
        toolbox.register("mate", self.cruzamento_blx)
        toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=num_torneio)
        
        # Avaliação paralela ou serial
        if parallel:
            # Configura o pool de processos para avaliação paralela
            pasta = self.dss.get_pasta()
            arquivo = self.dss.get_arquivo()
            # Configura o número de processos para o pool (deixando 2 núcleos livres)
            processos = n_jobs if (isinstance(n_jobs, int) and n_jobs > 0) else max(1, (os.cpu_count() or 2) - 2)
            ctx = mp.get_context("spawn")
            pool = ctx.Pool(processes=processos, initializer=_init_worker, 
                            initargs=(pasta, arquivo, self.kv_base_media_tensao))
            # Registra o pool para uso no toolbox
            toolbox.register("map", pool.map)
            toolbox.register("evaluate", _evaluate_individual_parallel)
        else:
            toolbox.register("map", map)
            toolbox.register("evaluate", self.avalia_individuo)
        
        dic_melhores_indiv = {"cromossomos": [], "fobs": []}
        all_best_fobs = []

        t0 = t.time()
        for rep in range(num_rep):
            print("\n", "="*40)
            print(f"{converte_tempo(t0)} - Iniciando AG... Repetição {rep + 1}/{num_rep}")
            
            populacao = toolbox.pop(n=num_pop)
            hof = tools.HallOfFame(1)

            # Avalia a população inicial (paralelo se registrado)
            fitnesses = toolbox.map(toolbox.evaluate, populacao)
            for ind, fit in zip(populacao, fitnesses):
                ind.fitness.values = fit
            
            hof.update(populacao)
            best_fobs_rep = [hof[0].fitness.values[0]]

            print("="*10 + " INICIO DAS GERAÇÕES " + "="*10)
            for gen in range(num_gen):
                # Elitismo: seleciona os melhores indivíduos para a próxima geração
                elite = tools.selBest(populacao, elite_size)
                elite = list(map(toolbox.clone, elite)) # Clona para não manter referências

                # Seleciona o restante da população para cruzamento e mutação
                offspring = toolbox.select(populacao, len(populacao) - elite_size)
                offspring = list(map(toolbox.clone, offspring))

                # Aplica cruzamento
                for c1, c2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < prob_cruz:
                        toolbox.mate(c1, c2)
                        del c1.fitness.values
                        del c2.fitness.values

                # Aplica mutação
                for mut in offspring:
                    if random.random() < prob_mut:
                        toolbox.mutate(mut)
                        del mut.fitness.values

                # Avalia os indivíduos que foram modificados (paralelo se registrado)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Cria a nova população
                populacao[:] = elite + offspring
                hof.update(populacao)
                melhor_fob = hof[0].fitness.values[0]
                best_fobs_rep.append(melhor_fob)

                if (gen + 1) % 50 == 0 or gen == num_gen - 1:
                    print(f"{converte_tempo(t0)} - Geração {gen + 1}/{num_gen} | Melhor FOB: {melhor_fob:.4f}")

            # Guarda o melhor indivíduo e seu FOB para esta repetição
            dic_melhores_indiv["cromossomos"].append(toolbox.clone(hof[0]))
            dic_melhores_indiv["fobs"].append(hof[0].fitness.values[0])
            all_best_fobs.append(best_fobs_rep)

        t1 = t.time()
        print(f"Tempo total de execução: {t1 - t0:.2f} segundos")

        # Encerra o pool, se criado
        if parallel:
            try:
                pool.close()
                pool.join()
            except Exception:
                pass

        return populacao, None, dic_melhores_indiv, all_best_fobs, self.barras
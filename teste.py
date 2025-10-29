import numpy as np
# a = np.array([[1,2],[3,4]])
# b = 1/2

#i = {'faseA':1, 'faseB':2, 'faseC':3}

#print(i[list(i.keys())[0]])



i = [[-18.48, 16.77], [-108.447, -121.23], [-79.133, -153.61]]

i = np.array(i)
pot = i * (4.16/1.732050807)
print(pot)

Ebat = 1000 * 1
e = np.zeros((3,2))

for fase in range(3):
    for i in range(2):
        if i == 0:  #==Primeiro valor de energia, considera q no indice -1 a bateria estava completamente carregada==#
            # Verifica se a bateria está sendo carregada ou descarregada
            if pot[fase][i] > 0:
                e[fase][i] = Ebat*0.8 + pot[fase][i]*1*0.95
            else:
                e[fase][i] = Ebat*0.8 + pot[fase][i]*1*(1/0.95)
        else:
            # Verifica se a bateria está sendo carregada ou descarregada
            if pot[fase][i] > 0:
                e[fase][i] = e[fase][i-1] + pot[fase][i]*1*0.95
            else:
                e[fase][i] = e[fase][i-1] + pot[fase][i]*1*(1/0.95)
                
print(e)

soc = e / Ebat
print(soc)
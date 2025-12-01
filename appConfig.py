import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
from appFunctions import AppFunctions


class App(tk.Tk):
    def __init__(self):
        #Setup inicial:
        super().__init__()
        self.title('Aplicativo IC02')
        self.geometry('1280x720')
        
        #Frames:
        self.mainFrame = MainFrame(self)
        
        #Loop da janela:
        self.mainloop()
   
        

class MainFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.creat_widgets()
        self.creat_layout()
        
    def creat_widgets(self):
        self.appFunctions = AppFunctions()
        
        #Estilos:
        styleBotRodar = ttk.Style()
        styleBotRodar.theme_use('default')
        styleBotRodar.configure('BotaoRodar.TButton', relief="flat" ,background="#029B07", foreground="#FFFFFF", font=('Arial', 10, 'bold'))
        styleBotRodar.map('BotaoRodar.TButton',
                        background=[('active', '#03C00A'), ('!disabled', '#029B07')]
                        )
        
        #Frames:
        self.frame_top = ttk.Frame(self)
        # self.frame_mid = ttk.Frame(self)
        self.frame_bot = ttk.Frame(self)
        self.frameResults = ttk.Frame(self.frame_bot)
        # self.frameGraficoDeseq = ttk.Frame(self.frame_bot)
        self.frameBotoes = ttk.Frame(self.frame_bot)
        
        # #Grafico Curva de Carga:
        # self.figCC, self.axCC = plt.subplots()
        # self.canvasCC = FigureCanvasTkAgg(self.figCC, master=self.frameGraficoDeseq)
        # self.axCC.set_ylabel('Porcentagem de carga')
        # self.axCC.set_xticks(range(24))
        # self.axCC.grid(True)
        
        #Gráfico Desequilibrio:
        self.figDeseq, self.axDeseq = plt.subplots()
        self.canvasDeseq = FigureCanvasTkAgg(self.figDeseq, master=self.frame_top)
        self.axDeseq.set_ylabel('Desequilíbrio Máximo')
        self.axDeseq.set_xticks(range(24))
        self.axDeseq.set_yticks(np.arange(0,2.5,0.5))
        self.axDeseq.grid(True)
        self.toolbarDeseq = NavigationToolbar2Tk(self.canvasDeseq, self.frame_top)
        self.toolbarDeseq.update()
        
        # #Entradas:
        # self.potMaLabel = tk.Label(self.frame_mid, text = 'Pma =')
        # self.potMbLabel = tk.Label(self.frame_mid, text = 'Pmb =')
        # self.potMcLabel = tk.Label(self.frame_mid, text = 'Pmc =')
        # self.potMaEntry = tk.Entry(self.frame_mid)
        # self.potMbEntry = tk.Entry(self.frame_mid)
        # self.potMcEntry = tk.Entry(self.frame_mid)
        
        #Treeview:
        self.tree = ttk.Treeview(self.frameResults)
        self.treescrolly = ttk.Scrollbar(self.frameResults, orient="vertical", command=self.tree.yview)
        self.treescrollx = ttk.Scrollbar(self.frameResults, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.treescrolly.set, xscrollcommand=self.treescrollx.set)
        
        #Combobox:
        self.comboSist = ttk.Combobox(self.frame_bot, values=['IEEE 4 bus', 'IEEE 13 bus', 'IEEE 34 bus', 'IEEE 123 bus'])
        self.comboSist.current(1)
        
        #Botoes:
        self.botao = ttk.Button(self.frameBotoes, text = 'exec', command=lambda: self.appFunctions.FunBotao(self.tree, self.axDeseq, self.canvasDeseq))
        self.botaoCSVFOBS = ttk.Button(self.frameBotoes, text = 'Salvar FOBs', command= self.appFunctions.FunBotaoCSVFOBS)
        self.botaoPlotFOB = ttk.Button(self.frameBotoes, text = 'Plotar FOB', command=lambda: self.appFunctions.FunBotaoPlotarFOB())
        self.botaoPlotCC = ttk.Button(self.frameBotoes, text = 'Curva de Carga', command=lambda: self.appFunctions.FunBotaoPlotar())
        self.botaoRodar = ttk.Button(self.frameBotoes, text = 'Rodar', style = 'BotaoRodar.TButton' , command=lambda: self.appFunctions.run_ag_in_thread(self.tree, self.axDeseq, self.canvasDeseq, self.comboSist.get()))
        
    def creat_layout(self):
        #Frame Principal:
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        
        #Frame Topo:
        self.frame_top.place(relx=0.02, rely=0.02, relwidth=0.96, relheight=0.39)
        # self.canvasCC.get_tk_widget().place(relx=0, rely=0, relheight=1, relwidth=1)
        
        # #Frame Meio:
        # self.frame_mid.place(relx=0, rely=0.43, relwidth=1, relheight=0.03)
        
        # self.potMaLabel.place(relx=0, rely=0, relwidth=0.1, relheight=1)
        # self.potMbLabel.place(relx=0.15, rely=0, relwidth=0.1, relheight=1)
        # self.potMcLabel.place(relx=0.3, rely=0, relwidth=0.1, relheight=1)
        
        # self.potMaEntry.place(relx=0.07, rely=0, relwidth=0.1, relheight=1)
        # self.potMbEntry.place(relx=0.22, rely=0, relwidth=0.1, relheight=1)
        # self.potMcEntry.place(relx=0.37, rely=0, relwidth=0.1, relheight=1)
        
        #Frame Inferior:
        self.frame_bot.place(relx=0.02, rely=0.42, relwidth=0.96, relheight=0.55)
        
        self.frameResults.place(relx=0.21, rely=0, relwidth=0.79, relheight=0.91)
        # self.frameGraficoDeseq.place(relx=0.5, rely=0, relwidth=0.5, relheight=0.92)
        self.frameBotoes.place(relx=0, rely=0.92, relwidth=1, relheight=0.08)
        
        self.botao.place(relx=0.46, rely=0, relwidth=0.1, relheight=1)
        self.botaoCSVFOBS.place(relx=0.57, rely=0, relwidth=0.1, relheight=1)
        self.botaoPlotFOB.place(relx=0.68, rely=0, relwidth=0.1, relheight=1)
        self.botaoPlotCC.place(relx=0.79, rely=0, relwidth=0.1, relheight=1)
        self.botaoRodar.place(relx=0.9, rely=0, relwidth=0.1, relheight=1)
        
        self.comboSist.place(relx=0, rely=0, relwidth=0.2, relheight=0.08)
        
        self.canvasDeseq.get_tk_widget().place(relx=0, rely=0, relwidth=1, relheight=0.9)
        self.toolbarDeseq.place(relx=0, rely=0.9, relwidth=1, relheight=0.1)
        
        self.tree.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.treescrolly.pack(side="right", fill='y')
        self.treescrollx.pack(side="bottom", fill='x')
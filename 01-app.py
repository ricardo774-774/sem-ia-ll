import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# ==============================
# CLASE PRINCIPAL DE LA INTERFAZ
# ==============================
class PerceptronGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        #Variables
        self.title("Perceptrón Clasificador")
        self.points = []  # Lista para almacenar los puntos (x1, x2)
        self.weights = None  # Tupla para almacenar (w1, w2, bias)

        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.setup_grid()  # Configuración de la cuadrícula y ejes

        # Agregar el gráfico a la interfaz
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Detectar clics del ratón para agregar puntos
        self.canvas.mpl_connect('button_press_event', self.add_point)

        control_frame = tk.Frame(self)
        control_frame.pack(pady=5)

        # Campos de entrada para pesos y bias (en una fila)
        tk.Label(control_frame, text="w1:").grid(row=0, column=0)
        self.w1_entry = tk.Entry(control_frame, width=5)
        self.w1_entry.grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="w2:").grid(row=0, column=2)
        self.w2_entry = tk.Entry(control_frame, width=5)
        self.w2_entry.grid(row=0, column=3, padx=5)

        tk.Label(control_frame, text="Bias w0:").grid(row=0, column=4)
        self.bias_entry = tk.Entry(control_frame, width=5)
        self.bias_entry.grid(row=0, column=5, padx=5)

        #========BOTONES===========
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)

        #Botón para clasificar puntos
        tk.Button(button_frame, text="Clasificar Puntos", command=self.classify_points).pack(side=tk.LEFT, padx=10)

        #Botón para resetear la gráfica y limpiar puntos
        tk.Button(button_frame, text="Resetear Gráfica", command=self.reset_graph).pack(side=tk.LEFT, padx=10)

        #Manejo del cierre de la aplicación
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    #Ajustes del grafico
    def setup_grid(self):
        """ Configura la cuadrícula del gráfico con ejes centrales """
        self.ax.set_xticks(np.arange(-10, 11, 1))
        self.ax.set_yticks(np.arange(-10, 11, 1))
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.axhline(0, color='black', linewidth=0.5)  # Eje X en negro
        self.ax.axvline(0, color='black', linewidth=0.5)  # Eje Y en negro

    #Captura de puntos con el ratón
    def add_point(self, event):
        """ Agrega puntos en la gráfica con un clic y los almacena en la lista """
        if event.inaxes:
            x, y = round(event.xdata), round(event.ydata)  # Redondeamos coordenadas
            self.points.append((x, y))  # Guardamos el punto en la lista
            self.ax.plot(x, y, 'ko', markersize=8)  # Punto negro
            self.canvas.draw()

    #Funcion de clasificacion y de salida
    def classify_points(self):
        """ Clasifica los puntos ingresados usando la ecuación del perceptrón """
        try:
            # Obtener valores ingresados por el usuario
            w1 = float(self.w1_entry.get())
            w2 = float(self.w2_entry.get())
            bias = float(self.bias_entry.get())
        except ValueError:
            # Mostrar alerta si los valores no son válidos
            messagebox.showwarning("Advertencia", "Por favor ingrese valores numéricos para los pesos y el bias.")
            return

        self.weights = (w1, w2, bias)  # Guardar pesos y bias

        # Redibujar el gráfico para actualizar la clasificación
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.setup_grid()

        # Clasificar cada punto y cambiar color
        for x, y in self.points:
            result = w1 * x + w2 * y + bias  # Cálculo de la salida del perceptrón
            color = 'red' if result >= 0 else 'blue'  # Clasificación binaria
            self.ax.plot(x, y, marker='o', color=color, markersize=8)

        # Dibujar el hiperplano de decisión
        if w2 != 0:  # Evitar división por cero
            x_vals = np.array(self.ax.get_xlim())  # Tomar los límites del eje X
            y_vals = - (w1 / w2) * x_vals - (bias / w2)  # Ecuación de la recta
            self.ax.plot(x_vals, y_vals, 'g--')  # Línea verde discontinua

        self.canvas.draw()

    # Funcion para resetear la grafica
    def reset_graph(self):
        """ Limpia la gráfica y borra los puntos ingresados """
        self.points = []  # Vaciar la lista de puntos
        self.ax.clear()  # Limpiar el gráfico
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.setup_grid()
        self.canvas.draw()

    #Manejo de cierre
    def on_closing(self):
        """ Cierra la ventana y la figura de Matplotlib """
        plt.close(self.figure)
        self.destroy()

if __name__ == "__main__":
    app = PerceptronGUI()
    app.mainloop()

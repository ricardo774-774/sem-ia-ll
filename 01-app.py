import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

# Configuración de la ventana principal
class PerceptronGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptrón Clasificador")

        # Inicializar variables: puntos y pesos
        self.points = []
        self.weights = None

        # Crear el gráfico de matplotlib
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.setup_grid()

        # Agregar gráfico a la interfaz
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Vincular clic del ratón para agregar puntos
        self.canvas.mpl_connect('button_press_event', self.add_point)

        # Contenedor para entradas y botones
        control_frame = tk.Frame(self)
        control_frame.pack(pady=5)

        # Crear entrada de texto para pesos y bias (una sola fila)
        tk.Label(control_frame, text="w1:").grid(row=0, column=0)
        self.w1_entry = tk.Entry(control_frame, width=5)
        self.w1_entry.grid(row=0, column=1, padx=5)

        tk.Label(control_frame, text="w2:").grid(row=0, column=2)
        self.w2_entry = tk.Entry(control_frame, width=5)
        self.w2_entry.grid(row=0, column=3, padx=5)

        tk.Label(control_frame, text="Bias w0:").grid(row=0, column=4)
        self.bias_entry = tk.Entry(control_frame, width=5)
        self.bias_entry.grid(row=0, column=5, padx=5)

        # Botones en una fila separada
        button_frame = tk.Frame(self)
        button_frame.pack(pady=5)

        tk.Button(button_frame, text="Clasificar Puntos", command=self.classify_points).pack(side=tk.LEFT, padx=10)
        tk.Button(button_frame, text="Resetear Gráfica", command=self.reset_graph).pack(side=tk.LEFT, padx=10)

        # Manejo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_grid(self):
        """ Configura la cuadrícula de la gráfica """
        self.ax.set_xticks(np.arange(-10, 11, 1))
        self.ax.set_yticks(np.arange(-10, 11, 1))
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)

    # Marcar puntos en el hiperplano
    def add_point(self, event):
        if event.inaxes:
            x, y = round(event.xdata), round(event.ydata)
            self.points.append((x, y))
            self.ax.plot(x, y, 'ko', markersize=8)
            self.canvas.draw()

    def classify_points(self):
        try:
            w1 = float(self.w1_entry.get())
            w2 = float(self.w2_entry.get())
            bias = float(self.bias_entry.get())
        except ValueError:
            messagebox.showwarning("Advertencia", "Por favor ingrese valores numéricos para los pesos y el bias.")
            return

        self.weights = (w1, w2, bias)

        # Redibujar puntos y clasificar
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.setup_grid()

        # Rojo y azul para las dos clases
        for x, y in self.points:
            result = w1 * x + w2 * y + bias
            color = 'red' if result >= 0 else 'blue'
            self.ax.plot(x, y, marker='o', color=color, markersize=8)

        # Dibujar hiperplano
        if w2 != 0:  # Evita división por cero
            x_vals = np.array(self.ax.get_xlim())
            y_vals = - (w1 / w2) * x_vals - (bias / w2)
            self.ax.plot(x_vals, y_vals, 'g--')  # Línea verde discontinua

        self.canvas.draw()

    def reset_graph(self):
        """ Reinicia la gráfica y limpia los puntos """
        self.points = []
        self.ax.clear()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.setup_grid()
        self.canvas.draw()

    # Cerrar figura y ventana
    def on_closing(self):
        plt.close(self.figure)
        self.destroy()

# Ejecutar aplicación
if __name__ == "__main__":
    app = PerceptronGUI()
    app.mainloop()

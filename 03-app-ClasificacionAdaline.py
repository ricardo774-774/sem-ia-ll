import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AdalineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Clasificador ADALINE")

        # Inicializar variables: puntos, pesos y tasa de aprendizaje
        self.points = []  # Lista para almacenar puntos y su clase (1 o -1)
        self.weights = np.random.uniform(-1, 1, 3)  # Pesos aleatorios iniciales (w1, w2, bias)
        self.learning_rate = 0.01  # Tasa de aprendizaje
        self.epoch = 0  # Contador de épocas

        # Crear el gráfico de matplotlib
        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        # Configurar la cuadrícula y los ejes
        self.ax.set_xticks(np.arange(-10, 11, 1))
        self.ax.set_yticks(np.arange(-10, 11, 1))
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)

        # Agregar gráfico a la interfaz
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('button_press_event', self.add_point)

        # Crear botón para avanzar época
        control_frame = tk.Frame(self)
        control_frame.pack()
        tk.Button(control_frame, text="Clasificar", command=self.advance_epoch).pack(side=tk.LEFT)

        # Espacio para mostrar los valores de los pesos y bias
        self.weights_frame = tk.Frame(self)
        self.weights_frame.pack()
        self.w1_label = tk.Label(self.weights_frame, text=f"Peso w1: {self.weights[0]:.2f}")
        self.w1_label.grid(row=0, column=0)
        self.w2_label = tk.Label(self.weights_frame, text=f"Peso w2: {self.weights[1]:.2f}")
        self.w2_label.grid(row=1, column=0)
        self.bias_label = tk.Label(self.weights_frame, text=f"Bias w0: {self.weights[2]:.2f}")
        self.bias_label.grid(row=2, column=0)

        # Inicializar las referencias a las líneas y contornos
        self.decision_line = None
        self.decision_contour = None

        # Manejo de cierre
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def add_point(self, event):
        if event.inaxes:
            x, y = round(event.xdata), round(event.ydata)
            # Clic izquierdo para clase -1 (azul), clic derecho para clase 1 (rojo)
            if event.button == 1:  # Clic izquierdo
                self.points.append((x, y, -1))  # Clase -1 para azul
                self.ax.plot(x, y, 'bo', markersize=12)  # Punto azul
            elif event.button == 3:  # Clic derecho
                self.points.append((x, y, 1))  # Clase 1 para rojo
                self.ax.plot(x, y, 'ro', markersize=12)  # Punto rojo
            self.canvas.draw()

    def advance_epoch(self):
        if not self.points:
            messagebox.showwarning("Advertencia", "No hay puntos en el plano.")
            return

        total_error = 0
        # Recorrer cada punto y aplicar el algoritmo ADALINE
        for x, y, label in self.points:
            input_vector = np.array([x, y, 1])  # Añadir bias como entrada constante (1)
            prediction = np.dot(self.weights, input_vector)  # Salida lineal del ADALINE
            error = label - prediction  # Calcular el error
            self.weights += self.learning_rate * error * input_vector  # Ajustar pesos
            total_error += error ** 2  # Sumar el error cuadrático

        # Eliminar la línea de decisión y contornos anteriores si existen
        if self.decision_line:
            self.decision_line.remove()
        if self.decision_contour:
            self.decision_contour.remove()

        # Dibujar el plano con el contorno degradado
        self.draw_decision_boundary()

        # Dibujar puntos nuevamente
        for x, y, label in self.points:
            color = 'ro' if label == 1 else 'bo'
            self.ax.plot(x, y, color, markersize=12)

        # Dibujar el hiperplano actualizado
        x_vals = np.array(self.ax.get_xlim())
        y_vals = - (self.weights[0] / self.weights[1]) * x_vals - (self.weights[2] / self.weights[1])
        self.decision_line, = self.ax.plot(x_vals, y_vals, 'g--')  # Línea verde discontinua

        # Actualizar etiquetas con los nuevos valores de pesos y bias (con dos decimales)
        self.w1_label.config(text=f"Peso w1: {self.weights[0]:.2f}")
        self.w2_label.config(text=f"Peso w2: {self.weights[1]:.2f}")
        self.bias_label.config(text=f"Bias w0: {self.weights[2]:.2f}")

        # Mostrar el error cuadrático medio para observar la convergencia
        mse = total_error / len(self.points)
        print(f"Época {self.epoch + 1}, Error cuadrático medio: {mse:.4f}")
        self.epoch += 1

        self.canvas.draw()

    def draw_decision_boundary(self):
        # Crear una cuadrícula de puntos
        x_min, x_max = -10, 10
        y_min, y_max = -10, 10
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
        
        # Calcular la salida para cada punto en la cuadrícula
        Z = self.weights[0] * xx + self.weights[1] * yy + self.weights[2]
        Z = np.tanh(Z)  # Usar tangente hiperbólica para suavizar

        # Dibujar el contorno degradado
        self.decision_contour = self.ax.contourf(xx, yy, Z, levels=50, cmap='coolwarm', alpha=0.3)

    def on_closing(self):
        plt.close(self.figure)
        self.destroy()

# Ejecutar aplicación
if __name__ == "__main__":
    app = AdalineGUI()
    app.mainloop()

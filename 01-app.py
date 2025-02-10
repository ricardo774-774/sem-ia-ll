import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np


class PerceptronGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Perceptrón Clasificador")

        self.points = []
        self.weights = None

        self.figure, self.ax = plt.subplots()
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        self.ax.set_xticks(np.arange(-10, 11, 1))
        self.ax.set_yticks(np.arange(-10, 11, 1))
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        self.canvas.mpl_connect('button_press_event', self.add_point)

        entry_frame = tk.Frame(self)
        entry_frame.pack()

        tk.Label(entry_frame, text="Peso w1:").grid(row=0, column=0)
        self.w1_entry = tk.Entry(entry_frame)
        self.w1_entry.grid(row=0, column=1)

        tk.Label(entry_frame, text="Peso w2:").grid(row=1, column=0)
        self.w2_entry = tk.Entry(entry_frame)
        self.w2_entry.grid(row=1, column=1)

        tk.Label(entry_frame, text="Bias w0:").grid(row=2, column=0)
        self.bias_entry = tk.Entry(entry_frame)
        self.bias_entry.grid(row=2, column=1)

        control_frame = tk.Frame(self)
        control_frame.pack()

        tk.Button(control_frame, text="Clasificar Puntos", command=self.classify_points).pack(side=tk.LEFT)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def add_point(self, event):
        if event.inaxes:
            x, y = round(event.xdata), round(event.ydata)
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

        self.ax.clear()
        self.ax.set_xticks(np.arange(-10, 11, 1))
        self.ax.set_yticks(np.arange(-10, 11, 1))
        self.ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        self.ax.axhline(0, color='black', linewidth=0.5)
        self.ax.axvline(0, color='black', linewidth=0.5)

        for x, y in self.points:
            result = w1 * x + w2 * y + bias
            color = 'ro' if result >= 0 else 'bo'
            self.ax.plot(x, y, color, markersize=12)

        x_vals = np.array(self)
        y_vals = - (w1 / w2) * x_vals - (bias / w2)
        self.ax.plot(x_vals, y_vals, 'g--')  # Línea verde discontinua
        self.canvas.draw()

    def on_closing(self):
        plt.close(self.figure)
        self.destroy()

if __name__ == "__main__":
    app = PerceptronGUI()
    app.mainloop()

import customtkinter as ctk
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np


class FFTViewer(ctk.CTkFrame):

    def __init__(self, master, click_callback=None):
        super().__init__(master)

        self.click_callback = click_callback

        self.canvas = tk.Canvas(self, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<Button-1>", self.on_click)

        self.photo = None
        self.spectrum = None

        self.display_width = 1
        self.display_height = 1

        self.original_width = 1
        self.original_height = 1

        self.points = []

    def set_spectrum(self, spectrum):

        self.spectrum = spectrum

        self.original_height, self.original_width = spectrum.shape

        image = Image.fromarray(spectrum)

        # fixed display size
        display_size = (700, 700)

        image = image.resize(display_size)

        self.display_width = display_size[0]
        self.display_height = display_size[1]

        self.photo = ImageTk.PhotoImage(image)

        self.canvas.delete("all")

        self.canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self.photo
        )

        self.canvas.config(
            width=self.display_width,
            height=self.display_height
        )

    def on_click(self, event):

        if self.spectrum is None:
            return

        # draw selected point
        r = 5

        self.canvas.create_oval(
            event.x - r,
            event.y - r,
            event.x + r,
            event.y + r,
            outline="red",
            width=2
        )

        # convert display coords → FFT coords
        scale_x = self.original_width / self.display_width
        scale_y = self.original_height / self.display_height

        u = int(event.x * scale_x)
        v = int(event.y * scale_y)

        self.points.append((u, v))

        print("Mapped FFT point:", u, v)

        if self.click_callback:
            self.click_callback(u, v)

    def clear_points(self):

        self.points = []

        if self.spectrum is not None:
            self.set_spectrum(self.spectrum)
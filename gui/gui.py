import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from image_io.image_loader import load_regular_image, load_dicom_image, get_dicom_tag
from image_io.metadata import build_metadata_text
from processing.interpolation.zoom import zoom_image
from processing.frequency.fft_utils import compute_fft
from processing.frequency.spectrum_display import magnitude_spectrum
from processing.frequency.notch_filters import apply_notch_filter
from processing.frequency.reconstruction import reconstruct_image

from pipeline.pipeline_manager import PipelineManager

from processing.filtering.linear_filters import apply_average, apply_gaussian
from processing.filtering.nonlinear_filters import median_filter
from processing.filtering.edge_detection import apply_edge_detection
from processing.histogram.local_equalization import (
    local_histogram_equalization,
    local_histogram_equalization_optimized
)
from processing.geometry.transformations import rotate, shear


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ScrollableImageView(ctk.CTkFrame):
    """
    A scrollable image viewer with panning via mouse drag.
    Displays either fitted (fit_to_window=True) or actual size (fit_to_window=False).
    """
    def __init__(self, master, bg_color="#2b2b2b", **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        # Canvas for image display
        self.canvas = tk.Canvas(self, bg=bg_color, highlightthickness=0)
        h_scroll = tk.Scrollbar(self, orient="horizontal", command=self.canvas.xview)
        v_scroll = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        
        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.image_on_canvas = None
        self.current_image_array = None
        self.fit_mode = True
        self.photo = None  
        self._resize_job = None

        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)

        
        self.bind("<Configure>", self._on_frame_configure)

        self._placeholder_text = "No image loaded"
        self.show_placeholder()

        self.frequency_points = []

        self.current_fft = None

        self.current_spectrum = None

    def _on_frame_configure(self, event=None):
        """
        Re-draw when the frame is resized, but debounce the redraw.
        Without this, the GUI may redraw the image many times while resizing/layout changes.
        """
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)

        self._resize_job = self.after(150, self._redraw_after_resize)

    def _redraw_after_resize(self):
        self._resize_job = None

        if self.current_image_array is not None and self.fit_mode:
            self.update_display()
        elif self.current_image_array is None:
            self.show_placeholder()

    def set_image(self, image_array, fit_to_window=True):
        """Load a new image into the viewer."""
        if image_array is None:
            self.clear()
            return
        self.current_image_array = image_array
        self.fit_mode = fit_to_window
        self.update_display()

    def update_display(self):
        """Render the image on the canvas according to current mode."""
        if self.current_image_array is None:
            self.show_placeholder()
            return

        if self.fit_mode:
           
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            if canvas_width <= 1 or canvas_height <= 1:
                self.after(50, self.update_display)
                return

            pil_img = Image.fromarray(self.current_image_array)
            img_w, img_h = pil_img.size
            scale = min(canvas_width / img_w, canvas_height / img_h)
            new_w, new_h = int(img_w * scale), int(img_h * scale)
            resized = pil_img.resize((new_w, new_h), Image.BILINEAR)
            self.photo = ImageTk.PhotoImage(resized)

            self.canvas.delete("all")
            self.canvas.create_image(canvas_width // 2, canvas_height // 2,
                                     anchor="center", image=self.photo)
            self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        else:
            pil_img = Image.fromarray(self.current_image_array)
            self.photo = ImageTk.PhotoImage(pil_img)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            self.canvas.configure(scrollregion=(0, 0, pil_img.width, pil_img.height))

        
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def show_placeholder(self):
        """Display centered placeholder text."""
        self.canvas.delete("all")
        self.current_image_array = None
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        if w <= 1 or h <= 1:
            self.after(50, self.show_placeholder)
            return
        self.canvas.create_text(w // 2, h // 2, text=self._placeholder_text,
                                fill="#aaaaaa", font=("Segoe UI", 14, "italic"))

    def clear(self):
        """Clear the displayed image and show placeholder."""
        self.current_image_array = None
        self.show_placeholder()

      #  funcs 3shan at7rk elimage wana b3ml zoom
    def _on_drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _on_drag_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)


class MedicalImageApp:
    def __init__(self):
        self.current_original = None
        self.current_processed = None
        self.zoom_factor = 1.0

        self.pipeline = PipelineManager()

        self.app = ctk.CTk()
        self.app.title("Clinical Image Analysis Workbench")
        self.app.geometry("1200x800")
        self.app.minsize(1000, 700)

        self.build_layout()
        self.build_left_panel()
        self.build_tabs()

    # ------------------------------ Layout ---------------------------------
    def build_layout(self):
        self.main_frame = ctk.CTkFrame(self.app)
        self.main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.left_panel = ctk.CTkFrame(self.main_frame, width=220)
        self.left_panel.pack(side="left", fill="y", padx=(0, 10))
        self.left_panel.pack_propagate(False)

        self.right_area = ctk.CTkFrame(self.main_frame)
        self.right_area.pack(side="right", fill="both", expand=True)

    def build_left_panel(self):
        ctk.CTkLabel(
            self.left_panel,
            text="Medical\nImage App",
            font=ctk.CTkFont(size=18, weight="bold")
        ).pack(pady=20)

        ctk.CTkFrame(self.left_panel, height=2, fg_color="#444").pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(self.left_panel, text="FILE",
                     font=ctk.CTkFont(size=11), text_color="#888").pack(pady=(8, 2))

        ctk.CTkButton(
            self.left_panel, text="Load Image", command=self.load_image,
            width=180, height=38, font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=4, padx=10)

        ctk.CTkButton(
            self.left_panel, text="Save Image", command=self.save_image,
            width=180, height=38, font=ctk.CTkFont(size=13, weight="bold"),
            fg_color="#2d6a2d", hover_color="#1a4a1a"
        ).pack(pady=4, padx=10)

        ctk.CTkFrame(self.left_panel, height=2, fg_color="#444").pack(fill="x", padx=10, pady=10)
       
        ctk.CTkLabel(
            self.left_panel,
            text="PIPELINE MODE",
            font=ctk.CTkFont(size=11),
            text_color="#888"
        ).pack(pady=(2, 4))

        self.pipeline_switch = ctk.CTkSwitch(
            self.left_panel,
            text="Stack operations",
            command=self.on_pipeline_toggle
        )
        self.pipeline_switch.pack(pady=4, padx=10)

        self.pipeline_mode_label = ctk.CTkLabel(
            self.left_panel,
            text="Mode: OFF - operations use original image",
            font=ctk.CTkFont(size=11),
            text_color="#aaaaaa",
            wraplength=180
        )
        self.pipeline_mode_label.pack(pady=4, padx=10)

        ctk.CTkButton(
            self.left_panel,
            text="Undo Last Step",
            command=self.undo_last_step,
            width=180,
            height=35,
            font=ctk.CTkFont(size=12),
            fg_color="#7a4f00",
            hover_color="#5c3c00"
        ).pack(pady=4, padx=10)

        ctk.CTkButton(
            self.left_panel,
            text="Reset to Original",
            command=self.reset_pipeline,
            width=180,
            height=35,
            font=ctk.CTkFont(size=12),
            fg_color="#6b1f1f",
            hover_color="#4d1717"
        ).pack(pady=4, padx=10)

        ctk.CTkFrame(self.left_panel, height=2, fg_color="#444").pack(fill="x", padx=10, pady=10)


        ctk.CTkLabel(self.left_panel, text="ZOOM",
                     font=ctk.CTkFont(size=11), text_color="#888").pack(pady=(2, 4))

        self.zoom_method = ctk.CTkOptionMenu(
            self.left_panel,
            values=["Bilinear", "Nearest Neighbor"],
            width=180, font=ctk.CTkFont(size=12)
        )
        self.zoom_method.pack(pady=4, padx=10)
        self.zoom_method.set("Bilinear")

        self.zoom_label = ctk.CTkLabel(
            self.left_panel, text="Zoom: 100%", font=ctk.CTkFont(size=12)
        )
        self.zoom_label.pack(pady=4)

        zoom_buttons_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        zoom_buttons_frame.pack(pady=4)

        ctk.CTkButton(
            zoom_buttons_frame, text="Zoom In +", command=self.zoom_in,
            width=85, height=35, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1a5276", hover_color="#154360"
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            zoom_buttons_frame, text="Zoom Out -", command=self.zoom_out,
            width=85, height=35, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#6e2c00", hover_color="#5a2500"
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            self.left_panel, text="Reset Zoom", command=self.reset_zoom,
            width=180, height=35, font=ctk.CTkFont(size=12),
            fg_color="#444", hover_color="#333"
        ).pack(pady=4, padx=10)

        self.status_label = ctk.CTkLabel(
            self.left_panel, text="Ready", font=ctk.CTkFont(size=11), text_color="#888"
        )
        self.status_label.pack(side="bottom", pady=10)

    def build_tabs(self):
        self.tab_view = ctk.CTkTabview(self.right_area)
        self.tab_view.pack(fill="both", expand=True)

        self.tab_view.add("Image Viewer")
        self.tab_view.add("Frequency Domain")
        self.tab_view.add("Pipeline Log")
        self.tab_view.add("Metadata")

        self.build_image_viewer_tab()
        self.build_frequency_tab()
        self.build_pipeline_log_tab()
        self.build_metadata_tab()

    def build_image_viewer_tab(self):
        viewer_tab = self.tab_view.tab("Image Viewer")

        # Main vertical layout:
        # top = images, bottom = operation controls
        main_viewer_frame = ctk.CTkFrame(viewer_tab)
        main_viewer_frame.pack(fill="both", expand=True, padx=10, pady=10)

        images_frame = ctk.CTkFrame(main_viewer_frame)
        images_frame.pack(fill="both", expand=True, padx=5, pady=(5, 8))

        original_frame = ctk.CTkFrame(images_frame)
        original_frame.pack(side="left", fill="both", expand=True, padx=5)

        ctk.CTkLabel(
            original_frame,
            text="Original Image",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=5)

        self.original_image_view = ScrollableImageView(
            original_frame,
            height=360,
            width=450
        )
        self.original_image_view.pack(fill="both", expand=True, padx=5, pady=5)

        processed_frame = ctk.CTkFrame(images_frame)
        processed_frame.pack(side="right", fill="both", expand=True, padx=5)

        ctk.CTkLabel(
            processed_frame,
            text="Processed Image",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=5)

        self.processed_image_view = ScrollableImageView(
            processed_frame,
            height=360,
            width=450
        )
        self.processed_image_view.pack(fill="both", expand=True, padx=5, pady=5)

        # Operation controls are now visible while the user sees the image.
        self.build_operation_controls(main_viewer_frame)

    def build_operation_controls(self, parent):
        # Fixed-height scrollable operations area.
        # This prevents the controls from being cut off when the window is not tall enough.
        controls_frame = ctk.CTkScrollableFrame(parent, height=230)
        controls_frame.pack(fill="x", padx=5, pady=(0, 5))

        ctk.CTkLabel(
            controls_frame,
            text="Operations",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(8, 4))

        # Container for the three operation sections.
        sections_frame = ctk.CTkFrame(controls_frame, fg_color="transparent")
        sections_frame.pack(fill="x", padx=8, pady=5)

        sections_frame.grid_columnconfigure(0, weight=1)
        sections_frame.grid_columnconfigure(1, weight=1)
        sections_frame.grid_columnconfigure(2, weight=1)

        # ---------------- Filtering section ----------------
        filtering_frame = ctk.CTkFrame(sections_frame)
        filtering_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            filtering_frame,
            text="Filtering",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        ctk.CTkLabel(
            filtering_frame,
            text="Kernel Size (odd number: 3, 5, 7...)",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.kernel_size_entry = ctk.CTkEntry(filtering_frame)
        self.kernel_size_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.kernel_size_entry.insert(0, "3")

        ctk.CTkLabel(
            filtering_frame,
            text="Gaussian Sigma / Standard Deviation",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.gaussian_sigma_entry = ctk.CTkEntry(filtering_frame)
        self.gaussian_sigma_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.gaussian_sigma_entry.insert(0, "1.0")

        buttons_row_1 = ctk.CTkFrame(filtering_frame, fg_color="transparent")
        buttons_row_1.pack(pady=3)

        ctk.CTkButton(
            buttons_row_1,
            text="Average",
            command=self.apply_average_filter,
            width=90
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            buttons_row_1,
            text="Gaussian",
            command=self.apply_gaussian_filter,
            width=90
        ).pack(side="left", padx=3)

        buttons_row_2 = ctk.CTkFrame(filtering_frame, fg_color="transparent")
        buttons_row_2.pack(pady=(3, 8))

        ctk.CTkButton(
            buttons_row_2,
            text="Median",
            command=self.apply_median_filter,
            width=90
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            buttons_row_2,
            text="Sobel Edges",
            command=self.apply_sobel_edges,
            width=90
        ).pack(side="left", padx=3)

        # ---------------- Histogram section ----------------
        histogram_frame = ctk.CTkFrame(sections_frame)
        histogram_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            histogram_frame,
            text="Histogram",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        ctk.CTkLabel(
            histogram_frame,
            text="Local Equalization Block Size",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.block_size_entry = ctk.CTkEntry(histogram_frame)
        self.block_size_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.block_size_entry.insert(0, "8")

        ctk.CTkLabel(
            histogram_frame,
            text="Odd Numbers",
            font=ctk.CTkFont(size=10),
            text_color="#999"
        ).pack(anchor="w", padx=10, pady=(0, 6))

        ctk.CTkButton(
            histogram_frame,
            text="Apply Local Equalization",
            command=self.apply_local_equalization,
            width=180
        ).pack(pady=(8, 10))

        # ---------------- Geometry section ----------------
        geometry_frame = ctk.CTkFrame(sections_frame)
        geometry_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)

        ctk.CTkLabel(
            geometry_frame,
            text="Geometry",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        ctk.CTkLabel(
            geometry_frame,
            text="Rotation Angle (degrees)",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.rotation_angle_entry = ctk.CTkEntry(geometry_frame)
        self.rotation_angle_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.rotation_angle_entry.insert(0, "30")

        ctk.CTkLabel(
            geometry_frame,
            text="Shear X Factor",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.shear_x_entry = ctk.CTkEntry(geometry_frame)
        self.shear_x_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.shear_x_entry.insert(0, "0.2")

        ctk.CTkLabel(
            geometry_frame,
            text="Shear Y Factor",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.shear_y_entry = ctk.CTkEntry(geometry_frame)
        self.shear_y_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.shear_y_entry.insert(0, "0.0")

        geometry_buttons = ctk.CTkFrame(geometry_frame, fg_color="transparent")
        geometry_buttons.pack(pady=(3, 8))

        ctk.CTkButton(
            geometry_buttons,
            text="Rotate",
            command=self.apply_rotation,
            width=90
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            geometry_buttons,
            text="Shear",
            command=self.apply_shearing,
            width=90
        ).pack(side="left", padx=3)

    def build_frequency_tab(self):
        freq_tab = self.tab_view.tab("Frequency Domain")

        freq_main_frame = ctk.CTkFrame(freq_tab)
        freq_main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        images_frame = ctk.CTkFrame(freq_main_frame)
        images_frame.pack(fill="both", expand=True)

        spectrum_frame = ctk.CTkFrame(images_frame)
        spectrum_frame.pack(side="left", fill="both", expand=True, padx=5)

        ctk.CTkLabel(
            spectrum_frame,
            text="Magnitude Spectrum",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=5)
        self.spectrum_view = ScrollableImageView(
            spectrum_frame,
        )
        self.spectrum_view.pack(fill="both", expand=True, padx=5, pady=5)

        result_frame = ctk.CTkFrame(images_frame)
        result_frame.pack(side="right", fill="both", expand=True, padx=5)
        ctk.CTkLabel(
            result_frame,
            text="Filtered Result",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=5)

        self.filtered_result_view = ScrollableImageView(
            result_frame,
        )
        self.filtered_result_view.pack(fill="both", expand=True, padx=5, pady=5)

        controls= ctk.CTkFrame(freq_main_frame)
        controls.pack(fill="x", padx=5, pady=(10, 5))

        ctk.CTkButton(
            controls,
            text="Apply Filter",
            command=self.show_fft_spectrum,
            width=180,
        ).pack(side="left", padx=5)

        #Filter Type
        self.notch_filter_type = ctk.CTkOptionMenu(
            controls,
            values=["Ideal", "Butterworth", "Gaussian"],
            width=160,
        )

        self.notch_filter_type.pack(side="left", padx=5)
        self.notch_filter_type.set("Gaussian")

        #Radius Slider
        self.notch_radius_slider = ctk.CTkSlider(
            controls,
            from_=2,
            to=50,
            number_of_steps=48,
            width=180,
        )
        self.notch_radius_slider.pack(side="left", padx=5)
        self.notch_radius_slider.set(10)

        #Radius Label
        self.radius_label = ctk.CTkLabel(
            controls,
            text="Radius: 10"
        )

        self.radius_label.pack(side="left", padx=5)

        self.notch_radius_slider.configure(
            command=lambda value: self.radius_label.configure(
                text=f"Radius: {int(value)}"
            )
        )

        # Apply Filter Button
        ctk.CTkButton(
            controls,
            text="Apply Notch Filter",
            command=self.apply_notch_filter_gui,
            width=180,
        ).pack(side="left", padx=5)
        # Clear Points Button

        ctk.CTkButton(
            controls,
            text="Clear Points",
            command=self.clear_frequency_points,
            width=140
        ).pack(side="left", padx=5)

        # ================= INSTRUCTIONS =================

        instructions = ctk.CTkLabel(
            freq_main_frame,
            text=(
                "Instructions:\n"
                "1. Load an image.\n"
                "2. Click 'Show FFT Spectrum'.\n"
                "3. Click on bright periodic noise points.\n"
                "4. Apply the notch filter.\n"
                "5. Compare the reconstructed image."
            ),
            justify="left",
            text_color="#bbbbbb",
            font=ctk.CTkFont(size=11)
        )

        instructions.pack(anchor="w", padx=10, pady=(0, 5))

        # ================= CLICK BINDING =================

        self.spectrum_view.canvas.bind(
            "<Button-1>",
            self.on_spectrum_click
        )

    # =========================================================
    # Frequency Domain Functions
    # =========================================================

    def show_fft_spectrum(self):
        if not self.pipeline.has_image():
            messagebox.showwarning(
                "No Image",
                "Please load an image first."
            )
            return

        try:
            image = self.pipeline.get_current()

            if image is None:
                messagebox.showwarning(
                    "No Image",
                    "No image available."
                )
                return

            self.status_label.configure(text="Computing FFT...")
            self.app.configure(cursor="watch")
            self.app.update_idletasks()

            fft_shifted = compute_fft(image)

            spectrum = magnitude_spectrum(fft_shifted)

            self.app.configure(cursor="")

            self.spectrum_view.current_fft = fft_shifted
            self.spectrum_view.current_spectrum = spectrum
            self.spectrum_view.frequency_points = []

            self.show_fit_image(
                self.spectrum_view,
                spectrum
            )

            self.tab_view.set("Frequency Domain")

            self.status_label.configure(
                text="FFT Spectrum displayed"
            )

        except Exception as e:
            self.app.configure(cursor="")

            messagebox.showerror(
                "FFT Error",
                f"Could not compute FFT spectrum.\nReason: {str(e)}"
            )

            self.status_label.configure(
                text="FFT failed"
            )

    def on_spectrum_click(self, event):
        if self.spectrum_view.current_spectrum is None:
            return

        canvas = self.spectrum_view.canvas

        x = int(canvas.canvasx(event.x))
        y = int(canvas.canvasy(event.y))

        self.spectrum_view.frequency_points.append((x, y))

        r = 5

        canvas.create_oval(
            x - r,
            y - r,
            x + r,
            y + r,
            outline="red",
            width=2
        )

        canvas.create_text(
            x + 10,
            y - 10,
            text=f"{len(self.spectrum_view.frequency_points)}",
            fill="yellow",
            font=("Segoe UI", 10, "bold")
        )

        self.status_label.configure(
            text=f"Selected frequency point ({x}, {y})"
        )

    def clear_frequency_points(self):
        self.spectrum_view.frequency_points = []

        if self.spectrum_view.current_spectrum is not None:
            self.show_fit_image(
                self.spectrum_view,
                self.spectrum_view.current_spectrum
            )

        self.status_label.configure(
            text="Frequency points cleared"
        )

    def apply_notch_filter_gui(self):
        if self.spectrum_view.current_fft is None:
            messagebox.showwarning(
                "No FFT",
                "Please compute FFT spectrum first."
            )
            return

        if len(self.spectrum_view.frequency_points) == 0:
            messagebox.showwarning(
                "No Points",
                "Please select frequency points first."
            )
            return

        try:
            radius = int(self.notch_radius_slider.get())

            filter_type = self.notch_filter_type.get()

            fft_shifted = self.spectrum_view.current_fft

            self.status_label.configure(
                text=f"Applying {filter_type} notch filter..."
            )

            self.app.configure(cursor="watch")
            self.app.update_idletasks()

            filtered_fft = apply_notch_filter(
                fft_shifted,
                self.spectrum_view.frequency_points,
                radius=radius,
                filter_type=filter_type
            )

            reconstructed = reconstruct_image(filtered_fft)

            self.app.configure(cursor="")

            self.current_processed = reconstructed

            self.pipeline.current_image = reconstructed

            self.show_fit_image(
                self.filtered_result_view,
                reconstructed
            )

            self.show_fit_image(
                self.processed_image_view,
                reconstructed
            )

            self.status_label.configure(
                text=f"{filter_type.capitalize()} notch filter applied"
            )

        except Exception as e:
            self.app.configure(cursor="")

            messagebox.showerror(
                "Filtering Error",
                f"Could not apply notch filter.\nReason: {str(e)}"
            )

            self.status_label.configure(
                text="Filtering failed"
            )

    def build_pipeline_log_tab(self):
        log_tab = self.tab_view.tab("Pipeline Log")

        ctk.CTkLabel(
            log_tab,
            text="Sequential Enhancement History",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=10)

        self.pipeline_log_box = ctk.CTkTextbox(
            log_tab,
            width=600,
            height=400,
            font=ctk.CTkFont(size=13)
        )
        self.pipeline_log_box.pack(fill="both", expand=True, padx=20, pady=10)
        self.pipeline_log_box.insert("1.0", "No operations applied yet.")
        self.pipeline_log_box.configure(state="disabled")        

    def build_metadata_tab(self):
        meta_tab = self.tab_view.tab("Metadata")
        ctk.CTkLabel(meta_tab, text="Image Information",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.metadata_box = ctk.CTkTextbox(
            meta_tab, width=600, height=400, font=ctk.CTkFont(size=13)
        )
        self.metadata_box.pack(fill="both", expand=True, padx=20, pady=10)
        self.metadata_box.insert("1.0", "Load an image to see its information here.")
        self.metadata_box.configure(state="disabled")

      # function bts3d anna n display elimage

    def show_fit_image(self, viewer, image_array):
        """Display image fitted within the viewer (no scrollbars needed)."""
        if image_array is None:
            viewer.clear()
        else:
            viewer.set_image(image_array, fit_to_window=True)

    def show_actual_image(self, viewer, image_array):
        """Display image at actual size (scrollbars appear if larger than view)."""
        if image_array is None:
            viewer.clear()
        else:
            viewer.set_image(image_array, fit_to_window=False)

    def show_zoomed_image(self, viewer, image_array, zoom_factor, method):
        """
        Display a zoomed version of the image relative to the fitted view.

        Important:
        - 100% means fitted-to-window view.
        - 125% means 1.25 times the fitted display size.
        - This uses our custom interpolation through zoom_image().
        - It does not modify the pipeline result.
        """
        if image_array is None:
            viewer.clear()
            return

        canvas_width = viewer.canvas.winfo_width()
        canvas_height = viewer.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            self.app.after(
                50,
                lambda: self.show_zoomed_image(viewer, image_array, zoom_factor, method)
            )
            return

        img_h, img_w = image_array.shape[:2]

        # This is the scale used by the normal fitted view.
        fit_scale = min(canvas_width / img_w, canvas_height / img_h)

        # Zoom relative to the fitted view, not relative to original pixel size.
        display_scale = fit_scale * zoom_factor

        # Prevent extremely tiny images from becoming 0 pixels wide/high.
        minimum_scale = max(1 / img_w, 1 / img_h)
        display_scale = max(display_scale, minimum_scale)

        zoomed = zoom_image(image_array, display_scale, method)

        viewer.set_image(zoomed, fit_to_window=False)            

    def show_metadata_text(self, text):
        self.metadata_box.configure(state="normal")
        self.metadata_box.delete("1.0", "end")
        self.metadata_box.insert("1.0", text)
        self.metadata_box.configure(state="disabled")

    def is_pipeline_enabled(self):
        return self.pipeline_switch.get() == 1


    def on_pipeline_toggle(self):
        if self.is_pipeline_enabled():
            self.pipeline_mode_label.configure(
                text="Mode: ON - operations stack on current result"
            )
            self.status_label.configure(text="Pipeline mode ON")
        else:
            self.pipeline_mode_label.configure(
                text="Mode: OFF - operations use original image"
            )
            self.status_label.configure(text="Pipeline mode OFF")


    def update_pipeline_log(self):
        if not hasattr(self, "pipeline_log_box"):
            return

        self.pipeline_log_box.configure(state="normal")
        self.pipeline_log_box.delete("1.0", "end")
        self.pipeline_log_box.insert("1.0", self.pipeline.get_log_text())
        self.pipeline_log_box.configure(state="disabled")


    def get_valid_odd_integer(self, entry, field_name):
        try:
            value = int(entry.get())
        except ValueError:
            raise ValueError(f"{field_name} must be an integer.")

        if value < 3:
            raise ValueError(f"{field_name} must be at least 3.")

        if value % 2 == 0:
            raise ValueError(f"{field_name} must be odd, such as 3, 5, or 7.")

        return value


    def get_valid_float(self, entry, field_name):
        try:
            value = float(entry.get())
        except ValueError:
            raise ValueError(f"{field_name} must be a number.")

        return value


    def apply_pipeline_operation(self, operation_function, operation_name):
        """
        Central function for all non-zoom processing operations.

        It decides whether to use:
        - original image, if pipeline mode is OFF
        - current processed image, if pipeline mode is ON
        """
        if not self.pipeline.has_image():
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            input_image = self.pipeline.get_input_image(self.is_pipeline_enabled())

            if not self.warn_if_large_image(input_image, operation_name):
                self.status_label.configure(text="Operation cancelled")
                return
            

            self.status_label.configure(text=f"Applying: {operation_name}...")
            self.app.configure(cursor="watch")
            self.app.update_idletasks()            

            result = operation_function(input_image)
            self.app.configure(cursor="")

            self.current_processed = self.pipeline.apply_result(result, operation_name)

            self.zoom_factor = 1.0
            self.zoom_label.configure(text="Zoom: 100%")

            # Use fit display for normal operations so the result does not look zoomed in.
            self.show_fit_image(self.processed_image_view, self.current_processed)

            self.update_pipeline_log()
            self.status_label.configure(text=f"Applied: {operation_name}")

            

        except Exception as e:
            self.app.configure(cursor="")
            messagebox.showerror(
                "Operation Error",
                f"Could not apply {operation_name}.\nReason: {str(e)}"
            )
            self.status_label.configure(text="Operation failed")        

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose an Image File",
            filetypes=[
                ("Medical Images", "*.jpg *.jpeg *.bmp *.dcm"),
                ("JPEG Images", "*.jpg *.jpeg"),
                ("BMP Images", "*.bmp"),
                ("DICOM Images", "*.dcm"),
                ("All Files", "*.*")
            ]
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith('.dcm'):
                pixel_array, dicom_data = load_dicom_image(file_path)
                self.current_original = pixel_array
                self.current_processed = pixel_array.copy()

                self.pipeline.set_original(pixel_array)
                self.update_pipeline_log()

                self.show_fit_image(self.original_image_view, self.current_original)
                self.show_fit_image(self.processed_image_view, self.current_processed)
                self.spectrum_view.clear()
                self.filtered_result_view.clear()
                self.spectrum_view.frequency_points = []
                self.spectrum_view.current_fft = None
                self.spectrum_view.current_spectrum = None

                h, w = pixel_array.shape
                text = build_metadata_text(
                    file_name=file_path.split("/")[-1],
                    width=w, height=h,
                    bit_depth=get_dicom_tag(dicom_data, "BitsAllocated"),
                    file_format="DICOM",
                    modality=get_dicom_tag(dicom_data, "Modality"),
                    patient_name=get_dicom_tag(dicom_data, "PatientName"),
                    patient_age=get_dicom_tag(dicom_data, "PatientAge"),
                    body_part=get_dicom_tag(dicom_data, "BodyPartExamined")
                )
                self.show_metadata_text(text)

            elif file_path.lower().endswith(('.jpg', '.jpeg', '.bmp')):
                image_array = load_regular_image(file_path)
                self.current_original = image_array
                self.current_processed = image_array.copy()

                self.pipeline.set_original(image_array)
                self.update_pipeline_log()

                self.show_fit_image(self.original_image_view, self.current_original)
                self.show_fit_image(self.processed_image_view, self.current_processed)

                h, w = image_array.shape
                file_name = file_path.split("/")[-1]
                text = build_metadata_text(
                    file_name=file_name,
                    width=w, height=h,
                    bit_depth=8,
                    file_format=file_name.split(".")[-1].upper()
                )
                self.show_metadata_text(text)

            else:
                messagebox.showerror("Unsupported Format",
                                     "Please use JPEG, BMP, or DICOM files only.")
                return

            self.zoom_factor = 1.0
            self.zoom_label.configure(text="Zoom: 100%")
            self.tab_view.set("Image Viewer")
            self.status_label.configure(text="Image loaded")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image.\nReason: {str(e)}")
            self.status_label.configure(text="Error loading image")

    def save_image(self):
        self.current_processed = self.pipeline.get_current()

        if self.current_processed is None:
            messagebox.showwarning("No Image", "Please load an image first!")
            return

        save_path = filedialog.asksaveasfilename(
            title="Save Image",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG Image", "*.jpg"),
                ("BMP Image", "*.bmp"),
                ("PNG Image", "*.png")
            ]
        )
        if not save_path:
            return

        try:
            Image.fromarray(self.current_processed).save(save_path)
            messagebox.showinfo("Saved!", f"Image saved to:\n{save_path}")
            self.status_label.configure(text="Image saved")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save.\nReason: {str(e)}")


    def apply_zoom(self):
        if not self.pipeline.has_image():
            messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
            return

        method = "nearest" if self.zoom_method.get() == "Nearest Neighbor" else "bilinear"

        try:
            input_image = self.pipeline.get_current()

            if input_image is None:
                messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
                return

            self.current_processed = input_image.copy()

            # At 100%, show the normal fitted image.
            if self.zoom_factor == 1.0:
                self.show_fit_image(self.processed_image_view, self.current_processed)
                self.status_label.configure(text="Viewing zoom: 100%")
                return

            # For other zoom levels, zoom relative to the fitted display size.
            self.show_zoomed_image(
                self.processed_image_view,
                self.current_processed,
                self.zoom_factor,
                method
            )

            self.status_label.configure(
                text=f"Viewing zoom: {int(self.zoom_factor * 100)}% ({method})"
            )

        except Exception as e:
            messagebox.showerror("Zoom Error", f"Could not apply zoom.\nReason: {str(e)}")
   
    def zoom_in(self):
        if not self.pipeline.has_image():
            messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
            return

        self.zoom_factor = min(round(self.zoom_factor + 0.25, 2), 4.0)
        self.zoom_label.configure(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        self.apply_zoom()

    def zoom_out(self):
        if not self.pipeline.has_image():
            messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
            return

        self.zoom_factor = max(round(self.zoom_factor - 0.25, 2), 0.25)
        self.zoom_label.configure(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        self.apply_zoom()

    def reset_zoom(self):
        if not self.pipeline.has_image():
            return

        self.zoom_factor = 1.0
        self.zoom_label.configure(text="Zoom: 100%")

        self.current_processed = self.pipeline.get_current()

        if self.current_processed is not None:
            self.show_fit_image(self.processed_image_view, self.current_processed)

        self.status_label.configure(text="Zoom view reset")
        
    def apply_average_filter(self):
        try:
            kernel_size = self.get_valid_odd_integer(
                self.kernel_size_entry,
                "Kernel size"
            )

            self.apply_pipeline_operation(
                lambda img: apply_average(img, kernel_size),
                f"Average Filter ({kernel_size}x{kernel_size})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_gaussian_filter(self):
        try:
            kernel_size = self.get_valid_odd_integer(
                self.kernel_size_entry,
                "Kernel size"
            )

            sigma = self.get_valid_float(
                self.gaussian_sigma_entry,
                "Gaussian sigma"
            )

            if sigma <= 0:
                raise ValueError("Gaussian sigma must be greater than 0.")

            self.apply_pipeline_operation(
                lambda img: apply_gaussian(img, kernel_size, sigma),
                f"Gaussian Filter ({kernel_size}x{kernel_size}, sigma={sigma})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_median_filter(self):
        try:
            kernel_size = self.get_valid_odd_integer(
                self.kernel_size_entry,
                "Kernel size"
            )

            self.apply_pipeline_operation(
                lambda img: median_filter(img, kernel_size),
                f"Median Filter ({kernel_size}x{kernel_size})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_sobel_edges(self):
        def sobel_operation(img):
            grad_x, grad_y, magnitude = apply_edge_detection(img)
            return magnitude

        self.apply_pipeline_operation(
            sobel_operation,
            "Sobel Edge Detection - Combined Magnitude"
        )


    def apply_local_equalization(self):
        try:
            block_size = self.get_valid_odd_integer(
                self.block_size_entry,
                "Block size"
            )

            self.apply_pipeline_operation(
                lambda img: local_histogram_equalization_optimized(img, block_size),
                f"Local Histogram Equalization (block size={block_size})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    def apply_rotation(self):
        try:
            angle = self.get_valid_float(
                self.rotation_angle_entry,
                "Rotation angle"
            )

            self.apply_pipeline_operation(
                lambda img: rotate(img, angle),
                f"Rotation ({angle} degrees)"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_shearing(self):
        try:
            shear_x = self.get_valid_float(
                self.shear_x_entry,
                "Shear X"
            )

            shear_y = self.get_valid_float(
                self.shear_y_entry,
                "Shear Y"
            )

            self.apply_pipeline_operation(
                lambda img: shear(img, shear_x=shear_x, shear_y=shear_y),
                f"Shearing (x={shear_x}, y={shear_y})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    def undo_last_step(self):
        if not self.pipeline.has_image():
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        previous_image, removed_operation = self.pipeline.undo()

        if previous_image is None:
            messagebox.showinfo("Undo", "There is no previous step to undo.")
            return

        self.current_processed = previous_image.copy()

        self.zoom_factor = 1.0
        self.zoom_label.configure(text="Zoom: 100%")

        self.show_fit_image(self.processed_image_view, self.current_processed)
        self.update_pipeline_log()

        if removed_operation is None:
            self.status_label.configure(text="Nothing to undo")
            messagebox.showinfo("Undo", "There is no operation to undo.")
        else:
            self.status_label.configure(text=f"Undid: {removed_operation}")

    def warn_if_large_image(self, image, operation_name):
        h, w = image.shape[:2]
        total_pixels = h * w

        if total_pixels > 1_000_000:
            return messagebox.askyesno(
                "Large Image Warning",
                f"{operation_name} may take time on this image "
                f"({w} x {h} pixels).\n\nDo you want to continue?"
            )

        return True

    def reset_pipeline(self):
        if not self.pipeline.has_image():
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        reset_image = self.pipeline.reset()

        self.current_processed = reset_image.copy()
        self.show_fit_image(self.processed_image_view, self.current_processed)
        self.update_pipeline_log()

        self.zoom_factor = 1.0
        self.zoom_label.configure(text="Zoom: 100%")

        self.status_label.configure(text="Reset to original")            

    def run(self):
        self.app.mainloop()

if __name__ == "__main__":
    app = MedicalImageApp()
    app.run()
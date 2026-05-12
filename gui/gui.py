import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from image_io.image_loader import load_regular_image, load_dicom_image, get_dicom_tag
from image_io.metadata import build_metadata_text
from processing.interpolation.zoom import zoom_image
from pipeline.pipeline_manager import PipelineManager
from gui.widgets.fft_viewer import FFTViewer
from gui.widgets.notch_panel import NotchPanel
from processing.frequency.reconstruction import reconstruct_image
from processing.frequency.fft_utils import compute_fft
from processing.frequency.spectrum_display import magnitude_spectrum
from processing.frequency.notch_filters import apply_notch_filter
from processing.filtering.linear_filters import apply_average, apply_gaussian
from processing.filtering.nonlinear_filters import median_filter
from processing.filtering.edge_detection import apply_edge_detection
from processing.histogram.local_equalization import (
    local_histogram_equalization,
    local_histogram_equalization_optimized
)
from processing.geometry.transformations import rotate, shear
from processing.morphology.binary_morphology import (
    global_threshold,
    create_structuring_element,
    erode,
    dilate,
    opening,
    closing,
    boundary_extraction
)
from processing.noise.noise import add_gaussian_noise, add_uniform_noise
from processing.roi.roi_tool import extract_roi, draw_roi_on_image
# Add this with your other imports at the top of gui.py
from processing.roi.roi_stats_window import show_roi_statistics
from processing.frequency.frequency_template_matching import (
    fourier_cross_correlate,
    fourier_cross_correlate_normalized,
)


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


# ──────────────────────────────────────────────────────────────────────────────
# Helper function for Template Matching tab
# ──────────────────────────────────────────────────────────────────────────────

def _tm_draw_on_canvas(canvas, image_array):
    """
    Fit image_array into canvas, return (photo, scale, ox, oy) for coordinate mapping.
    Returns None if canvas not yet realized.
    """
    cw = canvas.winfo_width()
    ch = canvas.winfo_height()
    if cw <= 1 or ch <= 1:
        return None

    ih, iw = image_array.shape[:2]
    scale = min(cw / iw, ch / ih)
    dw, dh = int(iw * scale), int(ih * scale)
    ox = (cw - dw) // 2
    oy = (ch - dh) // 2

    if image_array.ndim == 2:
        pil_img = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8), mode="L")
    else:
        pil_img = Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

    pil_img = pil_img.resize((dw, dh), Image.LANCZOS)
    return ImageTk.PhotoImage(pil_img), scale, ox, oy


class MedicalImageApp:
    def __init__(self):
        self.current_original = None
        self.current_processed = None
        self._tm_target_image = None
        self.zoom_factor = 1.0

        self.roi_start = None
        self.roi_end   = None
        self.roi_rect  = None

        # Template Matching state
        self._tm_template       = None
        self._tm_crop_start     = None
        self._tm_rect_id        = None
        self._tm_scale          = 1.0
        self._tm_offset_x       = 0
        self._tm_offset_y       = 0
        self._tm_photo_crop     = None
        self._tm_photo_result   = None
        self._tm_photo_corr     = None
        self._tm_photo_template = None
        self._tm_photo_target   = None
        self._tm_use_target_compare = False

        self.fft_shifted = None
        self.notch_points = []
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
        self.tab_view.add("Pipeline Log")
        self.tab_view.add("Metadata")
        self.tab_view.add("Template Matching")
        self.tab_view.add("Frequency Domain")


        self.build_image_viewer_tab()
        self.build_pipeline_log_tab()
        self.build_metadata_tab()
        self.build_template_matching_tab()
        self.build_frequency_tab()

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

        self.processed_image_view.canvas.bind("<ButtonPress-1>",   self._roi_drag_start)
        self.processed_image_view.canvas.bind("<B1-Motion>",       self._roi_drag_move)
        self.processed_image_view.canvas.bind("<ButtonRelease-1>", self._roi_drag_end)

    def build_operation_controls(self, parent):
        # Fixed-height tabbed operations area.
        # Each tab contains its own scrollable frame so controls do not get cut off.
        controls_frame = ctk.CTkFrame(parent)
        controls_frame.pack(fill="x", padx=5, pady=(0, 5))

        ctk.CTkLabel(
            controls_frame,
            text="Operations",
            font=ctk.CTkFont(size=15, weight="bold")
        ).pack(pady=(8, 4))

        self.operations_tab_view = ctk.CTkTabview(controls_frame, height=260)
        self.operations_tab_view.pack(fill="x", padx=8, pady=5)

        self.operations_tab_view.add("Enhancement")
        self.operations_tab_view.add("Morphology")
        self.operations_tab_view.add("Noise & ROI")

        enhancement_scroll = ctk.CTkScrollableFrame(
            self.operations_tab_view.tab("Enhancement"),
            height=220
        )
        enhancement_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        morphology_scroll = ctk.CTkScrollableFrame(
            self.operations_tab_view.tab("Morphology"),
            height=220
        )
        morphology_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        noise_roi_scroll = ctk.CTkScrollableFrame(
            self.operations_tab_view.tab("Noise & ROI"),
            height=220
        )
        noise_roi_scroll.pack(fill="both", expand=True, padx=5, pady=5)

        self.build_enhancement_controls(enhancement_scroll)
        self.build_morphology_controls(morphology_scroll)
        self.build_noise_roi_controls(noise_roi_scroll)

    def build_enhancement_controls(self, parent):
        sections_frame = ctk.CTkFrame(parent, fg_color="transparent")
        sections_frame.pack(fill="x", padx=8, pady=5)

        sections_frame.grid_columnconfigure(0, weight=1)
        sections_frame.grid_columnconfigure(1, weight=1)
        sections_frame.grid_columnconfigure(2, weight=1)
        sections_frame.grid_columnconfigure(3, weight=1)



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
            text="Kernel Size",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.kernel_size_entry = ctk.CTkEntry(filtering_frame)
        self.kernel_size_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.kernel_size_entry.insert(0, "3")

        ctk.CTkLabel(
            filtering_frame,
            text="Gaussian Sigma",
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
            text="Rotation Angle",
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

    def build_morphology_controls(self, parent):
        morphology_frame = ctk.CTkFrame(parent, fg_color="transparent")
        morphology_frame.pack(fill="x", padx=8, pady=5)

        # ---------------- Threshold section ----------------
        threshold_frame = ctk.CTkFrame(morphology_frame)
        threshold_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            threshold_frame,
            text="Thresholding",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        self.threshold_value_label = ctk.CTkLabel(
            threshold_frame,
            text="Threshold Value: 128",
            font=ctk.CTkFont(size=12)
        )
        self.threshold_value_label.pack(anchor="w", padx=10, pady=(4, 2))

        self.threshold_slider = ctk.CTkSlider(
            threshold_frame,
            from_=0,
            to=255,
            number_of_steps=255,
            command=self.on_threshold_slider_change
        )
        self.threshold_slider.pack(fill="x", padx=10, pady=8)
        self.threshold_slider.set(128)

        ctk.CTkButton(
            threshold_frame,
            text="Binarize",
            command=self.apply_threshold,
            width=150
        ).pack(pady=(8, 10))

        # ------------------------------------------------------------------
        # Bottom row: Structuring Element + Morphological Operations side by side
        # ------------------------------------------------------------------
        bottom_row = ctk.CTkFrame(morphology_frame, fg_color="transparent")
        bottom_row.pack(fill="x", padx=5, pady=5)

        bottom_row.grid_columnconfigure(0, weight=1)
        bottom_row.grid_columnconfigure(1, weight=1)

        # ---------------- Structuring element section ----------------
        se_frame = ctk.CTkFrame(bottom_row)
        se_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)

        ctk.CTkLabel(
            se_frame,
            text="Structuring Element",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        ctk.CTkLabel(
            se_frame,
            text="SE Size",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.se_size_entry = ctk.CTkEntry(se_frame)
        self.se_size_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.se_size_entry.insert(0, "3")

        ctk.CTkLabel(
            se_frame,
            text="SE Shape",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.se_shape_menu = ctk.CTkOptionMenu(
            se_frame,
            values=["Square", "Cross"]
        )
        self.se_shape_menu.pack(padx=10, pady=(2, 6), fill="x")
        self.se_shape_menu.set("Square")

        ctk.CTkLabel(
            se_frame,
            text="Size must be odd: 3, 5, 7...",
            font=ctk.CTkFont(size=10),
            text_color="#999"
        ).pack(anchor="w", padx=10, pady=(0, 8))

        # ---------------- Morphology operations section ----------------
        operations_frame = ctk.CTkFrame(bottom_row)
        operations_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 0), pady=5)

        ctk.CTkLabel(
            operations_frame,
            text="Morphological Operations",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        buttons_row_1 = ctk.CTkFrame(operations_frame, fg_color="transparent")
        buttons_row_1.pack(pady=3)

        ctk.CTkButton(
            buttons_row_1,
            text="Erosion",
            command=self.apply_erosion,
            width=120
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            buttons_row_1,
            text="Dilation",
            command=self.apply_dilation,
            width=120
        ).pack(side="left", padx=3)

        buttons_row_2 = ctk.CTkFrame(operations_frame, fg_color="transparent")
        buttons_row_2.pack(pady=3)

        ctk.CTkButton(
            buttons_row_2,
            text="Opening",
            command=self.apply_opening,
            width=120
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            buttons_row_2,
            text="Closing",
            command=self.apply_closing,
            width=120
        ).pack(side="left", padx=3)

        ctk.CTkButton(
            operations_frame,
            text="Boundary Extraction",
            command=self.apply_boundary_extraction,
            width=250,
            fg_color="#5b3f8c",
            hover_color="#432d69"
        ).pack(pady=(6, 10))

        ctk.CTkLabel(
            operations_frame,
            text="Tip: Binarize first. Morphology uses the current processed image.",
            font=ctk.CTkFont(size=10),
            text_color="#999",
            wraplength=260
        ).pack(pady=(0, 8))           

    def build_noise_roi_controls(self, parent):
        noise_roi_frame = ctk.CTkFrame(parent, fg_color="transparent")
        noise_roi_frame.pack(fill="x", padx=8, pady=5)

        # ---------------- Noise section ----------------
        noise_frame = ctk.CTkFrame(noise_roi_frame)
        noise_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            noise_frame,
            text="Noise Modeling",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        ctk.CTkLabel(
            noise_frame,
            text="Gaussian Std Dev",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.gaussian_std_entry = ctk.CTkEntry(noise_frame)
        self.gaussian_std_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.gaussian_std_entry.insert(0, "25")

        ctk.CTkLabel(
            noise_frame,
            text="Uniform Range (±)",
            font=ctk.CTkFont(size=11)
        ).pack(anchor="w", padx=10)

        self.uniform_range_entry = ctk.CTkEntry(noise_frame)
        self.uniform_range_entry.pack(padx=10, pady=(2, 6), fill="x")
        self.uniform_range_entry.insert(0, "50")

        noise_buttons = ctk.CTkFrame(noise_frame, fg_color="transparent")
        noise_buttons.pack(pady=(4, 10))

        ctk.CTkButton(
            noise_buttons,
            text="Add Gaussian",
            command=self.apply_gaussian_noise,
            width=120
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            noise_buttons,
            text="Add Uniform",
            command=self.apply_uniform_noise,
            width=120
        ).pack(side="left", padx=4)

        # ---------------- ROI section ----------------
        roi_frame = ctk.CTkFrame(noise_roi_frame)
        roi_frame.pack(fill="x", padx=5, pady=5)

        ctk.CTkLabel(
            roi_frame,
            text="ROI Analysis",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=(8, 6))

        ctk.CTkLabel(
            roi_frame,
            text="Draw ROI: click and drag on the processed image",
            font=ctk.CTkFont(size=11),
            text_color="#aaa",
            wraplength=260
        ).pack(padx=10, pady=(2, 4))

        self.roi_info_label = ctk.CTkLabel(
            roi_frame,
            text="No ROI selected",
            font=ctk.CTkFont(size=11),
            text_color="#aaa"
        )
        self.roi_info_label.pack(padx=10, pady=2)

        ctk.CTkButton(
            roi_frame,
            text="Isolate ROI",
            command=self.isolate_roi,
            width=180
        ).pack(pady=(8, 4))

        ctk.CTkButton(
            roi_frame,
            text="Show ROI Statistics",
            command=self.show_roi_stats,
            width=180,
            fg_color="#1a5c3a",
            hover_color="#134a2e"
        ).pack(pady=(0, 10))

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
    
    def build_frequency_tab(self):
        tab= self.tab_view.tab("Frequency Domain")
        
        #main controller
        main_frame = ctk.CTkFrame(tab)
        main_frame.pack(fill="both", expand=True)

        #Left: Spectrum viewer
        left_frame = ctk.CTkFrame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True)

        #FFT label
        ctk.CTkLabel(
            left_frame,
            text="Magnitude Spectrum",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=5)

        #FFT viewer
        self.fft_viewer = FFTViewer(left_frame, self.on_fft_click)
        self.fft_viewer.pack(fill="both", expand=True, padx=10, pady=10)

        #RIGHT: Notch filter controls
        right_frame = ctk.CTkScrollableFrame(main_frame, width=420)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        #Reconstructed image label
        ctk.CTkLabel(
            right_frame,
            text="Reconstructed Image",
            font=ctk.CTkFont(size=13, weight="bold")
        ).pack(pady=5)

        #Reconstructed image viewer
        self.frequency_result_view = ScrollableImageView(right_frame,width=380, height=300)
        self.frequency_result_view.pack(fill="both", expand=True, padx=10, pady=10)

        #Notch Panel
        self.notch_panel = NotchPanel(right_frame, self.apply_notch_filter_gui, self.clear_notch_points)
        self.notch_panel.pack(fill="x", padx=10, pady=10)
    
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

    def on_threshold_slider_change(self, value):
        threshold_value = int(float(value))
        self.threshold_value_label.configure(
            text=f"Threshold Value: {threshold_value}"
        )


    def get_threshold_value(self):
        return int(float(self.threshold_slider.get()))


    def get_structuring_element_from_gui(self):
        se_size = self.get_valid_odd_integer(
            self.se_size_entry,
            "Structuring element size"
        )

        se_shape = self.se_shape_menu.get()

        return create_structuring_element(se_size, se_shape), se_size, se_shape

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

    def apply_morphology_operation(self, operation_function, operation_name):
        """
        Central function for morphology operations.

        Morphology should work on the currently displayed processed image,
        not necessarily on the original image.

        This avoids the problem where:
            Binarize -> Erosion

        would fail when pipeline mode is OFF.

        Unlike general enhancement operations, morphology is naturally sequential:
        thresholding creates a binary mask, then erosion/dilation/opening/closing
        should work on that binary mask.
        """
        if not self.pipeline.has_image():
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            input_image = self.pipeline.get_current()

            if input_image is None:
                messagebox.showwarning("No Image", "Please load an image first.")
                return

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
                self.fft_shifted = compute_fft(self.current_original)

                spectrum = magnitude_spectrum(self.fft_shifted)

                self.notch_points = []

                if hasattr(self, "fft_viewer"):
                   self.fft_viewer.set_spectrum(spectrum)


                self.current_processed = pixel_array.copy()

                self.pipeline.set_original(pixel_array)
                self.update_pipeline_log()

                self.show_fit_image(self.original_image_view, self.current_original)
                self.show_fit_image(self.processed_image_view, self.current_processed)

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
                self.fft_shifted = compute_fft(self.current_original)
                spectrum = magnitude_spectrum(self.fft_shifted)
                self.notch_points = []
                if hasattr(self, "fft_viewer"):
                    self.fft_viewer.set_spectrum(spectrum)
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

    def _tm_load_target_image(self):
        file_path = filedialog.askopenfilename(
            title="Choose a Target Image for Template Matching",
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
                pixel_array, _ = load_dicom_image(file_path)
                target_image = pixel_array
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.bmp')):
                target_image = load_regular_image(file_path)
            else:
                messagebox.showerror("Unsupported Format", "Please use JPEG, BMP, or DICOM files only.")
                return

            self._tm_target_image = target_image
            self.tab_view.set("Template Matching")
            self._tm_redraw_crop_canvas()
            self._tm_result_canvas.delete("all")
            self._tm_corr_canvas.delete("all")
            self._tm_result_label.configure(text="Target image loaded. Turn on 'Use target image' if you want to compare against it.")
            self.status_label.configure(text="Target image loaded for template matching")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load target image.\nReason: {str(e)}")

    def _tm_on_use_target_toggle(self):
        self._tm_use_target_compare = bool(self._tm_use_target_toggle.get())
        if self._tm_use_target_compare and self._tm_target_image is None:
            messagebox.showinfo(
                "Target Image Missing",
                "Load a target image first, or turn the toggle off to compare against the same source image."
            )
            try:
                self._tm_use_target_toggle.deselect()
            except Exception:
                pass
            self._tm_use_target_compare = False

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

    def apply_threshold(self):
        try:
            threshold_value = self.get_threshold_value()

            self.apply_morphology_operation(
                lambda img: global_threshold(img, threshold_value),
                f"Global Thresholding (T={threshold_value})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_erosion(self):
        try:
            se, se_size, se_shape = self.get_structuring_element_from_gui()

            self.apply_morphology_operation(
                lambda img: erode(img, se),
                f"Erosion ({se_shape}, {se_size}x{se_size})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_dilation(self):
        try:
            se, se_size, se_shape = self.get_structuring_element_from_gui()

            self.apply_morphology_operation(
                lambda img: dilate(img, se),
                f"Dilation ({se_shape}, {se_size}x{se_size})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_opening(self):
        try:
            se, se_size, se_shape = self.get_structuring_element_from_gui()

            self.apply_morphology_operation(
                lambda img: opening(img, se),
                f"Opening ({se_shape}, {se_size}x{se_size})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_closing(self):
        try:
            se, se_size, se_shape = self.get_structuring_element_from_gui()

            self.apply_morphology_operation(
                lambda img: closing(img, se),
                f"Closing ({se_shape}, {se_size}x{se_size})"
            )

        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))


    def apply_boundary_extraction(self):
        try:
            se, se_size, se_shape = self.get_structuring_element_from_gui()

            self.apply_morphology_operation(
                lambda img: boundary_extraction(img, se),
                f"Boundary Extraction ({se_shape}, {se_size}x{se_size})"
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

    # ======================================================================
    #   TEMPLATE MATCHING TAB
    # ======================================================================

    def build_template_matching_tab(self):
        """
        Layout of the Template Matching tab
        ────────────────────────────────────
        Left half  → interactive crop canvas (rubber-band selection on the image)
        Right half → result viewer (original + bounding box) + correlation map
        """
        tab = self.tab_view.tab("Template Matching")

        # ── outer two-column frame ───────────────────────────────────────────────
        outer = ctk.CTkFrame(tab)
        outer.pack(fill="both", expand=True, padx=8, pady=8)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # ── LEFT — crop panel ───────────────────────────────────────────────────
        left_panel = ctk.CTkFrame(outer)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left_panel.rowconfigure(1, weight=1)
        left_panel.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left_panel, text="Step 1 — Draw crop rectangle on the image",
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, columnspan=2, pady=(6, 2), padx=6, sticky="w")

        # Canvas that shows the image and lets user draw a rectangle
        self._tm_crop_canvas = tk.Canvas(left_panel, bg="#2b2b2b", highlightthickness=0)
        self._tm_crop_canvas.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=6, pady=4)

        self._tm_crop_canvas.bind("<ButtonPress-1>",   self._tm_on_press)
        self._tm_crop_canvas.bind("<B1-Motion>",        self._tm_on_drag)
        self._tm_crop_canvas.bind("<ButtonRelease-1>",  self._tm_on_release)
        self._tm_crop_canvas.bind("<Configure>",        self._tm_redraw_crop_canvas)

        # Template preview strip
        ctk.CTkLabel(
            left_panel, text="Cropped template:",
            font=ctk.CTkFont(size=11), text_color="#aaa"
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(4, 0))

        self._tm_template_preview = tk.Canvas(
            left_panel, bg="#1e1e1e", height=70, highlightthickness=1,
            highlightbackground="#555"
        )
        self._tm_template_preview.grid(row=3, column=0, columnspan=2,
                                       sticky="ew", padx=6, pady=(0, 4))

        self._tm_info_label = ctk.CTkLabel(
            left_panel, text="No template selected yet.",
            font=ctk.CTkFont(size=11), text_color="#888"
        )
        self._tm_info_label.grid(row=4, column=0, columnspan=2,
                                  sticky="w", padx=8, pady=(0, 2))

        # Action buttons
        btn_row = ctk.CTkFrame(left_panel, fg_color="transparent")
        btn_row.grid(row=5, column=0, columnspan=2, pady=6, padx=6, sticky="ew")

        ctk.CTkButton(
            btn_row, text="Run Cross-Correlation",
            command=self._tm_run,
            height=36, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color="#1a5276", hover_color="#154360"
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        ctk.CTkButton(
            btn_row, text="Load Target Image",
            command=self._tm_load_target_image,
            height=36, font=ctk.CTkFont(size=12),
            fg_color="#2d6a2d", hover_color="#1f4f1f"
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        self._tm_use_target_toggle = ctk.CTkSwitch(
            btn_row,
            text="Use target image",
            command=self._tm_on_use_target_toggle
        )
        self._tm_use_target_toggle.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            btn_row, text="Clear",
            command=self._tm_clear,
            height=36, font=ctk.CTkFont(size=12),
            fg_color="#555", hover_color="#444"
        ).pack(side="left", expand=True, fill="x")

        # ── RIGHT — result panel ─────────────────────────────────────────────────
        right_panel = ctk.CTkFrame(outer)
        right_panel.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right_panel.rowconfigure(1, weight=3)
        right_panel.rowconfigure(3, weight=2)
        right_panel.columnconfigure(0, weight=1)
        right_panel.columnconfigure(1, weight=0)

        ctk.CTkLabel(
            right_panel, text="Step 2 — Match result (bounding box in red)",
            font=ctk.CTkFont(size=12, weight="bold")
        ).grid(row=0, column=0, pady=(6, 2), padx=6, sticky="w")

        self._tm_result_canvas = tk.Canvas(right_panel, bg="#2b2b2b", highlightthickness=0)
        self._tm_result_canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)

        ctk.CTkLabel(
            right_panel, text="Correlation map (brighter = better match):",
            font=ctk.CTkFont(size=11), text_color="#aaa"
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(4, 0))
        # Toggle: colored heatmap vs grayscale
        self._tm_color_toggle = ctk.CTkSwitch(
            right_panel, text="Colored heatmap"
        )
        try:
            self._tm_color_toggle.select()
        except Exception:
            pass
        self._tm_color_toggle.grid(row=2, column=1, sticky="e", padx=(0, 8), pady=(4, 0))
        self._tm_corr_canvas = tk.Canvas(
            right_panel, bg="#1e1e1e", height=110, highlightthickness=1,
            highlightbackground="#555"
        )
        self._tm_corr_canvas.grid(row=3, column=0, sticky="nsew", padx=6, pady=(0, 6))

        self._tm_result_info_frame = ctk.CTkFrame(right_panel, fg_color="transparent")
        self._tm_result_info_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 6))
        self._tm_result_info_frame.columnconfigure(0, weight=1)

        self._tm_result_label = ctk.CTkLabel(
            self._tm_result_info_frame,
            text="Run matching to see results here.",
            font=ctk.CTkFont(size=11),
            text_color="#888",
            wraplength=420,
            justify="left"
        )
        self._tm_result_label.grid(row=0, column=0, sticky="w")

        self._tm_score_label = ctk.CTkLabel(
            self._tm_result_info_frame,
            text="Best NCC Score: N/A",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#b7f3b7"
        )
        self._tm_score_label.grid(row=1, column=0, sticky="w", pady=(2, 0))

    def _tm_redraw_crop_canvas(self, event=None):
        """Redraw the crop canvas whenever it is resized or a new image is loaded."""
        canvas = self._tm_crop_canvas
        canvas.delete("all")

        if self.current_original is None:
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw > 1:
                canvas.create_text(
                    cw // 2, ch // 2,
                    text="Load an image first, then draw a crop rectangle here.",
                    fill="#aaaaaa", font=("Segoe UI", 11, "italic"),
                    width=cw - 20
                )
            return

        result = _tm_draw_on_canvas(canvas, self.current_original)
        if result is None:
            canvas.after(50, self._tm_redraw_crop_canvas)
            return

        photo, scale, ox, oy = result
        self._tm_photo_crop = photo        # keep reference
        self._tm_scale    = scale
        self._tm_offset_x = ox
        self._tm_offset_y = oy
        canvas.create_image(ox, oy, anchor="nw", image=self._tm_photo_crop)
        self._tm_rect_id  = None

    def _tm_on_press(self, event):
        """Record start of rubber-band rectangle."""
        if self.current_original is None:
            return
        self._tm_crop_start = (event.x, event.y)
        if self._tm_rect_id is not None:
            self._tm_crop_canvas.delete(self._tm_rect_id)
            self._tm_rect_id = None

    def _tm_on_drag(self, event):
        """Stretch the rubber-band rectangle while dragging."""
        if self._tm_crop_start is None:
            return
        if self._tm_rect_id is not None:
            self._tm_crop_canvas.delete(self._tm_rect_id)
        x0, y0 = self._tm_crop_start
        self._tm_rect_id = self._tm_crop_canvas.create_rectangle(
            x0, y0, event.x, event.y,
            outline="#00FF88", width=2, dash=(4, 2)
        )

    def _tm_on_release(self, event):
        """
        On mouse release: convert canvas coords → original image coords, crop template.
        """
        if self._tm_crop_start is None or self.current_original is None:
            return

        x0, y0 = self._tm_crop_start
        x1, y1 = event.x, event.y
        self._tm_crop_start = None

        # Ensure x0 < x1, y0 < y1
        cx0, cy0 = min(x0, x1), min(y0, y1)
        cx1, cy1 = max(x0, x1), max(y0, y1)

        # Map canvas coords back to original image coords
        ox, oy = self._tm_offset_x, self._tm_offset_y
        scale  = self._tm_scale
        ih, iw = self.current_original.shape[:2]

        img_x0 = int(np.clip((cx0 - ox) / scale, 0, iw - 1))
        img_y0 = int(np.clip((cy0 - oy) / scale, 0, ih - 1))
        img_x1 = int(np.clip((cx1 - ox) / scale, 0, iw))
        img_y1 = int(np.clip((cy1 - oy) / scale, 0, ih))

        if img_x1 - img_x0 < 2 or img_y1 - img_y0 < 2:
            messagebox.showwarning(
                "Template Too Small",
                "The selected region is too small.\nPlease drag a larger rectangle."
            )
            return

        self._tm_template = self.current_original[img_y0:img_y1, img_x0:img_x1]

        # Show template preview
        th, tw = self._tm_template.shape[:2]
        self._tm_info_label.configure(
            text=f"Template: {tw} × {th} px  (drag area in image coords: "
                 f"x={img_x0}–{img_x1}, y={img_y0}–{img_y1})"
        )

        # Fit template into preview strip
        prev_canvas = self._tm_template_preview
        pw = prev_canvas.winfo_width()
        ph = prev_canvas.winfo_height()
        if pw <= 1:
            pw, ph = 200, 70

        t_scale = min(pw / tw, ph / th)
        dw, dh  = max(1, int(tw * t_scale)), max(1, int(th * t_scale))
        if self._tm_template.ndim == 2:
            pil_t = Image.fromarray(
                np.clip(self._tm_template, 0, 255).astype(np.uint8), mode="L"
            )
        else:
            pil_t = Image.fromarray(
                np.clip(self._tm_template, 0, 255).astype(np.uint8)
            )
        pil_t = pil_t.resize((dw, dh), Image.LANCZOS)
        self._tm_photo_template = ImageTk.PhotoImage(pil_t)
        prev_canvas.delete("all")
        prev_canvas.create_image(pw // 2, ph // 2, anchor="center",
                                  image=self._tm_photo_template)

    def _tm_run(self):
        """Run Fourier cross-correlation and display result + correlation map."""
        if self._tm_template is None:
            messagebox.showwarning(
                "No Template",
                "Please draw a crop rectangle on the image to select a template."
            )
            return

        target_image = self.current_original
        target_label = "same source image"
        if self._tm_use_target_compare:
            if self._tm_target_image is None:
                messagebox.showwarning(
                    "No Target Image",
                    "Turn off 'Use target image' or load a target image first."
                )
                return
            target_image = self._tm_target_image
            target_label = "separate target image"

        if target_image is None:
            messagebox.showwarning("No Image", "Please load an image first.")
            return

        try:
            self.status_label.configure(text="Running cross-correlation…")
            self._tm_result_label.configure(text="Computing…")
            self.app.update_idletasks()

            # Use normalized cross-correlation for robustness on real images
            result_img, norm_corr, (pr, pc), (th, tw) = fourier_cross_correlate_normalized(
                target_image, self._tm_template
            )

            # ── Draw result image ────────────────────────────────────────────────
            res_canvas = self._tm_result_canvas
            res_canvas.delete("all")
            out = _tm_draw_on_canvas(res_canvas, result_img)
            if out is not None:
                photo, _, _, _ = out
                self._tm_photo_result = photo
                res_canvas.create_image(0, 0, anchor="nw", image=self._tm_photo_result)
                # Re-draw properly centred
                cw = res_canvas.winfo_width()
                ch = res_canvas.winfo_height()
                ih2, iw2 = result_img.shape[:2]
                s = min(cw / iw2, ch / ih2)
                dw2, dh2 = int(iw2 * s), int(ih2 * s)
                ox2, oy2 = (cw - dw2) // 2, (ch - dh2) // 2
                pil_r = Image.fromarray(result_img).resize((dw2, dh2), Image.LANCZOS)
                self._tm_photo_result = ImageTk.PhotoImage(pil_r)
                res_canvas.delete("all")
                res_canvas.create_image(ox2, oy2, anchor="nw",
                                        image=self._tm_photo_result)

            # ── Draw correlation map
            corr_canvas = self._tm_corr_canvas
            corr_canvas.delete("all")
            # Valid region size: out_h = ih - th + 1, out_w = iw - tw + 1
            ih_full, iw_full = target_image.shape[:2]
            out_h = max(1, ih_full - th + 1)
            out_w = max(1, iw_full - tw + 1)
            corr_valid = norm_corr[:out_h, :out_w]
            # Best (maximum) normalized cross-correlation score in the valid region
            try:
                max_score = float(np.max(corr_valid))
            except Exception:
                max_score = float('nan')
            # Create a colored heatmap (RGB) or grayscale depending on toggle.
            colored = True
            try:
                colored = bool(self._tm_color_toggle.get())
            except Exception:
                colored = True

            # Normalize corr_valid first.
            minv = float(np.min(corr_valid))
            maxv = float(np.max(corr_valid))
            if maxv - minv > 0:
                norm = (corr_valid - minv) / (maxv - minv)
            else:
                norm = np.zeros_like(corr_valid)
            if colored:
                # Try matplotlib colormap first; fall back to a simple jet-like numpy map.
                try:
                    import matplotlib.cm as cm
                    cmap = cm.get_cmap('viridis')
                    rgba = cmap(norm)
                    rgb = (rgba[..., :3] * 255).astype(np.uint8)
                except Exception:
                    v = norm
                    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
                    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
                    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
                    rgb = np.stack([r, g, b], axis=-1)
                    rgb = (rgb * 255).astype(np.uint8)

                corr_out = _tm_draw_on_canvas(corr_canvas, rgb)
                if corr_out is not None:
                    _, sc, ocx, ocy = corr_out
                    ccw = corr_canvas.winfo_width()
                    cch = corr_canvas.winfo_height()
                    ch_h, ch_w = rgb.shape[:2]
                    sc2 = min(ccw / ch_w, cch / ch_h)
                    cdw, cdh = int(ch_w * sc2), int(ch_h * sc2)
                    cocx, cocy = (ccw - cdw) // 2, (cch - cdh) // 2
                    pil_c = Image.fromarray(rgb, mode="RGB").resize((cdw, cdh), Image.LANCZOS)
                    self._tm_photo_corr = ImageTk.PhotoImage(pil_c)
                    corr_canvas.delete("all")
                    corr_canvas.create_image(cocx, cocy, anchor="nw", image=self._tm_photo_corr)
                    # Draw peak marker on the displayed valid-region
                    try:
                        disp_x = cocx + int(pc * sc2)
                        disp_y = cocy + int(pr * sc2)
                        size = max(3, int(4 * sc2))
                        corr_canvas.create_line(disp_x - size, disp_y, disp_x + size, disp_y, fill='lime', width=2)
                        corr_canvas.create_line(disp_x, disp_y - size, disp_x, disp_y + size, fill='lime', width=2)
                    except Exception:
                        pass
            else:
                corr_display = (norm * 255).astype(np.uint8)
                corr_out = _tm_draw_on_canvas(corr_canvas, corr_display)
                if corr_out is not None:
                    _, sc, ocx, ocy = corr_out
                    ccw = corr_canvas.winfo_width()
                    cch = corr_canvas.winfo_height()
                    ch_h, ch_w = corr_display.shape
                    sc2 = min(ccw / ch_w, cch / ch_h)
                    cdw, cdh = int(ch_w * sc2), int(ch_h * sc2)
                    cocx, cocy = (ccw - cdw) // 2, (cch - cdh) // 2
                    pil_c = Image.fromarray(corr_display, mode="L").resize((cdw, cdh), Image.LANCZOS)
                    self._tm_photo_corr = ImageTk.PhotoImage(pil_c)
                    corr_canvas.delete("all")
                    corr_canvas.create_image(cocx, cocy, anchor="nw", image=self._tm_photo_corr)
                    try:
                        disp_x = cocx + int(pc * sc2)
                        disp_y = cocy + int(pr * sc2)
                        size = max(3, int(4 * sc2))
                        corr_canvas.create_line(disp_x - size, disp_y, disp_x + size, disp_y, fill='lime', width=2)
                        corr_canvas.create_line(disp_x, disp_y - size, disp_x, disp_y + size, fill='lime', width=2)
                    except Exception:
                        pass

            # Update info label (include best NCC score)
            score_text = f"Best NCC Score: {max_score:.2f}" if not np.isnan(max_score) else "Best NCC Score: N/A"
            self._tm_score_label.configure(text=score_text)
            self._tm_result_label.configure(
                text=(f"Match found at image row={pr}, col={pc}  |  "
                      f"Bounding box: top-left=({pc}, {pr}), "
                      f"bottom-right=({pc + tw}, {pr + th})  |  "
                      f"Target: {target_label}")
            )
            self.status_label.configure(text="Template matching complete")

        except ValueError as ve:
            messagebox.showerror("Input Error", str(ve))
            self.status_label.configure(text="Matching failed")
        except Exception as e:
            messagebox.showerror("Error", f"Cross-correlation failed:\n{str(e)}")
            self.status_label.configure(text="Matching failed")

    def _tm_clear(self):
        """Reset all template matching state."""
        self._tm_template   = None
        self._tm_crop_start = None
        self._tm_target_image = None
        self._tm_use_target_compare = False
        try:
            self._tm_use_target_toggle.deselect()
        except Exception:
            pass
        if self._tm_rect_id is not None:
            self._tm_crop_canvas.delete(self._tm_rect_id)
            self._tm_rect_id = None

        self._tm_redraw_crop_canvas()
        self._tm_template_preview.delete("all")
        self._tm_result_canvas.delete("all")
        self._tm_corr_canvas.delete("all")
        self._tm_info_label.configure(text="No template selected yet.")
        self._tm_result_label.configure(text="Load a target image only if you want to compare against it.")
        self.status_label.configure(text="Template matching cleared")
        self.status_label.configure(text="Reset to original")  

    def on_fft_click(self, u, v):
        self.notch_points.append((u, v))
        print ("Selected notch point:", u, v )

    def  apply_notch_filter_gui(self, ftype, radius, order):
        if self.fft_shifted is None:
            return
        
        current_image = self.pipeline.get_current()

        self.fft_shifted = compute_fft(current_image)

        filtered, mask= apply_notch_filter(
            self.fft_shifted,
            self.notch_points,
            filter_type=ftype,
            radius= radius,
            order= order
        )
        result = reconstruct_image(filtered)
        print(result.dtype)
        print(result.min(), result.max())
        print("Filtered image shape:", result.shape)

        self.current_processed = self.pipeline.apply_result(
            result,
            f"{ftype.capitalize()} Notch Filter"
        )

        self.show_fit_image(self.processed_image_view, result)

        self.update_pipeline_log()

        self.show_fit_image(self.frequency_result_view, result)
        print("Applying notch filter...")
        print("Points:", self.notch_points)
        print("Radius:", radius)
        print("Type:", ftype)

        self.status_label.configure(text=f"Applied {ftype} notch filter with radius={radius} and order={order}")

    def clear_notch_points(self):
        self.notch_points = []
        
        if hasattr(self,"fft_viewer"):
            self.fft_viewer.clear_points()

    # ── Noise ──────────────────────────────────────────────────────────────

    def apply_gaussian_noise(self):
        try:
            std = float(self.gaussian_std_entry.get())
            if std <= 0:
                raise ValueError("Std dev must be greater than 0.")
            self.apply_pipeline_operation(
                lambda img: add_gaussian_noise(img, sigma=std),
                f"Gaussian Noise (std={std})"
            )
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    def apply_uniform_noise(self):
        try:
            r = float(self.uniform_range_entry.get())
            if r <= 0:
                raise ValueError("Range must be greater than 0.")
            self.apply_pipeline_operation(
                lambda img: add_uniform_noise(img, low=-r, high=r),
                f"Uniform Noise (range=±{r})"
            )
        except Exception as e:
            messagebox.showerror("Invalid Input", str(e))

    # ── ROI ────────────────────────────────────────────────────────────────

    def _roi_drag_start(self, event):
        self.roi_start = (event.x, event.y)
        self.roi_end   = None
        if self.roi_rect is not None:
            self.processed_image_view.canvas.delete(self.roi_rect)
            self.roi_rect = None

    def _roi_drag_move(self, event):
        if self.roi_start is None:
            return
        if self.roi_rect is not None:
            self.processed_image_view.canvas.delete(self.roi_rect)
        x1, y1 = self.roi_start
        self.roi_rect = self.processed_image_view.canvas.create_rectangle(
            x1, y1, event.x, event.y,
            outline="yellow", width=2, dash=(4, 2)
        )

    def _roi_drag_end(self, event):
        if self.roi_start is None:
            return
        self.roi_end = (event.x, event.y)
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        self.roi_info_label.configure(
            text=f"ROI: {w}×{h} px at ({min(x1,x2)}, {min(y1,y2)})"
        )

    def isolate_roi(self):
     if not self.pipeline.has_image():
        messagebox.showwarning("No Image", "Please load an image first.")
        return

     if self.roi_start is None or self.roi_end is None:
        messagebox.showwarning("No ROI", "Please draw an ROI on the processed image first.")
        return

     # Convert canvas coords → actual image pixel coords BEFORE clearing
     img_x1, img_y1 = self._canvas_to_image_coords(*self.roi_start)
     img_x2, img_y2 = self._canvas_to_image_coords(*self.roi_end)

     # Capture converted values into the lambda immediately (avoids reference bug)
     _x1, _y1, _x2, _y2 = img_x1, img_y1, img_x2, img_y2

     self.apply_pipeline_operation(
        lambda img: extract_roi(img, _x1, _y1, _x2, _y2),
        f"ROI Isolation ({abs(_x2-_x1)}×{abs(_y2-_y1)} px)"
     )

     # Clear the rectangle after isolating
     self.roi_start = None
     self.roi_end   = None
     if self.roi_rect is not None:
        self.processed_image_view.canvas.delete(self.roi_rect)
        self.roi_rect = None
     
     self.roi_info_label.configure(text="No ROI selected")
    
    def show_roi_stats(self):
     """
     Compute and display local histogram, mean, and variance
     for the currently drawn ROI — WITHOUT isolating it first.
     The user can draw the ROI, inspect stats, then decide to isolate.
     """
     if not self.pipeline.has_image():
      messagebox.showwarning("No Image", "Please load an image first.")
      return

     if self.roi_start is None or self.roi_end is None:
      messagebox.showwarning(
         "No ROI",
         "Please draw a rectangle on the processed image first,\n"
         "then click Show ROI Statistics."
      )
      return

     # Convert canvas coords → image pixel coords
     img_x1, img_y1 = self._canvas_to_image_coords(*self.roi_start)
     img_x2, img_y2 = self._canvas_to_image_coords(*self.roi_end)

     current_img = self.pipeline.get_current()
     if current_img is None:
        return
     # ──────────────────────────────────────────────────────────
     # Extract the ROI pixels from the current image
     from processing.roi.roi_tool import extract_roi  # ← WRONG (inside function)
     try:
         roi = extract_roi(current_img, img_x1, img_y1, img_x2, img_y2)
     except ValueError as e:
        messagebox.showwarning("ROI Too Small", str(e))
        return

     w = abs(img_x2 - img_x1)
     h = abs(img_y2 - img_y1)

     # Open the statistics popup window
     show_roi_statistics(roi, roi_label=f"{w}×{h} px")
     self.status_label.configure(text=f"ROI stats shown for {w}×{h} px region")


    def _canvas_to_image_coords(self, canvas_x, canvas_y):
   
      img = self.pipeline.get_current()
      if img is None:
        return canvas_x, canvas_y

      img_h, img_w = img.shape[:2]
      canvas_w = self.processed_image_view.canvas.winfo_width()
      canvas_h = self.processed_image_view.canvas.winfo_height()

      # This is the same scale factor used in update_display() fit mode
      scale = min(canvas_w / img_w, canvas_h / img_h)

      # The image is centred in the canvas — calculate the offset
      display_w = int(img_w * scale)
      display_h = int(img_h * scale)
      offset_x  = (canvas_w - display_w) // 2
      offset_y  = (canvas_h - display_h) // 2

      # Reverse the scale and offset
      img_x = int((canvas_x - offset_x) / scale)
      img_y = int((canvas_y - offset_y) / scale)

      # Clamp to image bounds
      img_x = max(0, min(img_x, img_w - 1))
      img_y = max(0, min(img_y, img_h - 1))

      return img_x, img_y
    # processing/roi/roi_stats_window.py


    def run(self):
        self.app.mainloop()
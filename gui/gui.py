import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tkinter as tk
import numpy as np
from image_loader import load_regular_image, load_dicom_image, get_dicom_tag
from metadata import build_metadata_text
from zoom import zoom_image

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

        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)

        
        self.bind("<Configure>", self._on_frame_configure)

        self._placeholder_text = "No image loaded"
        self.show_placeholder()

    def _on_frame_configure(self, event=None):
        """Re-draw when the frame is resized (only in fit mode)."""
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
            resized = pil_img.resize((new_w, new_h), Image.LANCZOS)
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

        self.app = ctk.CTk()
        self.app.title("Clinical Image Analysis Workbench")
        self.app.geometry("1100x700")
        self.app.minsize(900, 600)

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
        self.tab_view.add("Metadata")

        self.build_image_viewer_tab()
        self.build_metadata_tab()

    def build_image_viewer_tab(self):
        viewer_tab = self.tab_view.tab("Image Viewer")
        images_frame = ctk.CTkFrame(viewer_tab)
        images_frame.pack(fill="both", expand=True, padx=10, pady=10)

        original_frame = ctk.CTkFrame(images_frame)
        original_frame.pack(side="left", fill="both", expand=True, padx=5)
        ctk.CTkLabel(original_frame, text="Original Image",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)
        self.original_image_view = ScrollableImageView(original_frame, height=450, width=450)
        self.original_image_view.pack(fill="both", expand=True, padx=5, pady=5)

      
        processed_frame = ctk.CTkFrame(images_frame)
        processed_frame.pack(side="right", fill="both", expand=True, padx=5)
        ctk.CTkLabel(processed_frame, text="Processed Image",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)
        self.processed_image_view = ScrollableImageView(processed_frame, height=450, width=450)
        self.processed_image_view.pack(fill="both", expand=True, padx=5, pady=5)

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

    def show_metadata_text(self, text):
        self.metadata_box.configure(state="normal")
        self.metadata_box.delete("1.0", "end")
        self.metadata_box.insert("1.0", text)
        self.metadata_box.configure(state="disabled")
        self.tab_view.set("Metadata")

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
                self.current_processed = image_array.copy()

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
            self.status_label.configure(text="Image loaded")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load image.\nReason: {str(e)}")
            self.status_label.configure(text="Error loading image")

    def save_image(self):
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
        if self.current_original is None:
            messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
            return

        method = "nearest" if self.zoom_method.get() == "Nearest Neighbor" else "bilinear"

        try:
            zoomed = zoom_image(self.current_original, self.zoom_factor, method)
            self.current_processed = zoomed
            self.show_actual_image(self.processed_image_view, self.current_processed)
            self.status_label.configure(
                text=f"Zoom {int(self.zoom_factor * 100)}% ({method})"
            )
        except Exception as e:
            messagebox.showerror("Zoom Error", f"Could not apply zoom.\nReason: {str(e)}")

    def zoom_in(self):
        if self.current_original is None:
            messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
            return
        self.zoom_factor = min(round(self.zoom_factor + 0.25, 2), 4.0)   # max 400%
        self.zoom_label.configure(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        self.apply_zoom()

    def zoom_out(self):
        if self.current_original is None:
            messagebox.showwarning("Warning", "No image loaded. Please load an image first.")
            return
        self.zoom_factor = max(round(self.zoom_factor - 0.25, 2), 0.25)   # min 25%
        self.zoom_label.configure(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        self.apply_zoom()

    def reset_zoom(self):
        if self.current_original is None:
            return
        self.zoom_factor = 1.0
        self.zoom_label.configure(text="Zoom: 100%")
        self.current_processed = self.current_original.copy()
        self.show_fit_image(self.processed_image_view, self.current_processed)
        self.status_label.configure(text="Zoom reset")

    def run(self):
        self.app.mainloop()
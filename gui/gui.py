import customtkinter as ctk
from tkinter import filedialog, messagebox, Canvas, Scrollbar
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
from image_loader import load_regular_image, load_dicom_image, get_dicom_tag
from metadata import build_metadata_text
from zoom import zoom_image

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ScrollableImageCanvas(ctk.CTkFrame):
  

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

    
        self.canvas = Canvas(
            self,
            bg="#2b2b2b",
            highlightthickness=0,
            cursor="fleur"          
        )

        self.h_scroll = Scrollbar(self, orient="horizontal",
                                  command=self.canvas.xview)
        self.v_scroll = Scrollbar(self, orient="vertical",
                                  command=self.canvas.yview)

        self.canvas.configure(
            xscrollcommand=self.h_scroll.set,
            yscrollcommand=self.v_scroll.set
        )

       
        self.h_scroll.pack(side="bottom", fill="x")
        self.v_scroll.pack(side="right",  fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

    
        self.canvas.bind("<ButtonPress-1>",   self._on_drag_start)
        self.canvas.bind("<B1-Motion>",       self._on_drag_move)

        
        self.canvas.bind("<MouseWheel>",      self._on_mousewheel_vertical)
        self.canvas.bind("<Shift-MouseWheel>",self._on_mousewheel_horizontal)
        self.canvas.bind("<Button-4>",        self._on_mousewheel_vertical)
        self.canvas.bind("<Button-5>",        self._on_mousewheel_vertical)

        self._drag_start_x = 0
        self._drag_start_y = 0
        self._tk_image = None          
        self._placeholder_text = None  

        self._show_placeholder("No image loaded")


    def display_image(self, image_array):
       
        self.canvas.delete("all")

        pil_image = Image.fromarray(image_array)
        self._tk_image = ImageTk.PhotoImage(pil_image)  # must keep reference

        img_w, img_h = pil_image.size

      
        self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)

       
        self.canvas.configure(scrollregion=(0, 0, img_w, img_h))

        
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def display_fit(self, image_array, max_size=450):
    
        pil_image = Image.fromarray(image_array)
        pil_image.thumbnail((max_size, max_size))
        self._show_pil(pil_image, centre=True)

    def clear(self, message="No image loaded"):
        self.canvas.delete("all")
        self._tk_image = None
        self._show_placeholder(message)


    def _show_pil(self, pil_image, centre=False):
        self.canvas.delete("all")
        self._tk_image = ImageTk.PhotoImage(pil_image)
        img_w, img_h = pil_image.size

        if centre:
            cw = self.canvas.winfo_width()  or img_w
            ch = self.canvas.winfo_height() or img_h
            x = max(cw // 2, img_w // 2)
            y = max(ch // 2, img_h // 2)
            self.canvas.create_image(x, y, anchor="center", image=self._tk_image)
            self.canvas.configure(scrollregion=(0, 0, max(cw, img_w),
                                                     max(ch, img_h)))
        else:
            self.canvas.create_image(0, 0, anchor="nw", image=self._tk_image)
            self.canvas.configure(scrollregion=(0, 0, img_w, img_h))

        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)

    def _show_placeholder(self, text):
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, 1, 1))
        # Draw placeholder text in the middle of whatever space is available
        self.canvas.update_idletasks()
        w = self.canvas.winfo_width()  or 400
        h = self.canvas.winfo_height() or 400
        self._placeholder_text = self.canvas.create_text(
            w // 2, h // 2,
            text=text,
            fill="#888888",
            font=("Arial", 14)
        )

    # ---------------------------------------------
    #  funcs 3shan at7rk elimage wana b3ml zoom 
    # ---------------------------------------------

    def _on_drag_start(self, event):
        self.canvas.scan_mark(event.x, event.y)
        self._drag_start_x = event.x
        self._drag_start_y = event.y

    def _on_drag_move(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_mousewheel_vertical(self, event):
        # Windows: event.delta; Linux: event.num
        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_horizontal(self, event):
        if event.num == 4:
            self.canvas.xview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.xview_scroll(1, "units")
        else:
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")


class MedicalImageApp:

    def __init__(self):
        self.current_original  = None
        self.current_processed = None
        self.zoom_factor = 1.0

        self.app = ctk.CTk()
        self.app.title("Clinical Image Analysis Workbench")
        self.app.geometry("1200x750")
        self.app.minsize(900, 600)

        self.build_layout()
        self.build_left_panel()
        self.build_tabs()

   
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
            width=180,
            font=ctk.CTkFont(size=12)
        )
        self.zoom_method.pack(pady=4, padx=10)
        self.zoom_method.set("Bilinear")

        self.zoom_label = ctk.CTkLabel(
            self.left_panel, text="Zoom: 100%",
            font=ctk.CTkFont(size=12)
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

        
        ctk.CTkFrame(self.left_panel, height=2, fg_color="#444").pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(
            self.left_panel,
            text="Drag or use scrollbars\nto pan a zoomed image.",
            font=ctk.CTkFont(size=10),
            text_color="#888",
            justify="center"
        ).pack(pady=4, padx=10)

        self.status_label = ctk.CTkLabel(
            self.left_panel, text="Ready",
            font=ctk.CTkFont(size=11), text_color="#888"
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

        self.original_canvas = ScrollableImageCanvas(original_frame)
        self.original_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        # --- Processed image panel ---
        processed_frame = ctk.CTkFrame(images_frame)
        processed_frame.pack(side="right", fill="both", expand=True, padx=5)

        ctk.CTkLabel(processed_frame, text="Processed Image",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=5)

        self.processed_canvas = ScrollableImageCanvas(processed_frame)
        self.processed_canvas.pack(fill="both", expand=True, padx=4, pady=(0, 4))

    def build_metadata_tab(self):
        meta_tab = self.tab_view.tab("Metadata")
        ctk.CTkLabel(meta_tab, text="Image Information",
                     font=ctk.CTkFont(size=16, weight="bold")).pack(pady=10)

        self.metadata_box = ctk.CTkTextbox(
            meta_tab, width=600, height=400,
            font=ctk.CTkFont(size=13)
        )
        self.metadata_box.pack(fill="both", expand=True, padx=20, pady=10)
        self.metadata_box.insert("1.0", "Load an image to see its information here.")
        self.metadata_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # function bts3d anna n display elimage 
    # ------------------------------------------------------------------

    def show_fit_image(self, canvas_widget, image_array):
        """Display image scaled to fit the panel (no scrollbars needed)."""
        try:
            canvas_widget.display_fit(image_array, max_size=450)
        except Exception as e:
            canvas_widget.clear(f"Cannot display: {e}")

    def show_actual_image(self, canvas_widget, image_array):
        """
        Display image at its true pixel size.
        Scrollbars activate automatically when the image overflows the panel.
        """
        try:
            canvas_widget.display_image(image_array)
        except Exception as e:
            canvas_widget.clear(f"Cannot display: {e}")

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
                ("JPEG Images",    "*.jpg *.jpeg"),
                ("BMP Images",     "*.bmp"),
                ("DICOM Images",   "*.dcm"),
                ("All Files",      "*.*")
            ]
        )
        if not file_path:
            return

        try:
            if file_path.lower().endswith('.dcm'):
                pixel_array, dicom_data = load_dicom_image(file_path)
                self.current_original  = pixel_array
                self.current_processed = pixel_array.copy()

                self.show_fit_image(self.original_canvas,  self.current_original)
                self.show_fit_image(self.processed_canvas, self.current_processed)

                height, width = pixel_array.shape
                text = build_metadata_text(
                    file_name    = file_path.split("/")[-1],
                    width        = width,
                    height       = height,
                    bit_depth    = get_dicom_tag(dicom_data, "BitsAllocated"),
                    file_format  = "DICOM",
                    modality     = get_dicom_tag(dicom_data, "Modality"),
                    patient_name = get_dicom_tag(dicom_data, "PatientName"),
                    patient_age  = get_dicom_tag(dicom_data, "PatientAge"),
                    body_part    = get_dicom_tag(dicom_data, "BodyPartExamined")
                )
                self.show_metadata_text(text)

            elif file_path.lower().endswith(('.jpg', '.jpeg', '.bmp')):
                image_array = load_regular_image(file_path)
                self.current_original  = image_array
                self.current_processed = image_array.copy()

                self.show_fit_image(self.original_canvas,  self.current_original)
                self.show_fit_image(self.processed_canvas, self.current_processed)

                height, width = image_array.shape
                file_name = file_path.split("/")[-1]
                text = build_metadata_text(
                    file_name   = file_name,
                    width       = width,
                    height      = height,
                    bit_depth   = 8,
                    file_format = file_name.split(".")[-1].upper()
                )
                self.show_metadata_text(text)

            else:
                messagebox.showerror(
                    "Unsupported Format",
                    "Please use JPEG, BMP, or DICOM files only."
                )
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
                ("BMP Image",  "*.bmp"),
                ("PNG Image",  "*.png")
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

    # ---------------
    # zooming part 
    # ---------------

    def apply_zoom(self):
        if self.current_original is None:
            messagebox.showwarning("Warning", "No image — please load an image first.")
            return

        method = "nearest" if self.zoom_method.get() == "Nearest Neighbor" else "bilinear"
        try:
            zoomed = zoom_image(self.current_original, self.zoom_factor, method)
            # Use show_actual_image so scrollbars activate for large zooms
            self.show_actual_image(self.processed_canvas, zoomed)
            self.status_label.configure(
                text=f"Zoom {int(self.zoom_factor * 100)}% ({method}) — drag or scroll to pan"
            )
        except Exception as e:
            messagebox.showerror("Zoom Error", f"Could not apply zoom.\nReason: {str(e)}")

    def zoom_in(self):
        if self.current_original is None:
            messagebox.showwarning("Warning", "No image — please load an image first.")
            return
        self.zoom_factor = min(round(self.zoom_factor + 0.25, 2), 4.0)
        self.zoom_label.configure(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        self.apply_zoom()

    def zoom_out(self):
        if self.current_original is None:
            messagebox.showwarning("Warning", "No image — please load an image first.")
            return
        self.zoom_factor = max(round(self.zoom_factor - 0.25, 2), 0.25)
        self.zoom_label.configure(text=f"Zoom: {int(self.zoom_factor * 100)}%")
        self.apply_zoom()

    def reset_zoom(self):
        if self.current_original is None:
            return
        self.zoom_factor = 1.0
        self.zoom_label.configure(text="Zoom: 100%")
        self.show_fit_image(self.processed_canvas, self.current_original)
        self.status_label.configure(text="Zoom reset")

    def run(self):
        self.app.mainloop()
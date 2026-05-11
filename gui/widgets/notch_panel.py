import customtkinter as ctk


class NotchPanel(ctk.CTkFrame):
    """
    UI controls for notch filtering.
    """

    def __init__(self, master, apply_callback, clear_callback):
        super().__init__(master)

        self.apply_callback = apply_callback
        self.clear_callback = clear_callback

        # ---------------- filter type ----------------
        ctk.CTkLabel(self, text="Filter Type").pack(pady=(10, 2))

        self.filter_type = ctk.CTkOptionMenu(
            self,
            values=["ideal", "butterworth", "gaussian"]
        )
        self.filter_type.set("ideal")
        self.filter_type.pack(pady=5)

        # ---------------- radius ----------------
        ctk.CTkLabel(self, text="Radius").pack(pady=(10, 2))

        self.radius_entry = ctk.CTkEntry(self)
        self.radius_entry.insert(0, "10")
        self.radius_entry.pack(pady=5)

        # ---------------- order ----------------
        ctk.CTkLabel(self, text="Butterworth Order").pack(pady=(10, 2))

        self.order_entry = ctk.CTkEntry(self)
        self.order_entry.insert(0, "2")
        self.order_entry.pack(pady=5)

        # ---------------- buttons ----------------
        ctk.CTkButton(
            self,
            text="Apply Notch Filter",
            command=self.apply
        ).pack(pady=(15, 5))

        ctk.CTkButton(
            self,
            text="Clear Points",
            fg_color="#555",
            command=self.clear_callback
        ).pack(pady=5)

    # ---------------------------
    # trigger apply
    # ---------------------------
    def apply(self):
        try:
            radius = int(self.radius_entry.get())
            order = int(self.order_entry.get())

            self.apply_callback(
                self.filter_type.get(),
                radius,
                order
            )

        except Exception as e:
            print("Invalid input:", e)
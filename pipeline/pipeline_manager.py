import numpy as np


class PipelineManager:
    """
    Manages the sequential enhancement pipeline.

    Responsibilities:
    - Store the original image.
    - Store the current processed image.
    - Keep a history stack for undo.
    - Decide whether operations are applied on the original image
      or on the latest processed result.
    """

    def __init__(self):
        self.original_image = None
        self.current_image = None
        self.history = []
        self.operation_log = []

    def set_original(self, image):
        """
        Called whenever a new image is loaded.
        Resets the full pipeline.
        """
        if image is None:
            self.original_image = None
            self.current_image = None
            self.history = []
            self.operation_log = []
            return

        self.original_image = image.copy()
        self.current_image = image.copy()
        self.history = []
        self.operation_log = []

    def has_image(self):
        return self.original_image is not None

    def get_input_image(self, pipeline_enabled):
        """
        If pipeline mode is ON:
            apply next operation on the current processed image.

        If pipeline mode is OFF:
            apply next operation on the original image.
        """
        if self.original_image is None:
            return None

        if pipeline_enabled:
            return self.current_image.copy()

        return self.original_image.copy()

    def apply_result(self, result_image, operation_name):
        """
        Save the previous current image before replacing it.
        This makes undo possible.
        """
        if result_image is None:
            raise ValueError("Operation returned no image.")

        if self.current_image is not None:
            self.history.append(self.current_image.copy())

        result_image = np.clip(result_image, 0, 255).astype(np.uint8)

        self.current_image = result_image
        self.operation_log.append(operation_name)

        return self.current_image

    def undo(self):
        """
        Revert to the previous processed image.
        """
        if len(self.history) == 0:
            return self.current_image, None

        self.current_image = self.history.pop()

        removed_operation = None
        if len(self.operation_log) > 0:
            removed_operation = self.operation_log.pop()

        return self.current_image, removed_operation

    def reset(self):
        """
        Reset processed image back to original.
        """
        if self.original_image is None:
            return None

        self.current_image = self.original_image.copy()
        self.history = []
        self.operation_log = []

        return self.current_image

    def get_current(self):
        if self.current_image is None:
            return None
        return self.current_image.copy()

    def get_log_text(self):
        if len(self.operation_log) == 0:
            return "No operations applied yet."

        text = ""
        for index, operation in enumerate(self.operation_log, start=1):
            text += f"{index}. {operation}\n"

        return text
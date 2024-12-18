import cv2
import numpy as np
from tkinter import Tk, Label, Button, Scale, filedialog, HORIZONTAL, Canvas, Frame, Toplevel
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class ModernImageAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Custom Image Processing Techniques")
        self.master.geometry("900x700")
        self.master.configure(bg="#f0f0f5")

        self.image = None
        self.processed_image = None

        # Header Label
        self.header_label = Label(master, text="Custom Image Processing Techniques", font=("Helvetica", 18, "bold"), bg="#f0f0f5", fg="#333")
        self.header_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Left Column - Controls
        self.left_frame = Frame(master, bg="#f0f0f5")
        self.left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        # Upload Button
        self.upload_btn = Button(self.left_frame, text="Upload Image", command=self.upload_image, bg="#4CAF50", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        self.upload_btn.grid(row=1, column=0, pady=10)

        # Sliders for brightness, contrast, rotation, and resize
        self.brightness_slider = Scale(self.left_frame, from_=-100, to=100, label="Brightness", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.brightness_slider.grid(row=2, column=0, pady=5)
        self.brightness_slider.bind("<Motion>", self.update_brightness)

        self.contrast_slider = Scale(self.left_frame, from_=0.5, to=3.0, resolution=0.1, label="Contrast", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.contrast_slider.grid(row=3, column=0, pady=5)
        self.contrast_slider.bind("<Motion>", self.update_contrast)

        self.rotation_slider = Scale(self.left_frame, from_=-180, to=180, label="Rotate", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.rotation_slider.grid(row=4, column=0, pady=10)
        self.rotation_slider.bind("<Motion>", self.rotate_image)

        self.resize_slider = Scale(self.left_frame, from_=10, to=100, resolution=1, label="Resize (%)", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.resize_slider.grid(row=5, column=0, pady=10)
        self.resize_slider.bind("<Motion>", self.resize_image_with_slider)

        # Create other buttons
        self.create_buttons()

        # Right Column - Image Display
        self.canvas_frame = Frame(master, bg="#d9d9d9")
        self.canvas_frame.grid(row=1, column=1, padx=10, pady=10)

        self.canvas = Canvas(self.canvas_frame, width=800, height=500, bg="#d9d9d9", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.grid(row=0, column=0)

    def create_buttons(self):
        button_specs = [
            ("Noise Reduction", self.noise_reduction, "#2196F3"),
            ("Normalization", self.normalization, "#FF9800"),
            ("Color Correction", self.color_correction, "#3F51B5"),
            ("Enhancement", self.image_enhancement, "#009688"),
            ("Edge Detection", self.detect_edges_advanced, "#FF5722"),
            ("Histogram Equalization", self.histogram_equalization, "#8BC34A"),
            ("Show Histogram Page", self.show_histogram_page, "#00BCD4")
        ]

        for idx, (text, command, color) in enumerate(button_specs, start=6):  # Start after sliders
            Button(
                self.left_frame, text=text, command=command, bg=color, fg="white", font=("Helvetica", 10),
                padx=10, pady=5
            ).grid(row=idx, column=0, pady=5)

    def upload_image(self):
        filepath = filedialog.askopenfilename()
        if filepath:
            self.image = cv2.imread(filepath)
            self.image = self.resize_to_fit(self.image, 800, 500)
            self.processed_image = self.image.copy()
            self.display_image(self.image)

    def resize_to_fit(self, image, max_width, max_height):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def resize_image_with_slider(self, event=None):
        if self.image is not None:
            resize_percent = self.resize_slider.get()
            scale = resize_percent / 100.0
            new_w = int(self.image.shape[1] * scale)
            new_h = int(self.image.shape[0] * scale)
            self.processed_image = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.display_image(self.processed_image)

    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.canvas.create_image(400, 250, image=img, anchor="center")
        self.canvas.image = img

    def show_histogram_page(self):
        histogram_window = Toplevel(self.master)
        histogram_window.title("Image Histogram")
        histogram_window.geometry("600x600")

        # Display Histogram on the new window
        if self.image is not None:
            self.display_histogram(histogram_window, self.image)

    def display_histogram(self, window, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
        ax.plot(hist)
        ax.set_title("Image Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")
        
        canvas_hist = FigureCanvasTkAgg(fig, master=window)
        canvas_hist.get_tk_widget().pack()
        canvas_hist.draw()

    def update_brightness(self, event=None):
        if self.image is not None:
            beta = self.brightness_slider.get()
            self.processed_image = cv2.convertScaleAbs(self.image, alpha=1, beta=beta)
            self.display_image(self.processed_image)

    def update_contrast(self, event=None):
        if self.image is not None:
            alpha = self.contrast_slider.get()
            self.processed_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)
            self.display_image(self.processed_image)

    def noise_reduction(self):
        if self.image is not None:
            self.processed_image = cv2.GaussianBlur(self.image, (5, 5), 0)
            self.display_image(self.processed_image)

    def normalization(self):
        if self.image is not None:
            self.processed_image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)
            self.display_image(self.processed_image)

    def image_resizing(self):
        if self.image is not None:
            self.processed_image = self.resize_to_fit(self.image, 800, 500)
            self.display_image(self.processed_image)

    def color_correction(self):
        if self.image is not None:
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = cv2.equalizeHist(l)
            self.processed_image = cv2.merge((l, a, b))
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_LAB2BGR)
            self.display_image(self.processed_image)

    def image_enhancement(self):
        if self.image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            self.processed_image = cv2.filter2D(self.image, -1, kernel)
            self.display_image(self.processed_image)

    def rotate_image(self, event=None):
        if self.image is not None:
            angle = self.rotation_slider.get()
            h, w = self.image.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])

            bound_w = int(h * abs_sin + w * abs_cos)
            bound_h = int(h * abs_cos + w * abs_sin)

            rotation_matrix[0, 2] += bound_w / 2 - center[0]
            rotation_matrix[1, 2] += bound_h / 2 - center[1]

            self.processed_image = cv2.warpAffine(self.image, rotation_matrix, (bound_w, bound_h))
            self.display_image(self.processed_image)

    def detect_edges_advanced(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            self.processed_image = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image)

    def histogram_equalization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            equalized_image = cv2.equalizeHist(gray_image)
            self.processed_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
            self.display_image(self.processed_image)


if __name__ == "__main__":
    root = Tk()
    gui = ModernImageAnalyzerGUI(root)
    root.mainloop()

import cv2  
import numpy as np  
from tkinter import Tk, Label, Button, Scale, filedialog, HORIZONTAL, Canvas, Frame, Toplevel  #GUI components
from PIL import Image, ImageTk  
import matplotlib.pyplot as plt  
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 

class ModernImageAnalyzerGUI:
    def __init__(self, master):
        self.master = master  # Main window of the GUI
        self.master.title("Custom Image Processing Techniques") 
        self.master.geometry("900x700")  # Set the window size
        self.master.configure(bg="#f0f0f5") 

        self.image = None  # Original image
        self.processed_image = None  # Processed image

        # Header Label for the title
        self.header_label = Label(master, text="Custom Image Processing Techniques", font=("Helvetica", 18, "bold"), bg="#f0f0f5", fg="#333")
        self.header_label.grid(row=0, column=0, columnspan=2, pady=10)

        # Left Column for controls
        self.left_frame = Frame(master, bg="#f0f0f5")  # Frame for control elements
        self.left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="n")

        # Upload Image Button
        self.upload_btn = Button(self.left_frame, text="Upload Image", command=self.upload_image, bg="#4CAF50", fg="white", font=("Helvetica", 12), padx=10, pady=5)
        self.upload_btn.grid(row=1, column=0, pady=10)

        # Sliders for adjusting brightness, contrast, rotation, and resizing
        self.brightness_slider = Scale(self.left_frame, from_=-100, to=100, label="Brightness", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.brightness_slider.grid(row=2, column=0, pady=5)
        self.brightness_slider.bind("<Motion>", self.update_brightness)  # Bind slider movement to update function

        self.contrast_slider = Scale(self.left_frame, from_=0.5, to=3.0, resolution=0.1, label="Contrast", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.contrast_slider.grid(row=3, column=0, pady=5)
        self.contrast_slider.bind("<Motion>", self.update_contrast)  # Bind slider movement to update function

        self.rotation_slider = Scale(self.left_frame, from_=-180, to=180, label="Rotate", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.rotation_slider.grid(row=4, column=0, pady=10)
        self.rotation_slider.bind("<Motion>", self.rotate_image)  # Bind slider movement to update rotation

        self.resize_slider = Scale(self.left_frame, from_=10, to=100, resolution=1, label="Resize (%)", orient=HORIZONTAL, length=300, bg="#f0f0f5")
        self.resize_slider.grid(row=5, column=0, pady=10)
        self.resize_slider.bind("<Motion>", self.resize_image_with_slider)  # Bind slider movement to resize image

        # Create additional buttons for various image processing tasks
        self.create_buttons()

        # Right Column for displaying the image
        self.canvas_frame = Frame(master, bg="#d9d9d9")  # Frame for the image display area
        self.canvas_frame.grid(row=1, column=1, padx=10, pady=10)

        self.canvas = Canvas(self.canvas_frame, width=800, height=500, bg="#d9d9d9", highlightthickness=1, highlightbackground="#ccc")
        self.canvas.grid(row=0, column=0)

    # Method to create buttons for different image processing techniques
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

        # Create each button and add it to the left panel
        for idx, (text, command, color) in enumerate(button_specs, start=6):
            Button(
                self.left_frame, text=text, command=command, bg=color, fg="white", font=("Helvetica", 10),
                padx=10, pady=5
            ).grid(row=idx, column=0, pady=5)

    # Method to upload an image
    def upload_image(self):
        filepath = filedialog.askopenfilename()  # Open file dialog to select image
        if filepath:
            self.image = cv2.imread(filepath) 
            self.image = self.resize_to_fit(self.image, 800, 500)  # Resize to fit within the canvas
            self.processed_image = self.image.copy()  # Make a copy of the original image
            self.display_image(self.image)  # Display the uploaded image

    # Method to resize the image while maintaining aspect ratio
    def resize_to_fit(self, image, max_width, max_height):
        h, w = image.shape[:2]
        scale = min(max_width / w, max_height / h)  # Calculate scale to fit image
        new_w, new_h = int(w * scale), int(h * scale)  # Calculate new width and height
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Resize and return

    # Method to update the image brightness based on the slider value
    def update_brightness(self, event=None):
        if self.image is not None:
            beta = self.brightness_slider.get()  # Get brightness value from slider
            self.processed_image = cv2.convertScaleAbs(self.image, alpha=1, beta=beta)  # Adjust brightness
            self.display_image(self.processed_image)  # Display the processed image

    # Method to update the image contrast based on the slider value
    def update_contrast(self, event=None):
        if self.image is not None:
            alpha = self.contrast_slider.get()  # Get contrast value from slider
            self.processed_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=0)  # Adjust contrast
            self.display_image(self.processed_image)  # Display the processed image

    # Method to resize the image with a slider
    def resize_image_with_slider(self, event=None):
        if self.image is not None:
            resize_percent = self.resize_slider.get()  # Get the resize percentage from slider
            scale = resize_percent / 100.0  # Convert to scale factor
            new_w = int(self.image.shape[1] * scale)  # Calculate new width
            new_h = int(self.image.shape[0] * scale)  # Calculate new height
            self.processed_image = cv2.resize(self.image, (new_w, new_h), interpolation=cv2.INTER_AREA)  # Resize
            self.display_image(self.processed_image)  # Display the resized image

    # Method to display the image on the canvas
    def display_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert the image from BGR to RGB
        img = Image.fromarray(img)  # Convert the image to a PIL object
        img = ImageTk.PhotoImage(img)  # Convert PIL image to Tkinter-compatible format
        self.canvas.create_image(400, 250, image=img, anchor="center")  # Place the image on the canvas
        self.canvas.image = img  # Keep a reference to the image to avoid garbage collection

    # Method to show the histogram of the image in a new window
    def show_histogram_page(self):
        histogram_window = Toplevel(self.master)  # Create a new top-level window
        histogram_window.title("Image Histogram")  # Set window title
        histogram_window.geometry("600x600")  # Set window size

        # Display the histogram in the new window
        if self.image is not None:
            self.display_histogram(histogram_window, self.image)

    # Method to display the histogram of an image
    def display_histogram(self, window, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])  # Calculate histogram

        # Plot the histogram using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 4), dpi=80)
        ax.plot(hist)
        ax.set_title("Image Histogram")
        ax.set_xlabel("Pixel Intensity")
        ax.set_ylabel("Frequency")

        canvas_hist = FigureCanvasTkAgg(fig, master=window)  # Embed the matplotlib plot in Tkinter
        canvas_hist.get_tk_widget().pack()
        canvas_hist.draw()

    # Method to apply noise reduction using Gaussian blur
    def noise_reduction(self):
        if self.image is not None:
            self.processed_image = cv2.GaussianBlur(self.image, (5, 5), 0)  # Apply Gaussian blur
            self.display_image(self.processed_image)  # Display the processed image

    # Method to normalize the image to a specific range
    def normalization(self):
        if self.image is not None:
            self.processed_image = cv2.normalize(self.image, None, 0, 255, cv2.NORM_MINMAX)  # Normalize the image
            self.display_image(self.processed_image)  # Display the processed image

    # Method to apply color correction using LAB color space
    def color_correction(self):
        if self.image is not None:
            lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)  # Convert to LAB color space
            l, a, b = cv2.split(lab)  # Split LAB channels
            l = cv2.equalizeHist(l)  # Equalize the L channel (luminance)
            self.processed_image = cv2.merge((l, a, b))  # Merge channels back
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_LAB2BGR)  # Convert back to BGR
            self.display_image(self.processed_image)  # Display the processed image

    # Method to enhance the image using a kernel filter
    def image_enhancement(self):
        if self.image is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Define sharpening kernel
            self.processed_image = cv2.filter2D(self.image, -1, kernel)  # Apply filter to enhance image
            self.display_image(self.processed_image)  # Display the enhanced image

    # Method to rotate the image based on the slider value
    def rotate_image(self, event=None):
        if self.image is not None:
            angle = self.rotation_slider.get()  # Get rotation angle from slider
            h, w = self.image.shape[:2]  # Get image dimensions
            center = (w // 2, h // 2)  # Find the center of the image
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)  # Get rotation matrix

            # Calculate new dimensions to accommodate the rotated image
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            bound_w = int(h * abs_sin + w * abs_cos)
            bound_h = int(h * abs_cos + w * abs_sin)

            rotation_matrix[0, 2] += bound_w / 2 - center[0]  # Adjust for new image center
            rotation_matrix[1, 2] += bound_h / 2 - center[1]  # Adjust for new image center

            self.processed_image = cv2.warpAffine(self.image, rotation_matrix, (bound_w, bound_h))  # Apply rotation
            self.display_image(self.processed_image)  # Display the rotated image

    # Method to detect edges using the Canny edge detector
    def detect_edges_advanced(self):
        if self.image is not None:
            self.processed_image = cv2.Canny(self.image, 100, 200)  # Apply Canny edge detection
            self.display_image(self.processed_image)  # Display the processed image

    # Method to perform histogram equalization on the image
    def histogram_equalization(self):
        if self.image is not None:
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            self.processed_image = cv2.equalizeHist(gray_image)  # Apply histogram equalization
            self.display_image(self.processed_image)  # Display the processed image


if __name__ == "__main__":
    root = Tk()
    gui = ModernImageAnalyzerGUI(root)
    root.mainloop()

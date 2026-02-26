"""
Letter Recognition GUI

Draw a letter on the canvas and see the model's prediction
with a confidence bar chart for all 26 letters.
"""

import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow import keras
import threading


class LetterRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Letter Recognition")
        self.root.resizable(False, False)
        
        # Model will be loaded later
        self.model = None
        
        # Canvas settings (scaled up for easier drawing)
        self.canvas_size = 320  # Display size
        self.image_size = 32    # Model input size
        self.brush_size = 12    # Brush thickness
        
        # Create PIL image for drawing (white background)
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        
        # Last mouse position for smooth lines
        self.last_x = None
        self.last_y = None
        
        # Threading for predictions
        self.prediction_pending = False
        self.prediction_lock = threading.Lock()
        
        # Debug mode
        self.debug_mode = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Left side: Drawing canvas
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=(0, 10))
        
        # Model selection
        model_frame = ttk.Frame(left_frame)
        model_frame.grid(row=0, column=0, pady=(0, 5), sticky="ew")
        
        ttk.Label(model_frame, text="Model:").pack(side=tk.LEFT)
        self.model_label = ttk.Label(model_frame, text="No model loaded", foreground="red")
        self.model_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(model_frame, text="Load Model", command=self.load_model).pack(side=tk.RIGHT)
        
        # Canvas label
        ttk.Label(left_frame, text="Draw a letter (A-Z):", font=("Arial", 12)).grid(row=1, column=0, pady=(0, 5))
        
        # Drawing canvas
        self.canvas = tk.Canvas(
            left_frame, 
            width=self.canvas_size, 
            height=self.canvas_size,
            bg="white",
            cursor="crosshair",
            relief="solid",
            borderwidth=1
        )
        self.canvas.grid(row=2, column=0)
        
        # Bind mouse events
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # Buttons
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=3, column=0, pady=10)
        
        ttk.Button(button_frame, text="Predict", command=self.predict).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT, padx=5)
        
        # Prediction result label
        self.result_label = ttk.Label(
            left_frame, 
            text="Prediction: -", 
            font=("Arial", 16, "bold")
        )
        self.result_label.grid(row=4, column=0, pady=5)
        
        # Debug mode checkbox
        debug_frame = ttk.Frame(left_frame)
        debug_frame.grid(row=5, column=0, pady=5)
        ttk.Checkbutton(
            debug_frame, 
            text="Debug Mode (show preprocessed input)", 
            variable=self.debug_mode,
            command=self.toggle_debug
        ).pack()
        
        # Debug preview frame (hidden by default)
        self.debug_frame = ttk.LabelFrame(left_frame, text="Preprocessed Input (32x32)")
        self.debug_frame.grid(row=6, column=0, pady=5)
        self.debug_frame.grid_remove()  # Hidden initially
        
        # Debug canvas to show preprocessed image (scaled up for visibility)
        self.debug_canvas_size = 128
        self.debug_canvas = tk.Canvas(
            self.debug_frame,
            width=self.debug_canvas_size,
            height=self.debug_canvas_size,
            bg="black",
            relief="solid",
            borderwidth=1
        )
        self.debug_canvas.pack(padx=5, pady=5)
        self.debug_photo = None  # Keep reference to prevent garbage collection
        
        # Right side: Confidence plot
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1)
        
        ttk.Label(right_frame, text="Confidence:", font=("Arial", 12)).pack(pady=(0, 5))
        
        # Matplotlib figure for confidence bars (vertical layout)
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.fig.tight_layout(pad=2)
        
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=right_frame)
        self.plot_canvas.get_tk_widget().pack()
        
        # Initialize empty plot
        self.update_plot(np.zeros(26))
        
    def load_model(self):
        """Open file dialog to load a .keras model"""
        filepath = filedialog.askopenfilename(
            title="Select Model",
            filetypes=[("Keras Model", "*.keras"), ("H5 Model", "*.h5"), ("All Files", "*.*")],
            initialdir="."
        )
        if filepath:
            try:
                self.model = keras.models.load_model(filepath)
                model_name = filepath.split("/")[-1].split("\\")[-1]
                self.model_label.config(text=model_name, foreground="green")
                print(f"Model loaded: {filepath}")
            except Exception as e:
                self.model_label.config(text="Load failed", foreground="red")
                print(f"Error loading model: {e}")
    
    def start_draw(self, event):
        """Start drawing"""
        self.last_x = event.x
        self.last_y = event.y
        
    def draw_line(self, event):
        """Draw a line from last position to current position"""
        if self.last_x is not None and self.last_y is not None:
            # Draw on canvas
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=self.brush_size,
                fill="black",
                capstyle=tk.ROUND,
                smooth=True
            )
            # Draw on PIL image
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y],
                fill=0,
                width=self.brush_size
            )
        self.last_x = event.x
        self.last_y = event.y
        
        # Live prediction while drawing
        self.predict()
        
    def stop_draw(self, event):
        """Stop drawing"""
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        """Clear the drawing canvas"""
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Prediction: -")
        self.update_plot(np.zeros(26))
        
    def toggle_debug(self):
        """Toggle debug mode visibility"""
        if self.debug_mode.get():
            self.debug_frame.grid()
        else:
            self.debug_frame.grid_remove()
    
    def update_debug_preview(self, img_array):
        """Update the debug preview canvas with the preprocessed image"""
        if not self.debug_mode.get():
            return
        
        # img_array is normalized [0, 1], convert back to [0, 255] for display
        # The image is already inverted (white on black)
        display_array = (img_array[0] * 255).astype(np.uint8)
        
        # Handle channel dimension if present
        if len(display_array.shape) == 3:
            display_array = display_array[:, :, 0]
        
        # Create PIL image and scale up for visibility
        debug_img = Image.fromarray(display_array, mode='L')
        debug_img = debug_img.resize(
            (self.debug_canvas_size, self.debug_canvas_size), 
            Image.Resampling.NEAREST
        )
        
        # Convert to PhotoImage and display
        from PIL import ImageTk
        self.debug_photo = ImageTk.PhotoImage(debug_img)
        self.debug_canvas.delete("all")
        self.debug_canvas.create_image(0, 0, anchor=tk.NW, image=self.debug_photo)
    
    def preprocess_image(self):
        """Preprocess the drawn image for the model with centering"""
        # Convert to numpy for processing
        img_array = np.array(self.image, dtype=np.float32)
        
        # Invert first to find content (drawing is black=0 on white=255)
        inverted = 255.0 - img_array
        
        # Find bounding box of the drawn content
        rows = np.any(inverted > 10, axis=1)  # Threshold to ignore noise
        cols = np.any(inverted > 10, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            # Empty canvas - return blank image
            blank = np.zeros((1, self.image_size, self.image_size), dtype=np.float32)
            return blank
        
        # Get bounding box coordinates
        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]
        y_min, y_max = row_indices[0], row_indices[-1]
        x_min, x_max = col_indices[0], col_indices[-1]
        
        # Extract the content region
        content = self.image.crop((x_min, y_min, x_max + 1, y_max + 1))
        content_width, content_height = content.size
        
        # Calculate scaling to fit in target size with padding
        padding = 4  # Padding around the centered content
        target_size = self.image_size - 2 * padding
        
        # Scale to fit while maintaining aspect ratio
        scale = min(target_size / content_width, target_size / content_height)
        new_width = int(content_width * scale)
        new_height = int(content_height * scale)
        
        # Resize content
        content_resized = content.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new centered image (white background)
        centered_image = Image.new("L", (self.image_size, self.image_size), color=255)
        
        # Calculate position to center the content
        x_offset = (self.image_size - new_width) // 2
        y_offset = (self.image_size - new_height) // 2
        
        # Paste centered content
        centered_image.paste(content_resized, (x_offset, y_offset))
        
        # Convert to numpy array
        img_array = np.array(centered_image, dtype=np.float32)
        
        # Normalize to [0, 1]
        img_array = img_array / 255.0
        
        # Invert (white letters on black background, same as training data)
        img_array = 1.0 - img_array
        
        # Add batch dimension: (1, 32, 32)
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    def predict(self):
        """Run prediction on the drawn image (threaded)"""
        if self.model is None:
            self.result_label.config(text="Please load a model first!")
            return
        
        # Skip if prediction already in progress
        with self.prediction_lock:
            if self.prediction_pending:
                return
            self.prediction_pending = True
        
        # Run prediction in background thread
        thread = threading.Thread(target=self._predict_thread, daemon=True)
        thread.start()
    
    def _predict_thread(self):
        """Background thread for prediction"""
        try:
            # Preprocess image
            img_array = self.preprocess_image()
            
            # Update debug preview from main thread
            self.root.after(0, lambda: self.update_debug_preview(img_array))
            
            # Check if model expects channel dimension (CNN)
            if len(self.model.input_shape) == 4:
                img_array = img_array.reshape(-1, 32, 32, 1)
            
            # Get prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Get predicted class
            predicted_class = np.argmax(predictions)
            predicted_letter = chr(65 + predicted_class)  # Convert 0-25 to A-Z
            confidence = predictions[predicted_class] * 100
            
            # Update UI from main thread
            self.root.after(0, lambda: self._update_prediction_ui(predicted_letter, confidence, predictions))
        finally:
            with self.prediction_lock:
                self.prediction_pending = False
    
    def _update_prediction_ui(self, letter, confidence, predictions):
        """Update UI with prediction results (called from main thread)"""
        self.result_label.config(text=f"Prediction: {letter} ({confidence:.1f}%)")
        self.update_plot(predictions)
        
    def update_plot(self, confidences):
        """Update the confidence bar chart"""
        self.ax.clear()
        
        # Letter labels
        letters = [chr(65 + i) for i in range(26)]
        
        # Create vertical bar chart
        colors = ['#4CAF50' if c == max(confidences) and c > 0 else '#2196F3' for c in confidences]
        bars = self.ax.bar(letters, confidences, color=colors)
        
        # Formatting
        self.ax.set_ylim(0, 1.15)
        self.ax.set_ylabel("Confidence")
        self.ax.set_xlabel("Letter")
        self.ax.set_title("Letter Confidence Scores", pad=10)
        
        # Add percentage labels on bars
        for bar, conf in zip(bars, confidences):
            if conf > 0.05:
                self.ax.text(
                    bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + 0.02,
                    f'{conf*100:.0f}%',
                    ha='center',
                    fontsize=7
                )
        
        self.fig.tight_layout()
        self.plot_canvas.draw()


def main():
    root = tk.Tk()
    app = LetterRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

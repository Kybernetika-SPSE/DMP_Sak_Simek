import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import threading
import sys

class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        if message != '\n':  # Ignorovat prázdné nové řádky
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)

    def flush(self):
        pass  # Není potřeba implementovat pro Tkinter, stačí ignorovat

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("700x400")  # Velikost okna
        self.root.minsize(700, 400)  # Minimální velikost okna

        # Inicializace modelu TensorFlow Lite
        self.interpreter = tf.lite.Interpreter(model_path="efficientdet_lite0.tflite")
        self.interpreter.allocate_tensors()

        # Barevné schéma
        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)  # Zmenšení písma pro ovládací prvky
        font_bold = ("Helvetica", 8, "bold")

        self.root.configure(bg=bg_color)

        # Nadpis aplikace
        self.title = tk.Label(
            root, text="Rozpoznávání Objektů", font=("Helvetica", 12, "bold"), bg=bg_color, fg=accent_color
        )
        self.title.grid(row=0, column=0, columnspan=2, pady=10)

        # Rámec pro zobrazení obrázku a konzoli
        self.image_console_frame = tk.Frame(root, bg=frame_bg_color)
        self.image_console_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Rámec pro zobrazení obrázku
        self.image_frame = tk.Frame(self.image_console_frame, bg="#eeeeee", bd=1, relief="solid")
        self.image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.image_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Konzolový výstup a vstup
        self.console_frame = tk.Frame(self.image_console_frame, bg=frame_bg_color)
        self.console_frame.pack(fill="both", expand=True)

        self.console_output = tk.Text(self.console_frame, height=4, bg="#222222", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_output.pack(fill="both", expand=True, padx=5, pady=5)

        self.console_input = tk.Entry(self.console_frame, bg="#333333", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_input.pack(fill="x", padx=5, pady=5)
        self.console_input.bind("<Return>", self.execute_command)

        self.redirect_console_output()

        # Ovládací panel a graf
        self.controls_frame = tk.Frame(root, bg=bg_color, width=120)  # Zúžený ovládací panel
        self.controls_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")

        # Tlačítka pro načítání obrázku a videa
        self.load_image_btn = tk.Button(
            self.controls_frame, text="Načíst obrázek", command=self.load_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.load_image_btn.pack(fill="x", pady=5)

        self.load_video_btn = tk.Button(
            self.controls_frame, text="Načíst video", command=self.load_video,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.load_video_btn.pack(fill="x", pady=5)

        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.detect_btn.pack(fill="x", pady=5)

        # Inicializace dalších komponent
        self.image = None

    def redirect_console_output(self):
        sys.stdout = OutputRedirector(self.console_output)

    def run_command(self, command):
        def execute():
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for line in iter(process.stdout.readline, ""):
                self.insert_to_console(line)
            for line in iter(process.stderr.readline, ""):
                self.insert_to_console(line, error=True)
            process.stdout.close()
            process.stderr.close()
            process.wait()

        thread = threading.Thread(target=execute)
        thread.start()

    def insert_to_console(self, line, error=False):
        self.root.after(0, lambda: self.console_output.insert(tk.END, line if not error else f"[ERROR] {line}", "error" if error else None))
        self.root.after(0, lambda: self.console_output.see(tk.END))

    def execute_command(self, event):
        command = self.console_input.get()
        if command.strip():
            self.console_output.insert(tk.END, f"> {command}\n")
            self.console_input.delete(0, tk.END)
            self.run_command(command)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)

    def display_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((self.canvas.winfo_width(), self.canvas.winfo_height()), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        self.canvas.create_image(self.canvas.winfo_width() // 2, self.canvas.winfo_height() // 2, image=img_tk)
        self.canvas.image = img_tk  # Prevent garbage collection

    def detect_objects(self):
        if self.image is None:
            self.insert_to_console("Nejdříve načtěte obrázek.")
            return

        # Předzpracování obrazu pro detekci
        input_tensor = self.process_image(self.image)
        self.interpreter.set_tensor(self.interpreter.get_input_details()[0]['index'], input_tensor)
        self.interpreter.invoke()

        # Výsledky detekce
        boxes = self.interpreter.get_tensor(self.interpreter.get_output_details()[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.interpreter.get_output_details()[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.interpreter.get_output_details()[2]['index'])[0]

        self.insert_to_console("Detekce objektů dokončena.")

        # Filtrace a vykreslení výsledků
        detected_objects = []
        for i in range(len(scores)):
            if scores[i] > 0.5:
                detected_objects.append({
                    'box': boxes[i],
                    'class': classes[i],
                    'confidence': scores[i]
                })

        self.display_detected_objects(detected_objects)

    def process_image(self, img):
        # Preprocessing for object detection
        img_resized = cv2.resize(img, (300, 300))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_normalized = np.expand_dims(img_rgb, axis=0).astype(np.float32)
        img_normalized = img_normalized / 255.0  # Normalize
        return img_normalized

    def display_detected_objects(self, detections):
        # Draw bounding boxes and display class names
        for detection in detections:
            box = detection['box']
            class_id = int(detection['class'])
            confidence = detection['confidence']
            y1, x1, y2, x2 = box

            # Scale the box back to the original image size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            x1 = int(x1 * canvas_width)
            y1 = int(y1 * canvas_height)
            x2 = int(x2 * canvas_width)
            y2 = int(y2 * canvas_height)

            self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2)
            self.canvas.create_text(x1, y1, text=f"Confidence: {confidence:.2f}", fill="white", font=("Helvetica", 6))

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

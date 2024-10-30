import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import tensorflow as tf
import numpy as np

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("900x600")
        
        # Elegantní barevné schéma
        bg_color = "#fafafa"            # světlé pozadí aplikace
        frame_bg_color = "#e3e8f0"      # jemné modrošedé pozadí pro rámce
        accent_color = "#3a7bd5"        # akcentní barva tlačítek a nadpisů
        text_color = "#333333"          # tmavší textová barva
        
        self.root.configure(bg=bg_color)

        # Nadpis aplikace
        self.title = tk.Label(
            root, text="Rozpoznávání Objektů", font=("Arial", 24, "bold"), bg=bg_color, fg=accent_color
        )
        self.title.pack(pady=15)

        # Rámec pro zobrazení obrázku s obvodovým stylem a zaoblenými rohy
        self.image_frame = tk.Frame(root, bg=frame_bg_color, bd=0, highlightthickness=0)
        self.image_frame.place(x=20, y=80, width=560, height=420)
        
        self.canvas = tk.Canvas(self.image_frame, width=560, height=420, bg="black", bd=0, highlightthickness=0)
        self.canvas.pack()

        # Ovládací prvky v pravé části aplikace
        self.controls_frame = tk.Frame(root, width=300, bg=bg_color)
        self.controls_frame.place(x=600, y=80, width=280, height=400)

        # Tlačítko pro načtení obrázku s kulatým obvodem
        self.load_image_btn = tk.Button(
            self.controls_frame, text="Načíst obrázek", command=self.load_image,
            width=20, bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.load_image_btn.pack(pady=8, padx=15)

        # Tlačítko pro spuštění kamery
        self.start_camera_btn = tk.Button(
            self.controls_frame, text="Spustit kameru", command=self.start_camera,
            width=20, bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.start_camera_btn.pack(pady=8, padx=15)

        # Tlačítko pro spuštění detekce objektů
        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            width=20, bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.detect_btn.pack(pady=8, padx=15)

        # Slider pro nastavení citlivosti detekce
        self.sensitivity_label = tk.Label(
            self.controls_frame, text="Citlivost detekce:", bg=bg_color, fg=text_color, font=("Arial", 10, "bold")
        )
        self.sensitivity_label.pack(pady=10)

        self.sensitivity_slider = tk.Scale(
            self.controls_frame, from_=0, to=100, orient="horizontal", bg=frame_bg_color, fg=text_color,
            highlightthickness=0, relief="flat"
        )
        self.sensitivity_slider.set(50)
        self.sensitivity_slider.pack()

        # Výsledky detekce v rámci s kulatými rohy a světlým pozadím
        self.results_frame = tk.Frame(self.controls_frame, bg=frame_bg_color, bd=0, highlightthickness=0)
        self.results_frame.pack(pady=10, fill="both", expand=True)

        self.results_label = tk.Label(
            self.results_frame, text="Výsledky rozpoznávání:", bg=frame_bg_color, fg=text_color, anchor="nw",
            font=("Arial", 10, "bold")
        )
        self.results_label.pack(fill="x")

        self.results_text = tk.Text(self.results_frame, height=10, state="disabled", bg=frame_bg_color, fg=text_color, font=("Arial", 10))
        self.results_text.pack(fill="both", expand=True)

        # Načtení modelu TensorFlow Lite
        #self.load_model()

    #def load_model(self):
        # Načtení modelu TensorFlow Lite
        #self.interpreter = tf.lite.Interpreter(model_path="model.tflite")
        #self.interpreter.allocate_tensors()
        #self.input_details = self.interpreter.get_input_details()
        #self.output_details = self.interpreter.get_output_details()
    
    def load_image(self):
        # Načtení obrázku ze souboru
        file_path = filedialog.askopenfilename()
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)
    
    def start_camera(self):
        # Spuštění kamery
        self.cap = cv2.VideoCapture(0)
        self.show_frame()
    
    def show_frame(self):
        # Zobrazení snímku z kamery
        ret, frame = self.cap.read()
        if ret:
            self.display_image(frame)
            self.root.after(10, self.show_frame)
    
    def display_image(self, img):
        # Zobrazení obrázku v oblasti canvas
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.image = img_tk
    
    def detect_objects(self):
        # Místo pro skutečnou detekční logiku
        self.results_text.configure(state="normal")
        self.results_text.delete("1.0", tk.END)
        
        # Simulovaný příklad výstupu detekce
        detections = [
            {"object": "Kočka", "confidence": 0.95},
            {"object": "Auto", "confidence": 0.87}
        ]
        
        for detection in detections:
            self.results_text.insert(tk.END, f"{detection['object']}: {detection['confidence'] * 100:.2f}%\n")
        
        self.results_text.configure(state="disabled")

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

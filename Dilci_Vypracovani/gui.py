import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageEnhance, ImageFilter
import cv2
import subprocess
import threading
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
import numpy as np
import tensorflow as tf

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("700x400")  # Velikost okna
        self.root.minsize(700, 400)  # Minimální velikost okna

        # Barevné schéma
        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)  # Zmenšení písma pro ovládací prvky
        font_secondary = ("Helvetica", 6)  # Menší písmo pro textové komponenty
        font_bold = ("Helvetica", 8, "bold")

        self.root.configure(bg=bg_color)

        # Načteme model pro detekci objektů
        self.model = self.load_model()

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

        self.start_camera_btn = tk.Button(
            self.controls_frame, text="Spustit kameru", command=self.start_camera,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.start_camera_btn.pack(fill="x", pady=5)

        # Tlačítko pro rozpoznání objektů
        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.detect_btn.pack(fill="x", pady=5)

        # Možnost úprav obrázků
        self.edit_image_btn = tk.Button(
            self.controls_frame, text="Úpravy obrázku", command=self.edit_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.edit_image_btn.pack(fill="x", pady=5)

        # Rámec pro graf pod tlačítky
        self.chart_frame = tk.Frame(self.controls_frame, bg=bg_color)
        self.chart_frame.pack(fill="both", expand=True, pady=10)

        # Inicializace grafu
        self.create_chart()
        self.update_chart([])  # Inicializace prázdného grafu

        # Přidání události pro výběr oblasti
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        # Nastavení váhy sloupců
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)  # 3/5 pro obrazovku
        root.grid_columnconfigure(1, weight=2)  # 2/5 pro ovládací panel

    def redirect_console_output(self):
        # Přesměrování konzolového výstupu do Text widgetu
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")  # Simulace příkazu pro zobrazení výstupu

    def load_model(self):
        # Načte model EfficientDet Lite
        model_path = "/home/pi/examples/lite/examples/object_detection/raspberry_pi/efficientdet_lite0.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def run_command(self, command):
        # Spustí příkaz v PowerShellu a vypisuje jeho výstup do konzolového widgetu
        def execute():
            process = subprocess.Popen(["powershell", "-Command", command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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
        # Vložení výstupu do konzolového widgetu
        self.root.after(0, lambda: self.console_output.insert(tk.END, line if not error else f"[ERROR] {line}", "error" if error else None))
        self.root.after(0, lambda: self.console_output.see(tk.END))

    def execute_command(self, event):
        # Získá příkaz od uživatele a spustí ho v PowerShellu
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

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.show_frame_video()

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        self.show_frame()

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Převod snímku na RGB pro detekci objektů
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detekce objektů v aktuálním snímku
            detections = self.detect_objects_in_frame(frame)

            # Zobrazení výsledků detekce (obdélníky kolem objektů)
            self.display_image_with_detections(frame, detections)

            self.root.after(10, self.show_frame)

    def detect_objects_in_frame(self, frame):
        # Připravíme snímek pro detekci
        input_tensor = self.prepare_image(frame)

        # Načteme tensorové výstupy
        self.model.set_tensor(self.model.get_input_details()[0]['index'], input_tensor)
        self.model.invoke()

        # Získáme výsledky detekce
        boxes = self.model.get_tensor(self.model.get_output_details()[0]['index'])[0]  # Bounding boxes
        class_ids = self.model.get_tensor(self.model.get_output_details()[1]['index'])[0]  # Class IDs
        scores = self.model.get_tensor(self.model.get_output_details()[2]['index'])[0]  # Confidence scores

        # Filtrace výsledků na základě prahové hodnoty
        detections = []
        for i in range(len(scores)):
            if scores[i] > 0.5:  # Prahová hodnota pro rozpoznání (0.5 = 50%)
                y1, x1, y2, x2 = boxes[i]
                label = str(class_ids[i])
                confidence = scores[i]
                detections.append({
                    'object': label,
                    'confidence': confidence,
                    'bbox': (int(x1 * frame.shape[1]), int(y1 * frame.shape[0]), int(x2 * frame.shape[1]), int(y2 * frame.shape[0]))
                })

        return detections

    def prepare_image(self, frame):
        # Příprava snímku pro model (resize, normalizace atd.)
        img_resized = cv2.resize(frame, (300, 300))  # Změna velikosti pro model
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # RGB formát pro TensorFlow Lite
        input_tensor = np.expand_dims(img_rgb, axis=0).astype(np.float32)  # Přidání batch dimenze
        return input_tensor

    def display_image_with_detections(self, frame, detections):
        # Vykreslení výsledků detekce na obraz
        for detection in detections:
            bbox = detection['bbox']
            label = f"Objekt: {detection['object']} ({int(detection['confidence'] * 100)}%)"

            # Nakreslíme obdélník kolem detekovaného objektu
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

            # Přidáme text s názvem objektu a pravděpodobností
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Zobrazení snímku s detekcemi
        self.display_image(frame)

    def display_image(self, img):
        # Funkce pro zobrazení snímku na obrazovce (tato metoda zůstává stejná)
        self.original_image = img.copy()  # Uložení originálního obrázku pro úpravy
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((canvas_width, canvas_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
        self.canvas.image = img_tk  # Prevent garbage collection

# Spuštění aplikace
root = tk.Tk()
app = ObjectDetectionApp(root)
root.mainloop()

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

        # Barevné schéma
        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)  # Zmenšení písma pro ovládací prvky
        font_secondary = ("Helvetica", 6)  # Menší písmo pro textové komponenty
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

        # Nastavení váhy sloupců
        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)  # 3/5 pro obrazovku
        root.grid_columnconfigure(1, weight=2)  # 2/5 pro ovládací panel

    def redirect_console_output(self):
        # Přesměrování konzolového výstupu do Text widgetu
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")  # Simulace příkazu pro zobrazení výstupu

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

    def show_frame_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.display_image(frame)
            self.root.after(10, self.show_frame_video)
        else:
            self.video_capture.release()

    def display_image(self, img):
        self.original_image = img.copy()  # Uložení originálního obrázku pro úpravy
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((canvas_width, canvas_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
        self.canvas.image = img_tk  # Prevent garbage collection

    def create_chart(self):
        self.figure, self.ax = plt.subplots(figsize=(4, 2))  # Menší graf
        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=self.chart_frame)
        self.chart_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.chart_canvas.draw()

    def update_chart(self, detections):
        self.ax.clear()
        if detections:
            labels = [d['object'] for d in detections]
            values = [d['confidence'] * 100 for d in detections]
            self.ax.bar(labels, values, color='blue')
            self.ax.set_ylim(0, 100)
            self.ax.set_title('Detekce objektů')
        self.chart_canvas.draw()

    def detect_objects(self):
        if hasattr(self, 'image'):
            # Předpokládáme, že používáme image pro detekci
            image = self.image
        elif hasattr(self, 'video_capture') and self.video_capture.isOpened():
            # Pokud máme video, použijeme první snímek pro detekci
            ret, image = self.video_capture.read()
            if not ret:
                self.insert_to_console("[ERROR] Nepodařilo se načíst snímek z videa.")
                return
        else:
            self.insert_to_console("[ERROR] Žádný obrázek nebo video k detekci.")
            return

        # Detekce objektů (nebo zavolání vašeho modelu pro detekci)
        self.insert_to_console("[INFO] Detekce objektů běží...")
        command = "cd /home/pi/examples/lite/examples/object_detection/raspberry_pi && sudo python3 detect.py --model efficientdet_lite0.tflite"

        def execute_detection():
            try:
                process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                for line in iter(process.stdout.readline, ""):
                    self.insert_to_console(line)  # Vložit výstup do konzoly
                for line in iter(process.stderr.readline, ""):
                    self.insert_to_console(line, error=True)  # Vložit chyby do konzoly
                process.stdout.close()
                process.stderr.close()
                process.wait()
            except Exception as e:
                self.insert_to_console(f"[ERROR] {str(e)}", error=True)

        thread = threading.Thread(target=execute_detection)
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

    def edit_image(self):
        # Implementujte funkce pro úpravy obrázků (např. oříznutí, filtr, kontrast)
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

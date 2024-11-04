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

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("1200x700")
        self.root.minsize(1200, 700)

        # Barevné schéma
        bg_color = "#fafafa"
        frame_bg_color = "#e3e8f0"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        
        self.root.configure(bg=bg_color)

        # Nadpis aplikace
        self.title = tk.Label(
            root, text="Rozpoznávání Objektů", font=("Arial", 24, "bold"), bg=bg_color, fg=accent_color
        )
        self.title.pack(pady=10)

        # Rámec pro zobrazení obrázku a konzoli
        self.image_console_frame = tk.Frame(root, bg=frame_bg_color)
        self.image_console_frame.pack(side="left", fill="both", expand=True, padx=20, pady=10)

        # Rámec pro zobrazení obrázku
        self.image_frame = tk.Frame(self.image_console_frame, bg="black")
        self.image_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(self.image_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        # Konzolový výstup
        self.console_frame = tk.Frame(self.image_console_frame, bg=frame_bg_color)
        self.console_frame.pack(fill="both", expand=True)

        self.console_output = tk.Text(self.console_frame, height=10, bg="black", fg="white", font=("Courier", 10))
        self.console_output.pack(fill="both", expand=True)
        self.redirect_console_output()

        # Ovládací panel
        self.controls_frame = tk.Frame(root, bg=bg_color)
        self.controls_frame.pack(side="right", fill="y", padx=20, pady=10)

        # Tlačítka pro načítání obrázku a videa
        self.load_image_btn = tk.Button(
            self.controls_frame, text="Načíst obrázek", command=self.load_image,
            bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.load_image_btn.pack(fill="x", pady=5)

        self.load_video_btn = tk.Button(
            self.controls_frame, text="Načíst video", command=self.load_video,
            bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.load_video_btn.pack(fill="x", pady=5)

        self.start_camera_btn = tk.Button(
            self.controls_frame, text="Spustit kameru", command=self.start_camera,
            bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.start_camera_btn.pack(fill="x", pady=5)

        # Tlačítko pro rozpoznání objektů
        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.detect_btn.pack(fill="x", pady=5)

        # Možnost úprav obrázků
        self.edit_image_btn = tk.Button(
            self.controls_frame, text="Úpravy obrázku", command=self.edit_image,
            bg=accent_color, fg="white", font=("Arial", 11, "bold"), relief="flat"
        )
        self.edit_image_btn.pack(fill="x", pady=5)

        # Inicializace grafu
        self.create_chart()
        self.update_chart([])  # Inicializace prázdného grafu

        # Příprava pro oříznutí
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.original_image = None  # Uloží originální obrázek pro úpravy

        # Přidání události pro výběr oblasti
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def redirect_console_output(self):
        # Přesměrování konzolového výstupu do Text widgetu
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")  # Simulace příkazu pro zobrazení výstupu
    
    def run_command(self, command):
        # Spustí příkaz a vypisuje jeho výstup do konzolového widgetu
        def execute():
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            for line in iter(process.stdout.readline, b""):
                try:
                    # Přepnout na správný dekodér
                    self.insert_to_console(line)
                except UnicodeDecodeError:
                    self.insert_to_console(line, encoding='latin-1', replace=True)

            process.stdout.close()
            process.wait()

        thread = threading.Thread(target=execute)
        thread.start()

    def insert_to_console(self, line, encoding='utf-8', replace=False):
        # Vložení výstupu do konzolového widgetu
        if replace:
            decoded_line = line.decode(encoding, errors='replace')
        else:
            decoded_line = line.decode(encoding)

        self.root.after(0, lambda: self.console_output.insert(tk.END, decoded_line))
        self.root.after(0, lambda: self.console_output.see(tk.END))

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
            self.display_image(frame)
            self.root.after(10, self.show_frame)

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
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.chart_canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.chart_canvas.get_tk_widget().pack(side="right", fill="both", expand=True)
        self.chart_canvas.draw()

    def edit_image(self):
        if hasattr(self, 'original_image'):
            # Možnost úpravy jas, kontrast, nebo filtr
            brightness = simpledialog.askfloat("Jas", "Nastavte jas (0.0 - 2.0):", minvalue=0.0, maxvalue=2.0, initialvalue=1.0)
            if brightness is not None:
                img_pil = Image.fromarray(self.original_image)
                enhancer = ImageEnhance.Brightness(img_pil)
                self.original_image = np.array(enhancer.enhance(brightness))
            
            contrast = simpledialog.askfloat("Kontrast", "Nastavte kontrast (0.0 - 2.0):", minvalue=0.0, maxvalue=2.0, initialvalue=1.0)
            if contrast is not None:
                img_pil = Image.fromarray(self.original_image)
                enhancer = ImageEnhance.Contrast(img_pil)
                self.original_image = np.array(enhancer.enhance(contrast))

            filter_type = simpledialog.askstring("Filtr", "Vyberte filtr (rozmazání, ostření, žádný):").strip().lower()
            if filter_type == "rozmazání":
                img_pil = Image.fromarray(self.original_image).filter(ImageFilter.GaussianBlur(5))
                self.original_image = np.array(img_pil)
            elif filter_type == "ostření":
                img_pil = Image.fromarray(self.original_image).filter(ImageFilter.UnsharpMask(radius=2, percent=150))
                self.original_image = np.array(img_pil)

            self.display_image(self.original_image)  # Zobrazení upraveného obrázku
        else:
            messagebox.showwarning("Upozornění", "Nejdříve načtěte obrázek.")

    def on_button_press(self, event):
        # Uložení počátečního bodu pro oříznutí
        self.start_x = event.x
        self.start_y = event.y
        self.rect = None

    def on_mouse_drag(self, event):
        # Aktualizace obdélníku pro oříznutí
        if self.rect:
            self.canvas.delete(self.rect)

        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline="red")

    def on_button_release(self, event):
        # Oříznutí oblasti a zobrazení
        if self.rect:
            x0, y0, x1, y1 = (self.start_x, self.start_y, event.x, event.y)
            cropped_image = self.original_image[y0:y1, x0:x1]
            self.display_image(cropped_image)  # Zobrazení oříznutého obrázku

    def detect_objects(self):
        # Simulace detekce objektů a aktualizace grafu
        simulated_detections = [
            {'object': 'Objekt A', 'confidence': 0.9},
            {'object': 'Objekt B', 'confidence': 0.7}
        ]
        self.update_chart(simulated_detections)
        self.update_results(simulated_detections)

    def update_chart(self, detections):
        self.ax.clear()
        if detections:
            labels = [d['object'] for d in detections]
            values = [d['confidence'] * 100 for d in detections]
            self.ax.bar(labels, values, color='blue')
            self.ax.set_ylim(0, 100)
            self.ax.set_xlabel('Objekty')
            self.ax.set_ylabel('Důvěra (%)')
            self.ax.set_title('Důvěra detekovaných objektů')
        else:
            self.ax.bar([], [])
            self.ax.set_ylim(0, 100)
        self.chart_canvas.draw()

    def update_results(self, detections):
        # Zobrazení detekovaných objektů
        self.results_text.config(state="normal")
        self.results_text.delete(1.0, tk.END)  # Vyčistit předchozí výsledky
        for detection in detections:
            self.results_text.insert(tk.END, f"Objekt: {detection['object']}, Důvěra: {detection['confidence']:.2f}\n")
        self.results_text.config(state="disabled")

class OutputRedirector:
    def __init__(self, widget):
        self.widget = widget

    def write(self, message):
        self.widget.insert(tk.END, message)
        self.widget.see(tk.END)

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()

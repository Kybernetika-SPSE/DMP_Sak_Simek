import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
import utils
import tensorflow as tf
import numpy as np
import threading
import sys
import subprocess
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

class OutputRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        if message != '\n':
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)

    def flush(self):
        pass

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Rozpoznávání Objektů")
        self.root.geometry("700x400")
        self.root.minsize(700, 400)

        bg_color = "#f5f5f5"
        frame_bg_color = "#ffffff"
        accent_color = "#3a7bd5"
        text_color = "#333333"
        font_primary = ("Helvetica", 8)
        font_secondary = ("Helvetica", 6)
        font_bold = ("Helvetica", 8, "bold")

        self.root.configure(bg=bg_color)

        self.title = tk.Label(
            root, text="Rozpoznávání Objektů", font=("Helvetica", 12, "bold"), bg=bg_color, fg=accent_color
        )
        self.title.grid(row=0, column=0, columnspan=2, pady=10)

        self.image_console_frame = tk.Frame(root, bg=frame_bg_color)
        self.image_console_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        self.image_frame = tk.Frame(self.image_console_frame, bg="#eeeeee", bd=1, relief="solid")
        self.image_frame.pack(fill="both", expand=True, padx=5, pady=5)

        self.canvas = tk.Canvas(self.image_frame, bg="black")
        self.canvas.pack(fill="both", expand=True)

        self.console_frame = tk.Frame(self.image_console_frame, bg=frame_bg_color)
        self.console_frame.pack(fill="both", expand=True)

        self.console_output = tk.Text(self.console_frame, height=4, bg="#222222", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_output.pack(fill="both", expand=True, padx=5, pady=5)

        self.console_input = tk.Entry(self.console_frame, bg="#333333", fg="#ffffff", font=("Courier", 6), bd=1, relief="solid")
        self.console_input.pack(fill="x", padx=5, pady=5)
        self.console_input.bind("<Return>", self.execute_command)

        self.redirect_console_output()

        self.controls_frame = tk.Frame(root, bg=bg_color, width=120)
        self.controls_frame.grid(row=1, column=1, padx=10, pady=10, sticky="ns")

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
            self.controls_frame, text="Spustit kameru", 
            command=self.start_camera,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.start_camera_btn.pack(fill="x", pady=5)

        self.detect_btn = tk.Button(
            self.controls_frame, text="Rozpoznat objekty", command=self.detect_objects,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.detect_btn.pack(fill="x", pady=5)

        self.edit_image_btn = tk.Button(
            self.controls_frame, text="Úpravy obrázku", command=self.edit_image,
            bg=accent_color, fg="white", font=font_bold, relief="flat", height=1
        )
        self.edit_image_btn.pack(fill="x", pady=5)

        self.chart_frame = tk.Frame(self.controls_frame, bg=bg_color)
        self.chart_frame.pack(fill="both", expand=True, pady=10)

        self.create_chart()
        self.update_chart([])

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

        root.grid_rowconfigure(1, weight=1)
        root.grid_columnconfigure(0, weight=3)
        root.grid_columnconfigure(1, weight=2)

        model_path = "/home/pi/examples/lite/examples/object_detection/raspberry_pi/efficientdet_lite0.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        self.cap = None
        self.running = False

    def redirect_console_output(self):
        sys.stdout = OutputRedirector(self.console_output)
        self.run_command("echo Připojení k systému úspěšné")

    def run_command(self, command):
        def execute():
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
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

    def load_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if file_path:
            self.video_capture = cv2.VideoCapture(file_path)
            self.show_frame_video()
            
    def start_camera(self):
        if not self.running:
            self.running = True
            self.update_camera_frame()

    def update_camera_frame(self):
        if self.running:
            try:
                from picamera2 import Picamera2
                picam2 = Picamera2()
                picam2.configure(picam2.create_preview_configuration())
                picam2.start()
                frame = picam2.capture_array()

                if frame is not None:
                    self.display_image(frame)
                self.root.after(10, self.update_camera_frame)
            except ImportError:
                self.insert_to_console("[ERROR] Modul Picamera2 není nainstalován.")
                self.stop_camera()

    def show_frame_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            self.display_image(frame)
            self.root.after(10, self.show_frame_video)
        else:
            self.video_capture.release()

    def display_image(self, img):
        self.original_image = img.copy()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_resized = img_pil.resize((canvas_width, canvas_height), Image.LANCZOS)
        img_tk = ImageTk.PhotoImage(img_resized)

        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)
        self.canvas.image = img_tk

    def create_chart(self):
        self.figure, self.ax = plt.subplots(figsize=(4, 2))
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
        image = self.original_image

        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        input_shape = input_details[0]['shape'][1:3]

        image_resized = cv2.resize(image, (input_shape[1], input_shape[0]))
        input_data = np.expand_dims(image_resized, axis=0)
        input_data = (np.float32(input_data) - 127.5) / 127.5

        self.interpreter.set_tensor(input_details[0]['index'], input_data)
        self.interpreter.invoke()

        output_data = self.interpreter.get_tensor(output_details[0]['index'])
        detections = self.parse_detections(output_data)

        self.update_chart(detections)
        self.display_detections(image, detections)

    def parse_detections(self, output_data):
        detections = []
        for detection in output_data[0]:
            confidence = detection[2]
            if confidence > 0.5:
                ymin, xmin, ymax, xmax = detection[0:4]
                object_label = "Objekt"
                detections.append({
                    'object': object_label,
                    'confidence': confidence,
                    'bbox': (xmin, ymin, xmax, ymax)
                })
        return detections

    def display_detections(self, image, detections):
        for detection in detections:
            xmin, ymin, xmax, ymax = detection['bbox']
            start_point = (int(xmin * image.shape[1]), int(ymin * image.shape[0]))
            end_point = (int(xmax * image.shape[1]), int(ymax * image.shape[0]))
            cv2.rectangle(image, start_point, end_point, (0, 255, 0), 2)

        self.display_image(image)

    def edit_image(self):
        self.console_output.insert(tk.END, "[INFO] Funkce pro úpravu obrázku není implementována.\n")

    def on_button_press(self, event):
        pass

    def on_mouse_drag(self, event):
        pass

    def on_button_release(self, event):
        pass

    def stop_camera(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.stop_camera(), root.destroy()))
    root.mainloop()

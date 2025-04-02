# Aplikace pro Rozpoznávání Objektů

Toto je aplikace pro rozpoznávání objektů v Pythonu, která využívá model YOLO (You Only Look Once) pro detekci objektů v reálném čase. Aplikace umožňuje uživatelům načítat obrázky, zachytávat obrázky z webové kamery a detekovat objekty v těchto obrázcích. Detekované objekty jsou zvýrazněny obdélníky a popisky.

## Funkce

- Načítání obrázků z místního souborového systému.
- Zachytávání obrázků z webové kamery.
- Detekce objektů v obrázcích pomocí modelu YOLO.
- Zobrazení výsledků detekce s obdélníky a skóre důvěry.
- Export výsledků detekce do souboru CSV.
- Úprava obrázků pomocí Microsoft Paint.

## Požadavky

Pro spuštění této aplikace potřebujete mít nainstalovaný Python 3.7 nebo novější na vašem systému. Následující Python balíčky jsou vyžadovány:

- `opencv-python`
- `opencv-python-headless`
- `Pillow`
- `numpy`
- `torch`
- `torchvision`
- `matplotlib`
- `pandas`
- `ultralytics`

Všechny potřebné balíčky můžete nainstalovat pomocí poskytnutého souboru `requirements.txt`.

## Instalace

1. Klonujte repozitář do svého místního počítače:

   ```bash
   git clone https://github.com/yourusername/object-detection-app.git
   cd object-detection-app

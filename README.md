# GUI pro Rozpoznávání objektů

## Přehled
GUI aplikace pro rozpoznávání objektů pomocí TensorFlow Lite. Aplikace umožňuje uživatelům nahrávat obrázky nebo sledovat živé video z kamery s výpisem detekovaných objektů v reálném čase. 

### Klíčové funkce
- **Načtení obrázku a přístup ke kameře**: Možnost nahrát obrázek ze souboru nebo spustit živé video z kamery.
- **Rozpoznávání objektů pomocí TensorFlow Lite**: Efektivní a lehká implementace modelu pro reálnou detekci objektů.
- **Nastavitelná citlivost**: Citlivost detekce lze nastavit pomocí posuvníku.
- **Elegantní a moderní uživatelské rozhraní**: Minimalistický a přehledný design pomocí Tkinteru.

## Design uživatelského rozhraní a barevné schéma
GUI je navrženo s důrazem na jednoduchost a eleganci:
- **Barevné schéma**: Jemné modré a šedé tóny s akcentovými barvami pro profesionální vzhled.
- **Zaoblené hrany**: Tlačítka a rámečky se zaoblenými hranami pro moderní, hladký vzhled.
- **Přehledné rozvržení**: Tlačítka jsou jasně označená a umístěná pro snadnou orientaci.

### Barevná paleta
| Prvek               | Barva         | Hex kód    |
|---------------------|---------------|------------|
| Pozadí aplikace     | Světle šedá   | `#fafafa`  |
| Pozadí rámečků      | Modrošedá     | `#e3e8f0`  |
| Akcentní barva      | Modrá         | `#3a7bd5`  |
| Barva textu         | Tmavě šedá    | `#333333`  |

## Požadované knihovny
- opencv-python==4.10.0
- opencv-python-headless==4.10.0
- Pillow==9.2.0
- numpy==1.23.1
- torch==1.12.1
- torchvision==0.13.1
- matplotlib==3.5.1
- pandas==1.4.2
- ultralytics==8.0.0


## Návod na použití
1. **Načtení obrázku**: Klikněte na „Načíst obrázek“ a nahrajte obrázek pro detekci.
2. **Spuštění kamery**: Klikněte na „Spustit kameru“ pro aktivaci webkamery a zahájení živého záznamu.
3. **Rozpoznání objektů**: Stiskněte „Rozpoznat objekty“ pro spuštění rozpoznávání a zobrazení výsledků.
4. **Nastavení citlivosti**: Pomocí posuvníku citlivosti lze upravit míru detekce.

## Postup instalace
1. Naklonujte si repozitář a přejděte do složky projektu:
   ```bash
   git clone https://github.com/Kybernetika-SPSE/DMP_Sak_Simek
   cd ai-object-detection
2. Nainstalujte potřebné balíčky:
   ```bash
   pip install -r requirements.txt
3. Spusťte aplikaci:
   ```bash
   python app.py

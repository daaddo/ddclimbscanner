import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image
from transformers import Sam3TrackerProcessor, Sam3TrackerModel, Sam3Processor, Sam3Model

# --- 1. SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Caricamento Modelli
print("Caricamento modelli...")
processor_text = Sam3Processor.from_pretrained("facebook/sam3")
model_text = Sam3Model.from_pretrained("facebook/sam3").to(device)

processor_tracker = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
model_tracker = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)

image_path = "img.jpg"  # <--- INSERISCI LA TUA IMMAGINE
image = Image.open(image_path).convert("RGB")
W, H = image.size

# --- 2. STRUTTURA DATI GLOBALE ---
# Ogni elemento è un dizionario:
# {
#   'id': int,
#   'mask': np.array (H, W),
#   'box': list [x1, y1, x2, y2] (o None),
#   'points': list [[x,y], ...],
#   'labels': list [1, 0, ...]
# }
oggetti_scena = []
stato_interfaccia = {
    "modalita": "AGGIUNGI",  # "AGGIUNGI" o "MIGLIORA"
    "id_selezionato": None   # Indice dell'oggetto in fase di modifica
}

# --- 3. HELPER TRACKER ---
def esegui_tracker(box, punti, etichette):
    # Prepara gli input nel formato nidificato richiesto da SAM3
    # Box: [[ [x,y,x,y] ]] -> shape [1, 1, 4]
    fmt_box = [[box]] if box is not None else None
    
    # Points: [[ [[x,y], [x,y]] ]] -> shape [1, 1, N, 2]
    fmt_points = [[punti]] if len(punti) > 0 else None
    fmt_labels = [[etichette]] if len(etichette) > 0 else None

    # Se non abbiamo né box né punti, non possiamo fare nulla
    if fmt_box is None and fmt_points is None:
        return np.zeros((H, W), dtype=bool)

    inputs = processor_tracker(
        images=image,
        input_boxes=fmt_box,
        input_points=fmt_points,
        input_labels=fmt_labels,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model_tracker(**inputs)

    masks = processor_tracker.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"], 
        mask_threshold=0.0, 
        binarize=True
    )[0]
    
    # Ritorna la maschera migliore (indice 0)
    return masks[0, 0].numpy()

# --- 4. STEP 1: ANALISI TESTUALE ---
def inizializza_con_testo(prompt):
    global model_text
    print(f"Analisi testo: '{prompt}'...")
    inputs = processor_text(
        images=image,
        text=prompt,
        return_tensors="pt"
        ).to(device)
    with torch.no_grad():
        out = model_text(**inputs)
    
    # Otteniamo Box e Maschere iniziali
    results = processor_text.post_process_instance_segmentation(
        #TRESHOLD INIZIALE, MOLTO BASSO FORSE è MEGLIO
        out, target_sizes=[image.size[::-1]], threshold=0.14
    )[0]

    count = 0
    if "boxes" in results:
        for i in range(len(results["boxes"])):
            box = results["boxes"][i].cpu().numpy().tolist()
            mask = results["masks"][i].cpu().numpy()
            
            # Creiamo l'oggetto in memoria
            nuovo_obj = {
                'id': count,
                'mask': mask,
                'box': box,     # Importante: salviamo il box generato dal testo
                'points': [],   # Nessun punto manuale ancora
                'labels': []
            }
            oggetti_scena.append(nuovo_obj)
            count += 1
            
    print(f"Trovati {count} oggetti.")
    del model_text # Pulizia
    torch.cuda.empty_cache()

# Eseguiamo l'init
inizializza_con_testo("climbing holds") # <--- CAMBIA IL PROMPT

# --- 5. INTERFACCIA E LOGICA ---
fig, ax = plt.subplots(figsize=(10, 10))
plt.subplots_adjust(bottom=0.2) # Lascia spazio per i tasti

def show_mask_overlay(mask, ax, is_selected=False):
    # Colore diverso se selezionato (Giallo/Oro) o normale (Blu)
    if is_selected:
        color = np.array([255/255, 215/255, 0/255, 0.6]) 
    else:
        color = np.array([30/255, 144/255, 255/255, 0.4])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def ridisegna():
    ax.clear()
    ax.imshow(image)
    
    titolo = f"Modalità: {stato_interfaccia['modalita']}"
    
    for idx, obj in enumerate(oggetti_scena):
        selezionato = (idx == stato_interfaccia['id_selezionato'])
        show_mask_overlay(obj['mask'], ax, is_selected=selezionato)
        
        # Disegna i punti se presenti
        if obj['points']:
            pts = np.array(obj['points'])
            lbs = np.array(obj['labels'])
            # Verdi positivi, Rossi negativi
            ax.scatter(pts[lbs==1,0], pts[lbs==1,1], c='g', marker='*', s=100, edgecolors='w')
            ax.scatter(pts[lbs==0,0], pts[lbs==0,1], c='r', marker='x', s=100, edgecolors='w')

    if stato_interfaccia['modalita'] == "MIGLIORA" and stato_interfaccia['id_selezionato'] is None:
        titolo += "\nClicca su una maschera per selezionarla!"
    elif stato_interfaccia['modalita'] == "MIGLIORA":
        titolo += f"\nModifica oggetto ID {stato_interfaccia['id_selezionato']} (Sx: +, Dx: -)"
    else:
        titolo += "\nClicca nel vuoto per aggiungere nuovi oggetti"

    ax.set_title(titolo)
    ax.axis('off')
    fig.canvas.draw()

# --- GESTORE CLICK ---
def on_map_click(event):
    if event.inaxes != ax: return
    x, y = int(event.xdata), int(event.ydata)
    
    # 1. LOGICA MIGLIORA
    if stato_interfaccia['modalita'] == "MIGLIORA":
        # A. Se nessun oggetto è selezionato, cerchiamo chi abbiamo cliccato
        if stato_interfaccia['id_selezionato'] is None:
            found = False
            # Iteriamo al contrario per prendere l'ultimo disegnato (in cima)
            for i in range(len(oggetti_scena)-1, -1, -1):
                mask = oggetti_scena[i]['mask']
                # Controlliamo bounds e se il pixel è True
                if 0 <= y < H and 0 <= x < W and mask[y, x]:
                    stato_interfaccia['id_selezionato'] = i
                    print(f"Selezionato oggetto {i}")
                    found = True
                    break
            if not found:
                print("Nessuna maschera qui.")
            ridisegna()
            return

        # B. Oggetto selezionato -> Aggiungiamo punti di rifinitura
        idx = stato_interfaccia['id_selezionato']
        obj = oggetti_scena[idx]
        
        # Click sx (1) = Positivo, Dx (3) = Negativo
        lbl = 1 if event.button == 1 else 0
        obj['points'].append([x, y])
        obj['labels'].append(lbl)
        
        print(f"Raffinamento ID {idx}: Punti totali {len(obj['points'])}")
        
        # Ricalcolo Tracker usando Box originale + Tutti i punti accumulati
        nuova_mask = esegui_tracker(obj['box'], obj['points'], obj['labels'])
        obj['mask'] = nuova_mask
        ridisegna()

    # 2. LOGICA AGGIUNGI
    elif stato_interfaccia['modalita'] == "AGGIUNGI":
        print(f"Aggiunta nuovo oggetto in {x}, {y}")
        # Creiamo un nuovo oggetto da zero (senza box, solo punto)
        nuovo_obj = {
            'id': len(oggetti_scena),
            'mask': None, # Lo calcoliamo subito
            'box': None,
            'points': [[x, y]],
            'labels': [1]
        }
        # Calcolo iniziale
        mask = esegui_tracker(None, nuovo_obj['points'], nuovo_obj['labels'])
        nuovo_obj['mask'] = mask
        oggetti_scena.append(nuovo_obj)
        ridisegna()

cid = fig.canvas.mpl_connect('button_press_event', on_map_click)

# --- TASTI (MATPLOTLIB WIDGETS) ---
ax_add = plt.axes([0.15, 0.05, 0.2, 0.075])
ax_ref = plt.axes([0.40, 0.05, 0.2, 0.075])
ax_del = plt.axes([0.65, 0.05, 0.2, 0.075])

b_add = Button(ax_add, 'Aggiungi')
b_ref = Button(ax_ref, 'Migliora')
b_del = Button(ax_del, 'Elimina')

def set_mode_add(event):
    stato_interfaccia['modalita'] = "AGGIUNGI"
    stato_interfaccia['id_selezionato'] = None # Deseleziona
    ridisegna()

def set_mode_refine(event):
    stato_interfaccia['modalita'] = "MIGLIORA"
    stato_interfaccia['id_selezionato'] = None # Reset selezione
    ridisegna()

def delete_object(event):
    idx = stato_interfaccia['id_selezionato']
    if idx is not None:
        print(f"Eliminazione oggetto indice {idx}")
        oggetti_scena.pop(idx)
        stato_interfaccia['id_selezionato'] = None
        ridisegna()
    else:
        print("Seleziona un oggetto (in modalità Migliora) per eliminarlo.")

b_add.on_clicked(set_mode_add)
b_ref.on_clicked(set_mode_refine)
b_del.on_clicked(delete_object)

ridisegna()
plt.show()
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3TrackerProcessor, Sam3TrackerModel, Sam3Processor, Sam3Model
import random

# --- 1. SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Carichiamo entrambi i processori
# Sam3Processor -> Per il testo (Step 1)
processor_text = Sam3Processor.from_pretrained("facebook/sam3")
model_text = Sam3Model.from_pretrained("facebook/sam3").to(device)

# Sam3TrackerProcessor -> Per i click (Step 4)
processor_tracker = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
model_tracker = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)

image_path = "img.jpg" # <--- METTI LA TUA IMMAGINE
image = Image.open(image_path).convert("RGB")

# Lista globale per accumulare TUTTE le maschere (Text + Click)
# Ogni elemento sarà un array numpy (H, W) o tensore
maschere_globali = [] 

# --- STEP 1: ANALISI TRAMITE PROMPT ---
prompt_iniziale = "climbing holds"  # <--- CAMBIA IL PROMPT
print(f"Step 1: Analisi iniziale con prompt: '{prompt_iniziale}'")

inputs_text = processor_text(images=image, text=prompt_iniziale, return_tensors="pt").to(device)
with torch.no_grad():
    outputs_text = model_text(**inputs_text)

# Post-processing per ottenere le maschere binarie
# Nota: post_process_instance_segmentation ci dà maschere separate per ogni oggetto
risultati_text = processor_text.post_process_instance_segmentation(
    outputs_text, 
    threshold=0.2, 
    target_sizes=[image.size[::-1]]
)[0]

# Salviamo le maschere trovate dal testo nella lista globale
if "masks" in risultati_text:
    for mask in risultati_text["masks"]:
        # mask è (H, W), la convertiamo in numpy e salviamo
        maschere_globali.append(mask.cpu().numpy())
    print(f"--> Trovate {len(maschere_globali)} maschere col testo.")
else:
    print("--> Nessun oggetto trovato col testo.")

# Liberiamo RAM del modello testo (opzionale)
del model_text
torch.cuda.empty_cache()


# --- FUNZIONE HELPER TRACKER (STEP 4) ---
def calcola_nuovo_oggetto(click_coords):
    print(f"Step 4: Calcolo nuovo oggetto alle coordinate {click_coords}")
    
    # Input per il tracker: Solo il punto cliccato
    # Formato points: [image_lvl, object_lvl, point_lvl, coords]
    punti = [[ [click_coords] ]] 
    etichette = [[ [1] ]] # 1 = punto positivo

    inputs = processor_tracker(
        images=image,
        input_points=punti,
        input_labels=etichette,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model_tracker(**inputs)

    # Otteniamo la maschera del NUOVO oggetto
    masks = processor_tracker.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"], 
        mask_threshold=0.0, 
        binarize=True
    )[0] # Batch 0
    
    # Sam ritorna 3 maschere (multimask output), prendiamo la migliore (indice 0 solitamente)
    # Shape masks: (Num_objects, 1, H, W) -> prendiamo [0, 0]
    best_mask = masks[0, 0].numpy()
    
    return best_mask

# --- VISUALIZZAZIONE (STEP 2) ---
def show_mask(mask, ax, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# --- LOOP INTERATTIVO ---
fig, ax = plt.subplots(figsize=(10, 10))

def ridisegna_scena():
    ax.clear()
    ax.imshow(image)
    
    # Step 2: Mostra tutte le maschere accumulate (Testo + Click precedenti)
    for mask in maschere_globali:
        show_mask(mask, ax)
        
    ax.set_title(f"Prompt: '{prompt_iniziale}'\nOggetti totali: {len(maschere_globali)} (Clicca per aggiungerne)")
    ax.axis('off')
    fig.canvas.draw()

# Primo disegno (solo risultati testo)
ridisegna_scena()

# --- STEP 3: GESTIONE CLICK ---
def on_click(event):
    if event.inaxes != ax: return
    
    # Coordinate del click
    x, y = int(event.xdata), int(event.ydata)
    print(f"Step 3: Click utente su {x}, {y} (Oggetto mancato!)")
    
    # Feedback visivo "Sto calcolando"
    ax.plot(x, y, 'rx')
    fig.canvas.draw()
    
    # --- STEP 4: RICALCOLO TRAMITE TRACKER ---
    nuova_maschera = calcola_nuovo_oggetto([x, y])
    
    # Aggiungi la nuova maschera alla lista globale ("Passi tutte le maschere...")
    maschere_globali.append(nuova_maschera)
    
    # Ridisegna tutto ("mostri l'immagine")
    ridisegna_scena()

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
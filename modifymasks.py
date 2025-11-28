import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import Sam3TrackerProcessor, Sam3TrackerModel

# --- 1. SETUP ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" using cuda? :{torch.cuda.is_available()}")
processor = Sam3TrackerProcessor.from_pretrained("facebook/sam3")
model = Sam3TrackerModel.from_pretrained("facebook/sam3").to(device)

image_path = "img.jpg"
image = Image.open(image_path).convert("RGB")

# --- STATO GLOBALE DELL'OGGETTO (ID 0) ---
# punti correnti è una lista di lista, ciascuna inner list è costituita da due elementi
#  x,y che corrispondono ai punti, ogni elemento della lista esterna è un punto
#etichette correnti è una lista di dimensioni UGUALE a punti correnti, che riferisce se un punto è di addizione o sottrazione 
#ad esempio 1,0 -> aggiungi l oggetto sul punto primo punti_correnti , sottrai l oggetto dal secondo elemento in punti_correnti
punti_correnti = [[1200, 97]]  
etichette_correnti = [1]      



# Buffer per contare i click temporanei
nuovi_click_buffer = [] 

# --- FUNZIONE HELPER: ESEGUE SAM ---
def esegui_sam(punti, etichette):
    print(f"--> Esecuzione SAM su {len(punti)} punti totali...")
    
    
    inputs = processor(
        images=image,
        input_points=[[ punti ]],    
        input_labels=[[ etichette ]], 
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-processing
    nuova_maschera_batch = processor.post_process_masks(
        outputs.pred_masks.cpu(), 
        inputs["original_sizes"], 
        mask_threshold=0.0,
        binarize=True
    )[0]
    
    # Restituisce la maschera migliore (indice 0 delle 3 proposte)
    return nuova_maschera_batch[0]

# --- FUNZIONI DI VISUALIZZAZIONE ---
def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6]) 
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax):
    coords = np.array(coords)
    labels = np.array(labels)
    # Ora coords ha shape (N, 2) e labels (N,), quindi il filtro funziona
    pos_points = coords[labels==1]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=375, edgecolor='white', linewidth=1.25)


# --- 2. GENERAZIONE INIZIALE ---
print("Generazione maschera iniziale...")
maschera_attuale = esegui_sam(punti_correnti, etichette_correnti)


# --- 3. INTERFACCIA GRAFICA ---
fig = plt.figure(figsize=(10, 10))
ax = plt.gca()

def disegna_tutto():
    ax.clear() # Pulisce il vecchio grafico
    ax.imshow(image) # Ridisegna foto base
    
    # FIX 3: maschera_attuale è già la maschera 2D selezionata (vedi return di esegui_sam)
    # Quindi non serve [0] qui se l'abbiamo già preso dentro esegui_sam
    # Ma per sicurezza controlliamo: se è 3D (C, H, W) prendiamo la prima.
    mask_to_show = maschera_attuale
    if len(mask_to_show.shape) == 3:
        mask_to_show = mask_to_show[0]
        
    show_mask(mask_to_show.cpu().numpy(), ax) 
    show_points(punti_correnti, etichette_correnti, ax) 
    
    ax.set_title(f"Patata ID 0 - Punti totali: {len(punti_correnti)}\nClicca 2 volte per raffinare la maschera")
    ax.axis('on')
    fig.canvas.draw()

# Primo disegno
disegna_tutto()

# --- 4. GESTORE EVENTI ---
def on_click(event):
    if event.inaxes != ax: return
    
    x, y = int(event.xdata), int(event.ydata)
    print(f"Click registrato: {x}, {y}")
    
    # Feedback visivo immediato
    ax.plot(x, y, 'rx') 
    fig.canvas.draw()
    
    nuovi_click_buffer.append([x, y])
    
    if len(nuovi_click_buffer) % 7==0:
        print("\nRaggiunti 2 nuovi punti! Rigenerazione maschera...")
        
        # Aggiungi allo storico
        punti_correnti.extend(nuovi_click_buffer)
        etichette_correnti.extend([1] * len(nuovi_click_buffer))
        
        # Ricalcola
        global maschera_attuale
        maschera_attuale = esegui_sam(punti_correnti, etichette_correnti)
        
        # Aggiorna UI
        disegna_tutto()
        
        # Reset buffer
        nuovi_click_buffer.clear()
        print("Fatto.")

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()
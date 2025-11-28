from transformers import Sam3Processor, Sam3Model
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Funzione di supporto per disegnare le maschere ---
def show_mask(mask, ax, random_color=True):
    if random_color:
        # Genera un colore casuale (RGB) + Alpha (trasparenza)
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    
    h, w = mask.shape[-2:]
    # Reshape della maschera per applicare il colore
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# --- 2. Setup Modello e Immagine ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Sam3Model.from_pretrained("facebook/sam3").to(device)
processor = Sam3Processor.from_pretrained("facebook/sam3")

image_path = "img.jpg"  # Assicurati che l'immagine sia nella stessa cartella dello script
image = Image.open(image_path).convert("RGB")

# --- 3. Inferenza ---
# Modifica qui il testo per cercare oggetti diversi (es. "cat", "remote", "blanket")
text_prompt = "climbing holds" 
inputs = processor(
    images=image,
    text=text_prompt,
    return_tensors="pt",
    input_points= [[[1]]]
    ).to(device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs,
    threshold=0.5,
    mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist()
)[0]

print(f"Trovati {len(results['masks'])} oggetti per il prompt: '{text_prompt}'")

print(f"results sono: {results}")

# --- 4. Visualizzazione ---
plt.figure(figsize=(10, 10))
plt.imshow(image) # Mostra l'immagine base

# Itera su tutte le maschere trovate e disegnale
if len(results['masks']) > 0:
    for mask in results['masks']:
        # Convertiamo il tensore in numpy array per matplotlib
        show_mask(mask.cpu().numpy(), plt.gca())
else:
    print("Nessuna maschera trovata.")

plt.axis('off') # Nasconde gli assi (numeri sui lati)
plt.title(f"Segmentazione per: {text_prompt}")
plt.show()
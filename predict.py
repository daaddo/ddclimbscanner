# predict.py
from cog import BasePredictor, Input, Path
import torch
import numpy as np
from PIL import Image, ImageDraw
from transformers import Sam3Processor, Sam3Model, Sam3Config
from safetensors.torch import load_file
import os
import shutil
import time
from typing import List
import logging

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

class Predictor(BasePredictor):
    def setup(self):
        """Carica il modello in memoria (eseguito una volta sola all'avvio)"""
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("tokenizers").setLevel(logging.ERROR)
        logging.getLogger("datasets").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

        log("Inizio Setup...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log(f"Using device: {self.device}")
        

        # 1. Carica Processor e Config (leggeri)
        log(f"Caricamento configurazione da Davidinos/sam3data...")
        self.processor = Sam3Processor.from_pretrained("Davidinos/sam3data")
        log("processor caricato, caricamento modello")
        self.model_text =Sam3Model.from_pretrained("Davidinos/sam3data").to(self.device)
        
        log("Model loaded successfully.")

    def predict(
        self,

        image: Path = Input(description="Immagine da analizzare"),
        prompt: str = Input(description="Prompt testuale per la segmentazione", default="climbing holds"),
        threshold: float = Input(description="Soglia di confidenza", default=0.14)
    ) -> List[Path]:
        """Esegue la previsione e restituisce una lista di maschere individuali"""
        log(f"--- Nuova Richiesta ---")
        log(f"Prompt: '{prompt}' | Threshold: {threshold}")
        
        # Pulisci output precedenti se necessario
        output_dir = Path("/tmp/masks")
        if output_dir.exists():
            print("dir existing")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Carica immagine
        log(f"Caricamento immagine: {image}")
        pil_image = Image.open(image).convert("RGB")
        log(f"Dimensioni immagine: {pil_image.size}")
        
        # 2. Esegui il modello
        log("Esecuzione SAM3 Inference...")
        start_inf = time.time()
        inputs = self.processor(
            images=pil_image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model_text(**inputs)
        log(f"Inferenza completata in {time.time() - start_inf:.2f}s")

        # 3. Post-processing
        log("Post-processing risultati...")
        results = self.processor.post_process_instance_segmentation(
            outputs, 
            target_sizes=[pil_image.size[::-1]], 
            threshold=threshold
        )[0]

        # 4. Salva ogni maschera separatamente
        output_paths = []
        
        if "masks" in results:
            masks = results["masks"].cpu().numpy()
            log(f"Trovati {len(masks)} oggetti potenziali. Inizio salvataggio...")
            
            for i, mask in enumerate(masks):
                # mask è un array booleano (H, W) -> Convertiamo in uint8 (0 o 255)
                mask_uint8 = (mask * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_uint8, mode="L")
                
                # Salviamo il file singolo
                print(f"img = {output_dir / f"mask_{i:03d}.png"}")
                file_path = output_dir / f"mask_{i:03d}.png"
                mask_pil.save(file_path)
                output_paths.append(file_path)

        log(f"Completato. Restituisco {len(output_paths)} maschere.")
        return output_paths

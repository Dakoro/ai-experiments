# Fine-tuning de FLUX.1-dev avec LoRA : Style Alphonse Mucha

## Introduction

Ce notebook présente une implémentation complète du fine-tuning du modèle de génération d'images FLUX.1-dev en utilisant la technique LoRA (Low-Rank Adaptation) pour apprendre le style artistique d'Alphonse Mucha, célèbre artiste Art Nouveau.

## Architecture et Approche

### Modèle Base
- **FLUX.1-dev** : Modèle de diffusion de dernière génération développé par Black Forest Labs
- **Quantification 4-bit** : Utilisation de BitsAndBytesConfig pour réduire l'empreinte mémoire
- **LoRA** : Adaptation de rang faible pour un entraînement efficace sans modifier le modèle complet

### Dataset
- **Dataset utilisé** : `derekl35/alphonse-mucha-style`
- **Preprocessing** : Génération d'embeddings textuels pré-calculés pour optimiser l'entraînement
- **Résolution** : Images redimensionnées à 512x768 pixels

## Étapes du Pipeline

### 1. Génération des Embeddings (`get_embeddings`)

```python
# Chargement du pipeline FLUX avec quantification 4-bit
text_encoder = T5EncoderModel.from_pretrained(
    id, subfolder="text_encoder_2",
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)
```

**Fonctionnalités clés :**
- Encodage des prompts textuels avec T5
- Génération de hashes SHA256 pour chaque image
- Sauvegarde des embeddings dans un fichier Parquet optimisé

### 2. Dataset Personnalisé (`DreamBoothDataset`)

La classe `DreamBoothDataset` gère :
- **Chargement des images** avec transformations automatiques
- **Mapping des embeddings** pré-calculés via les hashes d'images
- **Preprocessing** : redimensionnement, crop aléatoire, normalisation

### 3. Entraînement avec LoRA

#### Configuration LoRA
```python
transformer_lora_config = LoraConfig(
    r=4,  # Rang de la décomposition
    lora_alpha=4,
    target_modules=["to_k", "to_q", "to_v", "to_out.0"]
)
```

#### Optimisations Techniques
- **Quantification NF4** : Réduit la mémoire de ~50%
- **Gradient Checkpointing** : Compromis mémoire/calcul
- **Mixed Precision FP16** : Accélération de l'entraînement
- **AdamW 8-bit** : Optimiseur efficace en mémoire

### 4. Boucle d'Entraînement Avancée

#### Sampling des Timesteps
```python
u = compute_density_for_timestep_sampling("none", bsz, 0.0, 1.0, 1.29)
indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
```

#### Fonction de Perte
- **Flow Matching** : Utilise le scheduler Euler discret
- **Weighting scheme** : Pondération adaptative des timesteps
- **Target** : `noise - model_input` (formulation flow matching)

## Optimisations Mémoire

### Techniques Employées
1. **Cache des Latents VAE** : Pré-calcul pour éviter l'encodage répétitif
2. **Quantification 4-bit** : Réduction drastique de l'empreinte mémoire
3. **LoRA** : Seulement ~1% des paramètres entraînables
4. **Gradient Accumulation** : Simulation de batch size plus important

### Gestion Mémoire
```python
del vae
free_memory()
torch.cuda.empty_cache()
```

## Configuration d'Entraînement

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| Learning Rate | 1e-4 | Taux d'apprentissage conservateur |
| Batch Size | 1 | Limité par la mémoire GPU |
| Gradient Accumulation | 4 | Simulation effective de batch=4 |
| LoRA Rank | 4 | Compromis performance/mémoire |
| Max Steps | 100 | Entraînement rapide de démonstration |
| Resolution | 512x768 | Format portrait adapté au style Mucha |

## Résultats et Utilisation

### Sauvegarde du Modèle
Le modèle entraîné est sauvegardé au format LoRA compatible avec la pipeline FLUX standard :

```python
FluxPipeline.save_lora_weights(
    output_dir, 
    transformer_lora_layers=lora_layers
)
```

### Chargement et Inférence
```python
# Chargement du modèle fine-tuné
pipeline = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
pipeline.load_lora_weights("./lora_flux_nf4")

# Génération d'images dans le style Mucha
image = pipeline("A woman in Art Nouveau style by Alphonse Mucha").images[0]
```

## Avantages de cette Approche

### Efficacité
- **Mémoire réduite** : Entraînement possible sur GPU 8-16GB
- **Temps d'entraînement** : Convergence rapide avec LoRA
- **Flexibilité** : Adaptation facile à d'autres styles artistiques

### Qualité
- **Préservation du modèle base** : Pas de dégradation des capacités générales
- **Spécialisation** : Apprentissage ciblé du style Mucha
- **Compatibilité** : Intégration seamless avec l'écosystème Diffusers

## Conclusion

Ce notebook démontre une implémentation moderne et efficace du fine-tuning de modèles de diffusion. L'utilisation combinée de LoRA, de la quantification 4-bit et des optimisations mémoire permet d'adapter FLUX.1-dev à des styles artistiques spécifiques avec des ressources computationnelles limitées.

Cette approche ouvre la voie à la démocratisation du fine-tuning de modèles de génération d'images de haute qualité, rendant accessible la création de modèles spécialisés pour des applications artistiques ou commerciales spécifiques.
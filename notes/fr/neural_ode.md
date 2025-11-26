# Rapport Technique : Neural ODE - Apprentissage de la Géométrie Temporelle

## Vue d'ensemble

Ce notebook démontre l'utilisation des **Neural Ordinary Differential Equations (Neural ODE)** pour l'apprentissage de systèmes dynamiques et l'extrapolation temporelle. L'hypothèse centrale est que l'apprentissage du "mouvement" (le champ vectoriel) plutôt que des "instantanés" (points statiques) permet au modèle d'extrapoler vers le futur.

## 1. Concept Fondamental

### Paradigme Neural ODE

Au lieu de prédire directement la position `y` à un temps `t`, le Neural ODE prédit la dérivée temporelle `dy/dt`. Le réseau apprend le **champ vectoriel** qui régit l'évolution du système, c'est-à-dire la "pente" de l'univers en tout point de l'espace d'états.

**Équation fondamentale** :
```
dy/dt = f(y, θ)
```

où `f` est un réseau de neurones paramétré par `θ`.

## 2. Architecture

### 2.1 Réseau ODEFunc (Moteur Physique)

```python
class ODEFunc(nn.Module):
    def __init__(self, hidden_dim=50):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),      # Entrée: position (x, y)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2)       # Sortie: vélocité (dx/dt, dy/dt)
        )
```

**Caractéristiques** :
- **Entrée** : État actuel (x, y) dans l'espace 2D
- **Sortie** : Vecteur vélocité (dx/dt, dy/dt)
- **Architecture** : MLP à 3 couches avec activations Tanh
- **Initialisation** : Poids gaussiens (μ=0, σ=0.1) pour des dynamiques lisses
- **Autonomie** : Le système est autonome (invariant par translation temporelle), le temps `t` n'est pas utilisé en entrée

### 2.2 Réseau MLP Standard (Comparaison)

Pour comparaison, un MLP standard est implémenté :

```python
class StandardMLP(nn.Module):
    def forward(self, t):
        # Entrée: temps t
        # Sortie: position (x, y)
```

Ce modèle apprend la relation **euclidienne** directe `y = f(t)` au lieu de la **géométrie temporelle** `dy/dt = f(y)`.

## 3. Génération des Données

### Spirale Dynamique

Les données d'entraînement sont générées à partir d'un système linéaire :

```python
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])
dy/dt = y @ A
y0 = [2., 0.]
t ∈ [0, 25]
```

**Propriétés** :
- **Trajectoire** : Spirale dans l'espace 2D
- **Bruit** : Gaussien (σ=0.1) pour simuler l'imperfection des capteurs
- **Points** : 1000 échantillons temporels
- **Division** : 60% passé (entraînement), 40% futur (test d'extrapolation)

## 4. Procédure d'Entraînement

### 4.1 Configuration

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `batch_time` | 10 | Longueur des trajectoires dans chaque batch |
| `batch_size` | 20 | Nombre de trajectoires par batch |
| `train_split` | 0.6 | Proportion des données pour l'entraînement |
| `niters` | 2000 | Nombre d'itérations |
| `optimizer` | RMSprop | lr=1e-3 pour Neural ODE |
| `optimizer` | Adam | lr=1e-3 pour MLP |

### 4.2 Algorithme d'Entraînement Neural ODE

1. **Échantillonnage** : Sélection aléatoire de `batch_size` points de départ dans le passé
2. **Intégration** : Calcul de la trajectoire prédite via `odeint(func, y0, t_batch)`
3. **Perte** : Erreur absolue moyenne entre trajectoire prédite et réelle
   ```python
   loss = mean(|pred_y - true_y|)
   ```
4. **Rétropropagation** : Mise à jour des poids via backpropagation à travers le solveur ODE
5. **Itération** : Répétition sur 2000 itérations

### 4.3 Algorithme d'Entraînement MLP

Le MLP standard apprend la correspondance directe `(t) → (x, y)` sur des points échantillonnés aléatoirement.

## 5. Résultats et Extrapolation

### 5.1 Capacité d'Extrapolation

Le Neural ODE est testé sur sa capacité à **prédire le futur** :
- **Condition initiale** : `y(t=0)`
- **Intégration** : Sur toute la plage temporelle `t ∈ [0, 25]`
- **Test critique** : Les positions dans `t ∈ [15, 25]` n'ont JAMAIS été vues pendant l'entraînement

```python
full_pred_y = odeint(func, true_y[0].unsqueeze(0), t)
```

### 5.2 Visualisation du Champ Vectoriel

Le notebook génère une **visualisation de la géométrie temporelle apprise** :

```python
# Grille 20×20 dans l'espace [-3, 3]²
X, Y = np.meshgrid(x_range, y_range)
velocities = func(0, grid_points)  # Calcul des vecteurs vitesse
plt.quiver(X, Y, U, V)  # Affichage des flèches
```

Cette visualisation révèle les "flèches invisibles" qui guident la dynamique du système.

## 6. Comparaison Neural ODE vs MLP Standard

### Expérience 2 : Géométrie Temporelle vs Géométrie Euclidienne

| Aspect | Neural ODE | MLP Standard |
|--------|-----------|--------------|
| **Paradigme** | Apprend le flux `dy/dt = f(y)` | Apprend la courbe `y = f(t)` |
| **Entrée** | État actuel (x, y) | Temps t |
| **Sortie** | Vitesse (dx/dt, dy/dt) | Position (x, y) |
| **Extrapolation** | Excellente (suit la dynamique) | Pauvre (interpolation polynomiale) |
| **Invariance** | Invariant par translation temporelle | Dépendant du temps absolu |
| **Interprétabilité** | Champ vectoriel visualisable | Boîte noire |

### Observations Attendues

D'après les commentaires du code :
- La **ligne noire (Neural ODE)** devrait suivre correctement la spirale dans la zone rouge (futur)
- La **ligne verte (MLP)** devrait diverger ou échouer à capturer la dynamique dans la zone non vue

## 7. Implémentation Technique

### 7.1 Bibliothèques Requises

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchdiffeq import odeint  # Solveur ODE différentiable
import matplotlib.pyplot as plt
import numpy as np
```

### 7.2 Solveur ODE

Le notebook utilise `torchdiffeq.odeint` qui implémente :
- **Méthode** : Runge-Kutta adaptatif (par défaut Dopri5)
- **Différentiabilité** : Backpropagation à travers le solveur via la méthode adjointe
- **Interface** : `odeint(func, y0, t)` retourne les états à tous les temps `t`

### 7.3 Stratégies d'Initialisation

```python
nn.init.normal_(m.weight, mean=0, std=0.1)
nn.init.constant_(m.bias, val=0)
```

Initialisation avec petits poids pour garantir des **dynamiques initiales lisses** et éviter les instabilités numériques.

## 8. Avantages et Limitations

### 8.1 Avantages du Neural ODE

1. **Extrapolation temporelle** : Capacité à prédire au-delà de l'horizon d'entraînement
2. **Efficacité mémoire** : Pas besoin de stocker tous les états intermédiaires
3. **Flexibilité temporelle** : Évaluation à n'importe quel instant `t`
4. **Interprétabilité** : Le champ vectoriel peut être visualisé et analysé
5. **Invariance** : Capture les lois physiques indépendantes du temps absolu

### 8.2 Limitations

1. **Coût computationnel** : Le solveur ODE est plus lent qu'un forward pass direct
2. **Stabilité numérique** : Peut diverger si le champ vectoriel est mal conditionné
3. **Hypothèse** : Suppose que le système est effectivement régi par une ODE
4. **Complexité** : Backpropagation à travers le solveur ajoute de la complexité

## 9. Cas d'Usage et Applications

Les Neural ODE sont particulièrement adaptés pour :

- **Séries temporelles irrégulières** : Données échantillonnées à intervalles non uniformes
- **Physique** : Systèmes dynamiques, mécanique, astronomie
- **Biologie** : Dynamiques de populations, réseaux génétiques
- **Finance** : Modélisation de processus continus
- **Robotique** : Planification de trajectoires, contrôle
- **Vision** : Flux optique, suivi d'objets

## 10. Conclusions

Ce notebook démontre un principe fondamental :

> **"En apprenant la géométrie de l'espace (les règles du mouvement), plutôt que les points individuels, le modèle peut naviguer vers des régions jamais vues."**

Le Neural ODE transforme un problème d'apprentissage supervisé en un problème d'**apprentissage des lois physiques**, ce qui confère au modèle une capacité d'extrapolation bien supérieure aux approches classiques.

### Perspective Philosophique

Le commentaire du code résume élégamment :
- **MLP Standard** : Apprend les "photographies" (géométrie euclidienne)
- **Neural ODE** : Apprend le "film" (géométrie temporelle)

Cette distinction capture l'essence de la différence entre **interpolation** (relier des points) et **simulation** (suivre des lois dynamiques).

## Références Techniques

- **Papier original** : "Neural Ordinary Differential Equations" (Chen et al., NeurIPS 2018)
- **Bibliothèque** : `torchdiffeq` - https://github.com/rtqichen/torchdiffeq
- **Méthode d'intégration** : Dormand-Prince (Dopri5) - solveur Runge-Kutta adaptatif d'ordre 5(4)

---

**Auteur du notebook** : Démonstration pédagogique de l'hypothèse "motion vs stills"
**Date du rapport** : 2025-11-26
**Fichier source** : `notebooks/neural_ode.ipynb`

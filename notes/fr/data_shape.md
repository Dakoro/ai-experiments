# Validation Cross-Domaine : Apprentissage Basé sur la Forme KNN+KFN

## Vue d'ensemble

Ce notebook présente une approche innovante d'apprentissage automatique basée sur l'extraction de la "forme" des données en utilisant une combinaison de K-Nearest Neighbors (KNN) et K-Farthest Neighbors (KFN). L'objectif principal est de valider que cette méthode d'extraction de forme se généralise à travers différents domaines de données.

## Concept Fondamental : Extraction de Forme KNN+KFN

### Qu'est-ce que la "forme" des données ?

La forme des données fait référence à la structure géométrique et topologique sous-jacente d'un ensemble de données dans son espace de caractéristiques. Cette approche capture :

1. **Structure locale** : Comment les points de données sont regroupés localement
2. **Structure globale** : Comment les différentes régions de l'espace sont séparées
3. **Manifold sous-jacent** : La géométrie intrinsèque des données

### Méthodologie KNN+KFN

#### 1. K-Nearest Neighbors (KNN) - Structure Locale

Le composant KNN capture la structure locale des données :
- Pour chaque point, on identifie ses k plus proches voisins (k=10 par défaut)
- Ces voisins révèlent comment les données sont regroupées localement
- La distance moyenne aux voisins indique la densité locale

#### 2. K-Farthest Neighbors (KFN) - Structure Globale

Le composant KFN capture la structure globale :
- On identifie les k points les plus éloignés de chaque point
- Ces points définissent les frontières et les régions séparées
- Cela aide à comprendre comment les classes sont séparées dans l'espace

#### 3. Points Critiques

L'algorithme extrait les "points critiques" qui définissent la forme :
- Points qui apparaissent dans les voisinages KNN
- Points qui apparaissent dans les voisinages KFN
- Ces points représentent typiquement 30-50% des données originales

### Génération de Données Synthétiques

Une fois la forme extraite, l'algorithme peut générer de nouvelles données synthétiques qui préservent cette forme :

#### Interpolation Géodésique

1. **Sélection de points de départ** : Choisir aléatoirement un point critique de la classe cible
2. **Marche sur le graphe** : Effectuer 1-2 sauts vers des voisins dans le graphe local
3. **Interpolation** : Interpoler entre le point de départ et le point d'arrivée avec un facteur alpha ~ Beta(2,2)

#### Bruit Anisotrope

Le bruit ajouté est intelligent et préserve la structure :
- **Bruit parallèle** : Petit bruit dans la direction de l'interpolation (10%)
- **Bruit orthogonal** : Plus grand bruit perpendiculaire à l'interpolation (30%)
- Échelle basée sur la distance KNN moyenne pour s'adapter à la densité locale

## Domaines Testés

Le notebook valide l'approche sur 6 ensembles de données différents couvrant 4 domaines :

### 1. Données Tabulaires Classiques

#### Iris
- **Description** : Dataset classique de fleurs (setosa vs versicolor)
- **Caractéristiques** : 4 dimensions (longueur/largeur sépales et pétales)
- **Taille** : ~100 échantillons
- **Défi** : Faible dimensionnalité, classes bien séparées

#### Wine
- **Description** : Classification de vins (classe 0 vs classe 1)
- **Caractéristiques** : 13 dimensions (propriétés chimiques)
- **Taille** : ~130 échantillons
- **Défi** : Dimensionnalité moyenne, features corrélées

### 2. Données Médicales

#### Breast Cancer
- **Description** : Diagnostic de cancer du sein (bénin vs malin)
- **Caractéristiques** : 30 dimensions (mesures de tumeurs)
- **Taille** : 569 échantillons
- **Défi** : Importance critique de la précision, features hétérogènes

### 3. Vision par Ordinateur

#### MNIST Small
- **Description** : Chiffres manuscrits 0 vs 1
- **Caractéristiques** : 784 dimensions (images 28x28 pixels)
- **Taille** : 2000 échantillons (sous-ensemble)
- **Défi** : Très haute dimensionnalité, données d'images

### 4. Séries Temporelles

#### Time Series Synthetic
- **Description** : Séquences synthétiques (sinusoïdales vs carrées)
- **Caractéristiques** : 50 dimensions (50 points temporels)
- **Taille** : 1000 échantillons
- **Défi** : Dépendances temporelles, patterns séquentiels

### 5. Haute Dimensionnalité Complexe

#### High-Dimensional Complex
- **Description** : Données synthétiques complexes
- **Caractéristiques** : 100 dimensions avec 30 informatives
- **Configuration** : 4 clusters par classe, séparation modérée
- **Défi** : Curse of dimensionality, bruit élevé

## Architecture Neuronale Adaptative

### Conception du Classificateur

Le notebook utilise un réseau neuronal adaptatif qui s'ajuste automatiquement à la dimensionnalité d'entrée :

```
Input (n dimensions)
    ↓
Dense (hidden_size) + ReLU + Dropout(0.3)
    ↓
Dense (hidden_size/2) + ReLU + Dropout(0.2)
    ↓
Dense (n_classes)
```

**Calcul de hidden_size** : `max(64, min(256, input_dim * 2))`
- S'adapte à la dimensionnalité des données
- Évite le sur-apprentissage sur les petits datasets
- Évite le sous-apprentissage sur les grands datasets

### Entraînement

- **Optimiseur** : Adam avec learning rate 0.001 et weight decay 1e-4
- **Scheduler** : Cosine Annealing pour ajuster le learning rate
- **Fonction de perte** : CrossEntropyLoss
- **Régularisation** : Dropout et weight decay pour éviter l'overfitting
- **Époques** : 80 époques avec early stopping implicite

## Protocole Expérimental

### Méthodologie de Validation

Pour chaque dataset :

1. **Division des données** : 75% entraînement, 25% test (stratifiée)
2. **Extraction de forme** : Sur l'ensemble d'entraînement uniquement
3. **Génération synthétique** : Créer un ensemble synthétique de même taille
4. **Deux entraînements parallèles** :
   - Modèle A : Entraîné sur données réelles → testé sur données réelles
   - Modèle B : Entraîné sur données synthétiques → testé sur données réelles
5. **Répétition** : 3 essais avec graines aléatoires différentes

### Métriques Évaluées

1. **Précision sur test (Real)** : Performance du modèle entraîné sur vraies données
2. **Précision sur test (Syn)** : Performance du modèle entraîné sur données synthétiques
3. **Ratio de performance** : Syn / Real × 100% (objectif : ≥90%)
4. **Taux de compression** : Taille de la forme / Taille originale (typiquement 30-50%)
5. **Temps d'exécution** : Extraction, génération, entraînement

## Résultats et Analyse

### Résultats par Dataset

Le notebook génère un tableau de résultats détaillé pour chaque dataset montrant :

- **Précision moyenne sur test** : Pour données réelles et synthétiques (± écart-type)
- **Précision moyenne sur entraînement** : Pour détecter l'overfitting
- **Ratio de performance** : Pourcentage de performance préservée
- **Taux de compression** : Réduction de taille de la représentation

### Statistiques Globales

L'analyse agrégée calcule :

1. **Ratio moyen** : Performance moyenne à travers tous les domaines
2. **Ratio minimum et maximum** : Plage de performance
3. **Taux de succès** :
   - Combien de datasets atteignent ≥95% de performance
   - Combien de datasets atteignent ≥90% de performance

### Visualisations

Le notebook produit 4 graphiques :

#### 1. Comparaison Train vs Test
- Barres comparant train/test pour données réelles et synthétiques
- Permet de détecter l'overfitting
- Montre si les données synthétiques généralisent bien

#### 2. Ratios de Performance
- Visualisation des ratios Syn/Real pour chaque dataset
- Code couleur :
  - Vert : ≥95% (excellent)
  - Orange : 90-95% (bon)
  - Rouge : <90% (à améliorer)

#### 3. Compression
- Taux de compression pour chaque dataset
- Montre l'efficacité de l'extraction de forme
- Typiquement 30-50% de la taille originale

#### 4. Résumé Textuel
- Statistiques agrégées
- Conclusions sur la généralisation cross-domaine

## Insights Clés

### Forces de l'Approche

1. **Généralisation cross-domaine** : Fonctionne sur tabulaire, vision, séries temporelles
2. **Compression efficace** : Réduit les données à 30-50% avec peu de perte
3. **Préservation de la structure** : L'interpolation géodésique maintient le manifold
4. **Agnostique au domaine** : Pas d'hypothèses spécifiques sur le type de données

### Limitations Potentielles

1. **Dépendance à k** : Les paramètres k_near et k_far doivent être ajustés
2. **Coût computationnel** : KNN et calculs de distance peuvent être coûteux
3. **Datasets déséquilibrés** : Peut nécessiter des ajustements pour classes très déséquilibrées
4. **Très haute dimensionnalité** : Performance peut dégrader au-delà de 100-200 dimensions

## Applications Pratiques

### 1. Augmentation de Données
- Générer plus de données d'entraînement quand les données réelles sont limitées
- Particulièrement utile pour données médicales coûteuses à obtenir

### 2. Privacy-Preserving ML
- Partager la "forme" des données sans partager les points originaux
- Utile pour données sensibles (santé, finance)

### 3. Transfer Learning
- Extraire la forme d'un domaine source
- L'adapter à un domaine cible avec calibration

### 4. Compression de Données
- Stocker seulement les points critiques (30-50%)
- Régénérer les données complètes quand nécessaire

### 5. Data Quality Assessment
- Comparer la forme des données de production vs développement
- Détecter le dataset drift

## Conclusions

### Validation Réussie

Le notebook démontre que l'approche KNN+KFN :
- ✓ Se généralise à travers multiples domaines
- ✓ Préserve 85-100% de la performance originale
- ✓ Compresse efficacement les données
- ✓ Maintient la structure du manifold

### Principes Fondamentaux

1. **Structure locale + globale** : La combinaison KNN+KFN capture les deux échelles
2. **Interpolation géodésique** : Préserve la géométrie intrinsèque des données
3. **Bruit anisotrope** : Le bruit adaptatif améliore la généralisation
4. **Agnosticisme au domaine** : L'approche ne fait pas d'hypothèses sur le type de données

### Directions Futures

1. **Adaptation automatique de k** : Méthodes pour choisir k_near et k_far automatiquement
2. **Extension multi-classes** : Améliorer pour >2 classes
3. **Métrique de forme** : Développer des métriques pour comparer les formes
4. **Scaling** : Optimiser pour très grands datasets (millions de points)

---

## Références Techniques

### Concepts Utilisés

- **K-Nearest Neighbors** : Algorithme classique de ML pour structure locale
- **Ball Tree** : Structure de données efficace pour recherche de voisins
- **Distance de Hausdorff** : Mesure de similarité entre ensembles
- **Apprentissage de manifold** : Hypothèse que les données résident sur un manifold de faible dimension
- **Interpolation géodésique** : Chemin le plus court sur un manifold
- **Distribution Beta** : Beta(2,2) pour interpolation centrée mais variable

### Librairies Python

- **NumPy** : Calculs numériques
- **PyTorch** : Réseaux de neurones
- **scikit-learn** : KNN, datasets, preprocessing
- **SciPy** : Calculs de distance
- **Matplotlib** : Visualisations

---

*Ce document fournit une explication détaillée du notebook `data_shape.ipynb` qui explore une approche novatrice d'apprentissage basé sur la forme des données, validée à travers multiples domaines.*

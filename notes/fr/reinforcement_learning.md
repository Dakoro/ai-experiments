# Implémentation d'Apprentissage par Renforcement avec Grid World

## Vue d'ensemble

Ce notebook implémente un système complet d'apprentissage par renforcement utilisant Q-learning pour résoudre un problème de navigation dans un monde en grille. L'implémentation démontre les concepts fondamentaux de l'AR incluant la conception d'environnement, l'entraînement d'agent et l'évaluation de politique.

## Architecture du Système

L'implémentation comprend deux composants principaux :

### 1. Environnement GridWorld (classe `GridWorld`)

Un environnement personnalisé qui simule un monde en grille 5x5 avec les caractéristiques suivantes :

- **Espace d'États** : 25 positions possibles (grille 5x5)
- **Espace d'Actions** : 4 actions discrètes (haut, droite, bas, gauche)
- **Position de Départ** : Coin supérieur gauche (0,0)
- **Position Objectif** : Coin inférieur droit (4,4)
- **Obstacles** : Fixés aux positions (1,1), (2,2) et (3,1)

#### Méthodes Clés :
- `reset()` : Initialise l'agent à la position de départ
- `step(action)` : Exécute l'action et retourne (état_suivant, récompense, terminé)
- `render(q_table)` : Visualise le monde en grille avec superposition optionnelle de la politique

#### Structure des Récompenses :
- **Objectif atteint** : +100 (récompense terminale)
- **Chaque étape** : -1 (encourage les chemins plus courts)
- **Obstacles** : -10 (pénalité, bien que l'agent ne puisse y entrer grâce à la vérification des limites)

### 2. Agent Q-Learning (classe `QLearningAgent`)

Implémente l'algorithme Q-learning avec les paramètres suivants :

- **Taux d'Apprentissage (α)** : 0.1 - Contrôle combien la nouvelle information remplace l'ancienne
- **Facteur de Remise (γ)** : 0.95 - Détermine l'importance des récompenses futures
- **Taux d'Exploration (ε)** : 0.1 - Probabilité de sélection d'action aléatoire

#### Méthodes Clés :
- `choose_action(state, training)` : Sélection d'action epsilon-greedy
- `update(state, action, reward, next_state, done)` : Mise à jour Q-valeur utilisant l'équation de Bellman

#### Règle de Mise à Jour Q-Learning :
```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

## Processus d'Entraînement

La fonction d'entraînement (`train_agent()`) s'exécute pendant 500 épisodes avec le workflow suivant :

1. **Initialisation d'Épisode** : Réinitialise l'environnement à l'état de départ
2. **Sélection d'Action** : Utilise la politique epsilon-greedy
3. **Interaction avec l'Environnement** : Exécute l'action, observe la récompense et l'état suivant
4. **Mise à Jour Q-Table** : Applique la règle de mise à jour Q-learning
5. **Terminaison** : L'épisode se termine quand l'objectif est atteint ou le maximum d'étapes (100) est dépassé

### Métriques d'Entraînement Suivies :
- **Récompenses d'Épisode** : Récompense totale accumulée par épisode
- **Étapes d'Épisode** : Nombre d'étapes requises pour atteindre l'objectif
- **Rapport de Progrès** : Toutes les 50 épisodes, affiche les moyennes mobiles

## Fonctionnalités Clés

### 1. Système de Visualisation
- **Rendu d'Environnement** : Montre la grille avec obstacles, départ (S) et objectif (G)
- **Visualisation de Politique** : Affiche la politique apprise sous forme de flèches directionnelles
- **Progrès d'Entraînement** : Trace les tendances de récompense et d'étapes avec moyennes mobiles

### 2. Framework de Test
- **Évaluation de Politique** : Teste l'agent entraîné pendant 5 épisodes sans exploration
- **Suivi de Trajectoire** : Enregistre la trajectoire complète du départ à l'objectif
- **Métriques de Performance** : Rapporte les étapes prises et la récompense totale

### 3. Implémentation Robuste
- **Vérification des Limites** : Empêche l'agent de sortir de la grille ou d'entrer dans les obstacles
- **Exploration Epsilon-Greedy** : Équilibre exploitation vs exploration
- **Surveillance de Convergence** : Suit le progrès d'entraînement via les moyennes mobiles

## Résultats d'Apprentissage Attendus

L'agent devrait apprendre à :
1. Naviguer du départ à l'objectif en évitant les obstacles
2. Découvrir le chemin optimal (route la plus courte évitant les obstacles)
3. Développer une politique cohérente qui généralise à travers les épisodes de test

## Détails d'Implémentation Technique

### Représentation d'État
- Les états sont représentés comme des tuples (ligne, colonne)
- Dimensions Q-table : (5, 5, 4) pour (lignes, colonnes, actions)

### Encodage d'Action
- 0 : Haut (-1, 0)
- 1 : Droite (0, 1)
- 2 : Bas (1, 0)
- 3 : Gauche (0, -1)

### Indicateurs de Convergence
- Diminution du nombre d'étapes sur les épisodes
- Augmentation des récompenses d'épisode
- Atteinte cohérente de l'objectif dans les épisodes de test

## Extensions Potentielles

1. **Obstacles Dynamiques** : Ajouter des obstacles mobiles ou placés aléatoirement
2. **Grille Plus Grande** : Étendre à des environnements plus grands
3. **Algorithmes Alternatifs** : Implémenter SARSA, Double Q-Learning ou Deep Q-Networks
4. **Espace d'Action Continu** : Étendre aux problèmes de contrôle continu
5. **Multi-Agent** : Ajouter plusieurs agents avec objectifs coopératifs/compétitifs

## Valeur Éducative

Cette implémentation sert d'excellente introduction aux concepts d'apprentissage par renforcement :
- **Processus de Décision Markoviens** : Structure claire état-action-récompense
- **Apprentissage par Différence Temporelle** : Q-learning comme approche sans modèle
- **Exploration vs Exploitation** : Démonstration de la stratégie epsilon-greedy
- **Évaluation de Politique** : Test et visualisation du comportement appris

L'implémentation complète fournit une base solide pour comprendre des algorithmes AR plus avancés et peut être facilement modifiée pour différents domaines de problèmes.
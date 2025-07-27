## Fondements Mathématiques des Méthodes de Monte Carlo

### Principe de Base

La méthode de Monte Carlo est fondamentalement basée sur la **Loi des Grands Nombres**. Si nous voulons estimer une espérance E[f(X)], nous pouvons l'approximer en utilisant :

$$\hat{I}_N = \frac{1}{N} \sum_{i=1}^{N} f(X_i)$$

où $X_1, X_2, ..., X_N$ sont des échantillons aléatoires indépendants et identiquement distribués (i.i.d.) de la distribution de X, et N est le nombre d'échantillons.

### Loi Forte des Grands Nombres

Mathématiquement, la Loi Forte des Grands Nombres énonce :

$$P\left(\lim_{N \to \infty} \frac{1}{N} \sum_{i=1}^{N} f(X_i) = E[f(X)]\right) = 1$$

Ceci garantit que notre estimation Monte Carlo converge vers la vraie valeur avec probabilité 1 quand N tend vers l'infini.

### Problèmes d'Intégration

De nombreuses applications Monte Carlo impliquent le calcul d'intégrales. Considérons l'intégrale :

$$I = \int_{\Omega} f(x) dx$$

Nous pouvons la réécrire comme une espérance :

$$I = \int_{\Omega} f(x) dx = |\Omega| \cdot E[f(X)]$$

où X est uniformément distribué sur Ω et |Ω| est le volume du domaine.

Pour un cas plus général avec une fonction de densité de probabilité p(x) :

$$I = \int_{\Omega} g(x) dx = \int_{\Omega} \frac{g(x)}{p(x)} p(x) dx = E\left[\frac{g(X)}{p(X)}\right]$$

### Variance et Analyse d'Erreur

La variance de l'estimateur Monte Carlo est :

$$\text{Var}(\hat{I}_N) = \text{Var}\left(\frac{1}{N} \sum_{i=1}^{N} f(X_i)\right) = \frac{\sigma^2}{N}$$

où $\sigma^2 = \text{Var}(f(X))$.

L'erreur standard est :

$$\text{ES}(\hat{I}_N) = \frac{\sigma}{\sqrt{N}}$$

Ceci montre le taux de convergence caractéristique $O(1/\sqrt{N})$ des méthodes de Monte Carlo.

### Théorème Central Limite

Pour N grand, le Théorème Central Limite nous dit que l'estimateur est approximativement distribué normalement :

$$\frac{\hat{I}_N - E[f(X)]}{\sigma/\sqrt{N}} \xrightarrow{d} \mathcal{N}(0,1)$$

Ceci nous permet de construire des intervalles de confiance :

$$P\left(\hat{I}_N - z_{\alpha/2}\frac{\sigma}{\sqrt{N}} \leq E[f(X)] \leq \hat{I}_N + z_{\alpha/2}\frac{\sigma}{\sqrt{N}}\right) \approx 1-\alpha$$

où $z_{\alpha/2}$ est la valeur critique de la distribution normale standard.

### Échantillonnage Préférentiel

L'échantillonnage préférentiel réduit la variance en échantillonnant à partir d'une distribution différente. Si nous voulons estimer :

$$I = \int f(x)p(x)dx = E_p[f(X)]$$

Nous pouvons échantillonner à partir d'une distribution différente q(x) et utiliser :

$$I = \int f(x)\frac{p(x)}{q(x)}q(x)dx = E_q\left[f(X)\frac{p(X)}{q(X)}\right]$$

La distribution d'échantillonnage préférentiel optimale minimise la variance et est proportionnelle à |f(x)|p(x).

### Monte Carlo par Chaînes de Markov (MCMC)

Pour des distributions complexes où l'échantillonnage direct est difficile, MCMC crée une chaîne de Markov avec la distribution stationnaire désirée π(x).

**Algorithme de Metropolis-Hastings :**

Étant donné l'état actuel $x_t$ :
1. Proposer un nouvel état $x'$ à partir de la distribution de proposition $q(x'|x_t)$
2. Calculer le ratio d'acceptation :
   $$\alpha = \min\left(1, \frac{\pi(x')q(x_t|x')}{\pi(x_t)q(x'|x_t)}\right)$$
3. Accepter $x_{t+1} = x'$ avec probabilité α, sinon $x_{t+1} = x_t$

**Condition de Balance Détaillée :**
L'algorithme satisfait :
$$\pi(x)P(x \to y) = \pi(y)P(y \to x)$$

Ceci assure que π(x) est la distribution stationnaire.

### Intégration Monte Carlo en Haute Dimension

Pour une intégrale en dimension d :

$$I = \int_{[0,1]^d} f(x_1, ..., x_d) dx_1...dx_d$$

Les méthodes numériques traditionnelles (comme la règle de Simpson) nécessitent $O(N^d)$ évaluations pour N points par dimension. Monte Carlo nécessite seulement O(N) évaluations avec une erreur $O(1/\sqrt{N})$ **indépendante de la dimension**.

### Techniques de Réduction de Variance

**Variables de Contrôle :**
Si nous connaissons E[g(X)] pour une fonction g corrélée avec f :

$$\hat{I}_{VC} = \hat{I}_N - c(\hat{g}_N - E[g(X)])$$

Le coefficient optimal est :
$$c^* = \frac{\text{Cov}(f(X), g(X))}{\text{Var}(g(X))}$$

**Variables Antithétiques :**
Utiliser des paires négativement corrélées. Si U ~ Uniforme(0,1), alors U et 1-U sont uniformes :

$$\hat{I}_{VA} = \frac{1}{2N} \sum_{i=1}^{N} [f(U_i) + f(1-U_i)]$$

**Échantillonnage Stratifié :**
Diviser le domaine en K strates avec probabilités $p_k$ :

$$\hat{I}_{ES} = \sum_{k=1}^{K} p_k \hat{I}_{N_k}^{(k)}$$

où $\hat{I}_{N_k}^{(k)}$ est l'estimation de la strate k avec $N_k$ échantillons.

### Quasi-Monte Carlo

Quasi-Monte Carlo utilise des suites déterministes à faible discrépance au lieu de nombres aléatoires. Pour une suite $(x_1, ..., x_N)$ dans $[0,1]^d$, la discrépance est :

$$D_N^* = \sup_{B \in \mathcal{B}} \left|\frac{1}{N}\sum_{i=1}^{N} \mathbf{1}_B(x_i) - \lambda(B)\right|$$

où $\mathcal{B}$ est l'ensemble des boîtes alignées sur les axes et λ est la mesure de Lebesgue.

L'inégalité de Koksma-Hlawka borne l'erreur d'intégration :

$$\left|\hat{I}_N - I\right| \leq V(f) \cdot D_N^*$$

où V(f) est la variation de f au sens de Hardy et Krause.

### Monte Carlo Multi-niveaux

Pour des problèmes avec plusieurs niveaux de précision, Monte Carlo multi-niveaux estime :

$$E[P_L] = E[P_0] + \sum_{l=1}^{L} E[P_l - P_{l-1}]$$

La variance de l'estimateur multi-niveaux avec $N_l$ échantillons au niveau l est :

$$\text{Var}(\hat{Y}) = \sum_{l=0}^{L} \frac{V_l}{N_l}$$

où $V_l = \text{Var}(P_l - P_{l-1})$.

### Diagnostics de Convergence pour MCMC

**Taille d'Échantillon Effective :**
Pour des échantillons MCMC autocorrélés :

$$TEE = \frac{N}{1 + 2\sum_{k=1}^{\infty} \rho_k}$$

où $\rho_k$ est l'autocorrélation au décalage k.

**Statistique de Gelman-Rubin :**
Pour m chaînes de longueur n :

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

où W est la variance intra-chaîne et $\hat{V}$ est une estimation de la variance postérieure.

### Probabilité d'Événements Rares

Pour estimer de petites probabilités $p = P(A)$, l'erreur relative est :

$$\frac{\text{ES}(\hat{p})}{\hat{p}} = \sqrt{\frac{1-p}{Np}} \approx \frac{1}{\sqrt{Np}}$$

Ceci montre qu'estimer des événements rares nécessite $O(1/p)$ échantillons pour une erreur relative fixe.

Ces fondements mathématiques démontrent pourquoi les méthodes de Monte Carlo sont puissantes : elles fournissent des garanties probabilistes de convergence, des bornes d'erreur quantifiables, et maintiennent leur efficacité en haute dimension où les méthodes traditionnelles échouent.
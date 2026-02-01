
# Mission - Partie 1 - Identifiez les causes d'attrition au sein d'une ESN

## Comment allez vous procéder ?

Cette mission suit un scénario de projet professionnel.

Vous pouvez suivre les étapes pour vous aider à réaliser vos livrables.

Avant de démarrer, nous vous conseillons de :
- lire toute la mission et ses documents liés ;
- prendre des notes sur ce que vous avez compris ;
- consulter les étapes pour vous guider ;
- préparer une liste de questions pour votre première session de mentorat.

Prêt à mener la mission ?

---

## Étapes

## Étape 1 - Effectuez une analyse exploratoire des fichiers de données

### Prérequis
- Avoir préparé un environnement virtuel Python avec tous les packages spécifiés dans la partie importation du notebook template.
- Avoir créé un notebook (via JupyterLab, Jupyter, VSCode ou Google Colab) dédié au projet.

### Résultats attendus
- Un DataFrame central, issu d’une jointure entre les fichiers de départ.
- Des cellules au sein du notebook pour calculer des statistiques descriptives sur les fichiers de départ et le fichier central, dans l’objectif de faire ressortir des différences clés entre les employés.
- Des cellules au sein du notebook pour générer des graphiques, dans l’objectif de faire ressortir des différences clés entre les employés.

### Recommandations
- Identifier et nettoyer les colonnes qui correspondent à des informations quantitatives ou qualitatives.
- Prendre l’habitude d’utiliser la méthode `.apply()` de Pandas pour nettoyer vos colonnes.
- Commencer par des statistiques descriptives simples sur chacun des trois fichiers.
- Identifier les colonnes permettant de réaliser des jointures entre les 3 fichiers.
- Bien choisir le type de graphique en fonction des colonnes que vous voulez analyser (quanti vs quanti, quanti vs quali, quali vs quali, etc.).

### Points de vigilance
- Ne pas se précipiter vers l’analyse avant d’avoir bien compris le libellé et le contenu des colonnes.
- Déterminer quel type de jointure sera le plus adapté pour rapprocher les fichiers (left, inner etc.).

### Ressources
- Le chapitre *Améliorez un jeu de données* du cours d'initiation au ML.
- Ce notebook exemple d’exploration de données en Python.

---

## Étape 2 - Préparez la donnée pour la modélisation

### Prérequis
- Avoir finalisé l’étape 1.

### Résultats attendus
- Un DataFrame contenant vos features prêtes pour la modélisation (autrement dit : un DataFrame qui peut être injecté dans une méthode `fit()` de sklearn sans erreur). Ce DataFrame est traditionnellement appelé **X**.
- Un Pandas Series contenant votre colonne cible (traditionnellement appelée **y**).
- Idéalement, des fonctions Python permettant de réaliser les transformations sur le fichier central de l’étape 1 pour obtenir les deux données X et y.

Prenez l’habitude le plus rapidement possible d’utiliser des fonctions pour reproduire les opérations récurrentes dans les projets ML, comme le feature engineering par exemple.

### Recommandations
- Utiliser une matrice de corrélation de Pearson, pour éliminer les fortes corrélations linéaires entre features.
- Tracer un pairplot pour mesurer l’intensité des corrélations non-linéaires s’il y en a.
- Compléter le pairplot avec une matrice de corrélation de Spearman.
- Sélectionner une méthode d’encoding adaptée pour les features qualitatives en fonction de leur sens métier.

### Points de vigilance
- Bien comprendre quand utiliser un OneHotEncoder ou une autre méthode d’encodage pour les features qualitatives.
- Bien comprendre quels types de features peuvent être inclus dans le calcul d’une matrice de corrélation.
- Déterminer quel type de jointure sera la plus adaptée pour rapprocher les fichiers (left, inner etc.).

### Ressources
- Le chapitre *Transformez les variables pour faciliter l’apprentissage du modèle* du cours *Initiez-vous au Machine Learning*.
- Le screencast sur la matrice de corrélation, issu du chapitre *Préparez vos données pour un modèle supervisé* du cours *Maîtrisez l’apprentissage supervisé*.

---

## Étape 3 - Réalisation d’un premier modèle de classification

### Prérequis
- Avoir réalisé l’étape 2.

### Résultats attendus
- Des jeux d’apprentissage et de test (X_train, X_test, y_train, y_test).
- Trois modèles entraînés sur les jeux d’apprentissage : un modèle Dummy, un modèle linéaire, un modèle non-linéaire.
- Des métriques d’évaluation calculées pour chaque modèle, sur le jeu d’apprentissage et le jeu de test.
- Des cellules de code permettant d’obtenir les résultats attendus.

### Recommandations
- Commencer par une séparation train/test simple ou une validation croisée simple.
- Entraîner d’abord un modèle Dummy, puis un modèle linéaire, avant un modèle non-linéaire.
- Utiliser des modèles à base d’arbre (RandomForest, XGBoost ou CatBoost).
- Calculer a minima : matrice de confusion, rappel et précision.
- Interpréter la performance du modèle par rapport au modèle Dummy.

### Points de vigilance
- Calculer les métriques sur le jeu d’apprentissage et de test pour mesurer l’overfitting.
- Ne pas confondre les métriques “accuracy” et “precision”.

### Ressources
- *Découvrez les concepts clés* du cours *Maîtrisez l’apprentissage supervisé*.
- *Commencez par un modèle étalon ou benchmark*.

---

## Étape 4 - Améliorez l'approche de classification

### Prérequis
- Avoir terminé l’étape 3.

### Résultat attendu
Une modélisation tenant compte du déséquilibre des classes et du contexte métier.

### Recommandations
- Formaliser l’approche de modélisation avant d’implémenter.
- Utiliser la stratification via sklearn.
- Exploiter la validation croisée.
- Utiliser la courbe précision-rappel.
- Réaliser du feature engineering.
- Pondérer les classes (`class_weights`).
- Utiliser l’undersampling ou l’oversampling avec `imblearn` si nécessaire.

### Points de vigilance
- Code très itératif : bien structurer les expérimentations.
- Nuancer l’interprétation selon le contexte métier.
- Attention à l’impact du déséquilibre des classes sur certaines métriques.

### Ressources
- *Préparez vos données pour un modèle supervisé*.
- *Évaluez de manière plus poussée vos modèles de classification*.
- Tutoriel sur l’undersampling.
- Tutoriel sur l’oversampling.

---

## Étape 5 - Optimisez et interprétez le comportement du modèle

### Prérequis
- Avoir terminé l’étape 4.

### Résultat attendu
- Un modèle non linéaire issu d’un fine-tuning.
- Des graphiques d’interprétation globale et locale.
- Des cellules pour le tuning et le calcul des feature importances.

### Recommandations
- Commencer par la feature importance globale.
- Comparer Permutation Importance et SHAP.
- Utiliser TreeExplainer ou KernelExplainer.
- Utiliser Waterfall Plot pour la locale.

### Points de vigilance
- Commencer par le fine-tuning avant l’interprétation.
- Éviter LIME.
- Bien comprendre les Shapley values.

### Ressource
- *Donnez de la transparence à votre modèle supervisé*.

---

## Étape 6 - Formalisez vos résultats

### Prérequis
- Avoir finalisé l’étape 5.

### Résultat attendu
Une présentation `.ppt` contenant :
- Les hypothèses de préparation de la donnée ;
- Les insights clés de l’analyse exploratoire ;
- La méthodologie de modélisation ;
- Les causes d’attrition (feature importance globale) ;
- Des exemples d’explication locale.

### Recommandations
- Rappeler les jeux de données initiaux.
- Sélectionner les messages les plus impactants.
- Adapter le discours à une audience non-technique.

### Ressources
- *Communiquez et formalisez vos idées par le storytelling*.
- *Améliorez l'impact de vos présentations*.

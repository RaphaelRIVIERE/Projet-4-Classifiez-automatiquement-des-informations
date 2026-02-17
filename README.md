# Identifiez les causes d'attrition au sein d'une ESN

Projet de classification supervisée visant a identifier et expliquer les facteurs de depart des employes d'une ESN (Entreprise de Services du Numerique).

## Contexte

Une ESN souhaite comprendre les causes de demission de ses employes. A partir de trois sources de donnees internes (SIRH, evaluations, sondages), l'objectif est de construire un modele de machine learning capable de predire l'attrition et d'en identifier les causes principales.

## Structure du projet

```
.
├── 1_notebook.ipynb          # Analyse exploratoire et preparation des donnees
├── 2_notebook.ipynb          # Modelisation, optimisation et interpretation
├── data/
│   ├── extrait_sirh.csv      # Donnees RH (anciennete, salaire, poste...)
│   ├── extrait_eval.csv      # Donnees d'evaluation des employes
│   ├── extrait_sondage.csv   # Donnees de sondage de satisfaction
│   └── data_clean.csv        # DataFrame central apres jointure et nettoyage
├── data_utils.py             # Fonctions utilitaires pour le chargement des donnees
├── preprocessing.py          # Pipeline de preprocessing et feature engineering
├── modelization.py           # Entrainement, evaluation et fine-tuning des modeles
├── visualizer.py             # Fonctions de visualisation (EDA et interpretation)
├── redundancy_analysis.py    # Analyse de redondance entre features
├── docs/
│   ├── contexte.md           # Enonce de la mission
│   └── fiche_auto_evaluation.pdf
└── pyproject.toml            # Dependances du projet (gestion via uv)
```

## Demarche

### 1. Analyse exploratoire (Notebook 1)

- Nettoyage et jointure des 3 fichiers sources en un DataFrame central
- Statistiques descriptives comparant les employes partis vs restes
- Visualisations : distributions, boxplots, heatmaps de correlation

### 2. Preparation des donnees (Notebook 1)

- Feature engineering a partir des donnees existantes
- Encodage des variables qualitatives (OrdinalEncoder, OneHotEncoder)
- Matrice de correlation de Pearson pour traiter les redondances
- Scaling des features

### 3. Modelisation baseline (Notebook 2)

- Separation train/test avec stratification
- Modele etalon (DummyClassifier)
- Modele lineaire (LogisticRegression)
- Modele non-lineaire (RandomForest, XGBoost)
- Metriques : matrice de confusion, precision, rappel, F1, PR-AUC, ROC-AUC

### 4. Amelioration et gestion du desequilibre (Notebook 2)

- Ponderation des classes (`class_weight`)
- Oversampling / undersampling (imblearn)
- Validation croisee
- Optimisation du seuil de probabilite via courbe precision-rappel

### 5. Fine-tuning et interpretation (Notebook 2)

- Optimisation des hyperparametres (GridSearchCV)
- Feature importance globale (Permutation Importance)
- Valeurs de Shapley (SHAP) : Beeswarm, Waterfall plots
- Interpretation locale et globale des causes d'attrition

## Installation

```bash
# Cloner le depot
git clone <url-du-repo>
cd projet-4

# Installer les dependances avec uv
uv sync

# Lancer JupyterLab
uv run jupyter lab
```

## Prerequis

- Python 3.12
- [uv](https://docs.astral.sh/uv/) pour la gestion des dependances

## Principales dependances

- pandas, numpy - Manipulation de donnees
- scikit-learn - Modelisation et evaluation
- xgboost - Gradient boosting
- imbalanced-learn - Gestion du desequilibre des classes
- shap - Interpretation des modeles
- matplotlib, seaborn - Visualisation

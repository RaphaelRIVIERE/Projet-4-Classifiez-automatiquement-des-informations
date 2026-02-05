from typing import List
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns


def cramers_v(col1: str, col2: str, df: pd.DataFrame) -> float:
    """
    Calcule le coefficient de Cramér's V entre deux colonnes.
    
    Le Cramér's V mesure l'association entre deux variables catégorielles.
    - 0 = aucune association
    - 1 = association parfaite
    
    Paramètres:
    -----------
    col1, col2 : str
        Noms des colonnes à comparer
    df : DataFrame
        Le DataFrame contenant les données
        
    Retour:
    -------
    float : valeur entre 0 et 1
    
    Exemple:
    --------
    >>> v = cramers_v('departement', 'poste', df)
    >>> print(f"Association: {v:.3f}")
    """
    # Créer un tableau de contingence (tableau croisé)
    contingence = pd.crosstab(df[col1], df[col2])
    
    # Test du Chi2
    chi2 = chi2_contingency(contingence)[0]
    n = contingence.sum().sum()  # nombre total d'observations
    min_dim = min(contingence.shape[0] - 1, contingence.shape[1] - 1)
    
    # Calculer le Cramér's V
    cramers = np.sqrt(chi2 / (n * min_dim))
    
    return cramers


def analyze_redundancy(columns: List[str], df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse la redondance entre plusieurs colonnes catégorielles.

    Paramètres:
    -----------
    columns : List[str]
        Liste des noms de colonnes à analyser
    df : pd.DataFrame
        Le DataFrame contenant les données

    Retour:
    -------
    pd.DataFrame : résultats avec colonnes, Cramér's V et interprétation

    Exemple:
    --------
    >>> results = analyze_redundancy(['dept', 'poste', 'domaine'], df)
    >>> print(results)
    """
    results = []

    # Comparer toutes les paires de colonnes
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1 = columns[i]
            col2 = columns[j]

            # Calculer le Cramér's V
            v = cramers_v(col1, col2, df)

            results.append({
                'Column_1': col1,
                'Column_2': col2,
                'Cramers_V': round(v, 3)
            })

    # Créer un DataFrame et trier par Cramér's V décroissant
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('Cramers_V', ascending=False)

    return df_results.reset_index(drop=True)


def visualize_redundancy(columns: List[str], df: pd.DataFrame) -> None:
    """
    Crée une heatmap pour visualiser la redondance entre colonnes.

    Paramètres:
    -----------
    columns : List[str]
        Liste des colonnes à visualiser
    df : pd.DataFrame
        Le DataFrame contenant les données

    Exemple:
    --------
    >>> visualize_redundancy(['dept', 'poste', 'domaine'], df)
    """
    # Créer une matrice vide
    n = len(columns)
    matrix = np.zeros((n, n))

    # Remplir la matrice avec les valeurs de Cramér's V
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 1.0  # Une colonne avec elle-même = 1
            else:
                matrix[i, j] = cramers_v(columns[i], columns[j], df)

    # Créer la heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn_r',
                xticklabels=columns,
                yticklabels=columns,
                vmin=0, vmax=1,
                square=True,
                cbar_kws={'label': "Cramér's V"})

    plt.title("Matrice de Redondance\n(0 = aucune, 1 = parfaite)",
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()



def full_analysis(columns: List[str], df: pd.DataFrame, plot: bool = True) -> pd.DataFrame:
    """
    Fait une analyse complète de redondance en une seule fonction.

    Paramètres:
    -----------
    columns : List[str]
        Colonnes à analyser
    df : pd.DataFrame
        Le DataFrame
    plot : bool
        Afficher la heatmap (défaut: True)

    Retour:
    -------
    pd.DataFrame : résultats de l'analyse

    Exemple:
    --------
    >>> results = full_analysis(['dept', 'poste', 'domaine'], df)
    """

    print("\n📊 STATISTIQUES DES COLONNES:")
    print("-" * 70)
    for col in columns:
        nunique = df[col].nunique()
        nmissing = df[col].isna().sum()
        print(f"  {col}: {nunique} valeurs uniques, {nmissing} manquantes")


    print("\n🔍 ANALYSE PAR PAIRES:")
    print("-" * 70)
    results = analyze_redundancy(columns, df)
    print(results.to_string(index=False))


    if plot:
        print("\n📈 Affichage de la heatmap...")
        visualize_redundancy(columns, df)

    return results
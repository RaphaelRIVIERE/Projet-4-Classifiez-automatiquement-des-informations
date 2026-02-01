import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import display
from typing import List

def analyze_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyse les valeurs manquantes dans un DataFrame.
    
    Args:
        df (pd.DataFrame): Le DataFrame à analyser.
    
    Returns:
        pd.DataFrame: DataFrame avec les statistiques de valeurs manquantes par colonne.
    """
    # Pourcentage global de cellules vides
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    pct_missing_global = (missing_cells / total_cells) * 100
    
    print(f"\n🌐 Pourcentage de cellules vides sur tout le DataFrame : {pct_missing_global:.2f}%")
    
    # Pourcentage par colonne
    missing_by_column = df.isna().sum()
    pct_by_column = (missing_by_column / len(df)) * 100
    
    # Créer un DataFrame pour les statistiques
    missing_df = pd.DataFrame({
        'Colonne': df.columns,
        'Valeurs manquantes': missing_by_column.values,
        'Pourcentage (%)': pct_by_column.values
    })
    
    # Trier par pourcentage décroissant
    missing_df = missing_df.sort_values('Pourcentage (%)', ascending=False)

    return missing_df


def plot_missing_values(missing_df: pd.DataFrame, top_n: int = 15, min_threshold: float = 0.1):
    """
    Visualise les valeurs manquantes sous forme de graphique à barres horizontales.
    
    Args:
        missing_df (pd.DataFrame): DataFrame retourné par analyze_missing_values().
        top_n (int): Nombre maximum de colonnes à afficher (par défaut: 15).
        min_threshold (float): Pourcentage minimum pour afficher une colonne (par défaut: 0.1%).
    """
    # Filtrer les colonnes selon le seuil
    missing_cols = missing_df[missing_df['Pourcentage (%)'] >= min_threshold].head(top_n).copy()
    
    if len(missing_cols) > 0:
        fig, ax = plt.subplots(figsize=(14, max(6, len(missing_cols) * 0.4)))
        
        # Créer le graphique horizontal
        bars = ax.barh(range(len(missing_cols)), missing_cols['Pourcentage (%)'])
        ax.set_yticks(range(len(missing_cols)))
        ax.set_yticklabels(missing_cols['Colonne'], fontsize=10)
        ax.set_xlabel('Pourcentage de valeurs manquantes (%)', fontsize=12, fontweight='bold')
        ax.set_title('Principales colonnes avec valeurs manquantes', fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Colorer les barres selon le niveau de gravité
        colors = ['#2ecc71' if x < 1 else '#f39c12' if x < 5 else '#e74c3c' 
                  for x in missing_cols['Pourcentage (%)']]
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # Ajouter les valeurs sur les barres
        for i, (idx, row) in enumerate(missing_cols.iterrows()):
            ax.text(row['Pourcentage (%)'] + 0.5, i, f"{row['Pourcentage (%)']:.2f}%", 
                    va='center', fontsize=9, fontweight='bold')
        
        legend_elements = [
            Patch(facecolor='#2ecc71', label='< 1% manquant (excellente couverture)'),
            Patch(facecolor='#f39c12', label='1-5% manquant (bonne couverture)'),
            Patch(facecolor='#e74c3c', label='> 5% manquant (attention requise)')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Ajuster les marges pour éviter les warnings
        plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
        plt.show()
    else:
        print(f"✅ Aucune colonne avec ≥ {min_threshold}% de valeurs manquantes !")
        


def explore_dataframe(df, show_missing=True):
    """
    Affiche les informations principales d'un DataFrame :
    - shape
    - head
    - info
    - describe
    - statistiques de valeurs manquantes

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame à analyser
    show_missing : bool, optional
        Affiche l'analyse des valeurs manquantes (default=True)
    """
    print("Shape :", df.shape)
    display(df.head())
    
    print("\n--- INFO ---")
    df.info()
    
    print("\n--- DESCRIBE ---")
    display(df.describe())
    
    print("\n--- MISSING VALUES ---")
    missing_stats = analyze_missing_values(df)
    if show_missing:
        display(missing_stats)
   


def distribution_column(df: pd.DataFrame, column: str):
    """
    Affiche la distribution des valeurs d'une colonne spécifique d'un DataFrame pandas.
    """
    print(f"\n 📊 Distribution de la colonne {column}")
    print("-" * 100)
    
    outlier_counts = df[column].value_counts(dropna=False)
    outlier_pct = df[column].value_counts(dropna=False, normalize=True) * 100
    
    outlier_summary = pd.DataFrame({
        'Effectif': outlier_counts,
        'Pourcentage': outlier_pct.round(2)
    })
    
    display(outlier_summary)

def display_columns_info(df: pd.DataFrame, columns: List[str]) -> None:
    """Affiche un résumé descriptif et visuel des colonnes sélectionnées d'un DataFrame pandas."""

    if df.empty:
        print("⚠️ DataFrame vide – aucune analyse possible.")
        return

    total_rows = len(df)

    for idx, col in enumerate(columns, 1):

        # Colonne inexistante
        if col not in df.columns:
            print(f"┌─ {idx}. {col}")
            print("│  ❌ Colonne inexistante")
            print("└" + "─" * 78)
            print()
            continue

        series = df[col]

        n_unique = series.nunique(dropna=True)
        n_missing = series.isna().sum()

        pct_unique = n_unique / total_rows * 100
        pct_missing = n_missing / total_rows * 100

        # En-tête colonne
        print(f"┌─ {idx}. {col}")
        print("│")

        # Type de données
        if pd.api.types.is_numeric_dtype(series):
            type_emoji = "🔢"
        elif pd.api.types.is_datetime64_any_dtype(series):
            type_emoji = "📅"
        else:
            type_emoji = "🔤"

        print(f"│  {type_emoji} Type: {series.dtype}")

        # Valeurs uniques
        print(f"│  🎯 Uniques: {n_unique:,} ({pct_unique:.1f}%)")

        # Valeurs manquantes
        if n_missing > 0:
            print(f"│  ⚠️ Manquantes: {n_missing:,} ({pct_missing:.1f}%)")
        else:
            print("│  ✅ Manquantes: 0 (0.0%)")

        # Valeurs explicites si peu nombreuses
        if 0 < n_unique <= 10:
            values = series.dropna().unique()
            values_str = ", ".join(map(str, values))
            if len(values_str) > 60:
                values_str = values_str[:60] + "..."
            print(f"│  📋 Valeurs: {values_str}")

        # Statistiques numériques
        if pd.api.types.is_numeric_dtype(series) and n_unique > 10:
            min_val = series.min()
            max_val = series.max()
            mean_val = series.mean()

            mean_str = f"{mean_val:.2f}" if pd.notna(mean_val) else "N/A"

            print(
                f"│  📈 Min: {min_val:.2f} | Max: {max_val:.2f} | Moyenne: {mean_str}"
            )

        print("└" + "─" * 78)
        print()

def remove_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Supprime les colonnes spécifiées du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame d'origine.
        columns (list): Liste des noms de colonnes à supprimer.

    Returns:
        pd.DataFrame: Le DataFrame sans les colonnes supprimées.
    """
    print(f"🗂️ Suppression de colonnes | shape initiale : {df.shape}")
    # Colonnes réellement présentes
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]

    df = df.drop(columns=existing_cols)

    print(
        f"✅ {len(existing_cols)} supprimée(s) | "
        f"{len(missing_cols)} inexistante(s) | "
        f"shape finale : {df.shape}"
    )

    if missing_cols:
        print(f"⚠️ Ignorées : {missing_cols}")

    return df
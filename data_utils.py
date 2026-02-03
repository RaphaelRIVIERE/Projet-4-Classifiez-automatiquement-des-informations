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


def split_ml_columns(
    df: pd.DataFrame,
    force_qualitative=None,
    force_ordinal=None,
    drop_cols=None,
    low_card_threshold=15
):
    """
    Séparation des colonnes pour pipeline ML avec distinction ordinal/catégoriel
    
    Parameters:
    -----------
    force_qualitative : list, optional
        Colonnes à forcer en catégoriel nominal
    force_ordinal : list, optional
        Colonnes à forcer en ordinal (ex: notes, satisfactions)
    low_card_threshold : int
        Seuil de cardinalité pour suggérer un type ordinal
    """
    
    force_qualitative = force_qualitative or []
    force_ordinal = force_ordinal or []
    drop_cols = drop_cols or []

    # Numériques
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    # Datetime
    datetime_cols = df.select_dtypes(include="datetime").columns.tolist()

    # Catégorielles naturelles (nominal)
    categorical_cols = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    # Numériques à faible cardinalité → ORDINAL potentiel
    low_card_numeric = [
        col for col in numeric_cols
        if df[col].nunique() <= low_card_threshold
        and col not in force_qualitative
    ]

    # Classification finale
    ordinal = sorted(set(low_card_numeric + force_ordinal) - set(drop_cols))
    
    qualitative = sorted(
        set(categorical_cols + force_qualitative) - set(ordinal) - set(drop_cols)
    )
    
    quantitative = sorted([
        col for col in numeric_cols
        if col not in ordinal 
        and col not in qualitative 
        and col not in drop_cols
    ])

    return {
        "quantitative": quantitative,
        "qualitative": qualitative,
        "ordinal": ordinal,
        "datetime": sorted(datetime_cols),
        "drop": drop_cols
    }

def check_duplicates(df: pd.DataFrame, subset=None):
    """
    Vérifie les doublons dans un DataFrame.
    """

    total_rows = len(df)
    duplicate_rows = df.duplicated(subset=subset).sum()

    return {
        "subset": subset if subset is not None else "all_columns",
        "total_rows": total_rows,
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": duplicate_rows / total_rows if total_rows > 0 else 0
    }




def explore_dataframe(df: pd.DataFrame, show_missing: bool=True):
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
    print("📋 INFORMATIONS GÉNÉRALES")
    print(f"• Lignes    : {df.shape[0]}")
    print(f"• Colonnes : {df.shape[1]}")

    print("\n--- INFO ---")
    df.info()
    
    print("\n--- DESCRIBE ---")
    display(df.describe())

    print("\n--- MISSING VALUES ---")
    missing_stats = analyze_missing_values(df)
    if show_missing:
        display(missing_stats)

    col_types = split_ml_columns(df)
    print("\n=== CLASSIFICATION DES VARIABLES ===")
    for k, v in col_types.items():
        print(f"{k.upper():<12} ({len(v)}): {v}")

    dup_info = check_duplicates(df)
    print("\n=== DUPLICATES ===")
    print(f"Lignes totales  : {dup_info['total_rows']}")
    print(f"Lignes dupliquées: {dup_info['duplicate_rows']}")
    print(f"Taux            : {dup_info['duplicate_ratio']:.2%}")



def distribution_column(
    df: pd.DataFrame, 
    column: str, 
    showtitle: bool = True, 
    max_rows: int = 20
) -> None:
    """
    Affiche la distribution des valeurs d'une colonne.
    
    Args:
        df: DataFrame pandas
        column: Nom de la colonne
        showtitle: Afficher le titre (défaut: True)
        max_rows: Nombre maximum de lignes à afficher (défaut: 20)
    """
    if showtitle:
        print(f"\n📊 Distribution de la colonne '{column}'")
        print("-" * 100)
    
    value_counts = df[column].value_counts(dropna=False)
    value_pct = (value_counts / len(df)) * 100
    
    distribution_summary = pd.DataFrame({
        'Effectif': value_counts,
        'Pourcentage': value_pct.round(2)
    })
    
    if len(distribution_summary) > max_rows:
        print(f"│  ℹ️  Affichage des {max_rows} valeurs les plus fréquentes (total: {len(distribution_summary)})")
        display(distribution_summary.head(max_rows))
    else:
        display(distribution_summary)


def display_single_column_info(
    df: pd.DataFrame, 
    col: str, 
    show_distribution: bool = False,
    max_distribution_rows: int = 10
) -> None:
    """Affiche un résumé descriptif et visuel d'une seule colonne.
    
    Args:
        df: DataFrame pandas
        col: Nom de la colonne à analyser
        show_distribution: Afficher la distribution détaillée (défaut: False)
        max_distribution_rows: Limite pour l'affichage de distribution (défaut: 10)
    """
    
    total_rows = len(df)
    
    if col not in df.columns:
        print(f"┌─ {col}")
        print("│  ❌ Colonne inexistante")
        print("└" + "─" * 78)
        print()
        return

    series = df[col]
    n_unique = series.nunique(dropna=True)
    n_missing = series.isna().sum()
    pct_unique = n_unique / total_rows * 100
    pct_missing = n_missing / total_rows * 100

    # En-tête
    print(f"┌─ {col}")
    print("│")

    # Type
    if pd.api.types.is_numeric_dtype(series):
        type_emoji = "🔢"
    elif pd.api.types.is_datetime64_any_dtype(series):
        type_emoji = "📅"
    else:
        type_emoji = "🔤"

    print(f"│  {type_emoji} Type: {series.dtype}")
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
        print(f"│  📈 Min: {min_val:.2f} | Max: {max_val:.2f} | Moyenne: {mean_str}")
    
    # Distribution détaillée (optionnelle et conditionnelle)
    if show_distribution and n_unique <= max_distribution_rows:
        print("│")
        distribution_column(df, col, showtitle=False, max_rows=max_distribution_rows)
    
    print("└" + "─" * 78)
    print()

def remove_columns(
    df: pd.DataFrame, 
    columns: List[str], 
    verbose: bool = True,
    strict: bool = False
) -> pd.DataFrame:
    """
    Supprime les colonnes spécifiées du DataFrame.

    Args:
        df: Le DataFrame d'origine
        columns: Liste des noms de colonnes à supprimer
        verbose: Afficher les messages de progression (défaut: True)
        strict: Si True, lève une erreur si une colonne n'existe pas (défaut: False)

    Returns:
        pd.DataFrame: Le DataFrame sans les colonnes supprimées
        
    Raises:
        KeyError: Si strict=True et qu'une colonne n'existe pas
    """
    if not columns:
        if verbose:
            print("⚠️ Aucune colonne à supprimer")
        return df
    
    if verbose:
        print(f"🗂️ Suppression de colonnes | shape initiale : {df.shape}")
    
    df = df.copy()
    
    # Colonnes réellement présentes
    existing_cols = [col for col in columns if col in df.columns]
    missing_cols = [col for col in columns if col not in df.columns]
    
    # Mode strict : lever une erreur si colonne manquante
    if strict and missing_cols:
        raise KeyError(f"Colonnes inexistantes : {missing_cols}")
    
    # Supprimer les colonnes existantes
    if existing_cols:
        df = df.drop(columns=existing_cols)
    
    # Affichage des résultats
    if verbose:
        if missing_cols:
            print(f"⚠️ Colonnes inexistantes (ignorées) : {missing_cols}")
        
        nb_supprimees = len(existing_cols)
        nb_ignorees = len(missing_cols)
        
        colonne_txt = "colonne" + ("s" if nb_supprimees > 1 else "")
        supprimee_txt = "supprimée" + ("s" if nb_supprimees > 1 else "")
        
        print(
            f"✅ {nb_supprimees} {colonne_txt} {supprimee_txt} | "
            f"{nb_ignorees} inexistante{'s' if nb_ignorees > 1 else ''} | "
            f"shape finale : {df.shape}"
        )
    
    return df


def compare_group_means(
	df: pd.DataFrame,
	target_col: str, 
	quanti_cols: list[str], 
	group_labels: dict[int | str, str] | None = None, 
	sort_by_gap: bool = True,
	decimals: int = 2
) -> pd.DataFrame:
    """
    Compare les moyennes de variables quantitatives entre groupes définis par une variable cible.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenant les données
    target_col : str
        Nom de la colonne cible (variable de groupement)
    quanti_cols : list
        Liste des colonnes quantitatives à comparer
    group_labels : dict, optional
        Dictionnaire pour renommer les groupes {valeur_originale: nouveau_label}
        Ex: {0: 'Restés (Non)', 1: 'Partis (Oui)'}
    sort_by_gap : bool, default=True
        Si True, trie les résultats par écart absolu décroissant
    decimales : int, default=2
        Nombre de décimales pour l'arrondi final
    
    Returns
    -------
    pd.DataFrame
        DataFrame avec les moyennes par groupe et l'écart en %
    """
    # Calculer les moyennes par groupe
    comparison = df.groupby(target_col)[quanti_cols].mean().T
    
    # Renommer les colonnes si labels fournis
    if group_labels:
        comparison.columns = [group_labels.get(col, col) for col in comparison.columns]
    
    # Calculer l'écart en % entre les deux groupes (suppose 2 groupes)
    cols = comparison.columns
    comparison['Écart (%)'] = (
        (comparison[cols[1]] - comparison[cols[0]]) / comparison[cols[0]] * 100
    ).round(1)
    
    # Trier par écart absolu si demandé
    if sort_by_gap:
        comparison = comparison.sort_values('Écart (%)', key=abs, ascending=False)
    
    # Arrondir le résultat final
    return comparison.round(decimals)
import pandas as pd
from typing import Tuple

def charger_donnees(path_eval: str, path_sirh: str, path_sondage: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les trois fichiers CSV"""
    print("📁 Chargement des données...")
    df_eval = pd.read_csv(path_eval)
    df_sirh = pd.read_csv(path_sirh)
    df_sondage = pd.read_csv(path_sondage)
    print(f"   ✓ Évaluations: {df_eval.shape}")
    print(f"   ✓ SIRH: {df_sirh.shape}")
    print(f"   ✓ Sondage: {df_sondage.shape}")
    return df_eval, df_sirh, df_sondage


def nettoyer_sirh(df_sirh):
    """Nettoie le fichier SIRH"""
    print("\n🧹 Nettoyage du fichier SIRH...")
    df = df_sirh.copy()
    
    # 1. Supprimer la colonne constante nombre_heures_travailless
    print("   • Suppression de la colonne constante 'nombre_heures_travailless'...")
    if 'nombre_heures_travailless' in df.columns:
        df = df.drop(columns=['nombre_heures_travailless'])

    # 2. Encoder le genre
    print("   • Encodage du genre (0=M, 1=F)...")
    df['genre_binaire'] = df['genre'].map({'M': 0, 'F': 1})
    
    # 3. One-hot encoding pour statut_marital
    print("   • One-hot encoding du statut marital...")
    statut_dummies = pd.get_dummies(df['statut_marital'], prefix='statut')
    df = pd.concat([df, statut_dummies], axis=1)
    
    # 4. One-hot encoding pour departement
    print("   • One-hot encoding du département...")
    dept_dummies = pd.get_dummies(df['departement'], prefix='dept')
    df = pd.concat([df, dept_dummies], axis=1)
    
    # 5. One-hot encoding pour poste
    print("   • One-hot encoding du poste...")
    poste_dummies = pd.get_dummies(df['poste'], prefix='poste')
    df = pd.concat([df, poste_dummies], axis=1)
    
    # 6. Supprimer les colonnes originales encodées
    colonnes_a_supprimer = ['genre', 'statut_marital', 'departement', 'poste']
    df = df.drop(columns=colonnes_a_supprimer)
    
    print(f"   ✓ Nettoyage terminé. Nouvelles dimensions: {df.shape}")
    return df


def pipeline(path_eval: str, path_sirh: str, path_sondage: str, save_output=True):
    """Exécute le pipeline complet de nettoyage et préparation"""
    print("="*80)
    print("PIPELINE DE NETTOYAGE ET PRÉPARATION DES DONNÉES")
    print("="*80)
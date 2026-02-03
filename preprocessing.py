import pandas as pd
from typing import Tuple
from data_utils import remove_columns

def load_data(path_sirh: str, path_eval: str, path_sondage: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Charge les trois fichiers CSV"""
    print("📁 Chargement des données...")
    df_sirh = pd.read_csv(path_sirh)
    df_eval = pd.read_csv(path_eval)
    df_sondage = pd.read_csv(path_sondage)
    print(f"   ✓ SIRH: {df_sirh.shape}")
    print(f"   ✓ Évaluations: {df_eval.shape}")
    print(f"   ✓ Sondage: {df_sondage.shape}")
    return df_sirh, df_eval, df_sondage


def clean_sirh_data(df_sirh: pd.DataFrame) -> pd.DataFrame:
    """
	Nettoie le fichier SIRH
	
	Args:
		df_sirh: DataFrame brut du SIRH
		
	Returns:
		DataFrame nettoyé
    """
    print("\n🧹 Nettoyage du fichier SIRH...")
    df = df_sirh.copy()
    df = remove_columns(df, ['nombre_heures_travailless'])
    
    print(f"   ✓ Nettoyage terminé. Nouvelles dimensions: {df.shape}")
    return df


def clean_eval_data(df_eval: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le fichier des évaluations"""
    print("\n🧹 Nettoyage du fichier des évaluations...")
    df = df_eval.copy()
    df['employee_id_extracted'] = df['eval_number'].str.extract(r'E_(\d+)')[0].astype(int)
    df['augementation_salaire_precedente'] = df['augementation_salaire_precedente'].apply(lambda x: int(x.replace('%', '').strip()))
    df['augmentation_salaire_precedente'] = df['augementation_salaire_precedente']

    df = remove_columns(df, ['eval_number', 'augementation_salaire_precedente'])
    print(f"   ✓ Nettoyage terminé. Nouvelles dimensions: {df.shape}")
    return df

def clean_sondage_data(df_sondage: pd.DataFrame) -> pd.DataFrame:
    """Nettoie le fichier des sondages"""
    print("\n🧹 Nettoyage du fichier des sondages...")
    df = df_sondage.copy()
    df = remove_columns(df, ['nombre_employee_sous_responsabilite', 'ayant_enfants'])
    print(f"   ✓ Nettoyage terminé. Nouvelles dimensions: {df.shape}")
    return df



def fusionner_datasets(df_sirh_clean: pd.DataFrame, df_eval_clean: pd.DataFrame, df_sondage_clean: pd.DataFrame):
    """Fusionne les trois datasets nettoyés avec validation"""
    print("\n🔗 Fusion des datasets...")
    
    df_sirh_clean, df_eval_clean, df_sondage_clean = df_sirh_clean.copy(), df_eval_clean.copy(), df_sondage_clean.copy()
    # Standardiser les noms de colonnes ID
    df_eval_clean = df_eval_clean.rename(columns={'employee_id_extracted': 'id_employee'})
    df_sondage_clean = df_sondage_clean.rename(columns={'code_sondage': 'id_employee'})
    
    # Étape 1: Fusionner EVAL et SIRH
    print("   • Fusion ÉVALUATIONS ↔ SIRH...")
    n_sirh, n_eval = len(df_sirh_clean), len(df_eval_clean)
    
    df_merged_1 = df_sirh_clean.merge(
        df_eval_clean,
        on='id_employee',
        how='left', #
        suffixes=('_sirh', '_eval'), # Suffixes pour éviter les conflits de noms
        validate='1:1'  # Chaque employé doit apparaître une seule fois dans chaque dataset
    )
    print(f"      → Résultat: {df_merged_1.shape}")
    print(f"      → Lignes perdues: SIRH={n_sirh - df_merged_1['id_employee'].nunique()}, "
          f"EVAL={n_eval - len(df_merged_1)}")
    
    # Étape 2: Fusionner avec SONDAGE
    print("   • Fusion avec SONDAGE...")
    n_before = len(df_merged_1)
    
    df_final = df_merged_1.merge(
        df_sondage_clean,
        on='id_employee',
        how='left',
        suffixes=('', '_sondage'),
        validate='1:1'
    )
    print(f"      → Résultat final: {df_final.shape}")
    print(f"      → Lignes perdues: {n_before - len(df_final)}")
    
    # Vérification finale
    print(f"\n✓ Employés uniques dans le dataset final: {df_final['id_employee'].nunique()}")
    
    return df_final
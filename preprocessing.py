import pandas as pd
from typing import Tuple, List
from data_utils import remove_columns
#Preprocess
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    df['annees_sous_responsable_actuel'] = df['annes_sous_responsable_actuel']
    df = remove_columns(df, ['nombre_employee_sous_responsabilite', 'ayant_enfants', 'annes_sous_responsable_actuel'])
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

    df_final = remove_columns(df_final, ['id_employee'])

    return df_final


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["genre"] = df["genre"].str.upper().map({"F": 0, "M": 1})
    df["heure_supplementaires"] = (
        df["heure_supplementaires"].str.capitalize().map({"Non": 0, "Oui": 1})
    )
    df["frequence_deplacement"] = (
        df["frequence_deplacement"].str.capitalize().map({
            "Aucun": 0,
            "Occasionnel": 1,
            "Frequent": 2
        })
    )

    return df


def features_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    ##### Features temporelles et de carrière
    # Mobilité interne (changements de poste)
    df['mobilite_interne'] = df['annees_dans_l_entreprise'] - df['annees_dans_le_poste_actuel']


    # Âge de début de carrière
    df['age_debut_carriere'] = df['age'] - df['annee_experience_totale']

    # Ratio temps sous responsable actuel / temps dans le poste
    # df['stabilite_management'] = df['annees_sous_responsable_actuel'] / (df['annees_dans_le_poste_actuel'] + 1)

    #####  Features de satisfaction et engagement

    # Score de satisfaction global (moyenne des satisfactions)
    satisfaction_cols = [
        'satisfaction_employee_environnement',
        'satisfaction_employee_nature_travail', 
        'satisfaction_employee_equipe',
        'satisfaction_employee_equilibre_pro_perso'
    ]
    df['score_satisfaction_global'] = df[satisfaction_cols].mean(axis=1)

    # Engagement formation
    # df['engagement_formation'] = df['nb_formations_suivies'] / (df['annees_dans_l_entreprise'] + 1)


    return df



def remove_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = remove_columns(df, [
        'niveau_hierarchique_poste',
        'annees_dans_l_entreprise',
        # 'annees_dans_le_poste_actuel',
        # 'annees_sous_responsable_actuel',
        # 'annees_depuis_la_derniere_promotion',
        'departement'
    ])

    return df



def prepare_ml_data(
    df: pd.DataFrame,
    target: str,
    binary_ordinal_features: List = []
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Prépare les données pour le ML : sépare X/y, encode la cible, crée le preprocessor.

    Args:
        df: DataFrame avec features + target
        target: Nom de la colonne cible (valeurs "Non"/"Oui")
        binary_ordinal_features: Liste des colonnes binaires/ordinales à ne pas scaler.
                                 Ces features passent en 'passthrough' dans le preprocessor.

    Returns:
        X: Features
        y: Target encodée (0/1)
        preprocessor: Pipeline de transformation
    """

    # Séparation features/target
    X = df.drop(columns=[target])
    y = df[target].map({"Non": 0, "Oui": 1})

    # Vérifications
    assert X.isna().sum().sum() == 0, "Il reste des valeurs manquantes dans X"
    assert y.notna().all(), f"Des valeurs de '{target}' n'ont pas été encodées"

    # Identification des types de features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_features = [
        col for col in X.select_dtypes(include=['int64', 'float64']).columns
        if col not in binary_ordinal_features
    ]

    # Création du preprocessor
    transformers = [
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]

    if binary_ordinal_features:
        transformers.append(('bin_ord', 'passthrough', binary_ordinal_features))

    preprocessor = ColumnTransformer(transformers=transformers)

    return X, y, preprocessor

from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score
)
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

def cross_validate_model(pipeline, X_train, y_train, cv, scoring):
    """
    Lance une validation croisée et agrège toutes les métriques (train et test).

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline contenant le preprocessor et le modèle.
    X_train : pd.DataFrame
        Features d'entraînement.
    y_train : pd.Series
        Cible d'entraînement.
    cv : cross-validation splitter
        Stratégie de découpage (ex: StratifiedKFold).
    scoring : str ou dict
        Métrique(s) d'évaluation.

    Returns
    -------
    dict avec les clés :
        - 'cv_results' : dict brut retourné par sklearn cross_validate
        - 'training_time_sec' : temps d'exécution
        - 'metrics_summary' : dict {metric_name: {test_mean, test_std, train_mean, train_std}}
    """
    start = time.time()

    try:
        cv_results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1
        )

        training_time = time.time() - start

        # Extraction des noms de métriques
        if isinstance(scoring, str):
            metric_names = [scoring]
        else:
            metric_names = list(scoring.keys())

        # Agrégation de toutes les métriques
        metrics_summary = {}
        for metric in metric_names:
            test_key = 'test_score' if isinstance(scoring, str) else f'test_{metric}'
            train_key = 'train_score' if isinstance(scoring, str) else f'train_{metric}'

            metrics_summary[metric] = {
                'test_mean': cv_results[test_key].mean(),
                'test_std': cv_results[test_key].std(),
                'train_mean': cv_results[train_key].mean(),
                'train_std': cv_results[train_key].std(),
            }

        return {
            'cv_results': cv_results,
            'training_time_sec': training_time,
            'metrics_summary': metrics_summary,
        }

    except Exception as e:
        return {
            'cv_results': None,
            'training_time_sec': time.time() - start,
            'metrics_summary': None,
            'error': str(e),
        }


def evaluate_model(pipeline, X_train, y_train, X_test, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Probabilités (si disponibles)
    y_proba = None
    pr_auc = None
    roc_auc = None
    if hasattr(pipeline, 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return {
        'pipeline': pipeline,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'report': report,
        'confusion_matrix': cm,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
    }


def build_comparison_df(all_cv_results=None, all_eval_results=None):
    """
    Construit un DataFrame comparatif des modèles à partir des résultats
    de la validation croisée et/ou de l'évaluation sur le jeu de validation.

    Parameters
    ----------
    all_cv_results : dict, optional
        {model_name: résultats de cross_validate_model()}
    all_eval_results : dict, optional
        {model_name: résultats de evaluate_model()}

    Returns
    -------
    pd.DataFrame
        Un modèle par ligne, métriques en colonnes.
    """
    all_cv_results = all_cv_results or {}
    all_eval_results = all_eval_results or {}

    model_names = list(dict.fromkeys(
        list(all_cv_results.keys()) + list(all_eval_results.keys())
    ))

    rows = []
    for model_name in model_names:
        row = {'model': model_name}

        # Métriques CV (train et test) avec écarts-types
        cv = all_cv_results.get(model_name)
        if cv and cv.get('metrics_summary'):
            for metric, scores in cv['metrics_summary'].items():
                row[f'cv_train_{metric}'] = scores['train_mean']
                row[f'cv_train_{metric}_std'] = scores['train_std']
                row[f'cv_test_{metric}'] = scores['test_mean']
                row[f'cv_test_{metric}_std'] = scores['test_std']
            row['cv_time_sec'] = cv.get('training_time_sec')

        # Métriques sur le jeu de test (per-class sur la classe positive "Quitte")
        # Dans modelization.py — build_comparison_df, à la fin du bloc "ev"

        ev = all_eval_results.get(model_name)
        if ev and ev.get('report'):
            report = ev['report']
            positive_class = report.get('1', {})
            row['test_accuracy'] = report.get('accuracy')
            row['test_precision'] = positive_class.get('precision')
            row['test_recall'] = positive_class.get('recall')
            row['test_f1'] = positive_class.get('f1-score')
            row['test_pr_auc'] = ev.get('pr_auc')       # NOUVEAU
            row['test_roc_auc'] = ev.get('roc_auc')      # NOUVEAU


        rows.append(row)

    return pd.DataFrame(rows).set_index('model')


def run_evaluation(pipelines, X_train, y_train, X_test, y_test, cv, scoring):
    """
    Évalue un ensemble de pipelines par validation croisée et sur le jeu de test.

    Parameters
    ----------
    pipelines : dict[str, Pipeline]
        {model_name: pipeline}
    X_train, y_train : données d'entraînement
    X_test, y_test : données de test
    cv : cross-validation splitter
    scoring : str ou dict

    Returns
    -------
    all_cv_results : dict
    all_eval_results : dict
    df_results : pd.DataFrame
    """
    all_cv_results = {}
    all_eval_results = {}
    for name, pipeline in pipelines.items():
        all_cv_results[name] = cross_validate_model(pipeline, X_train, y_train, cv=cv, scoring=scoring)
        all_eval_results[name] = evaluate_model(pipeline, X_train, y_train, X_test, y_test)
    df_results = build_comparison_df(all_cv_results=all_cv_results, all_eval_results=all_eval_results)
    return all_cv_results, all_eval_results, df_results


def get_summary_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un DataFrame de synthèse lisible (mean +/- std) pour les métriques CV et test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame indexé par nom de modèle, issu de build_comparison_df.

    Returns
    -------
    pd.DataFrame
        Tableau formaté avec CV et Test pour chaque métrique.
    """
    summary_rows = []
    for model_name in df.index:
        row = {'Modèle': model_name}
        for metric in ['average_precision', 'recall', 'precision', 'f1']:
            mean = df.loc[model_name, f'cv_test_{metric}']
            std = df.loc[model_name, f'cv_test_{metric}_std']
            display_name = 'PR-AUC' if metric == 'average_precision' else metric
            row[f'CV {display_name}'] = f"{mean:.3f} ± {std:.3f}"
            test_col = 'test_pr_auc' if metric == 'average_precision' else f'test_{metric}'
            row[f'Test {display_name}'] = f"{df.loc[model_name, test_col]:.3f}"
        summary_rows.append(row)

    return pd.DataFrame(summary_rows).set_index('Modèle')


def aggregate_shap_by_original_feature(shap_values, transformed_names, original_names):
    """
    Agrège les SHAP values des colonnes transformées (post OHE) vers les
    features originales, pour pouvoir comparer avec la permutation importance.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values calculées sur les données transformées.
    transformed_names : list[str]
        Noms des features après transformation (ex: 'poste_Manager').
    original_names : list[str]
        Noms des features brutes avant transformation (ex: 'poste').

    Returns
    -------
    pd.DataFrame
        Colonnes: feature, mean_shap — trié par importance décroissante.
    """
    shap_df = pd.DataFrame(shap_values.values, columns=transformed_names)

    # Mapper chaque colonne transformée vers la feature originale
    mapping = {}
    for col in transformed_names:
        matched = False
        for orig in original_names:
            if col == orig or col.startswith(orig + '_'):
                mapping[col] = orig
                matched = True
                break
        if not matched:
            mapping[col] = col

    shap_df.columns = [mapping[c] for c in shap_df.columns]

    # Sommer les |SHAP| des colonnes issues de la même feature originale
    agg = shap_df.abs().mean().groupby(level=0).sum().sort_values(ascending=False)

    return pd.DataFrame({
        'feature': agg.index,
        'mean_shap': agg.values,
    }).reset_index(drop=True)


def fine_tune_model(
    pipeline,
    param_grid,
    X_train, y_train,
    cv,
    scoring,
    refit,
    n_jobs=-1,
    verbose=1,
):
    """
    Fine-tune un pipeline via GridSearchCV (entraînement + CV uniquement).

    L'évaluation sur un jeu de test doit se faire séparément via
    evaluate_model() ou run_evaluation(), afin d'éviter tout risque
    de data leakage indirect.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline contenant preprocessor + modèle.
    param_grid : dict
        Grille d'hyperparamètres (préfixés par le nom du step, ex: 'model__max_depth').
    X_train, y_train : données d'entraînement
    cv : cross-validation splitter
    scoring : dict
        Métriques d'évaluation (même dict que pour run_evaluation).
    refit : str
        Métrique utilisée pour sélectionner le meilleur modèle.
    n_jobs : int
    verbose : int

    Returns
    -------
    dict avec :
        - 'grid_search': objet GridSearchCV fitté
        - 'best_params': meilleurs hyperparamètres
        - 'best_score': meilleur score CV (métrique refit)
        - 'best_pipeline': meilleur estimateur (déjà fitté sur X_train)
        - 'cv_results_df': DataFrame complet des résultats GridSearchCV
        - 'cv_results': dict compatible avec cross_validate_model()
    """
    start = time.time()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        refit=refit,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    training_time = time.time() - start
    best_pipeline = grid_search.best_estimator_

    # Construire un cv_results compatible avec cross_validate_model()
    # en extrayant les scores du meilleur modèle depuis GridSearchCV
    best_idx = grid_search.best_index_
    cv_results_raw = grid_search.cv_results_

    metric_names = list(scoring.keys()) if isinstance(scoring, dict) else [scoring]
    metrics_summary = {}
    cv_results_compat = {}

    for metric in metric_names:
        test_mean = cv_results_raw[f'mean_test_{metric}'][best_idx]
        test_std = cv_results_raw[f'std_test_{metric}'][best_idx]
        train_mean = cv_results_raw.get(f'mean_train_{metric}', [None])[best_idx]
        train_std = cv_results_raw.get(f'std_train_{metric}', [None])[best_idx]

        metrics_summary[metric] = {
            'test_mean': test_mean,
            'test_std': test_std,
            'train_mean': train_mean,
            'train_std': train_std,
        }

        # Reconstituer les scores par fold pour compatibilité boxplots
        n_splits = cv.get_n_splits()
        cv_results_compat[f'test_{metric}'] = np.array([
            cv_results_raw[f'split{i}_test_{metric}'][best_idx]
            for i in range(n_splits)
        ])
        if f'split0_train_{metric}' in cv_results_raw:
            cv_results_compat[f'train_{metric}'] = np.array([
                cv_results_raw[f'split{i}_train_{metric}'][best_idx]
                for i in range(n_splits)
            ])

    cv_output = {
        'cv_results': cv_results_compat,
        'training_time_sec': training_time,
        'metrics_summary': metrics_summary,
    }

    return {
        'grid_search': grid_search,
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_pipeline': best_pipeline,
        'cv_results_df': pd.DataFrame(cv_results_raw),
        'cv_results': cv_output,
    }



def compute_permutation_importance(pipeline, X_test, y_test, scoring: str,
                                   n_repeats=30, random_state=42, n_jobs=-1):
    perm_result = permutation_importance(
        pipeline,
        X_test,
        y_test,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    return pd.DataFrame({
        'feature': X_test.columns.tolist(),
        'importance_mean': perm_result.importances_mean,
        'importance_std': perm_result.importances_std
    }).sort_values('importance_mean', ascending=True)

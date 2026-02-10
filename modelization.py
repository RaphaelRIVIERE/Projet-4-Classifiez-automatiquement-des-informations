from sklearn.model_selection import cross_validate
from sklearn.metrics import (
    classification_report, confusion_matrix,
    average_precision_score, roc_auc_score
)
import matplotlib.pyplot as plt
import pandas as pd
import time


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
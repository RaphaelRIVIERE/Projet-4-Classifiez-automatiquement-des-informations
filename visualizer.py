import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
from typing import Literal
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay



def _apply_formatting(ax, title=None, xlabel=None, ylabel=None,
                      xticks_rotation=0, yticks_rotation=0, grid=False,
                      legend_title=None, show_legend=False):
    """Applique le formatage de base a un axe matplotlib."""
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)

    ax.tick_params(axis='x', rotation=xticks_rotation)
    ax.tick_params(axis='y', rotation=yticks_rotation)

    if grid:
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    if show_legend and ax.get_legend_handles_labels()[0]:
        ax.legend(title=legend_title, loc='best', fontsize=10)


def _annotate_bars(ax, fmt='{:.2f}', ylim_margin=1.15):
    """Ajoute les valeurs au-dessus des barres."""
    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt=fmt, padding=3, fontsize=10)

    ymin, ymax = ax.get_ylim()
    if ymax > 0:
        ax.set_ylim(ymin, ymax * ylim_margin)


def create_barplot(df, ax, x, y=None, hue=None, title=None, xlabel=None,
                   ylabel=None, xticks=45, palette=None, legend=None,
                   annot=True, alpha=1.0, edgecolor=None, linewidth=0.0,
                   width=0.8, errorbar=None, estimator='mean',
                   ylim_margin=1.15):
    """Cree un barplot ou un countplot."""
    if y is None:
        sns.countplot(data=df, x=x, hue=hue, palette=palette, ax=ax,
                      alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
    else:
        sns.barplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax,
                    alpha=alpha, edgecolor=edgecolor, linewidth=linewidth,
                    width=width, errorbar=errorbar, estimator=estimator)

    xticks_rotation = xticks if isinstance(xticks, int) else 0
    legend_title = legend if isinstance(legend, str) else None

    _apply_formatting(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                      xticks_rotation=xticks_rotation,
                      legend_title=legend_title,
                      show_legend=(hue is not None))

    if annot:
        _annotate_bars(ax, ylim_margin=ylim_margin)


def create_boxplot(df, ax, x=None, y=None, hue=None, title=None,
                   xlabel=None, ylabel=None, xticks=None, palette=None,
                   legend=None, log_scale=False, showfliers=True,
                   width=0.8, linewidth=1.5):
    """Cree un boxplot."""
    sns.boxplot(data=df, x=x, y=y, hue=hue, ax=ax, palette=palette,
                showfliers=showfliers, width=width, linewidth=linewidth)

    if log_scale:
        ax.set_yscale('log')
        if title:
            title = f"{title} (log scale)"

    xticks_rotation = xticks if isinstance(xticks, int) else 0
    legend_title = legend if isinstance(legend, str) else None

    _apply_formatting(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                      xticks_rotation=xticks_rotation,
                      legend_title=legend_title,
                      show_legend=(hue is not None))


def create_scatterplot(
    df, ax, x, y,
    hue=None, size=None, style=None,
    title=None, xlabel=None, ylabel=None,
    palette=None, color=None,
    alpha=None, s=None,
    edgecolor=None, linewidth=0,
    marker='o',
    regression=False,
    annotate_stats=False,
    grid=False,
    legend_title=None
):
    """
    Scatterplot Seaborn avec :
    - réduction du sur-plotting
    - jitter optionnel
    - régression linéaire (globale ou par hue)
    Compatible avec _apply_formatting (OpenClassrooms safe)
    """

    if s is None:
        s = 25

    if alpha is None:
        alpha = min(0.6, 500 / max(len(df), 1))

    plot_df = df.copy()


    sns.scatterplot(
        data=plot_df,
        x=x, y=y,
        hue=hue, size=size, style=style,
        ax=ax,
        palette=palette if hue else None,
        color=color if not hue else None,
        alpha=alpha,
        s=s,
        edgecolor=edgecolor,
        linewidth=linewidth,
        marker=marker
    )


    if regression:
        groups = [(None, plot_df)] if not hue else plot_df.groupby(hue)

        for name, group in groups:
            x_clean = group[x].dropna()
            y_clean = group.loc[x_clean.index, y].dropna()

            if len(x_clean) < 2:
                continue

            slope, intercept = np.polyfit(x_clean, y_clean, 1)
            x_vals = np.linspace(x_clean.min(), x_clean.max(), 200)
            y_vals = slope * x_vals + intercept

            ax.plot(
                x_vals, y_vals,
                linestyle='--',
                linewidth=2,
                label=f"Régression ({name})" if name is not None else "Régression"
            )

            if annotate_stats:
                r = np.corrcoef(x_clean, y_clean)[0, 1]
                ax.text(
                    0.02,
                    0.95 if name is None else 0.90,
                    f"{name}: r = {r:.2f}" if name else f"r = {r:.2f}",
                    transform=ax.transAxes,
                    fontsize=10,
                    verticalalignment='top'
                )

    _apply_formatting(
        ax,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        grid=grid,
        legend_title=legend_title,
        show_legend=(hue is not None or size is not None
                     or style is not None or regression)
    )


def create_heatmap(df, ax, annot=True, fmt='.2f', cmap='coolwarm',
                   title=None, xlabel=None, ylabel=None,
                   xticks=45, yticks=0, linewidths=0.5, linecolor='white',
                   vmin=None, vmax=None, center=None,
                   cbar=True, square=False, mask=None):
    """Cree une heatmap."""
    sns.heatmap(data=df, ax=ax, annot=annot, fmt=fmt, cmap=cmap,
                linewidths=linewidths, linecolor=linecolor,
                vmin=vmin, vmax=vmax, center=center,
                cbar=cbar, square=square, mask=mask)

    xticks_rotation = xticks if isinstance(xticks, int) else 0
    yticks_rotation = yticks if isinstance(yticks, int) else 0

    _apply_formatting(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                      xticks_rotation=xticks_rotation,
                      yticks_rotation=yticks_rotation)



def plot_contingency_analysis(
    df,
    *,
    rows,
    cols,
    axes,
    normalize: Literal['index', 'columns', 'all', 0, 1] = 'index',
    heatmap_title=None,
    barplot_title=None,
    xlabel=None,
    ylabel=None,
    legend_title=None,
):
    """
    Analyse de contingence avec heatmap (brute) et barplot (normalisé).
    
    Parameters
    ----------
    df : DataFrame
    rows : str - Variable en lignes
    cols : str - Variable en colonnes
    axes : array - Tableau de 2 axes matplotlib
    normalize : str - Type de normalisation ('index', 'columns', 'all')
    """
    assert len(axes) == 2, "axes doit contenir exactement 2 sous-graphiques"
    
    # Nettoyage
    df_clean = df[[rows, cols]].dropna()
    
    # Tables de contingence
    contingency = pd.crosstab(df_clean[rows], df_clean[cols])
    ct_norm = pd.crosstab(df_clean[cols], df_clean[rows], normalize=normalize) * 100
    ct_long = ct_norm.reset_index().melt(
        id_vars=cols,
        var_name=rows,
        value_name='pourcentage'
    )
    
    # Heatmap (distribution brute)
    create_heatmap(
        contingency,
        axes[0],
        title=heatmap_title,
        xlabel=xlabel or cols,
        ylabel=ylabel or rows,
    )
    
    # Barplot (répartition conditionnelle %)
    create_barplot(
        ct_long,
        axes[1],
        x=cols,
        y='pourcentage',
        hue=rows,
        title=barplot_title,
        legend=legend_title or rows,
        xlabel=xlabel or cols,
        ylabel="Pourcentage"
    )


def create_pairplot(df, columns, hue=None, title=None,
                    palette=None, legend=None,
                    xticks=0, yticks=0, grid=False,
                    diag_kind: Literal['auto', 'hist', 'kde'] = 'kde', alpha=0.4, s=15, corner=True):
    """Cree un pairplot Seaborn.

    Retourne l'objet PairGrid pour permettre des ajustements ulterieurs.
    """
    

    g = sns.pairplot(
        df[columns],
        hue=hue,
        palette=palette,
        diag_kind=diag_kind,
        plot_kws={'alpha': alpha, 's': s},
        corner=corner,
    )

    xticks_rotation = xticks if isinstance(xticks, int) else 0
    yticks_rotation = yticks if isinstance(yticks, int) else 0
    legend_title = legend if isinstance(legend, str) else None

    for ax in g.axes.flat:
        if ax is not None:
            _apply_formatting(
                ax,
                xticks_rotation=xticks_rotation,
                yticks_rotation=yticks_rotation,
                grid=grid,
                legend_title=legend_title,
                show_legend=False,
            )

    if title:
        g.figure.suptitle(title, y=1.02, fontsize=14, fontweight='bold')

    if hue is not None and g.legend is not None and legend_title:
        g.legend.set_title(legend_title)

    return g


def plot_metrics_comparison(df_results, ax, metric_cols=None, title=None,
                            figsize=(14, 6)):
    """
    Barplot groupé comparant plusieurs métriques pour chaque modèle.

    Parameters
    ----------
    df_results : pd.DataFrame
        DataFrame indexé par le nom du modèle, avec des colonnes de métriques.
    ax : matplotlib.axes.Axes
        Axe sur lequel dessiner.
    metric_cols : dict | None
        Mapping {nom_colonne: label_affiché}.
        Par défaut : Precision, Recall, F1, PR-AUC, ROC-AUC.
    title : str | None
        Titre du graphique.
    """
    if metric_cols is None:
        metric_cols = {
            'test_roc_auc': 'ROC-AUC',
            'test_pr_auc': 'PR-AUC',
            'test_recall': 'Recall',
            'test_f1': 'F1',
            'test_precision': 'Precision',
        }

    rows = []
    for model in df_results.index:
        for col, label in metric_cols.items():
            rows.append({
                'model': model,
                'metric': label,
                'score': df_results.loc[model, col],
            })

    df_long = pd.DataFrame(rows)

    create_barplot(
        df_long, ax, x='model', y='score', hue='metric',
        title=title,
    )


def visualize_cv_results(all_cv_results, df_all_results, metric='f1', suffix=None):
    """
    Visualise les résultats de cross-validation avec deux graphiques :
    - Distribution des scores par fold
    - Comparaison train vs test pour détecter l'overfitting

    Parameters
    ----------
    all_cv_results : dict
        Dictionnaire contenant les résultats CV pour chaque modèle
        Format: {model_name: {'cv_results': {'test_metric': [scores]}}}
    df_all_results : pd.DataFrame
        DataFrame avec les colonnes 'cv_train_{metric}' et 'cv_test_{metric}'
        et un index 'model'
    metric : str, default='f1'
        Métrique à visualiser (f1, accuracy, precision, recall, etc.)
    suffix : str, optional
        Suffixe ajouté aux titres (ex: "balanced", "SMOTE").
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    label = metric.upper().replace('_', '-')
    suffix_str = f" — {suffix}" if suffix else ""

    # Préparer les données pour le boxplot (distribution par fold)
    rows = []
    for name, res in all_cv_results.items():
        test_key = f'test_{metric}'
        if test_key in res['cv_results']:
            for score in res['cv_results'][test_key]:
                rows.append({'model': name, f'{metric}_score': score})

    df_cv_folds = pd.DataFrame(rows)

    # Préparer les données pour le barplot (train vs test)
    train_col = f'cv_train_{metric}'
    test_col = f'cv_test_{metric}'

    df_overfit = df_all_results[[train_col, test_col]].reset_index().melt(
        id_vars='model',
        var_name='set',
        value_name=f'{metric}_score'
    )

    create_boxplot(
        df_cv_folds, axes[0],
        x='model',
        y=f'{metric}_score',
        title=f'Distribution {label} par fold (CV){suffix_str}',
        ylabel=label
    )

    create_barplot(
        df_overfit, axes[1],
        x='model',
        y=f'{metric}_score',
        hue='set',
        title=f'{label} Train vs Test (diagnostic overfitting){suffix_str}'
    )

    plt.tight_layout()
    plt.show()
    


def create_pr_curves(all_eval_results, y_test, ax, title="Courbes Précision–Rappel"):
    """Trace les courbes Précision–Rappel."""
    for name, res in all_eval_results.items():
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue

        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_proba,
            name=f"{name} (AP={res['pr_auc']:.3f})",
            ax=ax
        )

    prevalence = y_test.mean()
    ax.axhline(
        y=prevalence,
        linestyle="--",
        color="grey",
        label=f"Prévalence ({prevalence:.1%})"
    )

    ax.set_title(title)
    ax.legend(loc="upper right")


def create_roc_curves(all_eval_results, y_test, ax, title="Courbes ROC"):
    """Trace les courbes ROC."""
    for name, res in all_eval_results.items():
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue

        RocCurveDisplay.from_predictions(
            y_test,
            y_proba,
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            ax=ax
        )

    ax.plot([0, 1], [0, 1], "grey", linestyle="--", label="Aléatoire")

    ax.set_title(title)
    ax.legend(loc="lower right")


def plot_pr_curves(all_eval_results, y_test, title="Courbes Précision–Rappel"):

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in all_eval_results.items():
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue

        PrecisionRecallDisplay.from_predictions(
            y_test,
            y_proba,
            name=f"{name} (AP={res['pr_auc']:.3f})",
            ax=ax
        )

    prevalence = y_test.mean()
    ax.axhline(
        y=prevalence,
        linestyle="--",
        color="grey",
        label=f"Prévalence ({prevalence:.1%})"
    )

    ax.set_title(title)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_roc_curves(all_eval_results, y_test, title="Courbes ROC"):

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, res in all_eval_results.items():
        y_proba = res.get("y_proba")
        if y_proba is None:
            continue

        RocCurveDisplay.from_predictions(
            y_test,
            y_proba,
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            ax=ax
        )

    ax.plot([0, 1], [0, 1], "grey", linestyle="--", label="Aléatoire")

    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()



def plot_roc_pr_curves(all_eval_results, y_test, suptitle="ROC vs PR"):
    """
    Trace les courbes ROC et Precision-Recall côte à côte pour tous les modèles.

    Parameters
    ----------
    all_eval_results : dict
        {model_name: {'y_proba': array, 'roc_auc': float, 'pr_auc': float}}
    y_test : array-like
        Vraies étiquettes du jeu de test.
    suptitle : str
        Titre principal de la figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for name, res in all_eval_results.items():
        y_proba = res.get('y_proba')
        if y_proba is None:
            continue

        RocCurveDisplay.from_predictions(
            y_test, y_proba,
            name=f"{name} (AUC={res['roc_auc']:.3f})",
            ax=axes[0]
        )
        PrecisionRecallDisplay.from_predictions(
            y_test, y_proba,
            name=f"{name} (AP={res['pr_auc']:.3f})",
            ax=axes[1]
        )

    axes[0].plot([0, 1], [0, 1], 'grey', linestyle='--', label='Aléatoire')
    axes[0].set_title("Courbes ROC")
    axes[0].legend(loc='lower right')

    prevalence = y_test.mean()
    axes[1].axhline(y=prevalence, color='grey', linestyle='--',
                    label=f'Prévalence ({prevalence:.1%})')
    axes[1].set_title("Courbes Précision-Rappel")
    axes[1].legend(loc='upper right')

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(all_eval_results, class_names, ncols=None):
    """
    Trace les matrices de confusion pour tous les modèles évalués.

    Parameters
    ----------
    all_eval_results : dict
        {model_name: {'confusion_matrix': array}}
    class_names : list[str]
        Noms des classes (ex: ['Reste', 'Quitte']).
    ncols : int, optional
        Nombre de colonnes. Par défaut = nombre de modèles.
    """
    n = len(all_eval_results)
    ncols = ncols or n
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for ax, (name, res) in zip(axes, all_eval_results.items()):
        cm_df = pd.DataFrame(res['confusion_matrix'],
                             index=class_names, columns=class_names)
        create_heatmap(cm_df, ax, fmt='d', cmap='Blues',
                       title=f'Matrice de confusion - {name}',
                       xlabel='Prédit', ylabel='Réel')

    # Masquer les axes inutilisés
    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()


def compare_model_versions(
    results,
    models,
    metrics=None,
    name_pattern='{model}_{version}',
    figsize=(18, 5),
):
    """
    Compare les performances de modèles entre différentes versions.

    Parameters
    ----------
    results : dict[str, pd.DataFrame]
        {version_name: df_results} - chaque DF indexé par nom du modèle.
    models : list[str]
        Noms courts des modèles (ex: ['LR', 'RF', 'XGB']).
    metrics : list[str], optional
        Colonnes à comparer. Défaut: ['test_pr_auc', 'test_recall', 'test_f1'].
    name_pattern : str
        Pattern pour retrouver l'index. Défaut: '{model}_{version}'.
    figsize : tuple
        Taille de la figure.
    """
    import matplotlib.pyplot as plt

    if metrics is None:
        metrics = ['test_roc_auc', 'test_pr_auc', 'test_recall']
		# metrics = ['test_pr_auc', 'test_recall', 'test_f1']


    versions = list(results.keys())
    df_combined = pd.concat(results.values())

    rows = []
    for model in models:
        for version in versions:
            index_key = name_pattern.format(model=model, version=version)
            for metric in metrics:
                label = metric.replace('test_', '').upper().replace('_', '-')
                rows.append({
                    'Modèle': model,
                    'Version': version,
                    'Métrique': label,
                    'Score': df_combined.loc[index_key, metric],
                })

    df_comp = pd.DataFrame(rows)
    metric_labels = [m.replace('test_', '').upper().replace('_', '-') for m in metrics]

    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    title_suffix = ' vs '.join(versions)
    for ax, metric_label in zip(axes, metric_labels):
        subset = df_comp[df_comp['Métrique'] == metric_label]
        create_barplot(
            subset, ax, x='Modèle', y='Score', hue='Version',
            title=f'{metric_label} : {title_suffix}'
        )

    plt.tight_layout()
    plt.show()



def create_barh(ax, labels, values, color='#3498db', title=None, xlabel=None, ylabel=None, grid=True):
    """Crée un barplot horizontal simple."""
    ax.barh(labels, values, color=color)
    _apply_formatting(ax, title=title, xlabel=xlabel, ylabel=ylabel, grid=grid)

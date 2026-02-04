import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib.container import BarContainer
from typing import Literal



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

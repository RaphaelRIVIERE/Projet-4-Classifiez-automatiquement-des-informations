import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer


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
                   ylim_margin=1.15, **kwargs):
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
                   width=0.8, linewidth=1.5, **kwargs):
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


def create_scatterplot(df, ax, x, y, hue=None, size=None, style=None,
                       title=None, xlabel=None, ylabel=None, palette=None,
                       color=None, alpha=0.7, s=None, edgecolor=None,
                       linewidth=0.5, marker='o', regression=False,
                       regression_kwargs=None, **kwargs):
    """Cree un scatterplot avec regression lineaire optionnelle."""
    if size is None and s is None:
        s = 36

    sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, style=style,
                    ax=ax,
                    palette=palette if hue else None,
                    color=color if not hue else None,
                    alpha=alpha, s=s, edgecolor=edgecolor,
                    linewidth=linewidth, marker=marker)

    if regression:
        x_data, y_data = df[x], df[y]
        mask = x_data.notna() & y_data.notna()
        x_clean, y_clean = x_data[mask], y_data[mask]

        if len(x_clean) >= 2:
            slope, intercept = np.polyfit(x_clean, y_clean, deg=1)
            x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
            y_line = slope * x_line + intercept
            default_kwargs = {'color': 'black', 'linestyle': '--',
                              'linewidth': 1.5, 'label': 'Regression lineaire'}
            ax.plot(x_line, y_line, **(regression_kwargs or default_kwargs))

    _apply_formatting(ax, title=title, xlabel=xlabel, ylabel=ylabel,
                      show_legend=(hue is not None or size is not None
                                   or style is not None or regression))

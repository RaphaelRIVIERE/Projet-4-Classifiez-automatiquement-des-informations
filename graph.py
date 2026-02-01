import pandas as pd
import seaborn as sns
from matplotlib.pyplot import Axes
from typing import Union, Optional

def create_hist(
    df: pd.DataFrame,
    ax: Axes,
    x: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotate_x: int = 0
):
    sns.histplot(
        data=df,
        x=x,
        ax=ax,
        # bins=20,
        # kde=True
    )

    if rotate_x:
        ax.tick_params(axis='x', rotation=rotate_x)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)

def create_boxplot(
    df: pd.DataFrame,
    ax: Axes,
    x: Union[str, None] = None,
    y: Union[str, None] = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    rotate_x: int = 0,
    log_scale: bool = False
):
    sns.boxplot(
        data=df,
        x=x,
        y=y,
        ax=ax
    )

    if log_scale:
        ax.set_yscale("log")
        title = title + " (log scale)"
        ylabel = ylabel + " (log scale)"

    if rotate_x:
        ax.tick_params(axis='x', rotation=rotate_x)


    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    

def create_heatmap_contingency(
    df: pd.DataFrame,
    ax: Axes,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    top_n_x: Optional[int] = None,
    top_n_y: Optional[int] = None,
    normalize: bool = False,
    annot: bool = False,
    fmt: Optional[str] = None
):
    df_work = df.copy()

    if top_n_x is not None:
        top_x = df_work[x].value_counts().head(top_n_x).index
        df_work = df_work[df_work[x].isin(top_x)]

    if top_n_y is not None:
        top_y = df_work[y].value_counts().head(top_n_y).index
        df_work = df_work[df_work[y].isin(top_y)]

    contingency = pd.crosstab(
        df_work[y],
        df_work[x],
        normalize='index' if normalize else False
    )

    if fmt is None:
        fmt = '.2f' if normalize else 'd'

    sns.heatmap(
        contingency if not normalize else contingency * 100,
        cmap='Blues',
        annot=annot,
        fmt=fmt,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)



def create_stacked_barplot(
    df: pd.DataFrame,
    ax: Axes,
    x: str,
    stack_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    legend_title: str = "",
    top_n_x: Optional[int] = None,
    top_n_stack: Optional[int] = None,
    rotate_x: int = 45,
    normalize: bool = False
):
    df_work = df.copy()

    # Limiter la cardinalité
    if top_n_x is not None:
        top_x = df_work[x].value_counts().head(top_n_x).index
        df_work = df_work[df_work[x].isin(top_x)]

    if top_n_stack is not None:
        top_stack = df_work[stack_col].value_counts().head(top_n_stack).index
        df_work = df_work[df_work[stack_col].isin(top_stack)]
    # Table de contingence
    contingency = (
        df_work
        .groupby([x, stack_col])
        .size()
        .unstack(fill_value=0)
    )

    # Normalisation optionnelle (proportions)
    if normalize:
        contingency = contingency.div(contingency.sum(axis=1), axis=0)

    # Barplot empilé
    bottom = pd.Series([0] * len(contingency), index=contingency.index)

    for category in contingency.columns:
        ax.bar(
            contingency.index,
            contingency[category],
            bottom=bottom,
            label=category
        )
        bottom += contingency[category]

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(axis="x", rotation=rotate_x)
    ax.legend(title=legend_title)

def annotate_bars(ax, fmt="{:.2f}", padding=3):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=padding)

def create_barplot(
    df: pd.DataFrame,
    ax: Axes,
    x: str,
    y: Optional[str] = None,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    palette=None,
    rotation=45,
    annotate=True
):
    if y is None:
        sns.countplot(data=df, x=x, hue=x, palette=palette, ax=ax)
    else:
        sns.barplot(data=df, x=x, y=y, hue=x, palette=palette, ax=ax)


    ax.tick_params(axis="x", rotation=rotation)
    ax.set_title(title if title else f"{y} par {x}")
    if xlabel:
    	ax.set_xlabel(xlabel)

    if ylabel:
        ax.set_ylabel(ylabel)

    if annotate:
        annotate_bars(ax)
        # Augmenter ylim pour laisser de l'espace aux annotations
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15 if ymax > 0 else ymax * 0.85)
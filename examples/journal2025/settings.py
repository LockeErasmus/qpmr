"""
Settings shared among all plots
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pathlib
import os


plt.rcParams.update({
    "font.size": 12,            # Default font size
    "axes.titlesize": 14,       # Axes title font size
    "axes.labelsize": 12,       # X/Y axis labels
    "xtick.labelsize": 10,      # X tick labels
    "ytick.labelsize": 10,      # Y tick labels
    "legend.fontsize": 10,      # Legend font size
    "figure.titlesize": 16,     # Figure title
    "text.usetex": True,
    "font.family": "serif",
})

# TWO column article -> LINE_WIDTH = one column width
LINE_WIDTH = 252.0 / 72 # 1 pt = 1/72 inches


# TODO save figure routine
PATH = pathlib.Path(__file__).parent.resolve() # path to figures
def save_figure(fig: Figure, name: str, path: os.PathLike[str]=PATH, **kwargs) -> None:
    default_kwargs = {"bbox_inches": "tight"}
    fname = pathlib.Path(path) / f"{name}.pdf"
    print(f"Saving figure: '{fname}'")
    fig.savefig(fname=fname, **(default_kwargs | kwargs))



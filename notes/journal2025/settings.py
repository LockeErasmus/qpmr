"""
Settings shared among all plots
"""
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import pathlib
import os


plt.rcParams.update({
    "font.size": 8,            # Default font size
    "axes.titlesize": 8,       # Axes title font size
    "axes.labelsize": 8,       # X/Y axis labels
    "xtick.labelsize": 7,      # X tick labels
    "ytick.labelsize": 7,      # Y tick labels
    "legend.fontsize": 7,      # Legend font size
    "figure.titlesize": 8,     # Figure title
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "serif",
})

# TWO column article -> LINE_WIDTH = one column width
LINE_WIDTH = 252.0 / 72 # 1 pt = 1/72 inches

SMALL_HEIGHT = LINE_WIDTH * 0.35
MEDIUM_HEIGHT = LINE_WIDTH * 1.0
LARGE_HEIGHT = LINE_WIDTH * 1.8


# TODO save figure routine
PATH = pathlib.Path(__file__).parent.resolve() # path to figures
def save_figure(fig: Figure, name: str, path: os.PathLike[str]=PATH, **kwargs) -> None:
    default_kwargs = {"bbox_inches": "tight"}
    fname = pathlib.Path(path) / f"{name}.pdf"
    print(f"Saving figure: '{fname}'")
    fig.savefig(fname=fname, **(default_kwargs | kwargs))



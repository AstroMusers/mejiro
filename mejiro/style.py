import matplotlib.pyplot as plt
from tdpy.plotting import configure_plot_style, normalize_plot_background

import mejiro


AAS_STYLE = f'{mejiro.__path__[0]}/aas.mplstyle'
TWO_COLUMN_WIDTH = 7.1
ONE_COLUMN_WIDTH = 3.4  # the exact value is 3.39375 inches


def set_aas_style(typeplotback='white'):
    """Set AAS journal typography with a white or dark plot background."""
    normalize_plot_background(typeplotback)
    plt.style.use(AAS_STYLE)
    configure_plot_style(typeplotback)
    
import matplotlib.pyplot as plt

import mejiro


AAS_STYLE = f'{mejiro.__path__[0]}/aas.mplstyle'
TWO_COLUMN_WIDTH = 7.1
ONE_COLUMN_WIDTH = 3.4  # the exact value is 3.39375 inches


def set_aas_style():
    """Set matplotlib style to AAS Journals style."""
    plt.style.use(AAS_STYLE)
    
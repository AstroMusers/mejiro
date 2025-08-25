__author__ = "Bryce Wedig"
__version__ = "2.0.0"


import os
import matplotlib.pyplot as plt


style_path = os.path.join(os.path.dirname(__file__), 'mejiro.mplstyle')
plt.style.use(style_path)

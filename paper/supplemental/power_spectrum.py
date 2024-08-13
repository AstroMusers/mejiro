#!/usr/bin/env python
# coding: utf-8

import os
import sys
from copy import deepcopy
from glob import glob

import hydra
import matplotlib.pyplot as plt
import numpy as np
from galsim import InterpolatedImage, Image, UniformDeviate
from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver
from lenstronomy.Util.correlation import power_spectrum_1d
from pyHalo.preset_models import CDM
from tqdm import tqdm




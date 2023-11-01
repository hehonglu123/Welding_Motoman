import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
from general_robotics_toolbox import *
from weld_dh2v import *

class LoglogModel(object):
    def __init__(self,material) -> None:
        
        self.material=material
        self.material_param=material_param[material]
        
    
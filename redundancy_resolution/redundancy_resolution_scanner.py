import numpy as np
import sys, traceback, time, copy
from general_robotics_toolbox import *
import matplotlib.pyplot as plt
from qpsolvers import solve_qp

sys.path.append('../toolbox')
from robot_def import *
from path_calc import *
# from utils import *

class redundancy_resolution(object):
	###robot1 hold weld torch, positioner hold welded part
	def __init__(self,robot,positioner,scan_path):
		# curve_sliced: list of sliced layers, in curve frame
		# robot: welder robot
		# positioner: 2DOF rotational positioner
		self.robot=robot
		self.positioner=positioner
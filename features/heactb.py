from constants import *
import numpy as np

def harmonize(value):
	if ( (value == REFUSAL) or (value == DONT_KNOW) or 
		(value ==  NOT_APPLICABLE)):
		return np.nan
	else:
		return value

def binarize(value):
	if ( (value == 3) or (value == 4)):
		return 0
	elif ( (value == 1) or (value == 2)):
		return 1
	elif np.isnan(value):
		return value

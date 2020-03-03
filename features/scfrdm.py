from constants import *
import numpy as np

def harmonize(value):
	if ( (value == REFUSAL) or (value == DONT_KNOW) or 
		(value ==  NOT_APPLICABLE) or (value== SCHD_NOT_APPLICABLE)):
		return np.nan
	return value

def binarize(value):
	if (value == 0):
		return 0
	elif (value>=3):
		return 1
	else:
		return 2

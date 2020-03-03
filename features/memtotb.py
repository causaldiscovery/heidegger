from constants import *
import numpy as np

def harmonize(value):
	if ( (value == REFUSAL) or (value == NOT_ASKED) or (value== SCHD_NOT_APPLICABLE) or 
		(value == NOT_APPLICABLE) or (value == -7)):
		return np.nan
	else:
		return value


def binarize(value):
	return value

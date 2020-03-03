from constants import *
import numpy as np

def harmonize(value):
	if ( (value == REFUSAL) or (value == DONT_KNOW) or 
		(value ==  NOT_APPLICABLE) or (value== SCHD_NOT_APPLICABLE)):
		return np.nan
	return value

def binarize(value):
	if (value==1)or (value==2) or (value==3) or (value==4) or (value==5)or (value==6) or (value==7):
		return 1
	elif  (value==8):
		return 0
	elif np.isnan(value):
		return value

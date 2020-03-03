from constants import *
import numpy as np

def harmonize(value):
	if ( (value == REFUSAL) or (value == DONT_KNOW) or 
		(value ==  NOT_APPLICABLE) or (value== SCHD_NOT_APPLICABLE)):
		return np.nan
	return value

def binarize(value):
	if (value==1 or value==2):
		return 1
	elif (value==3 or value==4  or value==5 or value==6):
		return 0;
	else:
		return 2

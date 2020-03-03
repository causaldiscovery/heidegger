from constants import *
import numpy as np

def harmonize(value):
	if ( (value == NOT_IMPUTED) or (value == NON_SAMPLE) or 
		(value ==  INST_RESPONDENT)):
		return np.nan
	return value

def binarize(value):
	return value
#import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np, os, sys, _pickle as pickle
from Settings import *


def plotWithMasses(uncertainties, quantity, markers):
	barlist = plt.bar(ALL_BLOCK_MASSES_TO_VALIDATE, uncertainties, label='Unseen block masses')
	#barlist[9].set_color('r')
	for marker in markers:
		barlist[int(marker)].set_color('r')
	plt.plot([],[], color='r', label='Seen block masses')
	plt.xlabel('Block Mass')
	plt.ylabel(quantity)
	#plt.title('Trained context is ' + str(identifier))
	plt.legend()

	plt.show()

	#plt.savefig(expert_file_name)
	#plt.close('all')
import numpy as np
from pyBKT.generate import synthetic_data
from pyBKT.generate import random_model, random_model_uni
from pyBKT.fit import EM_fit
from copy import deepcopy
from pyBKT.util import print_dot
import scipy.io as sio
import sys

#A = sio.loadmat('test2Student1.mat')
#B = sio.loadmat('test2Student2.mat')

#S1 = A['Xdot']
#S2 = B['Xdot']

#S = np.concatenate((S1, S2), axis=1)

#get number of students for each

valid_hapcap = 0
valid_comp = 0
num_questions = 23

for i in range(1, 120):
	string = 'HCS/DataMatrixBKTStudent' + str(i) + '.mat'
	try: 
		A = sio.loadmat(string)
		valid_hapcap += 1
	except:
		print ('no valid file')

	string = 'CS/DataMatrixBKTStudent' + str(i) + '.mat'
	try:
		A = sio.loadmat(string)
		valid_comp += 1
	except:
		print ('no valid file')

hapcap_transitions = np.zeros(valid_hapcap)
hapcap_guesses = np.zeros((valid_hapcap, num_questions))
hapcap_slips = np.zeros((valid_hapcap, num_questions))

comp_transitions = np.zeros(valid_comp)
comp_guesses = np.zeros((valid_comp, num_questions))
comp_slips = np.zeros((valid_comp, num_questions))

#put in Hapcap Students
hapcap_count = 0
for d in range(1, 120):
	string = 'HCS/DataMatrixBKTStudent' + str(d) + '.mat'
	try:
		A = sio.loadmat(string)
		S_full = A['DataMatrix']

		num_valid_files = 1

		num_subparts = S_full.shape[0]
		num_resources = 1
		num_fit_initializations = 25
		observation_sequence_lengths = np.full(50, 100, dtype=np.int)

		num_questions = S_full.shape[1]

		truemodel = {}

		truemodel["As"] = np.zeros((num_resources, 2, 2), dtype=np.float_)
		truemodel["As"][0, :, :] = np.transpose([[0.75, 0.25], [0.75, 0.25]])


		truemodel["learns"] = truemodel["As"][:, 1, 0]
		truemodel["forgets"] = truemodel["As"][:, 0, 1]

		truemodel["pi_0"] = np.array([[0.9], [0.1]]) #TODO: one prior per resource? does this array needs to be col?
		truemodel["prior"] = 0.1

		truemodel["guesses"] = np.full(num_subparts, 0.05, dtype=np.float_)
		truemodel["slips"] = np.full(num_subparts, 0.25, dtype=np.float_)

		truemodel["resources"] = np.random.randint(1, high = num_resources+1, size = sum(observation_sequence_lengths))

		starts = np.zeros((num_valid_files), dtype=np.int)
		lengths = np.zeros((num_valid_files), dtype=np.int)

		starts[0] = 1
		lengths[0] = S_full.shape[1]
		resources = np.ones((num_questions), dtype=np.int)

		data = {}
		#datastruct["stateseqs"] = {}
		data["data"] = S_full
		data["starts"] = starts
		data["lengths"] = lengths
		data["resources"] = resources

		best_likelihood = float("-inf")
		for i in range(num_fit_initializations):
		    print_dot.print_dot(i, num_fit_initializations)
		    fitmodel = random_model.random_model(num_resources, num_subparts)
		    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
		    if (log_likelihoods[-1] > best_likelihood):
		        best_likelihood = log_likelihoods[-1]
		        best_model = fitmodel


		hapcap_transitions[hapcap_count] = best_model['As'][0][0][0]
		hapcap_guesses[hapcap_count, :] = best_model['guesses']
		hapcap_slips[hapcap_count, :] = best_model['slips']
		print ('Done with Hapcap Model:')
		print (d)
		print (best_model['As'][0][0][0])
		print (best_model['guesses'])
		print (best_model['slips'])

		hapcap_count += 1

	except:
		print ('No file found')

#put in Computer Students
comp_count = 0
for d in range(1, 120):
	string = 'CS/DataMatrixBKTStudent' + str(d) + '.mat'
	try:
		A = sio.loadmat(string)
		S_full = A['DataMatrix']

		num_valid_files = 1

		num_subparts = S_full.shape[0]
		num_resources = 1
		num_fit_initializations = 25
		observation_sequence_lengths = np.full(50, 100, dtype=np.int)

		num_questions = S_full.shape[1]

		truemodel = {}

		truemodel["As"] = np.zeros((num_resources, 2, 2), dtype=np.float_)
		truemodel["As"][0, :, :] = np.transpose([[0.75, 0.25], [0.75, 0.25]])


		truemodel["learns"] = truemodel["As"][:, 1, 0]
		truemodel["forgets"] = truemodel["As"][:, 0, 1]

		truemodel["pi_0"] = np.array([[0.9], [0.1]]) #TODO: one prior per resource? does this array needs to be col?
		truemodel["prior"] = 0.1

		truemodel["guesses"] = np.full(num_subparts, 0.05, dtype=np.float_)
		truemodel["slips"] = np.full(num_subparts, 0.25, dtype=np.float_)

		truemodel["resources"] = np.random.randint(1, high = num_resources+1, size = sum(observation_sequence_lengths))

		starts = np.zeros((num_valid_files), dtype=np.int)
		lengths = np.zeros((num_valid_files), dtype=np.int)

		starts[0] = 1
		lengths[0] = S_full.shape[1]
		resources = np.ones((num_questions), dtype=np.int)

		data = {}
		#datastruct["stateseqs"] = {}
		data["data"] = S_full
		data["starts"] = starts
		data["lengths"] = lengths
		data["resources"] = resources

		best_likelihood = float("-inf")
		for i in range(num_fit_initializations):
		    print_dot.print_dot(i, num_fit_initializations)
		    fitmodel = random_model.random_model(num_resources, num_subparts)
		    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
		    if (log_likelihoods[-1] > best_likelihood):
		        best_likelihood = log_likelihoods[-1]
		        best_model = fitmodel

		comp_transitions[comp_count] = best_model['As'][0][0][0]
		comp_guesses[comp_count, :] = best_model['guesses']
		comp_slips[comp_count, :] = best_model['slips']
		print ('Done with Computer Model:')
		print (d)
		print (best_model['As'][0][0][0])
		print (best_model['guesses'])
		print (best_model['slips'])

		comp_count += 1

	except:
		print ('No file found')


#print out statistics of the results

print ('Learn Transition Hapcaps')
print ('Mean:')
print (np.mean(hapcap_transitions, axis=0))
print ('Variance:')
print (np.var(hapcap_transitions, axis=0))

print ('Learn Transition Computer')
print ('Mean:')
print (np.mean(comp_transitions, axis=0))
print ('Variance:')
print (np.var(comp_transitions, axis=0))

print ('Avg Guess per Question')
print ('Hapcap')
print (np.mean(hapcap_guesses, axis=0))
print ('Comp')
print (np.mean(comp_guesses, axis=0))

print ('Avg Slip per Question')
print ('Hapcap')
print (np.mean(hapcap_slips, axis=0))
print ('Comp')
print (np.mean(comp_slips, axis=0))



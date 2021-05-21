import sys
import os
import numpy as np
sys.path.append(os.path.abspath("/Users/surajmenon/Desktop/haptic_knowledge_tracing/pyBKT-master/source-py"))
from pyBKT.generate import synthetic_data
from pyBKT.generate import random_model, random_model_uni
from pyBKT.fit import EM_fit
from copy import deepcopy
from pyBKT.util import print_dot
import scipy.io as sio
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import copy

#A = sio.loadmat('test2Student1.mat')
#B = sio.loadmat('test2Student2.mat')

#S1 = A['Xdot']
#S2 = B['Xdot']

#S = np.concatenate((S1, S2), axis=1)

train_model = False

train_perc = .8
test_perc = .2

A = sio.loadmat('HCS/DataMatrixBKTStudent24.mat')
S_full = A['DataMatrix']
S_lengths = []
S_lengths.append(S_full.shape[1])

#put in Hapcap Students
for i in range(25, 120):
	string = 'HCS/DataMatrixBKTStudent' + str(i) + '.mat'
	try:
		A = sio.loadmat(string)
		S = A['DataMatrix']

		S_full = np.concatenate((S_full, S), axis=1)

		S_lengths.append(S.shape[1])

	except:
		print ('No file found')

hapcap_num_students = len(S_lengths)
hapcap_num_samples = np.sum(np.asarray(S_lengths), axis=0)

#put in Computer Students
for i in range(1, 120):
	string = 'CS/DataMatrixBKTStudent' + str(i) + '.mat'
	try:
		A = sio.loadmat(string)
		S = A['DataMatrix']

		S_full = np.concatenate((S_full, S), axis=1)

		S_lengths.append(S.shape[1])

	except:
		print ('No file found')

#swap polarity of data
for i in range(S_full.shape[0]):
	for j in range(S_full.shape[1]):
		if (S_full[i, j] == 1):
			S_full[i, j] = 2
		elif (S_full[i, j] == 2):
			S_full[i, j] = 1

computer_num_students = (len(S_lengths) - hapcap_num_students)
computer_num_samples = np.sum(np.asarray(S_lengths), axis=0) - hapcap_num_samples

#think about train/test split
hapcap_train_students = int(hapcap_num_students*train_perc)
hapcap_test_students = (hapcap_num_students-hapcap_train_students)

computer_train_students = int(computer_num_students*train_perc)
computer_test_students = (computer_num_students-computer_train_students)

hapcap_train_range = 0
for l in range(hapcap_train_students):
	student_samples = S_lengths[l]
	hapcap_train_range += student_samples

computer_train_range = 0
for l in range(computer_train_students):
	student_samples = S_lengths[l+hapcap_num_students]
	computer_train_range += student_samples

computer_train_range_S = computer_train_range + hapcap_num_samples

S_train_hapcap = S_full[:, :hapcap_train_range]
S_test_hapcap = S_full[:, hapcap_train_range:hapcap_num_samples]
S_train_computer = S_full[:, hapcap_num_samples:computer_train_range_S]
S_test_computer = S_full[:, computer_train_range_S:]

S_train = np.concatenate((S_train_hapcap, S_train_computer), axis=1)
resource_1_end = S_train_hapcap.shape[1]

S_squashed = copy.deepcopy(np.sum(S_full, axis=0))
S_squashed.astype(int)

 #since 2 means it was wrong
# for a in range(S_squashed.shape[0]):
# 	if (S_squashed[a] == 2):
# 		S_squashed[a] = 0

#Have 0 be wrong, 1 be correct
S_squashed[:] -= 1

S_squashed_hc = S_squashed[:hapcap_num_samples]
S_squashed_comp = S_squashed[hapcap_num_samples:]

#S_test_hapcap_squashed = np.sum(S_test_hapcap, axis=0).copy()
#S_test_computer_squashed = np.sum(S_test_computer, axis=0).copy()

S_test_hapcap_squashed = S_squashed[hapcap_train_range:hapcap_num_samples]
S_test_computer_squashed = S_squashed[computer_train_range_S:]

#we will run predictions on the last 25 questions
end_offset = 25
next_index = S_lengths[hapcap_train_students]
index = next_index-end_offset
S_test_hapcap_ends = S_test_hapcap[:, index:next_index]

for l in range(1, hapcap_test_students):
	next_index += S_lengths[l+hapcap_train_students]
	index = next_index-end_offset
	S_test_hapcap_end = S_test_hapcap[:, index:next_index]
	S_test_hapcap_ends = np.concatenate((S_test_hapcap_ends, S_test_hapcap_end), axis=1)

next_index = S_lengths[hapcap_num_students+computer_train_students]
index = next_index-end_offset
S_test_computer_ends = S_test_computer[:, index:next_index]

for l in range(1, computer_test_students):
	next_index += S_lengths[l+hapcap_num_students+computer_train_students]
	index = next_index-end_offset
	S_test_computer_end = S_test_computer[:, index:next_index]
	S_test_computer_ends = np.concatenate((S_test_computer_ends, S_test_computer_end), axis=1)

#create the running sums for the design matrix
S_runsum_hc = np.zeros(hapcap_num_samples)
S_runsum_comp = np.zeros(computer_num_samples)

running_sum = 0
index = 0
for l in range(hapcap_num_students):
	student_samples = S_lengths[l]
	next_index = (index+student_samples)
	running_sum = 0
	for m in range(index, next_index):
		running_sum = running_sum+S_squashed_hc[m]
		S_runsum_hc[m] = running_sum

	index = next_index


running_sum = 0
index = 0
for l in range(computer_num_students):
	student_samples = S_lengths[l+hapcap_num_students]
	next_index = (index+student_samples)
	running_sum = 0
	for m in range(index, next_index):
		running_sum = running_sum+S_squashed_comp[m]
		S_runsum_comp[m] = running_sum

	index = next_index

X_hapcap = np.zeros((hapcap_num_samples, 2))
y_hapcap = np.zeros(hapcap_num_samples)
X_computer = np.zeros((computer_num_samples, 2))
y_computer = np.zeros(computer_num_samples)

#Make Logistic Regression Design Matrix and result vector
current_index = 0
for l in range(hapcap_num_students):
	fill_values = S_lengths[l]
	time_values = np.arange(0, fill_values, 1)
	next_index = current_index+fill_values
	if (current_index != 0):
		X_hapcap[current_index:next_index, 0] = S_runsum_hc[(current_index-1):(next_index-1)]
		y_hapcap[current_index:next_index] = S_squashed_hc[current_index:next_index]
	else:
		X_hapcap[0, 0] = 0
		X_hapcap[current_index+1:next_index, 0] = S_runsum_hc[current_index:(next_index-1)]
		y_hapcap[current_index:next_index] = S_squashed_hc[current_index:next_index]

	X_hapcap[current_index:next_index, 1] = copy.deepcopy(time_values[:])
	current_index = next_index

current_index = 0
for l in range(computer_num_students):
	fill_values = S_lengths[l+hapcap_num_students]
	time_values = np.arange(0, fill_values, 1)
	next_index = current_index+fill_values
	if (current_index != 0):
		X_computer[current_index:next_index, 0] = S_runsum_comp[(current_index-1):(next_index-1)]
		y_computer[current_index:next_index] = S_squashed_comp[current_index:next_index]
	else:
		X_computer[0, 0] = 0
		X_computer[(current_index+1):next_index, 0] = S_runsum_comp[current_index:(next_index-1)]
		y_computer[current_index:next_index] = S_squashed_comp[current_index:next_index]

	X_computer[current_index:next_index, 1] = copy.deepcopy(time_values[:])
	current_index = next_index

print ('Data Test')
print (S_full.shape)
print (X_hapcap.shape)
print (X_computer.shape)

#split X and y into train/test
X_train_hapcap = X_hapcap[:hapcap_train_range, :]
X_test_hapcap = X_hapcap[hapcap_train_range:, :]
X_train_computer = X_computer[:computer_train_range, :]
X_test_computer = X_computer[computer_train_range:, :]

y_train_hapcap = y_hapcap[:hapcap_train_range]
y_test_hapcap = y_hapcap[hapcap_train_range:]
y_train_computer = y_computer[:computer_train_range]
y_test_computer = y_computer[computer_train_range:]

#Ok, now fit a Logistic Regression Model
logreg_hc = LogisticRegression().fit(X_train_hapcap, y_train_hapcap)
logreg_comp = LogisticRegression().fit(X_train_computer, y_train_computer)

pred_hc = logreg_hc.predict(X_test_hapcap[:, :])
predprob_hc = logreg_hc.predict_proba(X_test_hapcap[:, :])

pred_comp = logreg_comp.predict(X_test_computer[:, :])
predprob_comp = logreg_comp.predict_proba(X_test_computer[:, :])

#Calculate the Loss, Accuracy, and Confusion Matrix
loss_hc = np.sum(np.abs(y_test_hapcap[:] - predprob_hc[:, 1]), axis=0)
loss_comp = np.sum(np.abs(y_test_computer[:] - predprob_comp[:, 1]), axis=0)

#np.set_printoptions(threshold=sys.maxsize)
#print ('Test LogReg Pred')
#print (predprob_hc[:, 1])

print('Loss HC')
print (loss_hc)
print('Loss Comp')
print (loss_comp)

incorrect_hc = np.sum(np.abs(y_test_hapcap[:] - pred_hc[:]), axis=0)
incorrect_comp = np.sum(np.abs(y_test_computer[:] - pred_comp[:]), axis=0)

acc_hc = ((y_test_hapcap.shape[0]-incorrect_hc)/y_test_hapcap.shape[0])
acc_comp = ((y_test_computer.shape[0]-incorrect_comp)/y_test_computer.shape[0])

print ('Accuracy HC')
print (acc_hc)
print ('Accuracy Comp')
print (acc_comp)

cm_logreg = confusion_matrix(y_test_hapcap, pred_hc)
f1_logreg = f1_score(y_test_hapcap, pred_hc)

#plot1 HC and decision boundary
plt.figure(1)
plt.scatter(X_test_hapcap[:, 0], X_test_hapcap[:, 1], s=.1)

theta_hc = logreg_hc.coef_[0, :]
intercept_hc = logreg_hc.intercept_

y_hc = -(theta_hc[0]/theta_hc[1])*X_test_hapcap[:, 0] - (intercept_hc/theta_hc[1])
plt.plot(X_test_hapcap[:3000, 0], y_hc[:3000], color='r')
plt.ylim(-50, 2000)
plt.xlabel('Time')
plt.ylabel('Score (t-1)')
plt.title('Hapcap LogReg Decision Boundary')

#Add plot save here for LogReg Boundary if we need it
#plt.show()

#exit()

num_valid_students = (hapcap_train_students+computer_train_students)

num_subparts = S_train.shape[0]
num_resources = 1
num_fit_initializations = 25
observation_sequence_lengths = np.full(50, 100, dtype=np.int)

#num_questions_1 = S1.shape[1]
#num_questions_2 = S2.shape[1]
num_questions = S_full.shape[1]
num_hapcap_train_questions = S_train_hapcap.shape[1]
num_computer_train_questions = S_train_computer.shape[1]
num_train_questions = (num_hapcap_train_questions + num_computer_train_questions)

#Let us run the hapcap model and the computer model separately
#First the hapcap model

if (train_model):

	starts_hapcap = np.zeros((hapcap_train_students), dtype=np.int)
	lengths_hapcap = np.zeros((hapcap_train_students), dtype=np.int)

	running_sum = 1
	for i in range(hapcap_train_students):
		if (i == 0):
			starts_hapcap[i] = 1
		else:
			running_sum += S_lengths[i-1]
			starts_hapcap[i] = running_sum

		lengths_hapcap[i] = S_lengths[i]


	resources_hapcap = np.ones((num_hapcap_train_questions), dtype=np.int)
	#resources[resource_1_end:] += 1

	data = {}
	#datastruct["stateseqs"] = {}
	data["data"] = S_train_hapcap
	data["starts"] = starts_hapcap
	data["lengths"] = lengths_hapcap
	data["resources"] = resources_hapcap

	best_likelihood = float("-inf")
	for i in range(num_fit_initializations):
	    print_dot.print_dot(i, num_fit_initializations)
	    fitmodel = random_model.random_model(num_resources, num_subparts)
	    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
	    if (log_likelihoods[-1] > best_likelihood):
	        best_likelihood = log_likelihoods[-1]
	        best_model = fitmodel

	print('Printing Model Hapcap')
	print (best_model['As'])
	print (best_model['learns'])
	print (best_model['forgets'])
	print (best_model['guesses'])
	print (best_model['slips'])
	print (best_model['prior'])
	print (best_model['emissions'])
	print (best_model['pi_0'])

	#copy model
	hapcap_model = copy.deepcopy(best_model)

	#Now fit the computer model
	starts_computer = np.zeros((computer_train_students), dtype=np.int)
	lengths_computer = np.zeros((computer_train_students), dtype=np.int)

	running_sum = 1
	for i in range(computer_train_students):
		if (i == 0):
			starts_computer[i] = 1
		else:
			running_sum += S_lengths[(i-1)+hapcap_num_students]
			starts_computer[i] = running_sum

		lengths_computer[i] = S_lengths[i+hapcap_num_students]

	resources_computer = np.ones((num_computer_train_questions), dtype=np.int)

	data = {}
	#datastruct["stateseqs"] = {}
	data["data"] = S_train_computer
	data["starts"] = starts_computer
	data["lengths"] = lengths_computer
	data["resources"] = resources_computer

	best_likelihood = float("-inf")
	for i in range(num_fit_initializations):
	    print_dot.print_dot(i, num_fit_initializations)
	    fitmodel = random_model.random_model(num_resources, num_subparts)
	    (fitmodel, log_likelihoods) = EM_fit.EM_fit(fitmodel, data)
	    if (log_likelihoods[-1] > best_likelihood):
	        best_likelihood = log_likelihoods[-1]
	        best_model = fitmodel


	print('Printing Model Computer')
	print (best_model['As'])
	print (best_model['learns'])
	print (best_model['forgets'])
	print (best_model['guesses'])
	print (best_model['slips'])
	print (best_model['prior'])
	print (best_model['emissions'])
	print (best_model['pi_0'])

	#copy model
	computer_model = copy.deepcopy(best_model)
else:
	#fill in model from previous run
	hapcap_model = {}
	computer_model = {}

	#hapcap_model['As'] = np.array([[[0.99310107, 0.005],[0.00689893, 0.995]]])
	hapcap_model['As'] = np.array([[[0.995, 0.00556],[0.005, 0.99444]]])
	hapcap_model['pi_0'] = np.array([[0.62067546],[0.37932454]])
	hapcap_model['prior'] = 0.37932454
	#hapcap_model['prior'] = 0.237
	#hapcap_model['learns'] = np.array([0.00689893])
	hapcap_model['learns'] = np.array([0.00556])
	hapcap_model['forgets'] = np.array([0.005])
	hapcap_model['guesses'] = 	np.array([0.04232069, 0.0677057,  0.07775427, 0.05949924, 0.11337535, 0.06430935,
 										  0.24744096, 0.4,        0.05441852, 0.14066514, 0.05316346, 0.0669241,
 										  0.01881613, 0.0329436,  0.03353636, 0.24343568, 0.07302821, 0.25987954,
 										  0.30119346, 0.11039176, 0.10656615, 0.32065011, 0.31371927])
	hapcap_model['slips'] = 	np.array([3.02423089e-01, 2.81636616e-01, 1.54309966e-01, 1.33485057e-01,
										 2.52568298e-01, 1.74585157e-01, 1.26711222e-01, 4.45034929e-02,
										 3.05185138e-01, 1.59942455e-01, 4.00000000e-01, 2.91355318e-01,
										 4.00000000e-01, 3.42913893e-01, 1.97175042e-01, 8.13071619e-02,
										 1.05789207e-01, 5.57877602e-02, 2.99515857e-02, 1.45305150e-04,
										 5.84153424e-02, 5.74180498e-02, 1.37042995e-01])


	#computer_model['As'] = np.array([[[0.99457688, 0.005],[0.00542312, 0.995]]])
	computer_model['As'] = np.array([[[0.995, 0.0007523],[0.005, 0.9992477]]])
	computer_model['pi_0'] = np.array([[0.56781309],[0.43218691]])
	computer_model['prior'] = 0.4321869114096283
	#computer_model['prior'] = 0.43
	#computer_model['learns'] = np.array([0.00542312])
	computer_model['learns'] = np.array([0.0007523])
	computer_model['forgets'] = np.array([0.005])
	computer_model['guesses'] = np.array([0.04976697, 0.05495507, 0.06685592, 0.0619098,  0.07460955, 0.0924545,
 										0.31340662, 0.4,        0.05952836, 0.18114929, 0.05357501, 0.06451396,
 										0.0195442,  0.02404027, 0.07196168, 0.24346844, 0.09157146, 0.19233014,
 										0.25661663, 0.08232518, 0.08700777, 0.28230465, 0.23147651])
	computer_model['slips'] = 	np.array([0.4,      0.18395656, 0.13548965, 0.26687372, 0.15770516, 0.34860897,
 										0.20025422, 0.10202852, 0.18865225, 0.2006384,  0.25746249, 0.24013677,
 										0.21175613, 0.4,        0.17376782, 0.30500425, 0.26532981, 0.07427287,
 										0.25001907, 0.0879229,  0.17503192, 0.0817992, 0.14606756])

#model is fit, now lets try to evaluate it on the test set
hapcap_preds = np.zeros((S_test_hapcap_squashed.shape[0]))
computer_preds = np.zeros((S_test_computer_squashed.shape[0]))

hapcap_end_preds = np.zeros((2, end_offset*hapcap_test_students))
computer_end_preds = np.zeros((2, end_offset*computer_test_students))

#lets now do the predictions for hapcap students
index = 0
for i in range(hapcap_test_students):
	length = S_lengths[i+hapcap_train_students]
	next_index = index+length
	data = S_test_hapcap_squashed[index:next_index]

	alphas = np.zeros((length, 2))
	#params
	prior = hapcap_model['prior']
	learns = hapcap_model['learns'][0]
	forgets = hapcap_model['forgets'][0]
	guess = np.mean(hapcap_model['guesses'], axis=0)
	slip = np.mean(hapcap_model['slips'], axis=0)

	#fill in the alphas for this students series
	for j in range(length):
		if (j == 0):
			alphas[j, 0] = (1-prior)
			alphas[j, 1] = (prior)
		else:
			alphas[j, 0] = (alphas[(j-1), 0]*(1-learns)) + (alphas[(j-1), 1]*(forgets))
			alphas[j, 1] = (alphas[(j-1), 0]*(learns)) + (alphas[(j-1), 1]*(1-forgets))

	alphas_preds = alphas[:, 0]*guess + alphas[:, 1]*(1-slip)
	hapcap_preds[index:next_index] = copy.deepcopy(alphas_preds[:])

	#run predictions for the final offset
	alpha_end_preds = np.zeros((end_offset))
	questions = np.zeros((end_offset))

	for j in range(end_offset):
		for k in range(S_test_hapcap_ends.shape[0]):
			if (S_test_hapcap_ends[k, j] > 0):
				questions[j] = k

	for j in range(end_offset):
		offset = (end_offset-j)
		qi = questions[j].astype(int)
		alpha_end_preds[j] = alphas[(length-offset), 0]*hapcap_model['guesses'][qi] + alphas[(length-offset), 1]*(1-hapcap_model['slips'][qi])

	data_length = data.shape[0]
	hapcap_end_preds[0, (i*end_offset):(i*end_offset + end_offset)] = copy.deepcopy(data[(data_length-end_offset):])
	hapcap_end_preds[1, (i*end_offset):(i*end_offset + end_offset)] = copy.deepcopy(alpha_end_preds[:])

	index = next_index

index = 0
for i in range(computer_test_students):
	length = S_lengths[i+hapcap_num_students+computer_train_students]
	next_index = index+length
	data = S_test_computer_squashed[index:next_index]

	alphas = np.zeros((length, 2))

	#params
	prior = computer_model['prior']
	learns = computer_model['learns'][0]
	forgets = computer_model['forgets'][0]
	guess = np.mean(computer_model['guesses'], axis=0)
	slip = np.mean(computer_model['slips'], axis=0)

	#fill in the alphas for this students series
	for j in range(length):
		if (j == 0):
			alphas[j, 0] = (1-prior)
			alphas[j, 1] = (prior)
		else:
			alphas[j, 0] = (alphas[(j-1), 0]*(1-learns)) + (alphas[(j-1), 1]*(forgets))
			alphas[j, 1] = (alphas[(j-1), 0]*(learns)) + (alphas[(j-1), 1]*(1-forgets))

	alphas_preds = alphas[:, 0]*guess + alphas[:, 1]*(1-slip)
	computer_preds[index:next_index] = copy.deepcopy(alphas_preds[:])

	#run predictions for the final offset
	alpha_end_preds = np.zeros((end_offset))
	questions = np.zeros((end_offset))

	for j in range(end_offset):
		for k in range(S_test_computer_ends.shape[0]):
			if (S_test_computer_ends[k, j] > 0):
				questions[j] = k

	for j in range(end_offset):
		offset = (end_offset-j)
		qi = questions[j].astype(int)
		alpha_end_preds[j] = alphas[(length-offset), 0]*computer_model['guesses'][qi] + alphas[(length-offset), 1]*(1-computer_model['slips'][qi])

	data_length = data.shape[0]
	computer_end_preds[0, (i*end_offset):(i*end_offset + end_offset)] = copy.deepcopy(data[(data_length-end_offset):])
	computer_end_preds[1, (i*end_offset):(i*end_offset + end_offset)] = copy.deepcopy(alpha_end_preds[:])

	index = next_index

#now calculate predicted loss with BKT and accuracy
loss_hc_bkt = np.sum(np.abs(y_test_hapcap[:] - hapcap_preds[:]), axis=0)
loss_comp_bkt = np.sum(np.abs(y_test_computer[:] - computer_preds[:]), axis=0)

hapcap_pred_int = (hapcap_preds > .5).astype(int)
computer_pred_int = (computer_preds > .5).astype(int)

hapcap_incorr_bkt = np.sum(np.abs(y_test_hapcap[:] - hapcap_pred_int[:]), axis=0)
computer_incorr_bkt = np.sum(np.abs(y_test_computer[:] - computer_pred_int[:]), axis=0)

acc_hc_bkt = ((y_test_hapcap.shape[0]-hapcap_incorr_bkt)/y_test_hapcap.shape[0])
acc_comp_bkt = ((y_test_computer.shape[0]-computer_incorr_bkt)/y_test_computer.shape[0])

#calculate ends loss
loss_hc_ends = np.sum(np.abs(hapcap_end_preds[0, :] - hapcap_end_preds[1, :]), axis=0)
loss_comp_ends = np.sum(np.abs(computer_end_preds[0, :] - computer_end_preds[1, :]), axis=0)

hapcap_pred_int_ends = (hapcap_end_preds[1, :] > .5)
computer_pred_int_ends = (computer_end_preds[1, :] > .5)

hapcap_incorr_bkt_ends = np.sum(np.abs(hapcap_end_preds[0, :] - hapcap_pred_int_ends[:]), axis=0)
computer_incorr_bkt_ends = np.sum(np.abs(computer_end_preds[0, :] - computer_pred_int_ends[:]), axis=0)

acc_hc_bkt_ends = ((hapcap_end_preds.shape[1]-hapcap_incorr_bkt_ends)/hapcap_end_preds.shape[1])
acc_comp_bkt_ends = ((computer_end_preds.shape[1]-computer_incorr_bkt_ends)/computer_end_preds.shape[1])

print ('Hapcap Preds')
print (hapcap_preds)
print (hapcap_preds.shape)

print ('Computer Preds')
print (computer_preds)
print (computer_preds.shape)

print('Loss HC BKT')
print (loss_hc_bkt)
print('Loss Comp BKT')
print (loss_comp_bkt)

print ('Accuracy HC')
print (acc_hc_bkt)
print ('Accuracy Comp')
print (acc_comp_bkt)

print ('Loss HC Ends')
print (loss_hc_ends)

print ('Loss Comp Ends')
print (loss_comp_ends)

print ('Accuracy HC Ends')
print (acc_hc_bkt_ends)

print ('Accuracy Comp Ends')
print (acc_comp_bkt_ends)

#Let us now compare the confusion matrices from the two model for hapcaps

cm_bkt = confusion_matrix(y_test_hapcap, hapcap_pred_int)
f1_bkt = f1_score(y_test_hapcap, hapcap_pred_int)

cm_bkt_ends = confusion_matrix(hapcap_end_preds[0, :], hapcap_pred_int_ends[:])
f1_bkt_ends = f1_score(hapcap_end_preds[0, :], hapcap_pred_int_ends[:])

#Confusion Matrices
print ('Confusion Matrix LogReg')
print (cm_logreg)

print ('Confusion Matrix BKT')
print (cm_bkt)

#F1 Scores
print ('F1 Score LogReg')
print (f1_logreg)

print ('F1 Score BKT')
print (f1_bkt)

#ends
print ('Confusion Matrix Hapcap Ends')
print (cm_bkt_ends)

print ('F1 Score Hapcap Ends')
print (f1_bkt_ends)

#Now lets see if we can do some testing with random data generation

#Use the synthetic data library
observation_sequence_lengths = np.zeros((hapcap_test_students))
for i in range(hapcap_test_students):
	observation_sequence_lengths[i] = S_lengths[i+hapcap_train_students]

total_length = np.sum(observation_sequence_lengths, axis=0)
total_length = int(total_length)
resources_data = np.ones((total_length))
hapcap_model['resources'] = copy.deepcopy(resources_data)

data_fake = synthetic_data.synthetic_data(hapcap_model, observation_sequence_lengths, resources_data)

print ('Test Synthetic Data')
print (data_fake['data'].shape)

data_fake_mean = ((np.mean(data_fake['data'], axis=0)-1) > .5).astype(int)
data_mismatched = np.sum(np.abs(S_test_hapcap_squashed - data_fake_mean), axis=0)

data_accuracy = (S_test_hapcap_squashed.shape[0] - data_mismatched)/S_test_hapcap_squashed.shape[0]

print ('Data Mean Accuracy')
print (data_accuracy)

#Get average
data_fake_accuracies = np.zeros((data_fake['data'].shape[0]))

for i in range(data_fake['data'].shape[0]):
	data_f = (data_fake['data'][i, :] - 1)
	data_mm = np.sum(np.abs(S_test_hapcap_squashed - data_f), axis=0)

	data_fake_accuracies[i] = (S_test_hapcap_squashed.shape[0] - data_mm)/S_test_hapcap_squashed.shape[0]

print ('Max Accuracy')
print (np.max(data_fake_accuracies, axis=0))

print ('Avg Accuracy')
print (np.mean(data_fake_accuracies, axis=0))

cm_data = confusion_matrix(S_test_hapcap_squashed[:], data_fake_mean[:])
f1_data = f1_score(S_test_hapcap_squashed[:], data_fake_mean[:])

print ('Data Confusion Matrix')
print (cm_data)

print ('Data F1 Score')
print (f1_data)

#Get Expected Value Accuracy from Probabilities
ev_hapcap = np.sum((1-np.abs(y_test_hapcap[:] - hapcap_preds[:])), axis=0)/(y_test_hapcap.shape[0])

print ('Expected Value Accuracy')
print (ev_hapcap)

#Calculate an Average Parameter Error

#get max test length
max_length = 0
max_index = 0
for i in range(hapcap_test_students):
	if (S_lengths[hapcap_train_students+i] > max_length):
		max_length = S_lengths[hapcap_train_students+i]
		max_index = i

p_error_sums = np.zeros((max_length))
p_error_totals = np.zeros((max_length))

hapcap_alphas = np.zeros((max_length))
index = 0
for i in range(hapcap_test_students):
	length = S_lengths[hapcap_train_students+i]
	for j in range(length):
		p_error_sums[j] += y_test_hapcap[index+j]
		p_error_totals[j] += 1

	if (i == max_index):
		hapcap_alphas = hapcap_preds[index:(index+length)]

	index += length

avg_param_error = np.sum(np.abs((p_error_sums[:]/p_error_totals[:]) - hapcap_alphas[:]), axis=0)/(hapcap_alphas.shape[0])

print ('Average Parameter Error')
print (avg_param_error)

avg_param_scale = np.maximum(avg_param_error, (1-avg_param_error))
avg_param_scaled = (avg_param_error/avg_param_scale)

print ('Scaled Average Parameter Error')
print (avg_param_scaled)

hapcap_guess = np.mean(hapcap_model['guesses'], axis=0)
hapcap_guess_var = np.var(hapcap_model['guesses'], axis=0)
hapcap_slip = np.mean(hapcap_model['slips'], axis=0)
hapcap_slip_var = np.var(hapcap_model['slips'], axis=0)
computer_guess = np.mean(computer_model['guesses'], axis=0)
computer_guess_var = np.var(computer_model['guesses'], axis=0) 
computer_slip = np.mean(computer_model['slips'], axis=0)
computer_slip_var = np.var(computer_model['slips'], axis=0)

print ('Mean and Var Hapcap Guess')
print (hapcap_guess)
print (hapcap_guess_var)

print ('Mean and Var Hapcap Slip')
print (hapcap_slip)
print (hapcap_slip_var)

print ('Mean and Var Computer Guess')
print (computer_guess)
print (computer_guess_var)

print ('Mean and Var Computer Slip')
print (computer_slip)
print (computer_slip_var)

#Running the Forward Algorithm to compute the Likelihood of the Test Data

#model is fit, now lets try to evaluate it on the test set
hapcap_likelihoods = np.zeros((hapcap_test_students))
computer_likelihoods = np.zeros((computer_test_students))

hapcap_preds = np.zeros((S_test_hapcap_squashed.shape[0]))
computer_preds = np.zeros((S_test_computer_squashed.shape[0]))

#lets now do the predictions for hapcap students
index = 0
for i in range(hapcap_test_students):
	length = S_lengths[i+hapcap_train_students]
	next_index = index+length
	data = S_test_hapcap_squashed[index:next_index]

	alphas = np.zeros((length, 2))
	alphas_preds = np.zeros((length, 2))
	alphas_final_preds = np.zeros((length))

	#params
	prior = hapcap_model['prior']
	learns = hapcap_model['learns'][0]
	forgets = hapcap_model['forgets'][0]
	guesses = hapcap_model['guesses']
	slips = hapcap_model['slips']

	questions = np.zeros((length))

	for j in range(length):
		for k in range(S_test_hapcap.shape[0]):
			if (S_test_hapcap[k, j] > 0):
				questions[j] = k

	#fill in the alphas for this students series
	for j in range(length):
		if (j == 0):
			alphas[j, 0] = (1-prior)
			alphas[j, 1] = (prior)

			alphas_preds[j, 0] = (1-prior)
			alphas_preds[j, 1] = (prior)

			alphas_final_preds[j] = (prior)
		else:
			qi = questions[j].astype(int)
			if (data[j] == 0):
				alphas[j, 0] = (alphas[(j-1), 0]*(1-learns)*(1-guesses[qi])) + (alphas[(j-1), 1]*(forgets)*(1-guesses[qi]))
				alphas[j, 1] = (alphas[(j-1), 0]*(learns)*(slips[qi])) + (alphas[(j-1), 1]*(1-forgets)*(slips[qi]))
			else:
				alphas[j, 0] = (alphas[(j-1), 0]*(1-learns)*(guesses[qi])) + (alphas[(j-1), 1]*(forgets)*(guesses[qi]))
				alphas[j, 1] = (alphas[(j-1), 0]*(learns)*(1-slips[qi])) + (alphas[(j-1), 1]*(1-forgets)*(1-slips[qi]))

			alphas_preds[j, 0] = (alphas_preds[(j-1), 0]*(1-learns)) + (alphas_preds[(j-1), 1]*(forgets))
			alphas_preds[j, 1] = (alphas_preds[(j-1), 0]*(learns)) + (alphas_preds[(j-1), 1]*(1-forgets))
			alphas_final_preds[j] = alphas_preds[j, 0]*guesses[qi] + alphas_preds[j, 1]*(1-slips[qi])

	alphas_ll = np.sum(alphas[(length-1), :], axis=0)
	hapcap_likelihoods[i] = copy.deepcopy(alphas_ll)
	hapcap_preds[index:next_index] = copy.deepcopy(alphas_final_preds[:])

	index = next_index

#lets now do the predictions for computer students
index = 0
for i in range(computer_test_students):
	length = S_lengths[i+hapcap_num_students+computer_train_students]
	next_index = index+length
	data = S_test_computer_squashed[index:next_index]

	alphas = np.zeros((length, 2))
	alphas_preds = np.zeros((length, 2))
	alphas_final_preds = np.zeros((length))

	#params
	prior = computer_model['prior']
	learns = computer_model['learns'][0]
	forgets = computer_model['forgets'][0]
	guesses = computer_model['guesses']
	slips = computer_model['slips']

	questions = np.zeros((length))

	for j in range(length):
		for k in range(S_test_computer.shape[0]):
			if (S_test_computer[k, j] > 0):
				questions[j] = k

	#fill in the alphas for this students series
	for j in range(length):
		if (j == 0):
			alphas[j, 0] = (1-prior)
			alphas[j, 1] = (prior)

			alphas_preds[j, 0] = (1-prior)
			alphas_preds[j, 1] = (prior)

			alphas_final_preds[j] = (prior)
		else:
			qi = questions[j].astype(int)
			if (data[j] == 0):
				alphas[j, 0] = (alphas[(j-1), 0]*(1-learns)*(1-guesses[qi])) + (alphas[(j-1), 1]*(forgets)*(1-guesses[qi]))
				alphas[j, 1] = (alphas[(j-1), 0]*(learns)*(slips[qi])) + (alphas[(j-1), 1]*(1-forgets)*(slips[qi]))
			else:
				alphas[j, 0] = (alphas[(j-1), 0]*(1-learns)*(guesses[qi])) + (alphas[(j-1), 1]*(forgets)*(guesses[qi]))
				alphas[j, 1] = (alphas[(j-1), 0]*(learns)*(1-slips[qi])) + (alphas[(j-1), 1]*(1-forgets)*(1-slips[qi]))

			alphas_preds[j, 0] = (alphas_preds[(j-1), 0]*(1-learns)) + (alphas_preds[(j-1), 1]*(forgets))
			alphas_preds[j, 1] = (alphas_preds[(j-1), 0]*(learns)) + (alphas_preds[(j-1), 1]*(1-forgets))
			alphas_final_preds[j] = alphas_preds[j, 0]*guesses[qi] + alphas_preds[j, 1]*(1-slips[qi])

	alphas_ll = np.sum(alphas[(length-1), :], axis=0)
	computer_likelihoods[i] = copy.deepcopy(alphas_ll)
	computer_preds[index:next_index] = copy.deepcopy(alphas_final_preds[:])

	index = next_index

print ('Mean Likelihoods')

print ('Mean Hapcap Likelihood')
mean_hapcap_likelihood = np.mean(hapcap_likelihoods, axis=0)
print (mean_hapcap_likelihood)

print ('Mean Computer Likelihood')
mean_computer_likelihood = np.mean(computer_likelihoods, axis=0)
print (mean_computer_likelihood)

print ('Predictions')

print ('Hapcap Preds')
print (hapcap_preds)

print ('Computer Preds')
print (computer_preds)

#Now use these better alphas to capture a better Average Parameter Error

#get max test length
max_length = 0
max_index = 0
for i in range(hapcap_test_students):
	if (S_lengths[hapcap_train_students+i] > max_length):
		max_length = S_lengths[hapcap_train_students+i]
		max_index = i

p_error_sums = np.zeros((max_length))
p_error_totals = np.zeros((max_length))

hapcap_alphas = np.zeros((max_length))
index = 0
for i in range(hapcap_test_students):
	length = S_lengths[hapcap_train_students+i]
	for j in range(length):
		p_error_sums[j] += y_test_hapcap[index+j]
		p_error_totals[j] += 1

	if (i == max_index):
		hapcap_alphas = hapcap_preds[index:(index+length)]

	index += length

hapcap_avg_param_error = np.sum(np.abs((p_error_sums[:]/p_error_totals[:]) - hapcap_alphas[:]), axis=0)/(hapcap_alphas.shape[0])

print ('Better Hapcap Average Parameter Error')
print (hapcap_avg_param_error)

hapcap_avg_param_scale = np.maximum(hapcap_avg_param_error, (1-hapcap_avg_param_error))
hapcap_avg_param_scaled = (hapcap_avg_param_error/hapcap_avg_param_scale)

print ('Better Scaled Average Parameter Error')
print (hapcap_avg_param_scaled)

#get max test length
max_length = 0
max_index = 0
for i in range(computer_test_students):
	if (S_lengths[i+hapcap_num_students+computer_train_students] > max_length):
		max_length = S_lengths[i+hapcap_num_students+computer_train_students]
		max_index = i

p_error_sums = np.zeros((max_length))
p_error_totals = np.zeros((max_length))

computer_alphas = np.zeros((max_length))
index = 0
for i in range(computer_test_students):
	length = S_lengths[i+hapcap_num_students+computer_train_students]
	for j in range(length):
		p_error_sums[j] += y_test_computer[index+j]
		p_error_totals[j] += 1

	if (i == max_index):
		computer_alphas = computer_preds[index:(index+length)]

	index += length

computer_avg_param_error = np.sum(np.abs((p_error_sums[:]/p_error_totals[:]) - computer_alphas[:]), axis=0)/(computer_alphas.shape[0])

print ('Better Computer Average Parameter Error')
print (computer_avg_param_error)

computer_avg_param_scale = np.maximum(computer_avg_param_error, (1-computer_avg_param_error))
computer_avg_param_scaled = (computer_avg_param_error/computer_avg_param_scale)

print ('Better Scaled Average Parameter Error')
print (computer_avg_param_scaled)



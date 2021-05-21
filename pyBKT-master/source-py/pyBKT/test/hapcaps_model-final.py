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
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import copy
import seaborn as sns


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

#Have 0 be wrong, 1 be correct
S_squashed[:] -= 1

S_squashed_hc = S_squashed[:hapcap_num_samples]
S_squashed_comp = S_squashed[hapcap_num_samples:]

print (S_squashed_hc.shape)
print (S_squashed_comp.shape)

print (hapcap_num_students)
print (computer_num_students)
exit()

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

cm_comp_logreg = confusion_matrix(y_test_computer, pred_comp)
f1_comp_logreg = f1_score(y_test_computer, pred_comp)

#plot1 HC and decision boundary
plt.figure(1)
plt.scatter(X_test_hapcap[:, 0], X_test_hapcap[:, 1], s=.1)

theta_hc = logreg_hc.coef_[0, :]
intercept_hc = logreg_hc.intercept_

y_hc = -(theta_hc[0]/theta_hc[1])*X_test_hapcap[:, 0] - (intercept_hc/theta_hc[1])
# plt.plot(X_test_hapcap[:3000, 0], y_hc[:3000], color='r')
# plt.ylim(-50, 2000)
# plt.xlabel('Time')
# plt.ylabel('Score (t-1)')
# plt.title('Hapcap LogReg Decision Boundary')

# #Add plot save here for LogReg Boundary if we need it
# plt.savefig('logreg.png')


num_valid_students = (hapcap_train_students+computer_train_students)

num_subparts = S_train.shape[0]
num_resources = 1
num_fit_initializations = 25
observation_sequence_lengths = np.full(50, 100, dtype=np.int)


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

	hapcap_model['As'] = np.array([[[0.99460754, 0.001], [0.00539246, 0.999]]])
	hapcap_model['pi_0'] = np.array([[0.5810146], [0.4189854]])
	#hapcap_model['prior'] = 0.41898539816814395
	hapcap_model['prior'] = 0.41898539816814395
	hapcap_model['learns'] = np.array([0.00539246])
	#hapcap_model['learns'] = np.array([0.00539246])
	hapcap_model['forgets'] = np.array([0.001])
	hapcap_model['guesses'] = 	np.array([0.03968328, 0.06520551, 0.0781403,  0.06044178, 0.11366427, 0.06430346,
 										0.25020116, 0.45393107, 0.05554176, 0.13477844, 0.05460504, 0.06777157,
 										0.0193162,  0.03521147, 0.03328427, 0.24341709, 0.07249608, 0.2554319,
 										0.30134754, 0.1084103,  0.10333275, 0.32652712, 0.30896418])
	hapcap_model['slips'] = 	np.array([0.36722798, 0.31898363, 0.20615389, 0.30374915, 0.37327547, 0.18645334,
 										0.13924249, 0.08145794, 0.33794405, 0.18154819, 0.5,        0.38398256,
 										0.5,        0.37867858, 0.26950601, 0.09406755, 0.1226853,  0.07751296,
 										0.10634889, 0.040095,   0.15335273, 0.10603025, 0.16524907])


	computer_model['As'] = np.array([[[0.99584343, 0.001], [0.00415657, 0.999]]])
	computer_model['pi_0'] = np.array([[0.44442436], [0.55557564]])
	#computer_model['prior'] = 0.5555756424122067
	computer_model['prior'] = 0.5555756424122067
	computer_model['learns'] = np.array([0.00415657])
	#computer_model['learns'] = np.array([0.00435657])
	computer_model['forgets'] = np.array([0.001])
	# computer_model['guesses'] = 	np.array([0.03968328, 0.06520551, 0.0781403,  0.06044178, 0.11366427, 0.06430346,
 # 										0.25020116, 0.45393107, 0.05554176, 0.13477844, 0.05460504, 0.06777157,
 # 										0.0193162,  0.03521147, 0.03328427, 0.24341709, 0.07249608, 0.2554319,
 # 										0.30134754, 0.1084103,  0.10333275, 0.32652712, 0.30896418])
	# computer_model['slips'] = 	np.array([0.36722798, 0.31898363, 0.20615389, 0.30374915, 0.37327547, 0.18645334,
 # 										0.13924249, 0.08145794, 0.33794405, 0.18154819, 0.5,        0.38398256,
 # 										0.5,        0.37867858, 0.26950601, 0.09406755, 0.1226853,  0.07751296,
 # 										0.10634889, 0.040095,   0.15335273, 0.10603025, 0.16524907])
	computer_model['guesses'] = np.array([0.0506464,  0.05740003, 0.06818031, 0.06279636, 0.07468154, 0.09690254,
 										0.31596187, 0.5,        0.06210698, 0.18757193, 0.05518057, 0.06798196,
 										0.02044008, 0.02423187, 0.07335546, 0.24625346, 0.09407903, 0.19767698,
 										0.25980498, 0.08917975, 0.08874032, 0.30269874, 0.23289485])
	computer_model['slips'] = 	np.array([0.5,        0.18349963, 0.1387788,  0.23271912, 0.13819418, 0.35076591,
 										0.20039586, 0.11274801, 0.1852813,  0.21107716, 0.24395035, 0.25027954,
 										0.26575247, 0.48915967, 0.17281178, 0.30662723, 0.2589953,  0.07589817,
 										0.24796133, 0.09235821, 0.17168115, 0.09512988, 0.14697441])

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

hapcap_preds = np.zeros((S_test_hapcap_squashed.shape[0]))
computer_preds = np.zeros((S_test_computer_squashed.shape[0]))

hapcap_probs = np.zeros((S_test_hapcap_squashed.shape[0]))
computer_probs = np.zeros((S_test_computer_squashed.shape[0]))

hapcap_avg_preds = np.zeros((S_test_hapcap_squashed.shape[0]))
computer_avg_preds = np.zeros((S_test_computer_squashed.shape[0]))

#lets now do the predictions for hapcap students
index = 0
for i in range(hapcap_test_students):
	length = S_lengths[i+hapcap_train_students]
	next_index = index+length
	data = S_test_hapcap_squashed[index:next_index]

	alphas = np.zeros((length, 2))
	alphas_avg = np.zeros((length, 2))

	alphas_final_preds = np.zeros((length))
	alphas_probs = np.zeros((length))
	alphas_avg_preds = np.zeros((length))

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
		qi = questions[j].astype(int)

		#calculate prior up to this point
		if (j == 0):
			alphas[0, 0] = (1-prior)
			alphas[0, 1] = prior

			alphas_avg[0, 0] = (1-prior)
			alphas_avg[0, 1] = prior
		else:
			alphas[j, 0] =  (alphas[(j-1), 0]*(1-learns)) + (alphas[(j-1), 1]*(forgets))
			alphas[j, 1] = (alphas[(j-1), 0]*(learns)) + (alphas[(j-1), 1]*(1-forgets))

			alphas_avg[j, 0] =  (alphas_avg[(j-1), 0]*(1-learns)) + (alphas_avg[(j-1), 1]*(forgets))
			alphas_avg[j, 1] = (alphas_avg[(j-1), 0]*(learns)) + (alphas_avg[(j-1), 1]*(1-forgets))

		#record this probability
		alphas_probs[j] = alphas[j, 1]

		#calculate the prediction for this time
		alphas_final_preds[j] = alphas[j, 0]*guesses[qi] + alphas[j, 1]*(1-slips[qi])

		#calculate w/ the avg guess/slip
		alphas_avg_preds[j] = alphas_avg[j, 0]*hapcap_guess + alphas_avg[j, 1]*(1-hapcap_slip)

		#calculate the posteriors (given the data)
		if (data[j] == 0):
			alphas[j, 1] = (alphas[(j), 1]*(slips[qi]))/((alphas[(j), 1]*(slips[qi]))+((1-alphas[(j), 1])*(1-guesses[qi])))
			alphas[j, 0] = 1-alphas[j, 1]

			#alphas_avg[j, 1] = (alphas[(j), 1]*(hapcap_slip))/((alphas[(j), 1]*(hapcap_slip))+((1-alphas[(j), 1])*(1-hapcap_guess)))
			#alphas_avg[j, 0] = 1-alphas_avg[j, 1]
		else:
			alphas[j, 1] = (alphas[(j), 1]*(1-slips[qi]))/((alphas[(j), 1]*(1-slips[qi]))+((1-alphas[(j), 1])*guesses[qi]))
			alphas[j, 0] = 1-alphas[j, 1]

			#alphas_avg[j, 1] = (alphas[(j), 1]*(1-hapcap_slip))/((alphas[(j), 1]*(1-hapcap_slip))+((1-alphas[(j), 1])*hapcap_guess))
			#alphas_avg[j, 0] = 1-alphas_avg[j, 1]

	hapcap_probs[index:next_index] = copy.deepcopy(alphas_probs[:])
	hapcap_preds[index:next_index] = copy.deepcopy(alphas_final_preds[:])
	hapcap_avg_preds[index:next_index] = copy.deepcopy(alphas_avg_preds[:])

	index = next_index

#lets now do the predictions for computer students
index = 0
for i in range(computer_test_students):
	length = S_lengths[i+hapcap_num_students+computer_train_students]
	next_index = index+length
	data = S_test_computer_squashed[index:next_index]

	alphas = np.zeros((length, 2))
	alphas_avg = np.zeros((length, 2))

	alphas_final_preds = np.zeros((length))
	alphas_probs = np.zeros((length))
	alphas_avg_preds = np.zeros((length))

	#params
	prior = computer_model['prior']
	learns = computer_model['learns'][0]
	forgets = computer_model['forgets'][0]
	guesses = computer_model['guesses']
	slips = computer_model['slips']

	questions = np.zeros((length))

	for j in range(length):
		for k in range(S_test_hapcap.shape[0]):
			if (S_test_hapcap[k, j] > 0):
				questions[j] = k

	#fill in the alphas for this students series

	for j in range(length):
		qi = questions[j].astype(int)

		#calculate prior up to this point
		if (j == 0):
			alphas[0, 0] = (1-prior)
			alphas[0, 1] = prior

			alphas_avg[0, 0] = (1-prior)
			alphas_avg[0, 1] = prior
		else:
			alphas[j, 0] = (alphas[(j-1), 0]*(1-learns)) + (alphas[(j-1), 1]*(forgets))
			alphas[j, 1] = (alphas[(j-1), 0]*(learns)) + (alphas[(j-1), 1]*(1-forgets))

			alphas_avg[j, 0] = (alphas_avg[(j-1), 0]*(1-learns)) + (alphas_avg[(j-1), 1]*(forgets))
			alphas_avg[j, 1] = (alphas_avg[(j-1), 0]*(learns)) + (alphas_avg[(j-1), 1]*(1-forgets))

		#record this probability
		alphas_probs[j] = alphas[j, 1]

		#calculate the prediction for this time
		alphas_final_preds[j] = alphas[j, 0]*guesses[qi] + alphas[j, 1]*(1-slips[qi])

		#calculate w/ the avg guess/slip
		alphas_avg_preds[j] = alphas_avg[j, 0]*hapcap_guess + alphas_avg[j, 1]*(1-hapcap_slip)

		#calculate the posteriors (given the data)
		if (data[j] == 0):
			alphas[j, 1] = (alphas[(j), 1]*(slips[qi]))/((alphas[(j), 1]*(slips[qi]))+((1-alphas[(j), 1])*(1-guesses[qi])))
			alphas[j, 0] = 1-alphas[j, 1]

			#alphas_avg[j, 1] = (alphas[(j), 1]*(hapcap_slip))/((alphas[(j), 1]*(hapcap_slip))+((1-alphas[(j), 1])*(1-hapcap_guess)))
			#alphas_avg[j, 0] = 1-alphas_avg[j, 1]
		else:
			alphas[j, 1] = (alphas[(j), 1]*(1-slips[qi]))/((alphas[(j), 1]*(1-slips[qi]))+((1-alphas[(j), 1])*guesses[qi]))
			alphas[j, 0] = 1-alphas[j, 1]

			#alphas_avg[j, 1] = (alphas[(j), 1]*(1-hapcap_slip))/((alphas[(j), 1]*(1-hapcap_slip))+((1-alphas[(j), 1])*hapcap_guess))
			#alphas_avg[j, 0] = 1-alphas_avg[j, 1]


	computer_probs[index:next_index] = copy.deepcopy(alphas_probs[:])
	computer_preds[index:next_index] = copy.deepcopy(alphas_final_preds[:])
	computer_avg_preds[index:next_index] = copy.deepcopy(alphas_avg_preds[:])

	index = next_index

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

	index += length

hapcap_stats = (p_error_sums[:]/p_error_totals[:])

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

	index += length

computer_stats = (p_error_sums[:]/p_error_totals[:])

#Lets calculate here the KL Divergence and compare with the APE

def kl(p, pm):
	eps = 1e-6
	div = ((1-p+eps)*np.log(1-p+eps) - (1-p+eps)*np.log(1-pm+eps)) + ((p+eps)*np.log(p+eps) - (p+eps)*np.log(pm+eps))
	return div

hapcap_errors = np.zeros((hapcap_preds.shape[0]))
hapcap_kl = np.zeros((hapcap_preds.shape[0]))

index = 0
for i in range(hapcap_test_students):
	length = S_lengths[i+hapcap_train_students]
	for j in range(length):
		hapcap_errors[index+j] = np.abs(hapcap_stats[j] - hapcap_preds[index+j])
		hapcap_kl[index+j] = kl(hapcap_stats[j], hapcap_preds[index+j])

	index += length

hapcap_mean_error = np.mean(hapcap_errors, axis=0)
hapcap_mean_kl = np.mean(hapcap_kl, axis=0)

hapcap_mean_error_scale = np.maximum(hapcap_mean_error, (1-hapcap_mean_error))
hapcap_mean_error_scaled = (hapcap_mean_error/hapcap_mean_error_scale)

computer_errors = np.zeros((computer_preds.shape[0]))
computer_kl = np.zeros((computer_preds.shape[0]))

index = 0
for i in range(computer_test_students):
	length = S_lengths[i+hapcap_num_students+computer_train_students]
	for j in range(length):
		computer_errors[index+j] = np.abs(computer_stats[j] - computer_preds[index+j])
		computer_kl[index+j] = kl(computer_stats[j], computer_preds[index+j])

	index += length

computer_mean_error = np.mean(computer_errors, axis=0)
computer_mean_kl = np.mean(computer_kl, axis=0)

computer_mean_error_scale = np.maximum(computer_mean_error, (1-computer_mean_error))
computer_mean_error_scaled = (computer_mean_error/computer_mean_error_scale)

print ('Hapcap Mean Error, Scaled Error, and KL')
print (hapcap_mean_error)
print (hapcap_mean_error_scaled)
print (hapcap_mean_kl)

print ('Computer Mean Error, Scaled Error, and KL')
print (computer_mean_error)
print (computer_mean_error_scaled)
print (computer_mean_kl)

#now calculate predicted loss with BKT and accuracy
loss_hc_bkt = np.sum(np.abs(y_test_hapcap[:] - hapcap_preds[:]), axis=0)
loss_comp_bkt = np.sum(np.abs(y_test_computer[:] - computer_preds[:]), axis=0)

hapcap_pred_int = (hapcap_preds > .5).astype(int)
computer_pred_int = (computer_preds > .5).astype(int)

hapcap_incorr_bkt = np.sum(np.abs(y_test_hapcap[:] - hapcap_pred_int[:]), axis=0)
computer_incorr_bkt = np.sum(np.abs(y_test_computer[:] - computer_pred_int[:]), axis=0)

acc_hc_bkt = ((y_test_hapcap.shape[0]-hapcap_incorr_bkt)/y_test_hapcap.shape[0])
acc_comp_bkt = ((y_test_computer.shape[0]-computer_incorr_bkt)/y_test_computer.shape[0])


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


#Let us now compare the confusion matrices from the two model for hapcaps
cm_bkt = confusion_matrix(y_test_hapcap, hapcap_pred_int)
f1_bkt = f1_score(y_test_hapcap, hapcap_pred_int)

recall_hapcap_bkt = recall_score(y_test_hapcap, hapcap_pred_int)
precision_hapcap_bkt = precision_score(y_test_hapcap, hapcap_pred_int)

cm_comp_bkt = confusion_matrix(y_test_computer, computer_pred_int)
f1_comp_bkt = f1_score(y_test_computer, computer_pred_int)

recall_computer_bkt = recall_score(y_test_computer, computer_pred_int)
precision_computer_bkt = precision_score(y_test_computer, computer_pred_int)

rms_hapcap = mean_squared_error(y_test_hapcap, hapcap_pred_int, squared=False)
rms_comp = mean_squared_error(y_test_computer, computer_pred_int, squared=False)

#Recall, Precision
print ('Recall Hapcap BKT')
print (recall_hapcap_bkt)

print ('Recall Computer BKT')
print (recall_computer_bkt)

print ('Precision Hapcap BKT')
print (precision_hapcap_bkt)

print ('Precision Computer BKT')
print (precision_computer_bkt)

#Confusion Matrices
print ('Confusion Matrix Hapcap LogReg')
print (cm_logreg)

print ('Confusion Matrix Computer LogReg')
print (cm_comp_logreg)

print ('Confusion Matrix Hapcap BKT')
print (cm_bkt)

print ('Confusion Matrix Computer BKT')
print (cm_comp_bkt)

#F1 Scores
print ('F1 Score Hapcap LogReg')
print (f1_logreg)

print ('F1 Score Computer LogReg')
print (f1_comp_logreg)

print ('F1 Score Hapcap BKT')
print (f1_bkt)

print ('F1 Score Computer BKT')
print (f1_comp_bkt)

print ('RMSE Hapcap BKT')
print (rms_hapcap)

print ('RMSE Computer BKT')
print (rms_comp)

# plt.title('HapCaps BKT Confusion Matrix')
# cm_hapcap_heat = sns.heatmap(cm_bkt/np.sum(cm_bkt), annot=True, 
# fmt='.2%', cmap='Blues')
# fig = cm_hapcap_heat.get_figure()
# string = 'cm_hapcap.png'
# fig.savefig(string)

# plt.title('Trackpad BKT Confusion Matrix')
# cm_computer_heat = sns.heatmap(cm_comp_bkt/np.sum(cm_comp_bkt), annot=True, 
# fmt='.2%', cmap='Blues')
# fig = cm_computer_heat.get_figure()
# string = 'cm_computer.png'
# fig.savefig(string)

print ('Total Number Students')
print ('Hapcap')
print (hapcap_train_students+hapcap_test_students)
print ('Computer')
print (computer_train_students+computer_test_students)

total_hapcap_test_length = 0
total_computer_test_length = 0
max_hapcap_length = 0
max_computer_length = 0

for i in range (hapcap_test_students):
	length = S_lengths[i+hapcap_train_students]
	if (length >= max_hapcap_length):
		max_hapcap_length = length

	total_hapcap_test_length += length

for i in range(computer_test_students):
	length = S_lengths[i+hapcap_num_students+computer_train_students]
	if (length >= max_computer_length):
		max_computer_length = length

	total_computer_test_length += length

print (total_hapcap_test_length)
print (total_computer_test_length)

test_hapcap_probs = np.zeros((max_hapcap_length))
test_computer_probs = np.zeros((max_computer_length))

test_hapcap_results = np.zeros((max_hapcap_length))
test_computer_results = np.zeros((max_computer_length))

test_hapcap_avg_results = np.zeros((max_hapcap_length))
test_computer_avg_results = np.zeros((max_computer_length))

num_hapcap_results = np.zeros((max_hapcap_length))
num_computer_results = np.zeros((max_computer_length))

total_length = 0
for i in range (hapcap_test_students):
	length = S_lengths[i+hapcap_train_students]

	temp = total_length+length

	test_hapcap_probs[:length] += hapcap_probs[total_length:temp]
	test_hapcap_results[:length] += hapcap_preds[total_length:temp]
	test_hapcap_avg_results[:length] += hapcap_avg_preds[total_length:temp]

	total_length = temp

	ones = np.ones((length))
	num_hapcap_results[:length] += ones


avg_hapcap_probs = (test_hapcap_probs/num_hapcap_results)
avg_hapcap_results = (test_hapcap_results/num_hapcap_results)
avg_hapcap_avg_results = (test_hapcap_avg_results/num_hapcap_results)

total_length = 0
for i in range (computer_test_students):
	length = S_lengths[i+hapcap_num_students+computer_train_students]

	temp = total_length+length

	test_computer_probs[:length] += computer_probs[total_length:temp]
	test_computer_results[:length] += computer_preds[total_length:temp]
	test_computer_avg_results[:length] += computer_avg_preds[total_length:temp]

	total_length = temp

	ones = np.ones((length))
	num_computer_results[:length] += ones


avg_computer_probs = (test_computer_probs/num_computer_results)
avg_computer_results = (test_computer_results/num_computer_results)
avg_computer_avg_results = (test_computer_avg_results/num_computer_results)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

#what about just the first 1000 steps, and with a moving average
trim_val = 1100

avg_hapcap_results_trim = np.zeros((trim_val))
avg_hapcap_results_trim[:] = avg_hapcap_results[:trim_val]

avg_computer_results_trim = np.zeros((trim_val))
avg_computer_results_trim[:] = avg_computer_results[:trim_val]

avg_hapcap_probs_trim = np.zeros((trim_val))
avg_hapcap_probs_trim = avg_hapcap_probs[:trim_val]

avg_computer_probs_trim = np.zeros((trim_val))
avg_computer_probs_trim = avg_computer_probs[:trim_val]

avg_hapcap_avg_results_trim = np.zeros((trim_val))
avg_hapcap_avg_results_trim[:] = avg_hapcap_avg_results[:trim_val]

avg_computer_avg_results_trim = np.zeros((trim_val))
avg_computer_avg_results_trim[:] = avg_computer_avg_results[:trim_val]

hapcap_offset = 0
computer_offset = 0

mv_avg_n = 100

mavg_hapcap_results = moving_average(avg_hapcap_results_trim, mv_avg_n) + hapcap_offset
mavg_computer_results = moving_average(avg_computer_results_trim, mv_avg_n) + computer_offset

mavg_hapcap_probs = moving_average(avg_hapcap_probs_trim, mv_avg_n) + hapcap_offset
mavg_computer_probs = moving_average(avg_computer_probs_trim, mv_avg_n) + computer_offset

mavg_hapcap_avg_preds = moving_average(avg_hapcap_avg_results_trim, mv_avg_n) + hapcap_offset
mavg_computer_avg_preds = moving_average(avg_computer_avg_results_trim, mv_avg_n) + computer_offset

x = np.linspace(1, trim_val, num=trim_val)

mvg_hapcap_num = mavg_hapcap_results.shape[0]
x = np.linspace(1, mvg_hapcap_num, num=mvg_hapcap_num)

plt.figure(5)
plt.plot(x[50:], mavg_hapcap_results[50:], color='red')
plt.plot(x[50:], mavg_computer_results[50:], color='blue')
plt.ylim(0, .4)
plt.xlabel('Timestep')
plt.ylabel('P(C)')
plt.legend(['Hapcap', 'Trackpad'])
plt.title('Student Improvement via Both Methods')
plt.show()

#np.savetxt("x.csv", x, delimiter=",")
#np.savetxt("hapcap_results.csv", mavg_hapcap_results, delimiter=",")
#np.savetxt("computer_results.csv", mavg_computer_results, delimiter=",")

plt.figure(6)
plt.plot(x[50:], mavg_hapcap_probs[50:], color='red')
plt.plot(x[50:], mavg_computer_probs[50:], color='blue')
plt.ylim(0, .4)
plt.xlabel('Timestep')
plt.ylabel('P(C)')
plt.legend(['Hapcap', 'Trackpad'])
plt.title('Student Improvement via Both Methods')
plt.show()

plt.figure(7)
plt.plot(x[0:], mavg_hapcap_avg_preds[0:], color='red')
plt.plot(x[0:], mavg_computer_avg_preds[0:], color='blue')
plt.ylim(.4, .7)
plt.xlabel('Timestep')
plt.ylabel('P(C)')
plt.legend(['Hapcap', 'Trackpad'])
plt.title('Student Improvement via Both Methods')
plt.show()

np.savetxt("x.csv", x, delimiter=",")
np.savetxt("hapcap_results.csv", mavg_hapcap_avg_preds, delimiter=",")
np.savetxt("computer_results.csv", mavg_computer_avg_preds, delimiter=",")


# plt.figure(6)
# bar_width = .35

# color_hapcap = ['green']
# color_computer = ['red']
# prior_hapcap = np.array(['Hapcap Prior'])
# prior_computer = np.array(['Computer Prior'])
# hapcap_priors = np.array([hapcap_model['prior']])
# computer_priors = np.array([computer_model['prior']])

# plt.bar(prior_hapcap, hapcap_priors, bar_width, color=color_hapcap)
# plt.bar(prior_computer, computer_priors, bar_width, color=color_computer)
# plt.ylabel('Probability')
# plt.ylim(0, 1)
# plt.grid(True)
# plt.show()


















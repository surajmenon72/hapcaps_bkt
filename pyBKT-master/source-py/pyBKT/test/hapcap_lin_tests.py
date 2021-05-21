import numpy as np
from copy import deepcopy
import scipy.io as sio
import sys
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression

#Run hypothesis testing on pre, post tests
#Run regression analysis on pre,post tests
#Run prediction comparison on time series data w/ LogReg vs. BKT

#get pre,post data imported

hapcap_rooms = [16, 18, 20]
computer_rooms = [32, 6]

#make pre
hapcap_pre = np.zeros((1, 31))
hapcap_pre_lengths = [1]
computer_pre = np.zeros((1, 31))
computer_pre_lengths = [1]

room_sizes = {}
for i in range(1, 50):
	string = 'dataprepost/r' + str(i) + 'Pre.npy'

	try:
		A = np.load(string)

		if (i in hapcap_rooms):
			hapcap_pre = np.concatenate((hapcap_pre, A), axis=0)
			hapcap_pre_lengths.append(A.shape[0])
			room_sizes[str(i)] = A.shape[0]
		elif (i in computer_rooms):
			computer_pre = np.concatenate((computer_pre, A), axis=0)
			computer_pre_lengths.append(A.shape[0])
			room_sizes[str(i)] = A.shape[0]
	except:
		print ('No File Found')

#make post
hapcap_post = np.zeros((1, 31))
hapcap_post_lengths = [1]
computer_post = np.zeros((1, 31))
computer_post_lengths = [1]
for i in range(1, 50):
	string = 'dataprepost/r' + str(i) + 'Post.npy'

	try:
		A = np.load(string)

		if (i in hapcap_rooms):
			hapcap_post = np.concatenate((hapcap_post, A), axis=0)
			hapcap_post_lengths.append(A.shape[0])
		elif (i in computer_rooms):
			computer_post = np.concatenate((computer_post, A), axis=0)
			computer_post_lengths.append(A.shape[0])
	except:
		print ('No File Found')

#make the vectors for just the final counts, assume in the last index
#assume we have a data file for both pre and post for each piece
hapcap_correspondence = True
for i in range(len(hapcap_pre_lengths)):
	if (i > 0):
		#make sure our results correspond to the same students
		if (hapcap_pre_lengths[i] != hapcap_post_lengths[i]):
			hapcap_correspondence = False


if (hapcap_correspondence):
	hapcap_pre_vec = hapcap_pre[:, 30]
	hapcap_post_vec = hapcap_post[:, 30]

computer_correspondence = True
for i in range(len(computer_pre_lengths)):
	if (i > 0):
		#make sure our results correspond to the same students
		if (computer_pre_lengths[i] != computer_post_lengths[i]):
			computer_correspondence = False


if (computer_correspondence):
	computer_pre_vec = computer_pre[:, 30]
	computer_post_vec = computer_post[:, 30]

if ((hapcap_correspondence == False) or (computer_correspondence == False)):
	print ('Pre and Post do not match, exiting')
	exit()

#have the actual improvement be what is modeled
hapcap_y = np.abs(hapcap_post_vec-hapcap_pre_vec)
computer_y = np.abs(computer_post_vec-computer_pre_vec)

#Ok, now lets consider a regression with some basic features

room32NumStudents = room_sizes['32']
room20NumStudents = room_sizes['20']
room18NumStudents = room_sizes['18']
room16NumStudents = room_sizes['16']
room6NumStudents = room_sizes['6']
female = 1
genderRoom32 = [1,1,0,1,0,0,0,0,1,0,1,1,1,0,0,1]
genderRoom16 = [0,1,0,1,0,1,0,0,0,1,1,1,1,0,0]
genderRoom6 = [0,1,1,1,0,1,1,1,1,0,1,0,0,1,0,1,0,1,1]
genderRoom18 = [1,1,0,1,0,0,1,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1]
genderRoom20 = [1,0,0,1,1,0,0,1,1,0,0,0,0,0,1,0,1,0,1,1]

n_genderRoom32 = np.asarray(genderRoom32)
n_genderRoom16 = np.asarray(genderRoom16)
n_genderRoom6 = np.asarray(genderRoom6)
n_genderRoom18 = np.asarray(genderRoom18)
n_genderRoom20 = np.asarray(genderRoom20)

#check shapes

room32shape = (room32NumStudents == n_genderRoom32.shape[0])
room16shape = (room16NumStudents == n_genderRoom16.shape[0])
room6shape = (room6NumStudents == n_genderRoom6.shape[0])
room18shape = (room18NumStudents == n_genderRoom18.shape[0])
room20shape = (room20NumStudents == n_genderRoom20.shape[0])

if (room32shape and room16shape and room6shape and room18shape and room20shape):
	print ('Features align')
else:
	print ('Features do not align')
	exit()

#rooms 16, 18, 20
gender_hapcap = np.zeros(1)
gender_hapcap = np.concatenate((gender_hapcap, n_genderRoom16), axis=0)
gender_hapcap = np.concatenate((gender_hapcap, n_genderRoom18), axis=0)
gender_hapcap = np.concatenate((gender_hapcap, n_genderRoom20), axis=0)

#rooms 32, 6
gender_computer = np.zeros(1)
gender_computer = np.concatenate((gender_computer, n_genderRoom32), axis=0)
gender_computer = np.concatenate((gender_computer, n_genderRoom6), axis=0)

#for now assume this is our only feature.  Now find the diff of pre, post tests
num_features = 1
num_hapcap_students = room16NumStudents+room18NumStudents+room20NumStudents+1
num_computer_students = room32NumStudents+room6NumStudents+1
X_hapcap = np.zeros((num_hapcap_students, num_features))
X_computer = np.zeros((num_computer_students, num_features))

#attach the features
X_hapcap[:, 0] = gender_hapcap
X_computer[:, 0] = gender_computer

print (X_hapcap.shape)
print (hapcap_y.shape)
print (X_computer.shape)
print (computer_y.shape)

#Ok, now we have the data, lets run the hypothesis test
t_hapcap, p_hapcap = stats.ttest_ind(hapcap_pre_vec, hapcap_post_vec)
t_computer, p_computer = stats.ttest_ind(computer_pre_vec, computer_post_vec)
t_diff, p_diff = stats.ttest_ind(hapcap_y, computer_y)

#compare reg.coeff, reg.intercept, plots and stuff
print ('T Test Results')
print ('Hapcap Pre vs. Post')
print ('T:', t_hapcap, 'P value:', p_hapcap)
print ('Computer Pre vs. Post')
print ('T:', t_computer, 'P value:', p_computer)
print ('Diff Hapcap Vs. Computer')
print ('T:', t_diff,'P value:', p_diff)

print (' ')

#plot 1
print ('T Test between Hapcap Pre and Post')
plt.figure(1)
zeros = np.zeros(hapcap_pre_vec.shape[0])
ones = np.ones(hapcap_post_vec.shape[0])

pre = np.vstack((zeros, hapcap_pre_vec)).T
post = np.vstack((ones, hapcap_post_vec)).T

data = np.vstack((pre,post))

plt.scatter(data[:, 0], data[:, 1])
label = 'T: ' + str(t_hapcap) + '     P value:' + str(p_hapcap)
plt.xlabel(label)
plt.ylabel('Score')
plt.title('T Test between Hapcap Pre and Post')
plt.show()

#plot 2
print ('T test between Computer Pre and Post')
plt.figure(2)
zeros = np.zeros(computer_pre_vec.shape[0])
ones = np.ones(computer_post_vec.shape[0])

pre = np.vstack((zeros, computer_pre_vec)).T
post = np.vstack((ones, computer_post_vec)).T

data = np.vstack((pre,post))

plt.scatter(data[:, 0], data[:, 1])
label = 'T: ' + str(t_computer) + '     P value:' + str(p_computer)
plt.xlabel(label)
plt.ylabel('Score')
plt.title('T Test between Computer Pre and Post')
plt.show()

#plot 3
print ('T test between Hapcap and Comp Diff')
plt.figure(3)
zeros = np.zeros(hapcap_y.shape[0])
ones = np.ones(computer_y.shape[0])

pre = np.vstack((zeros, hapcap_y)).T
post = np.vstack((ones, computer_y)).T

data = np.vstack((pre,post))

plt.scatter(data[:, 0], data[:, 1])
label = 'T: ' + str(t_diff) + '     P value:' + str(p_diff)
plt.xlabel(label)
plt.ylabel('Diff between Post and Pre Scores')
plt.title('T Test between Hapcap and Comp Diff')
plt.show()

print (' ')

#And the Regressions
reg_hapcap = LinearRegression().fit(X_hapcap, hapcap_y)
reg_computer = LinearRegression().fit(X_computer, computer_y)

print ('Hapcap Regression')
print (reg_hapcap.coef_)
print (reg_hapcap.intercept_)

print ('Computer Regression')
print (reg_computer.coef_)
print (reg_computer.intercept_)

#plot 4
plt.figure(4)
plt.scatter(X_hapcap, hapcap_y)
x_line = np.linspace(0, 1, num=10).reshape(-1, 1)
y_line = reg_hapcap.predict(x_line)
plt.plot(x_line, y_line, color='red')
label = 'Gender: Male = 0, Female = 1.  theta =' + str(reg_hapcap.coef_)
plt.xlabel(label)
plt.ylabel('Diff between Post and Pre Scores')
plt.title('Regression of Hapcap Pre vs. Post Data by Gender')
plt.show()

#plot 5
plt.figure(5)
plt.scatter(X_computer, computer_y)
x_line = np.linspace(0, 1, num=10).reshape(-1, 1)
y_line = reg_computer.predict(x_line)
plt.plot(x_line, y_line, color='red')
label = 'Gender: Male = 0, Female = 1.  theta =' + str(reg_computer.coef_)
plt.xlabel(label)
plt.ylabel('Diff between Post and Pre Scores')
plt.title('Regression of Computer Pre vs. Post Data by Gender')
plt.show()










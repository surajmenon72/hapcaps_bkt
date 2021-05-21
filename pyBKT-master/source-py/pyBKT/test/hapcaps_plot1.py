import numpy as np
import matplotlib.pyplot as plt


hapcap_avg_t = 0.9052840265409683
hapcap_var_t = 0.005056714496268578

comp_avg_t = 0.9046329057456984
comp_var_t = 0.0068137908403066785
#Avg Guess per Question
#Hapcap
hapcap_guess = [0.72522582, 0.60786242, 0.47273889, 0.75036638, 0.66515549, 0.53484804,
 				0.38536503, 0.424204,   0.62460019, 0.58951826, 0.68330475, 0.75278817,
 				0.64824595, 0.5027727,  0.80716903, 0.38218361, 0.61044862, 0.34107215,
 				0.55602609, 0.64408618, 0.64728886, 0.37221206, 0.39753219]
#Comp
comp_guess = [0.7814131,  0.70642212, 0.57899753, 0.76360613, 0.68171775, 0.58790428,
 			  0.53825579, 0.23633649, 0.49046465, 0.70274461, 0.66253123, 0.69059426,
			  0.64875616, 0.68930421, 0.81207501, 0.51177459, 0.62336256, 0.40862179,
			  0.41556987, 0.67283708, 0.70183631, 0.4430947,  0.48290897]
#Avg Slip per Question
#Hapcap
hapcap_slip = [0.07779578, 0.12261548, 0.11930765, 0.0605531,  0.11200321, 0.14364938,
			   0.30525045, 0.57471626, 0.08316147, 0.1790163,  0.07414969, 0.12794752,
			   0.02326036, 0.03129882, 0.0857465,  0.30032475, 0.16127622, 0.33543798,
			   0.38827422, 0.16471795, 0.13879829, 0.3805423,  0.47626009]
#Comp
comp_slip = [0.05281108, 0.08046016, 0.09363044, 0.04876081, 0.09264934, 0.11872019,
			 0.32502947, 0.53838773, 0.0681894,  0.16664019, 0.05976212, 0.1124457,
			 0.01665728, 0.01497798, 0.07602847, 0.28811854, 0.12131864, 0.24075379,
			 0.34787576, 0.15752019, 0.11324493, 0.30545397, 0.42556853]





hc_g_a = np.array(hapcap_guess)
c_g_a = np.array(comp_guess)

hc_s_a = np.array(hapcap_slip)
c_s_a = np.array(comp_slip)

"""
data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, axisbg="1.0")

for data, color, group in zip(data, colors, groups):
x, y = data
ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
"""
x = np.linspace(1, 23, num=23)
bar_width = .35

plt.figure(1)
#plt.subplot(2, 1, 1)
plt.bar(x, hc_g_a, bar_width, color='red', label='hapcap')
#plt.subplot(2, 1, 2)
plt.bar(x+bar_width, c_g_a, bar_width, color='blue', label='computer')
plt.legend()
plt.xlabel('Question')
plt.ylabel('Probability of Guess')
#plt.show()

plt.figure(2)
#plt.subplot(2, 1, 1)
plt.bar(x, hc_s_a, bar_width, color='red', label='hapcap')
#plt.subplot(2, 1, 2)
plt.bar(x+bar_width, c_s_a, bar_width, color='blue', label='computer')
plt.legend()
plt.xlabel('Question')
plt.ylabel('Probability of Slip')

plt.show()



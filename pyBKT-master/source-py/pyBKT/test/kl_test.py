import numpy as np 
import matplotlib.pyplot as plt

def kl(p, pm):
	eps = 1e-6
	div = ((1-p+eps)*np.log(1-p+eps) - (1-p+eps)*np.log(1-pm+eps)) + ((p+eps)*np.log(p+eps) - (p+eps)*np.log(pm+eps))
	return div

p = .5
pm = np.linspace(0, 1, 101)

errs = np.zeros(pm.shape[0])
kls = np.zeros(pm.shape[0])

for i in range(pm.shape[0]):
	errs[i] =  np.abs(p - pm[i])
	kls[i] = kl(p, pm[i])

plt.figure(1)
plt.plot(pm, errs)
plt.plot(pm, kls)
plt.ylim(0, 1)
plt.show()
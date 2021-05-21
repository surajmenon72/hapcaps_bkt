import numpy as np 

states = 2
outputs = 2
states_perm = 4

A_0 = np.zeros(states)
A = np.zeros((states, states))
B = np.zeros((states, outputs))

A_0[:] = [.4, .6]
A[:, :] = [[.4, .6], [.6, .4]]
B[:, :] = [[.25, .75], [.75, .25]]

t_s = 10
x_t = np.zeros(t_s, dtype=int)
x_t_n = np.zeros(t_s, dtype=int)
x_t[:] = [0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
x_t_n[:] = (1-x_t)

alphas = np.zeros((states, t_s))
betas = np.zeros((states, t_s+1))

gammas_init = np.zeros(states)
gammas = np.zeros((states_perm, t_s))

iters = 10

for i in range(iters):
	print ('Iter')
	print (i)
	
	alphas[0, 0] = A_0[0]
	alphas[1, 0] = A_0[1]

	betas[0, (t_s-1)] = A_0[0]
	betas[1, (t_s-1)] = A_0[1]

	betas[0, t_s] = 1
	betas[1, t_s] = 1

	#compute alphas, betas
	for a in range(states):
		for c in range(1, t_s):
			for b in range(states):
				alphas[a, c] += alphas[a, (c-1)]*A[b, a]*B[a, x_t[c]]

				beta_ts = (t_s-c)-1
				betas[a, beta_ts] += betas[a, (beta_ts+1)]*A[b, a]*B[a, x_t[beta_ts]]


	#compute gammas

	gammas_init[0] = A_0[0]*B[0, x_t[0]]*betas[0, 1]
	gammas_init[1] = A_0[1]*B[1, x_t[1]]*betas[1, 1]
	for a in range(states):
		for b in range(states):
			for c in range(t_s):
				gamma_i = (2*a + b)
				gammas[gamma_i, c] = alphas[a, c]*A[a, b]*B[b, x_t[c]]*betas[b, c+1]



	#update init param
	A_0[0] = gammas_init[0] / np.sum(gammas_init, axis=0)
	A_0[1] = gammas_init[1] / np.sum(gammas_init, axis=0)


	#update A params
	for a in range(states):
		for b in range(states):
			gamma_i = (2*a + b)
			gamma_i0 = (2*a + 0)
			gamma_i1 = (2*a + 1)

			A[a, b] = np.sum(gammas[gamma_i, :]) / (np.sum(gammas[gamma_i0, :]) + np.sum(gammas[gamma_i1, :]))


	for a in range(states):
		for b in range(outputs):
			gamma_i0 = (2*0 + a)
			gamma_i1 = (2*1 + a)

			if (b == 0):
				gammas_0 = np.multiply(gammas[gamma_i0, :], x_t_n)
				gammas_1 = np.multiply(gammas[gamma_i1, :], x_t_n)
			else:
				gammas_0 = np.multiply(gammas[gamma_i0, :], x_t)
				gammas_1 = np.multiply(gammas[gamma_i1, :], x_t)

			num = (np.sum(gammas_0, axis=0)+np.sum(gammas_1, axis=0))
			den = (np.sum(gammas[gamma_i0, :])+np.sum(gammas[gamma_i1, :]))

			B[a, b] = num/den



print ('Done')
print ('A_0')
print (A_0)
print ('A')
print (A)
print ('B')
print (B)




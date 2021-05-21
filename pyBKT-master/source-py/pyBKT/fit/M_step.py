import numpy as np

def run(model, trans_softcounts, emission_softcounts, init_softcounts):

    #is this right??
    trans_softcounts[np.sum(trans_softcounts, axis=1) == 0,:] = 1
    emission_softcounts[np.sum(emission_softcounts, axis=2) == 0,:] = 1
    assert (trans_softcounts.shape[1] == 2)
    assert (trans_softcounts.shape[2] == 2)

    temp = np.sum(trans_softcounts, axis=1)

    for i in range(model['As'].shape[0]):
    	model['As'][i] = trans_softcounts[i] / np.sum(trans_softcounts, axis=1)[i]

    if (model['As'][:, 0, 1] > .05):
         model['As'][:, 0, 1] = .001
         model['As'][:, 1, 1] = .999

    model['learns'] = model['As'][:, 1, 0]

    model['forgets'] = model['As'][:, 0, 1]

    temp = np.sum(emission_softcounts, axis=2)

    #model['emissions'] = emission_softcounts / np.sum(emission_softcounts, axis=1)
    model['emissions'] = emission_softcounts / temp[:, :, None]

    emission_shape = model['emissions'].shape[0]
    for i in range(emission_shape):
        if (model['emissions'][i, 0, 1] > .5):
            model['emissions'][i, 0, 1] = .5
            model['emissions'][i, 0, 0] = .5
        if (model['emissions'][i, 1, 0] > .5):
            model['emissions'][i, 1, 0] = .5
            model['emissions'][i, 1, 1] = .5

    #the expand dims is very weird.
    #model['guesses'] = np.expand_dims(model['emissions'][:, 0, 1].squeeze(), axis=0)
    #model['slips'] = np.expand_dims(model['emissions'][:, 1, 0].squeeze(), axis=0)
    model['guesses'] = model['emissions'][:, 0, 1]
    model['slips'] = model['emissions'][:, 1, 0]

    temp = np.sum(init_softcounts[:])
    model['pi_0'] = init_softcounts[:] / np.sum(init_softcounts[:])

    model['prior'] = model['pi_0'][1][0]

    return(model)

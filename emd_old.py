import numpy as np

q = [.125, .125, .125, .125, .125, .125, .125, .125]
p = [0., 0., .16666667, .16666667, .16666667, .16666667, .16666667, .16666667]
# p = [[[0.,0.16666667], [0.16666667, 0.16666667]],[[0.16666667, 0.16666667],[0.16666667,0.16666667]]]
# q = [[[.125, .125],[.125,.125]],[[.125,.125],[.125,.125]]]

def emd(p,q):
    emd = 0
    td = 0
    for i in range(0,len(p)-1):
        emd = p[i] + emd - q[i]
        td += abs(emd)
    return td

def kullback_leibner(p,q):
    DKL = 0
    for i in range(0,len(p)-1):
        if p[i] == 0:
            DKL += 0
        else:
            DKL += p[i]*np.log(p[i]/q[i])
    return DKL


dist_emd = emd(p,q)
dist_dkl = kullback_leibner(p,q)


print "EMD(p,q) = ", dist_emd
print "Kullback_Leibner(p,q) = ", dist_dkl

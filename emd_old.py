import numpy as np
import matplotlib.pyplot as pl
from itertools import product

q = [.125, .125, .125, .125, .125, .125, .125, .125]
p = [0., 0., .16666667, .16666667, .16666667, .16666667, .16666667, .16666667]
# p = [[[0.,0.16666667], [0.16666667, 0.16666667]],[[0.16666667, 0.16666667],[0.16666667,0.16666667]]]
# q = [[[.125, .125],[.125,.125]],[[.125,.125],[.125,.125]]]


def emd(p,q):
    emd = 0
    td = 0
    for i in range(0, len(p)-1):
        emd = p[i] + emd - q[i]
        td += abs(emd)
    return td


def kullback_leibner(p, q):
    DKL = 0
    for i in range(0, len(p)-1):
        if p[i] == 0:
            DKL += 0
        else:
            DKL += p[i]*np.log(p[i]/q[i])
    return DKL


def variations_generator(x, x_len):
    sample = list()
    for i in product(x, repeat=x_len):
        sample.append(i)
    return sample


def and_gate(list):
    result = []
    for i in range(0, len(list)):
        a = list[i][0]
        b = list[i][1]
        if a == 1 and b == 1:
            result.append(1)
        else:
            result.append(0)

    return result


def or_gate(list):
    result = []
    for i in range(0, len(list)):
        a = list[i][0]
        b = list[i][1]
        if a == 0 and b == 0:
            result.append(0)
        else:
            result.append(1)

    return result


def xor_gate(list):
    result = []
    for i in range(0, len(list)):
        a = list[i][0]
        b = list[i][1]
        if (a == 1 and b == 1) or (a == 0 and b == 0):
            result.append(0)
        else:
            result.append(1)

    return result


bin_set = [0, 1]

my_3_space = variations_generator(bin_set, 3)
my_2_space = variations_generator(bin_set, 2)
my_1_space = [(1, 0), (1, 1)]

dist_emd = emd(p, q)
dist_dkl = kullback_leibner(p, q)

and_g = and_gate(my_2_space)
or_g = or_gate(my_2_space)
xor_g = xor_gate(my_2_space)

and_g2 = and_gate(my_1_space)
or_g2 = or_g
xor_g2 = xor_gate(my_1_space)

pA0 = float(or_g.count(0))/float(len(or_g))
pA1 = float(or_g.count(1))/float(len(or_g))
pB0 = float(and_g.count(0))/float(len(and_g))
pB1 = float(and_g.count(1))/float(len(and_g))
pC0 = float(xor_g.count(0))/float(len(xor_g))
pC1 = float(xor_g.count(1))/float(len(xor_g))

unc_fut = []
unc_fut.append(pA0*pB0*pC0)
unc_fut.append(pA1*pB0*pC0)
unc_fut.append(pA0*pB1*pC0)
unc_fut.append(pA1*pB1*pC0)
unc_fut.append(pA0*pB0*pC1)
unc_fut.append(pA1*pB0*pC1)
unc_fut.append(pA0*pB1*pC1)
unc_fut.append(pA1*pB1*pC1)

pA0 = float(or_g2.count(0))/float(len(or_g2))
pA1 = float(or_g2.count(1))/float(len(or_g2))
pB0 = float(and_g2.count(0))/float(len(and_g2))
pB1 = float(and_g2.count(1))/float(len(and_g2))
pC0 = float(xor_g2.count(0))/float(len(xor_g2))
pC1 = float(xor_g2.count(1))/float(len(xor_g2))

con_fut = []

con_fut.append(pA0*pB0*pC0)
con_fut.append(pA1*pB0*pC0)
con_fut.append(pA0*pB1*pC0)
con_fut.append(pA1*pB1*pC0)
con_fut.append(pA0*pB0*pC1)
con_fut.append(pA1*pB0*pC1)
con_fut.append(pA0*pB1*pC1)
con_fut.append(pA1*pB1*pC1)

print(unc_fut)
print("")
print(con_fut)

#t = pl.plot(unc_fut)
#pl.show(unc_fut)
#q = pl.hist(unc_fut, bins=5)
#pl.show(q)

dist_emd2 = emd(unc_fut, con_fut)
dist_dkl2 = kullback_leibner(unc_fut, con_fut)

print("")
print("EMD1(p,q) = ", dist_emd)
print("")
print("Kullback_Leibner1(p,q) = ", dist_dkl)
print("")
print("EMD2(p,q) = ", dist_emd2)
print("")
print("Kullback_Leibner2(p,q) = ", dist_dkl2)
print("")
print(my_3_space)
print("")
print(my_2_space)
print("")
print(my_1_space)
print("")
print("AND: ", and_g)
print("")
print("OR: ", or_g)
print("")
print("XOR: ", xor_g)
print("")
print("AND2: ", and_g2)
print("")
print("OR2: ", or_g2)
print("")
print("XOR2: ", xor_g2)
print("")


p1 = [0., 0., 0., .5, .5, 0., 0., 0.]
q1 = [.25, 0., 0., .25, .25, 0., 0., .25]
q2 = [.16666667, .16666667, .16666667, .16666667, .16666667, 0., .16666667, 0.]
q3 = [0., 0., .16666667, .16666667, .16666667, .16666667, .16666667, .16666667]
q4 = [.125, .125, .125, .125, .125, .125, .125, .125]

dist_emd3 = emd(p1, q1)
dist_dkl3 = kullback_leibner(p1, q1)
dist_emd4 = emd(p1, q2)
dist_dkl4 = kullback_leibner(p1, q2)
dist_emd5 = emd(p1, q3)
dist_dkl5 = kullback_leibner(p1, q3)
dist_emd6 = emd(p1, q4)
dist_dkl6 = kullback_leibner(p1, q4)

print("EMD3=",dist_emd3)
print("")
print("KL3=",dist_dkl3)
print("")
print("EMD4=",dist_emd4)
print("")
print("KL4=",dist_dkl4)
print("")
print("EMD5=",dist_emd5)
print("")
print("KL5=",dist_dkl5)
print("")
print("EMD6=",dist_emd6)
print("")
print("KL6=",dist_dkl6)
print("")

import numpy as np
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
    for i in (0, len(list)-1):
        a = list[i][0]
        b = list[i][1]
        print(list[i])
        print("a:",a)
        print("b:",b)
        if a == 1 and b == 1:
            result.append(1)
        else:
            result.append(0)

    return result


def or_gate(a, b):
    result = []
    if a == 0 and b == 0:
        result.append(0)
    else:
        result.append(1)
    return result


def xor_gate(a, b):
    result = []
    if (a == 1 and b == 1) or (a == 0 and b == 0):
        result.append(0)
    else:
        result.append(1)
    return result

bin_set = [0, 1]

my_3_space = variations_generator(bin_set, 3)
my_2_space = variations_generator(bin_set, 2)

dist_emd = emd(p, q)
dist_dkl = kullback_leibner(p, q)


print("")
print("EMD(p,q) = ", dist_emd)
print("")
print("Kullback_Leibner(p,q) = ", dist_dkl)
print("")
print(my_3_space)
print("")
print(my_2_space)
print("")
print(and_gate(my_2_space))

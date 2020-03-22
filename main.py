

import numpy as np
import fractions
import numpy.linalg as ln
import sympy as sy
# setto il print di numpy per stampare le frazioni
np.set_printoptions(formatter={'all': lambda x:
                    str(fractions.Fraction(x).limit_denominator())})


A = [[2/3, -4/3, 2],
     [5/6, 4/3, -2],
     [5/6, -2/3, 0]]

B = [[2/3],
     [-2/3],
     [1/3]]

C = [0, -1, 1]


# converto le matrici in ndarray
A = np.matrix(A)
B = np.matrix(B)
C = np.matrix(C)

dimA = A.shape
dimB = B.shape
dimC = C.shape

# calcolo le matrici R e O
R = np.concatenate((B, A*B, A*A*B), axis=1)
O = np.concatenate((C, C*A, C*A*A), axis=0)

# calcolo la dimensione di XR e di XNO
dimXR = ln.matrix_rank(R)
dimXNO = 3 - ln.matrix_rank(O)


# trovo una base di XR e una base di XNO
XR = np.matrix(sy.Matrix(R).columnspace(), dtype=np.float64).T
XNO = np.matrix(sy.Matrix(O).nullspace(), dtype=np.float64).T

print("dimensione di XR = ", dimXR)
print("base di XR")
print(XR)

print("dimensione di XNO = ", dimXNO)
print("base di XNO")
print(XNO)

# calcolo la dimensione di XR + XNO
dimXR_plus_XNO = ln.matrix_rank(np.concatenate((XR, XNO), axis=1))

# calcolo la dimensione di XR intersecato XNO
dimXR_int_XNO = dimXR + dimXNO - dimXR_plus_XNO
print("dimensione di XR intersecato XNO = ", dimXR_int_XNO)

# trovo una base di XR intersecato XNO

# dichiaro i parametri per il sistema lineare
alpha = ()
for i in range(np.size(XR, 1)):
    alpha = alpha + (sy.symbols("alpha" + str(i)),)
beta = ()
for i in range(np.size(XNO, 1)):
    beta = beta + (sy.symbols("beta" + str(i)),)
param = (alpha + beta)
paramb = param

# costruisco la matrice del sistema con la base di XR concatenata
# alla base di XNO e al vettore nullo --> M = [XR|-XNO|0]
M = sy.Matrix(np.concatenate((XR, XNO*(-1), [[0], [0], [0]]), axis=1))

# risolvo il sistema lineare, che restituira' un numero di parametri pari
# alla dimensione del sottospazio XR intersecato XNO
(param,) = sy.linsolve(M, param)

# sostituisco a tutti i parametri 1
paramc = []
for i in range(len(param)):
    tmp = param[i]
    for j in range(len(paramb)):
        tmp = tmp.subs(paramb[j], 1)
    paramc.append([tmp])

# moltiplico la base di XR per il vettore contenente i parametri
XR_int_XNO = np.dot(XR, np.array((paramc[:dimXR]), dtype=np.float64))

print("base di XR intersecato XNO")
print(XR_int_XNO)

# costruisco T

# T sarà composta da:
#   - dimXR_int_XNO colonne di XR_int_XNO
#   - (dimXR - dimXR_int_XNO) colonne di XR
#   - (dimXNO - dimXR_int_XNO) colonne di XNO
#   - if dim(T) < 3: completamento a base

T = np.concatenate((XR_int_XNO, XR[:, dimXR - dimXR_int_XNO],
                    XNO[:, dimXNO - dimXR_int_XNO]), axis=1)

i = 0
while np.size(T, 1) < dimA[0]:
    T = np.concatenate((T, np.matrix(np.eye(dimA[0]))[:, i]), axis=1)
    i += 1

print("matrice di cambiamento di base T")
print(T)

# devo cabiare base ad A, B, C
Ad = T.I*A*T
print("la matrice A decomposta")
print(Ad)

Bd = T.I*B
print("la matrice B decomposta")
print(Bd)

Cd = C*T
print("la matrice C decomposta")
print(Cd)

# ora devo far vedere i diversi sistemi

# sottosistema raggiungibile
A_rag = Ad[0:dimXR, 0:(dimXR)]
B_rag = Bd[0:dimXR]
C_rag = Cd[:, 0:dimXR]

print("il sottosistema raggiungibile è:")
print("la matrice A del sottosistema raggiungibile")
print(A_rag)
print("la matrice B del sottosistema raggiungibile")
print(B_rag)
print("la matrice C del sottosistema raggiungibile")
print(C_rag)

# sottosistema osservabile
A_oss = Ad[dimXR_int_XNO:dimXR, 0:(dimXR)]
B_oss = Bd[0:(dimXR)]
C_oss = Cd[:, 0:(dimXR)]

print("il sottosistema osservabile è:")
print("la matrice A del sottosistema osservabile")
print(A_oss)
print("la matrice B del sottosistema osservabile")
print(B_oss)
print("la matrice C del sottosistema osservabile")
print(C_oss)

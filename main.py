

import numpy as np
import fractions
import numpy.linalg as ln
import sympy as sy

# setto il print di numpy per stampare le frazioni
np.set_printoptions(formatter={'all': lambda x:
                    str(fractions.Fraction(x).limit_denominator(1000))})


def continue_to_elaborate():
    while(input("you whant to continue? [y/n]") != "y"):
        pass
    print()


# A = [[2/3, -4/3, 2],
#     [5/6, 4/3, -2],
#     [5/6, -2/3, 0]]

# B = [[2/3],
#     [-2/3],
#     [1/3]]

# C = [0, -1, 1]

A = [[-1, -2/3, 2/3],
     [-1, 2/3, 1/3],
     [1/2, -1/3, -5/3]]

B = [[0],
     [-1],
     [0]]

C = [1/2, -1, 0]

# dim = 4
# A = np.random.random_sample((dim, dim))
# B = np.random.random_sample((dim, 1))
# C = np.random.random_sample((1, dim))

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
dimXNO = dimA[0] - ln.matrix_rank(O)


# trovo una base di XR

print("dimensione di XR = ", dimXR)
# inizializzo XR ad una colonna vuota nel caso in cui la dimensione sia 0
XR = np.zeros((dimA[0], 0))
if not dimXR == 0:
    XR = np.matrix(sy.Matrix(R).columnspace(), dtype=np.float64).T
    print("base di XR")
    print(XR)

print("-"*50)
continue_to_elaborate()


# e una base di XNO

print("dimensione di XNO = ", dimXNO)
# inizializzo XNO ad una colonna vuota nel caso in cui la dimensione sia 0
XNO = np.zeros((dimA[0], 0))
if not dimXNO == 0:
    XNO = np.matrix(sy.Matrix(O).nullspace(), dtype=np.float64).T
    print("base di XNO")
    print(XNO)

print("-"*50)
continue_to_elaborate()


# calcolo la dimensione di XR + XNO
dimXR_plus_XNO = ln.matrix_rank(np.concatenate((XR, XNO), axis=1))

# calcolo la dimensione di XR intersecato XNO
dimXR_int_XNO = dimXR + dimXNO - dimXR_plus_XNO
print("dimensione di XR intersecato XNO = ", dimXR_int_XNO)

# trovo una base di XR intersecato XNO
XR_int_XNO = np.zeros((dimA[0], 0))
if not dimXR_int_XNO == 0:
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

print("-"*50)
continue_to_elaborate()


# costruisco T

# T sarà composta da:
#   - dimXR_int_XNO colonne di XR_int_XNO
#   - (dimXR - dimXR_int_XNO) colonne di XR
#   - (dimXNO - dimXR_int_XNO) colonne di XNO
#   - if dim(T) < 3: completamento a base

# calcolo il numero di colonne di ogni sottosistema
n_col_A11 = dimXR_int_XNO
n_col_A22 = dimXR - dimXR_int_XNO
n_col_A33 = dimXNO - dimXR_int_XNO
n_col_A44 = dimA[0] - n_col_A11 - n_col_A22 - n_col_A33

# calcolo i limiti di ogni sottosistema per poter suddividere poi
# la matrice piu' facilmente
stop_A11 = n_col_A11
stop_A22 = n_col_A11 + n_col_A22
stop_A33 = n_col_A11 + n_col_A22 + n_col_A33
stop_A44 = n_col_A11 + n_col_A22 + n_col_A33 + n_col_A44

# print(stop_A11, stop_A22, stop_A33, stop_A44)

T = np.concatenate((XR_int_XNO, XR[:, 0:n_col_A22], XNO[:, 0:n_col_A33]),
                   axis=1)

# colplemento a base aggiungendo vettori della base canonica
for i in range(n_col_A44):
    T = np.concatenate((T, np.matrix(np.eye(dimA[0]))[:, i]), axis=1)


print("matrice di cambiamento di base T")
print(T)
print("-"*50)
continue_to_elaborate()


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

print("-"*50)
continue_to_elaborate()


# ora devo far vedere i diversi sistemi

# sottosistema raggiungibile
rag = np.r_[0:stop_A22]
A_rag = Ad[tuple(np.meshgrid(rag, rag))].T
B_rag = Bd[rag]
C_rag = Cd[:, rag]

print("il sottosistema raggiungibile è:")
print("la matrice A del sottosistema raggiungibile")
print(A_rag)
print("la matrice B del sottosistema raggiungibile")
print(B_rag)
print("la matrice C del sottosistema raggiungibile")
print(C_rag)
print("-"*50)
continue_to_elaborate()


# sottosistema osservabile

# seleziono le colonne necessarie
oss = np.r_[stop_A11:stop_A22, stop_A33:stop_A44]
A_oss = Ad[tuple(np.meshgrid(oss, oss))].T
B_oss = Bd[oss]
C_oss = Cd[:, oss]

print("il sottosistema osservabile è:")
print("la matrice A del sottosistema osservabile")
print(A_oss)
print("la matrice B del sottosistema osservabile")
print(B_oss)
print("la matrice C del sottosistema osservabile")
print(C_oss)
print("-"*50)
continue_to_elaborate()


# sottosistema non osservabile

# seleziono le colonne necessarie
n_oss = np.r_[0:stop_A11, stop_A22:stop_A33]
A_n_oss = Ad[tuple(np.meshgrid(n_oss, n_oss))].T
B_n_oss = Bd[n_oss]
C_n_oss = Cd[:, n_oss]

print("il sottosistema non osservabile è:")
print("la matrice A del sottosistema non osservabile")
print(A_n_oss)
print("la matrice B del sottosistema non osservabile")
print(B_n_oss)
print("la matrice C del sottosistema non osservabile")
print(C_n_oss)
print("-"*50)
continue_to_elaborate()


# sottosistema raggiungibile e osservabile

# seleziono le colonne necessarie
rag_oss = np.r_[stop_A11:stop_A22]
A_rag_oss = Ad[tuple(np.meshgrid(rag_oss, rag_oss))].T
B_rag_oss = Bd[rag_oss]
C_rag_oss = Cd[:, rag_oss]

print("il sottosistema raggiungibile e osservabile è:")
print("la matrice A del sottosistema raggiungibile e osservabile")
print(A_rag_oss)
print("la matrice B del sottosistema raggiungibile e osservabile")
print(B_rag_oss)
print("la matrice C del sottosistema raggiungibile e osservabile")
print(C_rag_oss)
print("-"*50)
continue_to_elaborate()


# secondo esercizio
print("secondo esercizio:")

# trovo gli autovalori e autovettori
eigvalue, eigvectors = ln.eig(A)
eigvectors = np.matrix(eigvectors)

print("gli autovalori sono:")
for i, l in enumerate(eigvalue):
    print("λ" + str(i) + " = " + str(fractions.Fraction(l)
                                     .limit_denominator(1000)))

print("-"*50)
continue_to_elaborate()


print("gli autovettori associati sono:")
for i in range(dimA[0]):
    print("autovettore associato all'autovalore λ" + str(i) + ":")
    print(eigvectors[:, i])

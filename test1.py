# #importing python libs
#
# import sympy as sym
#
# import numpy as np
# from math import pi
# import math
# import matplotlib.pyplot as plt
# from sympy import I, Matrix, symbols
# from sympy.physics.quantum import TensorProduct, Dagger
# import scipy.optimize
# import scipy.integrate
# import scipy.constants as const
#
# #import qutip
#
# from matplotlib.colors import Normalize as Norm
#
# import time
# #define some s pre/post operators
#
# def spre(m):
#     return TensorProduct(sym.eye(m.shape[0]),m)
#
# def spost(m):
#     return TensorProduct(m.T, sym.eye(m.shape[0]))
#
# def collapse(c):
#     tmp = Dagger(c)*c/2
#     return spre(c)*spost(Dagger(c))-spre(tmp)-spost(tmp)
#
#
# s13=Matrix([[0,0,1],[0,0,0],[0,0,0]])
# s23=Matrix([[0,0,0],[0,0,1],[0,0,0]])
# s12=Matrix([[0,1,0],[0,0,0],[0,0,0]])
#
# s31=s13.T
# s32=s23.T
# s21=s12.T
#
# s11 = s12*s21
# s22 = s21*s12
# s33 = s31*s13
#
# delo,delm=sym.symbols('delta_o delta_mu', real=True)
# #delao, delam =sym.symbols('delta_a_o delta_a_mu') #detunings between atom and cavity
# gamma13,gamma23,gamma2d,gamma3d,nbath,gammamu=sym.symbols('gamma_13 gamma_23 gamma_2d gamma_3d n_b gamma_mu', real=True, negative=False) #energy decay for atom levels
# Omega=sym.symbols('Omega', real=False, negative=False) #pump Rabi frequency
# rho11, rho12, rho13, rho21, rho22, rho23, rho31, rho32, rho33=sym.symbols('rho_11 rho_12 rho_13 rho_21 rho_22 rho_23 rho_31 rho_32 rho_33') #Density matrix elements
# a, b = sym.symbols('a b') #classical amplitudes of the optical and microwave fields
# #ar,ai=sym.symbols('a_r a_i', real=True)
# go, gm=sym.symbols('g_o, g_mu',real=False, negative=False) #coupling strengths for optical and microwave fields
# lam=sym.symbols('lambda')
#
# H=Omega*s32+gm*s21*b+go*s31*a
# H=H+Dagger(H)
# H=H+(delo)*s33+(delm)*s22
#
# np.savez('output_H_test',H=H)
#
#
# npzfile=np.load('output_H_test.npz')
#
# H_load=npzfile['H']
# print(H_load)
import time
print('honhonhon' + time.ctime())

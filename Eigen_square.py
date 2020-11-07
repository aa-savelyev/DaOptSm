# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Собственные числа #
#
# Пример с квадратом

# +
# Imports 
import sys
import warnings
import numpy as np
import numpy.linalg as LA

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm  # Colormaps
import seaborn as sns
# -

# Styles, fonts
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set_style('whitegrid')
matplotlib.rcParams['font.size'] = 13

# ---

# ## Собственные числа и собственные векторы ##
#
# Вектор имеет длину и направление.
# Общее влияние проеобразования $\mathbf{C}$ на вектор $\mathbf{x}$ &mdash; это сочетание вращения и растяжения.
# Например, оно изменяет  длину и направление вектора $\mathbf{x_1}$.
# Однако вектор $\mathbf{x_2}$ после преобразования изменяет только длину.
# Фактически, матрица $\mathbf{C}$ растягивает $\mathbf{x_2}$ в том же направлении и даёт вектор $\mathbf{t_2}$.
# Единственный способ изменить величину вектора без изменения его направления &mdash; умножить его на скаляр.
# Таким образом, для вектора $\mathbf{x_2}$ эффект умножения на $\mathbf{C}$ подобен умножению на скалярное число $\lambda$:
# $$ \mathbf{Cx_2} = \lambda \mathbf{x_2}. $$

# Но почему для нас важны собственные векторы?
# Как уже упоминалось ранее, собственный вектор превращает умножение матриц в умножение скаляров.
# Кроме того, они обладают еще некоторыми интересными свойствами.
# Воспользуемся модулем `numpy.linalg` и найдём собственные числа и собственные векторы матрицы $\mathbf{C}$.
# Далее нарисуем собственные векторы.

# +
C = np.array([[3, 2],
              [0, 2]])

lmbd, U = LA.eig(C)
CU = C @ U
print('lambda = ', np.round(lmbd, 4))
# print('u = \n', np.round(u, 4))
print('U = ')
np.disp(np.round(U, 4))
# -

# Теперь расммотрим квадрат.

origin = [[0,0], [0,0]] # origin point
l1x = np.linspace(-1, 1)
l1y = np.ones_like(l1x)
l2x = np.ones_like(l1x)
l2y = np.linspace(1, -1)
l3x = np.linspace(1, -1)
l3y = -1*np.ones_like(l1x)
l4x = -1*np.ones_like(l1x)
l4y = np.linspace(-1, 1)
L = np.vstack((np.hstack((l1x, l2x, l3x, l4x)), np.hstack((l1y, l2y, l3y, l4y))))

S = np.array([[-1,1], [1,1], [1,-1], [-1,-1], [-1,1]]).T

# +
S1 = LA.inv(U) @ S
S2 = np.diag(lmbd) @ S1
S3 = U @ S2
Sn = [S, S1, S2, S3]
Sn_str = ["$\mathbf{S}$", "$\mathbf{U^{-1}S}$",
         " $\mathbf{\Lambda U^{-1}S}$", "$\mathbf{U\Lambda U^{-1}S}$"]

U1 = LA.inv(U) @ U
U2 = np.diag(lmbd) @ U1
U3 = U @ U2
Un = [U, U1, U2, U3]

# +
fig, ax = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4)

for i, axi in enumerate(ax.flatten()):
    axi.plot(Sn[i][0,:], Sn[i][1,:], color='b')
    axi.quiver(*origin, Un[i][0,:], Un[i][1,:], color=['g'],
               width=0.012, angles='xy', scale_units='xy', scale=1)
    axi.set_xlabel('x', fontsize=14)
    axi.set_ylabel('y', fontsize=14)
    axi.set_xlim([-4, 4])
    axi.set_ylim([-4, 4])
    axi.set_aspect('equal')
    axi.grid(True)
    # axi.set_title("Original vectors")
    axi.axhline(y=0, color='k')
    axi.axvline(x=0, color='k')
    axi.text(*(Un[i].T[0]+[.1,.1]), "$\mathbf{u_1}$", fontsize=14)
    axi.text(*(Un[i].T[1]+[.1,.1]), "$\mathbf{u_2}$", fontsize=14)
    axi.text(1.4, -2, Sn_str[i], color='b', fontsize=14)

# +
u, d, vt = LA.svd(C)
v = vt.T
print('u = ')
np.disp(u)
print('d = ')
np.disp(d)
print('vt = ')
np.disp(vt)

S1 = vt @ S
S2 = np.diag(d) @ S1
S3 = u @ S2
# S2 = u @ np.diag(d) @ S1
# S3 = C @ S
Sn = [S, S1, S2, S3]
Sn_str = ["$\mathbf{x}$", "$\mathbf{V^Tx}$",
          "$\mathbf{DV^Tx}$", "$\mathbf{UDV^Tx}$"]

v1 = vt @ v
v2 = np.diag(d) @ v1
v3 = u @ v2
vn = [v, v1, v2, v3]

# +
fig, ax = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4)

for i, axi in enumerate(ax.flatten()):
    axi.plot(Sn[i][0,:], Sn[i][1,:], color='b')
    axi.quiver(*origin, vn[i][0,:], vn[i][1,:], color=['g'],
               width=0.01, angles='xy', scale_units='xy', scale=1)
    axi.quiver(*origin, u[0,:], u[1,:], color=['r'],
               width=0.01, angles='xy', scale_units='xy', scale=1)
    axi.set_xlabel('x', fontsize=14)
    axi.set_ylabel('y', fontsize=14)
    axi.set_xlim([-4, 4])
    axi.set_ylim([-4, 4])
    axi.set_aspect('equal')
    axi.grid(True)
    # axi.set_title("Original vectors")
    axi.axhline(y=0, color='k')
    axi.axvline(x=0, color='k')
#     axi.text(*(un[i].T[0]+[.1,.1]), "$\mathbf{u_1}$", fontsize=14)
#     axi.text(*(un[i].T[1]+[.1,.1]), "$\mathbf{u_2}$", fontsize=14)
    axi.text(1.5, -2, Sn_str[i], color='b', fontsize=14)
# -


U, d, Vt = LA.svd(C)
print(U)
print(d)
print(Vt)



# ---

# ## Литература ##
#
# 1. Воронцов К.В. [Математические методы обучения по прецендентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). 141 c.

# Versions used
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))
print('scipy: {}'.format(sp.__version__))



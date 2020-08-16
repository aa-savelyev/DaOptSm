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

# # Разложения матриц #
#
# Необходимы какие-то сведения из линейной алгебры.

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

# ## Линейные отображения и их матрицы ##
#
# Для понимания SVD нам необходимо сначала понять разложение матрицы по собственному значению (*спектарльное разложение?*).
#
# Можно представить матрицу $\mathbf{A}$ как преобразование, которое действует на вектор $\mathbf{x}$ путем умножения для получения нового вектора $\mathbf{Ax}$.
#
# Примером матрицы линейного отображения является матрица поворота:
# $$
#   \mathbf{A} = 
#   \begin{pmatrix}
#     \cos{\theta} & -\sin{\theta} \\
#     \sin{\theta} &  \cos{\theta}
#   \end{pmatrix}.
# $$
#
# Или матрица растяжения:
# $$
#   \mathbf{B} = 
#   \begin{pmatrix}
#     k & 0 \\
#     0 & 1
#   \end{pmatrix}.
# $$

# +
x = np.array([1,0]) # Original vector
theta = 30 * np.pi / 180 # degress in radian
A = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]]) # Rotation matrix for theta=30 degrees
B = np.array([[3,0],[0,1]]) # Stretching matrix

Ax = A @ x  # y1 is the rotated vector
Bx = B @ x  # y2 is the stretched vector

# Reshaping and storing both x and Ax in t1 to be plotted as vectors
t1 = np.concatenate([x.reshape(1,2), Ax.reshape(1,2)])
# Reshaping and storing both x and Bx in t2 to be plotted as vectors
t2 = np.concatenate([x.reshape(1,2), Bx.reshape(1,2)])
origin = [0], [0] # origin point

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,15))
plt.subplots_adjust(wspace=0.4)

# Plotting t1
ax1.quiver(*origin, t1[:,0], t1[:,1], color=['g', cm.tab10(3)],
           width=0.013, angles='xy', scale_units='xy', scale=1)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('y', fontsize=14)
ax1.set_xlim([-0.5,1.5])
ax1.set_ylim([-0.5,1])
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_title("Rotation transform")
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.text(1, 0.1, "$\mathbf{x}$", fontsize=16)
ax1.text(0.8, 0.6, "$\mathbf{Ax}$", fontsize=16)

# Plotting t2
ax2.quiver(*origin, t2[:,0], t2[:,1], color=['g', cm.tab10(3)],
           width=0.013, angles='xy', scale_units='xy', scale=1)
ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('y', fontsize=14)
ax2.set_xlim([-0.5,3.5])
ax2.set_ylim([-1.5,1.5])
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_title("Stretching transform")
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')
ax2.text(1, 0.2, "$\mathbf{x}$", fontsize=16)
ax2.text(3, 0.2, "$\mathbf{Bx}$", fontsize=16)

plt.show()
# -

# Теперь попробуем другую матрицу:
# $$
#   \mathbf{C} = 
#   \begin{pmatrix}
#     3 & 2 \\
#     0 & 2
#   \end{pmatrix}.
# $$
#
# На рисунке показано преобразование множества точек $\mathbf{x}$ (окружность) и, в частности, двух векторов $\mathbf{x_1}$ и $\mathbf{x_2}$.
# Начальные векторы $\mathbf{x}$ с левой стороны образуют окружность, матрица преобразования изменяет эту окружность и превращает её в эллипс.
# Векторы выборки $\mathbf{x_1}$ и $\mathbf{x_2}$ в окружности преобразуются в $\mathbf{t_1}$ и $\mathbf{t_2}$ соответственно.

# +
# Creating the vectors for a circle and storing them in x
r = 1
phi = np.linspace(0, 2*np.pi, 100)
xi = r*np.cos(phi)
yi = r*np.sin(phi)
x = np.vstack((xi, yi))

C = np.array([[3, 2],
              [0, 2]])
t = C @ x  # Vectors in t are the transformed vectors of x

# getting a sample vector from x
xs_1 = x[:, 15]
xs_2 = x[:, 0]
ts_1 = C @ xs_1
ts_2 = C @ xs_2

# +
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,15))

plt.subplots_adjust(wspace=0.4)
delta1 = (0.1, 0.2)
delta2 = (-0.4, 0.3)

# Plotting x
ax1.plot(x[0,:], x[1,:], color='b')
ax1.quiver(*origin, xs_1[0], xs_1[1], color=['g'], width=0.012, angles='xy', scale_units='xy', scale=1)
ax1.quiver(*origin, xs_2[0], xs_2[1], color=['g'], width=0.012, angles='xy', scale_units='xy', scale=1)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('y', fontsize=14)
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_axisbelow(True)
ax1.set_title("Original vectors")
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.text(0.6, 1.0, "$\mathbf{x_1}$", fontsize=14)
ax1.text(1.1, 0.2, "$\mathbf{x_2}$", fontsize=14)
ax1.text(0.3, -1.3, "$\mathbf{x}$", color='b', fontsize=14)

# Plotting t
ax2.plot(t[0, :], t[1, :], color='b')
ax2.quiver(*origin, ts_1[0], ts_1[1], color=cm.tab10(3), width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.quiver(*origin, ts_2[0], ts_2[1], color=cm.tab10(3), width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.quiver(*origin, xs_1[0], xs_1[1], color=['g'], width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.quiver(*origin, xs_2[0], xs_2[1], color=['g'], width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('y', fontsize=14)
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_axisbelow(True)
ax2.set_title("New vectors after transformation")
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')
ax2.text(3.3, 1.8, "$\mathbf{t_1}$", fontsize=14)
ax2.text(3.0, -0.5, "$\mathbf{t_2}$", fontsize=14)
ax2.text(1.0, -1.7, "$\mathbf{Cx}$", color='b', fontsize=14)

# plt.savefig('1.png', dpi=300, bbox_inches='tight')
plt.show()
# -

# На рисунке показано преобразование множества точек $\mathbf{x}$ (окружность) и, в частности, двух векторов $\mathbf{x_1}$ и $\mathbf{x_2}$.
# Начальные векторы $\mathbf{x}$ с левой стороны образуют окружность, матрица преобразования изменяет эту окружность и превращает её в эллипс.
# Векторы выборки $\mathbf{x_1}$ и $\mathbf{x_2}$ в окружности преобразуются в $\mathbf{t_1}$ и $\mathbf{t_2}$ соответственно.

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

lam, u = LA.eig(C)
lu = C @ u
print('lam = ', np.round(lam, 4))
print('u = \n', np.round(u, 4))

# +
t = C @ x  # Vectors in t are the transformed vectors of x

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,15))
plt.subplots_adjust(wspace=0.4)

# Plotting x
ax1.plot(x[0,:], x[1,:], color='b')
ax1.quiver(*origin, u[0,:], u[1,:], color=['g'], width=0.012, angles='xy', scale_units='xy', scale=1)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('y', fontsize=14)
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title("Original picture")
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.text(1, 0.3, "$\mathbf{u_1}$", fontsize=14)
ax1.text(-1.6, 0.5, "$\mathbf{u_2}$", fontsize=14)
ax1.text(0.3, -1.3, "$\mathbf{x}$", color='b', fontsize=14)

# Plotting t
ax2.plot(t[0, :], t[1, :], color='b')
ax2.quiver(*origin, lu[0,:], lu[1,:], color=cm.tab10(3), width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.quiver(*origin, u[0,:], u[1,:], color=['g'], width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('y', fontsize=14)
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_title("After transformation")
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')
# ax2.text(1, 0.3, "$\mathbf{u_1}$", fontsize=14)
# ax2.text(-0.8, 0.6, "$\mathbf{u_2}$", fontsize=14)
ax2.text(2.2, 0.3, "$\lambda_1 \mathbf{u_1}$", fontsize=14)
ax2.text(-2.6, 1.2, "$\lambda_2 \mathbf{u_2}$", fontsize=14)
ax2.text(1.0, -1.7, "$\mathbf{Cx}$", color='b', fontsize=14)

plt.show()

# +
x1 = LA.inv(u) @ x
x2 = np.diag(lam) @ x1
x3 = u @ x2
xn = [x, x1, x2, x3]
xn_str = ["$\mathbf{x}$", "$\mathbf{U^{-1}x}$",
          "$\mathbf{\Lambda U^{-1}x}$", "$\mathbf{U\Lambda U^{-1}x}$"]

u1 = LA.inv(u) @ u
u2 = np.diag(lam) @ u1
u3 = u @ u2
un = [u, u1, u2, u3]

# +
fig, ax = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4)

for i, axi in enumerate(ax.flatten()):
    axi.plot(xn[i][0,:], xn[i][1,:], color='b')
    axi.quiver(*origin, un[i][0,:], un[i][1,:], color=['g'],
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
    axi.text(*(un[i].T[0]+[.1,.1]), "$\mathbf{u_1}$", fontsize=14)
    axi.text(*(un[i].T[1]+[.1,.1]), "$\mathbf{u_2}$", fontsize=14)
    axi.text(1, -2, xn_str[i], color='b', fontsize=14)
# -

# Теперь расммотрим квадрат.

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

S1 = LA.inv(u) @ S
S2 = np.diag(lam) @ S1
S3 = u @ S2
Sn = [S, S1, S2, S3]
Sn_str = ["$\mathbf{S}$", "$\mathbf{U^{-1}S}$",
         " $\mathbf{\Lambda U^{-1}S}$", "$\mathbf{U\Lambda U^{-1}S}$"]

# +
fig, ax = plt.subplots(2, 2, figsize=(10,10))
plt.subplots_adjust(wspace=0.4)

for i, axi in enumerate(ax.flatten()):
    axi.plot(Sn[i][0,:], Sn[i][1,:], color='b')
    axi.quiver(*origin, un[i][0,:], un[i][1,:], color=['g'],
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
    axi.text(*(un[i].T[0]+[.1,.1]), "$\mathbf{u_1}$", fontsize=14)
    axi.text(*(un[i].T[1]+[.1,.1]), "$\mathbf{u_2}$", fontsize=14)
    axi.text(1.4, -2, Sn_str[i], color='b', fontsize=14)
# -

CC = np.array([[2, 1],
               [2, 2]])

# +
u, d, vt = LA.svd(C)
u[:,0] *= -1
v = vt.T
print('u = ', np.round(u, 4))
print('d = ', np.round(d, 4))
print('vt = ', np.round(vt, 4))

S1 = vt @ S
S2 = np.diag(d) @ S1
S3 = u @ S2
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





# ## Симметричные матрицы ##
#
# Рассмотрим симметричную матрицу:
# $$
#   \mathbf{D} = 
#   \begin{pmatrix}
#     3 & 1 \\
#     1 & 2
#   \end{pmatrix}.
# $$
#
# Найдём собст

D = np.array([[3, 1],
              [1, 2]])
lam, u = LA.eig(D)
lu = D @ u
print('lam = ', np.round(lam, 4))
print('u = \n', np.round(u, 4))

# +
t = D @ x   # Vectors in t are the transformed vectors of x
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,15))

plt.subplots_adjust(wspace=0.4)

# Plotting x
ax1.plot(x[0,:], x[1,:], color='b')
ax1.quiver(*origin, u[0,:], u[1,:], color=['g'], width=0.012, angles='xy', scale_units='xy', scale=1)
ax1.set_xlabel('x', fontsize=14)
ax1.set_ylabel('y', fontsize=14)
ax1.set_xlim([-4,4])
ax1.set_ylim([-4,4])
ax1.set_aspect('equal')
ax1.grid(True)
ax1.set_title("Original vectors")
ax1.axhline(y=0, color='k')
ax1.axvline(x=0, color='k')
ax1.text(1, 0.3, "$\mathbf{u_1}$", fontsize=14)
ax1.text(-1.2, 1.0, "$\mathbf{u_2}$", fontsize=14)
ax1.text(0.3, -1.3, "$\mathbf{x}$", color='b', fontsize=14)

# Plotting t
ax2.plot(t[0, :], t[1, :], color='b')
ax2.quiver(*origin, lu[0,:], lu[1,:], color=cm.tab10(3), width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.quiver(*origin, u[0,:], u[1,:], color=cm.tab10(2), width=0.012, angles='xy', scale_units='xy', scale=1)
ax2.set_xlabel('x', fontsize=14)
ax2.set_ylabel('y', fontsize=14)
ax2.set_xlim([-4,4])
ax2.set_ylim([-4,4])
ax2.set_aspect('equal')
ax2.grid(True)
ax2.set_title("New vectors after transformation")
ax2.axhline(y=0, color='k')
ax2.axvline(x=0, color='k')
# ax2.text(1, 0.3, "$\mathbf{u_1}$", fontsize=14)
# ax2.text(-1.2, 0.5, "$\mathbf{u_2}$", fontsize=14)
ax2.text(2.7, 2.3, "$\lambda_1 \mathbf{u_1}$", fontsize=14)
ax2.text(-2.0, 1.4, "$\lambda_2 \mathbf{u_2}$", fontsize=14)
ax2.text(0.9, -1.5, "$\mathbf{Dx}$", color='b', fontsize=14)

plt.show()
# -

# На этот раз у собственных векторов есть интересное свойство.
# Мы видим, что собственные векторы находятся вдоль главных осей эллипса.
# Таким образом, матрица $\mathbf{D}$ преобразует начальную окружность, растягивая её вдоль собственных векторов $\mathbf{u_1}$ и $\mathbf{u_2}$ в $\lambda_1$ и $\lambda_2$ раз соотвественно.
# Если абсолютное значение собственного значения больше 1, то вдоль него происходит растяжение, а если меньше &mdash; сжитие.
# Отрицательные собственные значения соответствуют зеркальному отражению.













U, d, Vt = LA.svd(C)
print(U)
print(d)
print(Vt)





# +
def pol(a, x):
    return np.sum([a[i]*x**i for i in range(len(a))], axis=0)

def sin(a, x):
    return np.sin(a*x)

def xsin(a, x):
    return x*np.sin(a*x)


# +
# Define the data
np.random.seed(123)
Ns = 10 # Number of samples
# x_lim = np.array([0, 1])
x_lim = np.array([-1, 1])

# Underlying functional relation
fun = pol
a = [2, 3]
label = f'{a[0]} + {a[1]}x'
# fun = sin
# a = 5
# label = f'sin({a}x)'

# Noise
e_std = 0.5  # Standard deviation of the noise
err = e_std * np.random.randn(Ns)  # Noise

# Features and output
x = np.random.uniform(x_lim[0], x_lim[1], Ns)  # Independent variable x
# x = np.linspace(x_lim[0], x_lim[1], Ns)  # Independent variable x
y = fun(a, x) + err  # Dependent variable

# Show data
X = np.linspace(x_lim[0], x_lim[1], 100)
plt.figure(figsize=(8, 5))
plt.title('Noisy data samples from linear line')
plt.plot(x, y, 'o', ms=4, label='data: (x, y)')
plt.plot(X, fun(a, X), 'k--', label=label)
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend()
plt.show()
# -

# ### Полиномиальная регрессия ###

# +
# The number of fetures
Nf = 7
# Stack X with ones to be fitted by vectorized methods such as OLS and gradient descent
F = np.ones_like(x)
for i in range(1, Nf):
    F = np.vstack((F, x**i))
F = F.T
# print(F)
    
# Fit parameters with OLS
Alpha = np.linalg.inv(F.T @ F) @ F.T @ y
print(Alpha)

# Function representing fitted line
f = lambda x: sum([Alpha[i]*x**i for i in range(Nf)])
# -

# Show OLS fitted line
plt.figure(figsize=(8, 5))
plt.title('Ordinary least squares regression fit')
plt.plot(x, y, 'o', ms=4, label='data: (x, y)')
plt.plot(X, fun(a, X), 'k--', label=label)
plt.plot(X, f(X), '-', label='OLS')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-2, 6)
plt.show()

# ---

# ## Сингулярное разложение ##
#
# Произвольную матрицу $F$ размерностью $l \times n$ ранга $n$ можно представить в виде сингулярного разложения (singular value decomposition, SVD)
# $$ F = VDU^\top. $$
#
# **Свойства сингулярного разложения:**
# 1. матрица $D$ размером $n \times n$ диагональна, $ D = \mathrm{diag}\left( \sqrt{\lambda_1}, \ldots, \sqrt{\lambda_n}\right) $, где $\lambda_1, \ldots, \lambda_n$ &mdash; общие ненулевые собственные значения матриц $F^\top F$ и $FF^\top$;
# 2. матрица $V = (v_1, \ldots, v_n)$ размером $l \times n$ ортогональна, $V^\top V = I_n$, столбцы $v_j$ являются собственными векторами матрицы $FF^\top$, соответствующими $\lambda_1, \ldots, \lambda_n$;
# 3. матрица $U = (u_1, \ldots, u_n)$ размером $n \times n$ ортогональна, $U^\top U = I_n$, столбцы $u_j$ являются собственными векторами матрицы $F^\top F$, соответствующими $\lambda_1, \ldots, \lambda_n$.
#
# Имея сингулярное разложение, легко записать
#  - псевдообратную матрицу:
# $$ F^{+} = (UDV^\top VDU^\top)^{-1}UDV^\top = UD^{-1}V^\top = \sum_{j=1}^n \frac{1}{\sqrt{\lambda_j} }u_j v_j^\top;  \label{eq:psevdo}\tag{1} $$
#  
#  - вектор МНК-решения:
# $$ \alpha^* = F^{+} y  = UD^{-1}V^\top y = \sum_{j=1}^n \frac{1}{\sqrt{\lambda_j}}u_j (v_j^\top y);  \label{eq:alpha-res}\tag{2} $$
#  
#  - вектор $F\alpha^*$ &mdash; МНК-аппроксимацию целевого вектора $y$:
# $$ F\alpha^* = P_F y = (VDU^\top)UD^{-1}V^\top y = VV^\top y = \sum_{j=1}^n v_j (v_j^\top y);  \label{eq:F-alpha-res}\tag{3} $$
#  
#  - норму вектора коэффициентов:
# $$ \Vert \alpha^* \Vert^2 = y^\top VD^{-1}U^\top UD^{-1}V^\top y = y^\top VD^{-2}V^\top y = \sum_{j=1}^n \frac{1}{\lambda_j} (v_j^\top y)^2.  \label{eq:alpha-res-norm}\tag{4} $$

# +
V, d, Ut = np.linalg.svd(F, full_matrices=False)
# print(V)
print(f'd = {d}')
# print(Ut.T)

mu = (d[0]/d[-1])**2
print(f'число обусловленности mu = {mu}')

# +
# Fit parameters with SVD
Alpha_svd = sum([1/d[i] * Ut[i] * (V.T[i] @ y) for i in range(Nf)])
print(Alpha_svd)

# Function representing fitted line
f_svd = lambda x: sum([Alpha_svd[i]*x**i for i in range(Nf)])
# -

# Show OLS fitted line
plt.figure(figsize=(8, 5))
plt.title('Ordinary least squares regression fit')
plt.plot(x, y, 'o', ms=4, label='data: (x, y)')
plt.plot(X, fun(a, X), 'k--', label=label)
plt.plot(X, f(X), '-', label='OLS')
plt.plot(X, f_svd(X), ':', label='OLS-svd')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-2, 6)
plt.show()

# ---

# ## Проблема мультиколлинеарности ##
#
# Если ковариационная матрица $\Sigma = F^\top F$ имеет неполный ранг, то её обращение невозможно.
# Тогда приходится отбрасывать линейно зависимые признаки или применять описанные ниже методы &mdash; регуляризацию или метод главных компонент.
# На практике чаще встречается проблема *мультиколлинеарности* &mdash; когда матрица $\Sigma$ имеет полный ранг, но близка к некоторой матрице неполного ранга.
# Тогда говорят, что $\Sigma$ &mdash; матрица неполного псевдоранга или что она плохо обусловлена.
# Геометрически это означает, что объекты выборки сосредоточены вблизи линейного подпространства меньшей размерности $m < n$.
# Признаком мультиколлинеарности является наличие у матрицы $\Sigma$ собственных значений, близких к нулю.
#
# **Определение.** *Число обусловленности* матрицы $\Sigma$ есть
# $$ \mu(\Sigma) = \left\Vert \Sigma \right\Vert \Vert \Sigma^{-1} \Vert = \frac{\underset{u:\, \Vert u\Vert = 1}{\max} \Vert \Sigma u \Vert}{\underset{u:\, \Vert u\Vert = 1}{\min} \Vert \Sigma u \Vert} = \frac{\lambda_{\max}}{\lambda_{\min}}, $$
#
# где $\lambda_{\max}$ и $\lambda_{\min}$ &mdash; максимальное и минимальное собственные значения матрицы $\Sigma$.
#
# Матрица считается плохо обусловленной, если $\mu(\Sigma) \gtrsim 10^2 \div 10^4$.
# Обращение такой матрицы численно неустойчиво.
# При умножении обратной матрицы на вектор, $z = \Sigma^{-1}u$, относительная погрешность усиливается в $\mu(\Sigma)$ раз:
# $$ \frac{\Vert \delta z \Vert}{\Vert z \Vert} \le \mu(\Sigma) \frac{\Vert \delta u \Vert}{\Vert u \Vert}.$$
#
# Именно это и происходит с МНК-решением в случае плохой обусловленности.
# В формуле ([4](#mjx-eqn-eq:alpha-res-norm)) близкие к нулю собственные значения оказываются в знаменателе, в результате увеличивается разброс коэффициентов $\alpha^*$, появляются большие по абсолютной величине положительные и отрицательные коэффициенты.
# МНК-решение становится неустойчивым &mdash; малые погрешности измерения признаков или ответов
# у обучающих объектов могут существенно повлиять на вектор решения $\alpha^*$, а погрешности измерения признаков у тестового объекта $x$ &mdash; на значения функции регрессии $g(x, \alpha^*)$.
# Мультиколлинеарность влечёт не только неустойчивость и переобучение, но и неинтерпретируемость коэффициентов, так как по абсолютной величине коэффициента $\alpha_j$ становится невозможно судить о степени важности признака $f_j$.
# Проблема мультиколленеарности никак не проявляется на обучающих данных: вектор $F\alpha^*$ не зависит от собственных значений $\lambda$ (см. формулу ([3](#mjx-eqn-eq:F-alpha-res))).

# ---

# ## Гребневая регрессия ##
#
# Для решения проблемы мультиколлинеарности добавим к функционалу $Q$ регуляризатор, штрафующий большие значения нормы вектора весов $\Vert \alpha \Vert$:
#
# $$ Q_\tau(\alpha) = \Vert F\alpha - y \Vert^2 + \tau \Vert \alpha \Vert^2, $$
#
# где $\tau$ &mdash; неотрицательный параметр.
# В случае мультиколлинеарности имеется бесконечно много векторов $\alpha$, доставляющих функционалу $Q$ значения, близкие к минимальному.
# Штрафное слагаемое выполняет роль регуляризатора, благодаря которому среди них выбирается решение с минимальной нормой.
# Приравнивая нулю производную $Q_\tau (\alpha)$ по параметру $\alpha$, находим:
#
# $$ \alpha_\tau^\ast = (F^\top F + \tau I_n)^{-1} F^\top y. $$
#
# Таким образом, перед обращением матрицы к ней добавляется &laquo;гребень&raquo; &mdash; диагональная матрица $\tau I_n$.
# Отсюда и название метода &mdash; гребневая регрессия (ridge regression).
# При этом все её собственные значения увеличиваются на $\tau$ , а собственные векторы не изменяются.
# В результате матрица становится хорошо обусловленной, оставаясь в то же время &laquo;похожей&raquo; на исходную.
#
# Выразим регуляризованное МНК-решение через сингулярное разложение:
# $$ \alpha_\tau^* = (UD^2 U^\top + \tau I_n)^{-1} UDV^\top y = U(D^2 + \tau I_n)^{-1} DV^\top y = \sum_{j=1}^n \frac{\sqrt{\lambda_j}}{\lambda_j + \tau} u_j (v_j^\top y).  \label{eq:alpha-tau-res}\tag{5} $$
#
# В формуле \eqref{eq:alpha-tau-res} используется так называемое &laquo;проталкивающее равенство&raquo; ([push-through identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity)): $A(I + BA)^{-1} = (I + AB)^{-1}A$.
#
# Теперь найдём регуляризованную МНК-аппроксимацию целевого вектора $y$:
# $$ F\alpha_\tau^\ast = VDU^\top \alpha_\tau^* = V \mathrm{diag}\left( \frac{\lambda_j}{\lambda_j + \tau} \right) V^\top y = \sum_{j=1}^n \frac{\lambda_j}{\lambda_j + \tau} v_j (v_j^\top y).  \label{eq:F-alpha-tau-res}\tag{6} $$
#
# Как и прежде в ([3](#mjx-eqn-eq:F-alpha-res)), МНК-аппроксимация представляется в виде разложения целевого вектора $y$ по базису собственных векторов матрицы $FF^\top$.
# Только теперь проекции на собственные векторы сокращаются, умножаясь на $\frac{\lambda_j}{\lambda_j + \tau} \in (0, 1)$. В сравнении с ([4](#mjx-eqn-eq:alpha-res-norm)) уменьшается и норма вектора коэффициентов:
#
# $$ \Vert \alpha_\tau^{\ast} \Vert^2 = \Vert U(D^2 + \tau I_n)^{-1} DV^\top y \Vert^2  = \sum_{j=1}^n \frac{\lambda_j}{(\lambda_j + \tau)^2} (v_j^\top y)^2 < \sum_{j=1}^n \frac{1}{\lambda_j} (v_j^\top y)^2 = \Vert \alpha^{\ast} \Vert^2. \label{eq:alpha-tau-es-norm}\tag{7} $$
#
#
# ### Выбор константы регуляризации ###
#
# Из формулы \eqref{eq:alpha-tau-res} видно, что при $\tau \to 0$ регуляризованное решение стремится к МНК-решению: $\alpha_\tau^\ast \to \alpha^\ast$.
# При $\tau \to \infty$ чрезмерная регуляризации приводит к вырожденному решению: $\alpha^\ast_\tau \to 0$.
# Оба крайних случая нежелательны, поэтому оптимальным является некоторое промежуточное значение $\tau^\ast$.
# Для его нахождения можно применять, например, скользящий контроль.
#
# Известна практическая рекомендация брать $\tau$ в отрезке [0.1, 0.4], если столбцы матрицы $F$ заранее
# стандартизованы (центрированы и нормированы).
# Ещё одна эвристика &mdash; выбрать $\tau$ так, чтобы число обусловленности приняло заданное не слишком большое значение: $M_0 = \mu(F^\top F + \tau I_n) = \frac{\lambda_\max + \tau}{\lambda_\min + \tau}$, откуда следует рекомендация $\tau^\ast \approx \lambda_\max/M_0$.

# +
# Fit parameters with ridge regression
M_0 = 1e2    # desired condition number
tau = max(d)**2/M_0
print(f'd = {d}')
print(f'tau = {tau}')
Alpha_r = sum([d[i]/(d[i]**2+tau) * Ut[i] * (V.T[i] @ y) for i in range(Nf)])
print(f'Alpha_r = {Alpha_r}')

# Function representing fitted line
ridge = lambda x: sum([Alpha_r[i]*x**i for i in range(Nf)])
# -

# Show OLS fitted line
plt.figure(figsize=(8, 5))
plt.title('OLS vs ridge regression')
plt.plot(x, y, 'o', ms=4, label='data: (x, y)')
plt.plot(X, fun(a, X), 'k--', label=label)
plt.plot(X, f(X), '-', label='OLS')
plt.plot(X, ridge(X), '-', label='ridge')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-2, 6)
plt.show()

# ---

# ## Лассо Тибширани ##
#
# Ещё один метод регуляризации внешне похож на гребневую регрессию, но приводит к качественно иному поведению вектора коэффициентов.
# Вместо добавления штрафного слагаемого к функционалу качества вводится ограничение-неравенство, запрещающее слишком большие абсолютные значения коэффициентов:
#
# $$
# \left\{
# \begin{align}
#     & Q(\alpha) = \Vert F\alpha - y \Vert^2 \to \min_\alpha \\
#     & \sum\limits_{j=1}^n |\alpha_j| \le \chi
# \end{align}
# \right.,
# \label{eq:lasso}\tag{8}
# $$
# где $\chi$ &mdash; параметр регуляризации.
# При больших значениях $\chi$ ограничение \eqref{eq:lasso} становится строгим неравенством, и решение совпадает с МНК-решением.
# Чем меньше $\chi$, тем больше коэффициентов $\alpha_j$ обнуляются.
# Происходит отбор (селекция) признаков, поэтому параметр $\chi$ называют ещё *селективностью*.
# Образно говоря, параметр $\chi$ зажимает вектор коэффициентов, лишая его избыточных степеней свободы.
# Отсюда и название метода &mdash; *лассо* (LASSO, least absolute shrinkage and selection operator). 

import scipy as sp
from scipy.optimize import minimize
from scipy.optimize import NonlinearConstraint


# +
def Q_obj(a):
    '''Q - objective function'''
    return np.linalg.norm((F @ a - y))**2

def constr(a):
    '''Constraint'''
    return np.sum(abs(a))

def solve_lasso(Q_obj, constr, chi):
    '''solve LASSO optimization task'''
    nonlinear_constraint = NonlinearConstraint(constr, 0., chi)
    N_ms = 10   # multistart
    res = []
    for i in range(N_ms):
    # Alpha_0 = np.zeros(Nf)    # initial approximation
        Alpha_0 = 10*np.random.rand(Nf) - 5
        res.append(minimize(Q_obj, Alpha_0, method='SLSQP', constraints=nonlinear_constraint))
    argmin = np.argmin([item.fun for item in res])
    return res[argmin]


# -

# solve LASSO optimization task
chi = 5    # the max constraint for the decision vector
res = solve_lasso(Q_obj, constr, chi)
Alpha_l = res.x
lasso = lambda x: sum([Alpha_l[i]*x**i for i in range(Nf)])

# Show OLS fitted line
plt.figure(figsize=(8, 5))
plt.title('OLS, ridge and LASSO regression fits')
plt.plot(x, y, 'o', ms=4, label='data: (x, y)')
plt.plot(X, fun(a, X), 'k-', label=label)
plt.plot(X, f(X), '-', label='f(x)')
plt.plot(X, ridge(X), '-', label='ridge')
plt.plot(X, lasso(X), '-', label='lasso')
plt.legend()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.ylim(-2, 6)
plt.show()

print(f'OLS:   a = {Alpha}  \n norm(a) = {constr(Alpha)},   Q = {Q_obj(Alpha_svd)}\n')
print(f'ridge: a = {Alpha_r}\n norm(a) = {constr(Alpha_r)}, Q = {Q_obj(Alpha_r)}\n')
print(f'LASSO: a = {Alpha_l}\n norm(a) = {constr(Alpha_l)}, Q = {Q_obj(Alpha_l)}\n')

# ---

# ## Сравнение лассо и гребневой регрессии ##
#
# Оба метода успешно решают проблему мультиколлинеарности.
# Гребневая регрессия использует все признаки, стараясь &laquo;выжать максимум&raquo; из имеющейся информации. Лассо производит отбор признаков, что предпочтительнее, если среди признаков есть шумовые или измерения признаков связаны с ощутимыми затратами.
#
# Ниже приводится сравнение гребневой регресии и лассо.
# Ослабление регуляризации (рост параметров $\sigma = 1/\tau$ и $\chi$) ведёт к уменьшению ошибки на обучении и увеличению нормы вектора коэффициентов.
# При этом ошибка на контроле в какой-то момент проходит через минимум, и далее только возрастает &mdash; это и есть переобучение (проверить самостоятельно).

eps = 1e-2
Sigma = np.arange(eps, 1e3*eps, eps)
AA = np.empty((len(Sigma), len(d)))
for i, sigma in enumerate(Sigma):
    AA[i] = sum([d[i]/(d[i]**2+1/sigma) * Ut[i] * (V.T[i] @ y) for i in range(Nf)])

# +
# Show OLS fitted line
plt.figure(figsize=(8, 5))
plt.title('Ridge regression')

for i, aa in enumerate(AA.T):
    plt.plot(Sigma, aa, '-', label=f'$x^{i}$')
plt.xlabel('$\sigma$')
plt.ylabel(r'$\alpha$')
# plt.ylim((1, 6))
plt.legend(loc=2)
plt.show()

# +
Chi = np.arange(1, 11, 1)
BB = np.empty((len(Chi), len(d)))

for i, chi in enumerate(Chi):
    res = solve_lasso(Q_obj, constr, chi)
    BB[i] = res.x
# print(res) 

# +
# Show OLS fitted line
plt.figure(figsize=(8, 5))
plt.title('Lasso')

for i, bb in enumerate(BB.T):
    plt.plot(Chi, bb, '-', label=f'$x^{i}$')
plt.xlabel(r'$\chi$')
plt.ylabel(r'$\alpha$')
# plt.ylim((1, 6))
plt.legend(loc=2)
plt.show()
# -



# ---

# ### Проверка сингулярного разложения

# +
A1 = np.dot(F, F.T)
w1, V1 = LA.eig(A1)
ind = np.argsort(w1)[::-1]

for i, v in enumerate(V.T):
    print('SVD:')
    print('l = ', d[i]**2)
#     print('s = ', v)
#     print('Ax = ', np.dot(A1, v))
#     print('lx = ', d[i]**2 * v)
    
    j = ind[i]
    v1 = V1.T[j]
    print('EIG:')
    print('l = ', w1[j])
#     print('s = ', v1)
#     print('Ax = ', np.dot(A1, v1))
#     print('lx = ', w1[j]*v)
    print()

# +
A2 = np.dot(F.T, F)

# print(A1)
w1, V1 = LA.eig(A1)
print(w1)
print(V1)
# print(np.linalg.matrix_rank(V1))
# print(np.linalg.det(V1))
# for v in v1:
#     print(np.linalg.norm(v))

# print(A2)
w2, V2 = LA.eig(A2)
print(w2)
print(V2)
# print(np.linalg.matrix_rank(V2))
# -

for i, v in enumerate(V1.T):
    print(np.dot(A1, v))
    print(w1[i]*v)

# D = np.vstack([np.diag(d), np.zeros((N_obj-len(s), N_features))])
D = np.diag(d)
print(D)
# display(np.dot(np.dot(V, D), U.T) - F)

# ## Метод главных компонент ##

G = np.dot(V, D)
display(G)
L = np.dot(G.T, G)
display(L)
display(L.diagonal())



# ## Литература ##
#
# 1. Воронцов К.В. [Математические методы обучения по прецендентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). 141 c.

# Versions used
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(sns.__version__))
print('scipy: {}'.format(sp.__version__))



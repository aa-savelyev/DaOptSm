# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Суррогатное моделирование


# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Одномерный-случай" data-toc-modified-id="Одномерный-случай-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Одномерный случай</a></span><ul class="toc-item"><li><span><a href="#Данные-без-шума" data-toc-modified-id="Данные-без-шума-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Данные без шума</a></span></li><li><span><a href="#Данные-с-шумом" data-toc-modified-id="Данные-с-шумом-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Данные с шумом</a></span></li></ul></li><li><span><a href="#Двумерный-случай" data-toc-modified-id="Двумерный-случай-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Двумерный случай</a></span><ul class="toc-item"><li><span><a href="#Данные-без-шума" data-toc-modified-id="Данные-без-шума-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Данные без шума</a></span></li><li><span><a href="#Данные-с-шумом" data-toc-modified-id="Данные-с-шумом-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Данные с шумом</a></span></li></ul></li><li><span><a href="#Задание" data-toc-modified-id="Задание-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Задание</a></span></li></ul></div>
# -

# Imports
import numpy as np
import matplotlib.pyplot as plt

# +
# Styles
import matplotlib
matplotlib.rcParams['font.size'] = 14
cm = plt.cm.tab10  # Colormap

import seaborn
from IPython.display import Image
im_width = 800


# +
# # %config InlineBackend.figure_formats = ['pdf']
# # %config Completer.use_jedi = False
# -

def set_constants(obj_fun):
    '''Set bounds and optimum point'''
    
    fun_name = obj_fun.__name__
    if fun_name == 'ParSin':
        X_LIM = [0., 1.]
        F_LIM = [0, obj_fun(X_LIM[1])]
        X_OPT = 0.757
    
    elif fun_name == 'rosen':
        X_LIM = [-1.2, 1.2]
        F_LIM = [0, 680]
        X_OPT = [1., 1.]
    
    else:
        raise ValueError(f"Unknown objective function: '{fun_name}'")
    
    return np.array(X_LIM), np.array(F_LIM), np.array(X_OPT)


# ---

# ## Одномерный случай

# **Парабола $\times$ синус (ParSin)**
#
# $$ f(x) = (6x-2)^2 \cdot \sin\left(12x-4\right) $$
#
# Глобальный минимум: $x = 0.757$, $f(x) = -6.021$\
# Локальный минимум:  $x = 0.143$, $f(x) = -0.986$\
# Точка перегиба:     $x = 0.333$, $f(x) = 0.0$

def ParSin(x):
    '''Parabola times sine'''
    return (6*x-2)**2 * np.sin(12*x-4)
ParSin.__name__ = 'ParSin'


def graph_fun(fun, trajectory=[], figname='', noisy=False):
    '''Plot function'''
    seaborn.set_style('whitegrid')
    
    plt.figure(figsize=(8, 5))
    X_test = np.linspace(*X_LIM, 401)
    
    # function contours
    if noisy:
        plt.plot(X_test, fun(X_test), 'kx', alpha=.5, label='Objective function')
    else:
        plt.plot(X_test, fun(X_test), 'k-', label='Objective function')
    
    # points
    plt.plot(X_OPT, fun(X_OPT), '*', ms=20, c=cm(3), label='Minimum')
    if (len(trajectory) != 0):
        X = trajectory
        plt.plot(X[0],  fun(X[0]),   'o', c=cm(0), ms=8)
        plt.plot(X,     fun(X),     '-o', c=cm(0), ms=3.5)
        plt.plot(X[-1], fun(X[-1]),  '+', c=cm(0), mew=2., ms=15)
        
    plt.legend()
    plt.tight_layout()
    if (figname):
        plt.savefig(figname, dpi=200, bbox_inches='tight')


# ### Данные без шума

# Выбор задачи и установка констант

# +
obj_fun = ParSin
X_LIM, F_LIM, X_OPT = set_constants(obj_fun)

print(f'obj_fun = {obj_fun.__name__}')
print(f'X_OPT = {X_OPT}, obj_fun(X_OPT) = {obj_fun(X_OPT):.3f}')
# -

# Отрисовка графика выбранной целевой функции

graph_fun(obj_fun)


# ### Данные с шумом

# Теперь добавим к целевой функции шум:
# $$
#   f_{noisy} = f(x) + \sigma_{n} \xi.
# $$
#
# Здесь $\xi$ &mdash; нормальная случайная величина, переменная $\sigma_{n}$ задаёт амплитуду шума.

def add_noise(fun, sigma_n):
    def ret_fun(x):
        xi = np.random.randn(*x.shape)
        return fun(x) + sigma_n * xi
    return ret_fun


sigma_n = 1.0
obj_fun_noisy = add_noise(obj_fun, sigma_n)

graph_fun(obj_fun_noisy, noisy=True)


# ---

# ## Двумерный случай

# **Функция Розенброка**
#
# $$ f(x_1, x_2) = 100(x_2 - x_1^2)^2 + (1-x_1)^2 $$
#
# Функция имеет единственный минимум, находящийся внутри узкой параболической долины в точке $x = (1, 1)$ и равный $0$.

def rosen(x):
    '''Rosenbrock function'''
    # 2D: f = 100*(x2 - x1**2)**2 + (1 - x1)**2
    return sum(100.0*(x[1:]-x[:-1]**2.0)**2.0 + (1-x[:-1])**2.0)
rosen.__name__ = 'rosen'


# functions for visualization
def fun_2d(X1, X2, fun):
    array_2d = np.zeros((len(X1), len(X2)))
    for i, x2 in enumerate(X2):
        for j, x1 in enumerate(X1):
            array_2d[i, j] = fun(np.array([x1, x2]))
    return array_2d


def fun_contours(fun, points=[], constr=None, trajectory=[], figname=''):
    '''Draw function 2D contours'''
    seaborn.set_style('white')
    
    plt.figure(figsize=(7, 7))
    X1 = X2 = np.linspace(*X_LIM, 401)

    # function contours
    z_lines = np.linspace(0, F_LIM[1]**0.5, 20)**2
    if (F_LIM[0] < 0):
        z_lines_1 = np.linspace(F_LIM[0], 0, 20)
        z_lines = np.concatenate((z_lines_1[:-1], z_lines))
    
    contours = plt.contour(X1, X2, fun_2d(X1,X2,fun), z_lines,
                           linewidths=1., colors='k', alpha=0.9)
    plt.clabel(contours, fontsize=8, fmt='%.0f')

    # points
    for point in points:
        plt.plot(*point, 'x', c=cm(3), mew=2., ms=15)
    
    # trajectory
    if (len(trajectory) != 0):
        plt.plot(*trajectory[:,0],   'o', c=cm(0), ms=8)
        plt.plot(*trajectory,       '-o', c=cm(0), ms=3.5)
        plt.plot(*trajectory[:,-1],  '+', c=cm(0), mew=2., ms=15)
    
    # constraint
    if constr:
        plt.contour(X1,X2,fun_2d(X1,X2,constr),0,linewidths=1.,colors=cm(1))

    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$", rotation='horizontal', horizontalalignment='center')
    plt.xlim(*X_LIM)
    plt.ylim(*X_LIM)
    plt.tight_layout()
    plt.show()
    if (figname):
        plt.savefig(figname, dpi=200, bbox_inches='tight')


# ### Данные без шума

# +
obj_fun = rosen
X_LIM, F_LIM, X_OPT = set_constants(obj_fun)
F_OPT = obj_fun(X_OPT)

print(f'obj_fun = {obj_fun.__name__}')
print(f'X_OPT = {X_OPT}, F_OPT = {F_OPT:.3f}')
# -

fun_contours(obj_fun, points=[X_OPT])


# ### Данные с шумом

def add_noise(fun, sigma_n):
    def ret_fun(x):
        xi = np.random.randn()
        return fun(x) + sigma_n * xi
    return ret_fun


sigma_n = 1.0
obj_fun_noisy = add_noise(obj_fun, sigma_n)

fun_contours(obj_fun_noisy, points=[X_OPT])

# ---

# ## Задание
#
# Провести сравнение двух методов построения суррогатных моделей: полиномиальная регрессия (PR) и регрессия на основе гауссовских процессов (GPR).
# Рассмотреть два случая: одномерный (функция ParSin) и двумерный (функция Розенброка).
#
# Задачи:
#
# 1. Сделать план экспериментов, рекомендуется использовать метод LHS из библиотеки `pyDOE2` или `DOEpy`.
# 1. Построить суррогатные модели методами: PR и GPR. Количество признаков для полиномиальной регрессии выбрать самостоятельно.
# 1. Построить зависимость ошибки по норме $L_2$ от количества точек в обучающей выборке $N_{train}$ (1D: $N_{train, 1D} = [4, 8, 16]$, 2D: $N_{train, 2D} = [8, 16, 32]$). Ошибку считать по методу кросс-валидации.
#
# 1. Повторить процедуру для данных с шумом $\sigma_n = 1.0$.

# ---

# Versions used
import sys
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))



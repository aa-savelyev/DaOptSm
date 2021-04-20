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

# + [markdown] slideshow={"slide_type": "slide"}
# # Гауссовские процессы #

# + [markdown] toc=true slideshow={"slide_type": "subslide"}
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Случайные-процессы" data-toc-modified-id="Случайные-процессы-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Случайные процессы</a></span><ul class="toc-item"><li><span><a href="#Базовые-понятия-и-определения" data-toc-modified-id="Базовые-понятия-и-определения-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Базовые понятия и определения</a></span></li><li><span><a href="#Пример" data-toc-modified-id="Пример-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Пример</a></span></li><li><span><a href="#Моментные-характеристики-процессов" data-toc-modified-id="Моментные-характеристики-процессов-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Моментные характеристики процессов</a></span></li></ul></li><li><span><a href="#Гауссовские-процессы" data-toc-modified-id="Гауссовские-процессы-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Гауссовские процессы</a></span><ul class="toc-item"><li><span><a href="#Базовые-понятия-и-определения" data-toc-modified-id="Базовые-понятия-и-определения-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Базовые понятия и определения</a></span></li><li><span><a href="#Ковариационные-функции" data-toc-modified-id="Ковариационные-функции-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Ковариационные функции</a></span></li><li><span><a href="#Генерация-случайной-выборки-гауссовских-процессов" data-toc-modified-id="Генерация-случайной-выборки-гауссовских-процессов-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Генерация случайной выборки гауссовских процессов</a></span></li><li><span><a href="#Ещё-примеры-траекторий" data-toc-modified-id="Ещё-примеры-траекторий-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Ещё примеры траекторий</a></span></li></ul></li><li><span><a href="#Литература" data-toc-modified-id="Литература-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Литература</a></span></li></ul></div>

# + slideshow={"slide_type": "skip"}
# Imports
import sys
sys.path.append('../modules')
import graph_support
import numpy as np
import scipy

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# + slideshow={"slide_type": "skip"}
# Styles
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['lines.linewidth'] = 2.
matplotlib.rcParams['lines.markersize'] = 7
# cm = plt.cm.tab10  # Colormap

import seaborn
seaborn.set_style('whitegrid')
figscale = 2

# + slideshow={"slide_type": "skip"} language="html"
# <style>
#     .container.slides .celltoolbar, .container.slides .hide-in-slideshow {display: None ! important;}
# </style>

# + [markdown] slideshow={"slide_type": "slide"}
# ## Случайные процессы ##
#
# Что такое гауссовский процесс?
# Как можно догадаться из названия, это процесс, состоящий из случайных величин, распределённых по Гауссу.
# Точное определение гласит, что гауссовский процесс &mdash; это случайный процесс, все конечномерные распределения которого гауссовские.

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Базовые понятия и определения ###
#
# **Определение 1.** *Случайным процессом* называется семейство случайных величин $X(\omega, t)$, $\omega \in \Omega$, заданных на одном вероятностном пространстве $(\Omega, \mathcal{F}, \mathrm{P})$ и зависящих от параметра $t$, принимающего значения из некоторого множества $T \in \mathbb{R}$. Параметр $t$ обычно называют *временем*.
#
# К случайному процессу всегда следует относиться как к функции двух переменных: исхода $\omega$ и времени $t$. Это независимые переменные.
#
# **Определение 2.** При фиксированном времени $t = t_0$ случайная величина $X(\omega, t_0)$ называется *сечением процесса* в точке $t_0$. При фиксированном исходе $\omega = \omega_0$ функция времени $X(\omega_0, t)$ называется *траекторией* (*реализацией*, *выборочной функцией*) процесса.

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Пример ###
#
# Известным примером стохастического процесса является модель броуновского движения (известная также как винеровский процесс).
# Броуновское движение &mdash; это случайное движение частиц, взвешенных в жидкости.
# Такое движение может рассматриваться как непрерывное случайное движение, при котором частица перемещается в жидкости из-за случайного столкновения с ней других частиц.
# Мы можем моделировать этот процесс во времени $t$ в одном измерении $d$, начиная с точки $0$ и перемещая частицу за определенное количество времени $\Delta t$ на случайное расстояние $\Delta d$ от предыдущего положения.
# Случайное расстояние выбирается из нормального распределения со средним $0$ и дисперсией $\Delta t$. Выборку $\Delta d$ из этого нормального распределения обозначим как $\Delta d \sim \mathcal{N}(0, \Delta t)$. Позицию $d(t)$ изменяется со временем по следующему закону $d(t + \Delta t) = d(t) + \Delta d$.
#
# На следующем рисунке мы смоделировали 10 различных траекторий броуновского движения, каждый из которых проиллюстрирован разным цветом.

# + slideshow={"slide_type": "subslide"}
# 1D simulation of the Brownian motion process
total_time = 1.
nb_steps = 500
delta_t = total_time / nb_steps
nb_processes = 10  # Simulate N different motions
mean = 0.  # Mean of each movement
stdev = np.sqrt(delta_t)  # Standard deviation of each movement

# Simulate the brownian motions in a 1D space by cumulatively making a new movement delta_d
# Move randomly from current location to N(0, delta_t)
distances = np.cumsum(np.random.normal(mean, stdev, (nb_processes, nb_steps)), axis=1)

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
plt.figure(figsize=(figscale*8, figscale*5))
# Make the plots
t = np.arange(0, total_time, delta_t)
for i in range(nb_processes):
    plt.plot(t, distances[i,:])
plt.title(
    f'Процесс броуновского движения,\n\
    траектории {nb_processes} реализаций процесса'
)
# plt.fill_between(t, -2*t**0.5, 2*t**0.5, color='grey', alpha=0.1)
plt.xlabel('$t$')
plt.ylabel('$d(t)$')
plt.xlim([0, total_time])
plt.tight_layout()
plt.show()


# + [markdown] slideshow={"slide_type": "subslide"}
# ### Моментные характеристики процессов ###
#
# На рисунке выше можно видеть несколько траекторий стохастического процесса.  Каждая реализация определяет позицию $d$ для каждого возможного временного шага $t$. Таким образом, каждая реализация соответствует функции $f(t) = d$.
#
# Это означает, что случайный процесс можно интерпретировать как случайное распределение функции. Мы можем получить реализацию функции с помощью стохастического процесса. Однако каждая реализация функции будет различной из-за случайности стохастического процесса.
#
# **Определение 3.** *Математическим ожиданием* случайного процесса $X(t)$ называется функция $m_x : T \rightarrow \mathbb{R}$, значение который в каждый момент времени равно математическому ожиданию соответствующего сечения, т.е. $\forall t \in T \; m_x(t) = \mathrm{E}X(t)$.
#
# **Определение 4.** *Корреляционной функцией* случайного процесса $X(t)$ называется функция двух переменных $K_x : T \times T \rightarrow \mathbb{R}$, которая каждой паре моментов времени сопоставляет корреляционный момент соответствующих сечений процесса, т.е.
# $$
#     K_x(t_1, t_2) = \mathrm{E} \left[ \left(X(t_1) - \mathrm{E}X(t_1)\right) \cdot \left(X(t_2) - \mathrm{E}X(t_2)\right) \right].
# $$

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Гауссовские процессы ##

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Базовые понятия и определения ###
#
# **Определение 5.** Случайный процесс $\{X(t), \; t \ge 0\}$ называется *гауссовским*, если для любого $n \ge 1$ и точек $0 \le t_1 < \ldots < t_n$ вектор $(X(t_1), \, \ldots \,, X(t_n))$ является нормальным случайным вектором.
#
#
# Гауссовские процессы &mdash; это распределение функций $f(x)$, которое определяется средней функцией $m(t)$ и положительной ковариационной функцией $k(t,t')$, где $t$ &mdash; параметр функции, а $(t,t')$ &mdash; все возможные пары из области определения. Обозначаются гауссовские процессы так:
#
# $$ f(t) \sim \mathcal{GP}(m(t), k(t,t')). $$

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Ковариационные функции ###
#
# Гауссовский процесс полноcтью определяется функцией среднего и ковариационной функцией.
# Ковариационная функция $k(x_a, x_b)$ моделирует совместную изменчивость случайных переменных гауссовского процесса, она возвращает значение ковариации между каждой парой в $x_a$ и $x_b$.
#
# Спецификация ковариационной функции (также известной как функция ядра) неявно задаёт распределение по функциям $f(x)$. Выбирая конкретный вид функции ядра $k$, мы задаём апроиорную информацию о данном распределении. Функция ядра должна быть симметричной и положительно-определённой.

# + [markdown] slideshow={"slide_type": "subslide"}
# Мы будем использовать квадратичное экспоненциальтное ковариационную функцию (также известную как гауссовское ядро):
#
# $$ k(x_a, x_b) = \sigma_f^2 \exp{ \left( -\frac{1}{2l^2} \lVert x_a - x_b \rVert^2 \right) }. $$
#
# Параметр длины $l$ контролирует гладкость функции, а $\sigma_f$ &mdash; вертикальную вариацию. Для простоты используется один и тот же параметр длины $l$ для всех входных размеров (изотропное ядро).
#
# Могут быть определены и другие функции ядра, приводящие к различным свойствам распределения гауссовского процесса.

# + slideshow={"slide_type": "subslide"}
# Isotropic squared exponential kernel.
def kernel(X1, X2, l=1.0, sigma_f=1.0):
    '''
    Isotropic squared exponential kernel. Computes 
    a covariance matrix from points in X1 and X2.
    
    Args:
        X1: Array of m points (m x d).
        X2: Array of n points (n x d).

    Returns:
        Covariance matrix (m x n).
    '''
    
    sqdist = np.sum(X1**2,1).reshape(-1,1) + np.sum(X2**2,1) - 2*np.dot(X1, X2.T)
    return sigma_f**2 * np.exp(-0.5 / l**2 * sqdist)


# + [markdown] slideshow={"slide_type": "subslide"}
# Пример ковариационной матрицы с гауссовским ядром приведён на рисунке слева внизу.
# Справа показана значение ковариационной функции, если одна из переменных равна $0$: $k(0,x)$. \
# Обратите внимание, что по мере удаления $x$ от $0$ ковариация уменьшается экспоненциально.

# + slideshow={"slide_type": "subslide"}
# Illustrate covariance matrix and function
xlim = (-3, 3)
X = np.expand_dims(np.linspace(*xlim, 25), 1)
Sigma = kernel(X, X)

# +
graph_support.hide_code_in_slideshow()
# Show covariance matrix example from exponentiated quadratic
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(figscale*9., figscale*4.))

# Plot covariance matrix
im = ax1.imshow(Sigma, cmap=cm.viridis_r)
cbar = plt.colorbar(
    im, ax=ax1, fraction=0.045, pad=0.05)
cbar.ax.set_ylabel('$k(x_1,x_2)$')
ax1.set_title(
    'Пример ковариационной матрицы\n\
    для квадратичного экспоненциального ядра',
    pad=10)
ax1.set_xlabel('$x_1$')
ax1.set_ylabel('$x_2$')
ticks = list(range(xlim[0], xlim[1]+1))
ax1.set_xticks(np.linspace(0, len(X)-1, len(ticks)))
ax1.set_yticks(np.linspace(0, len(X)-1, len(ticks)))
ax1.set_xticklabels(ticks)
ax1.set_yticklabels(ticks)
ax1.grid(False)

# Show covariance with X=0
X = np.expand_dims(np.linspace(*xlim, num=100), 1)
zero = np.array([[0]])
Sigma_0 = kernel(X, zero)
# Make the plots
ax2.plot(X[:,0], Sigma_0[:,0], label='$k(x,0)$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$k(x)$')
ax2.set_title(
    'Ковариация между $x$ и $0$',
    pad=10)
# ax2.set_ylim([0, 1.1])
ax2.set_xlim(*xlim)
ax2.legend(loc=1)

fig.tight_layout()
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# ### Генерация случайной выборки гауссовских процессов ###

# + [markdown] slideshow={"slide_type": "subslide"}
# Конечное размерное подмножество распределения гауссовского процесса приводит к частному распределению, которое является гауссовским распределением $\mathbf{y} \sim \mathcal{N}(\mathbf{\mu}, \Sigma)$ со средним вектором $\mathbf{\mu} = m(X)$ и ковариационной матрицей $\Sigma = k(X, X)$.
#
# На рисунке ниже приведена выборка из 5 различных функциональных реализаций гауссовского процесса с экспоненциальным квадратичным ядром без каких-либо наблюдаемых данных.
# Для этого мы нарисуем коррелированные образцы из 41-мерного гауссианы $\mathcal{N}(0, k(X, X))$ с $X = [X_1, \ldots, X_{41}]$.

# + slideshow={"slide_type": "subslide"}
# Sample from the Gaussian process distribution
nb_of_samples = 41  # Number of points in each function
number_of_functions = 5  # Number of functions to sample

# Independent variable samples
X = np.expand_dims(np.linspace(0, 8, nb_of_samples), 1)
Sigma = kernel(X, X)  # Kernel of data points

# Draw samples from the prior at our data points.
# Assume a mean of 0 for simplicity
ys = np.random.multivariate_normal(
    mean=np.zeros(nb_of_samples), cov=Sigma, size=number_of_functions)

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
# Plot the sampled functions
plt.figure(figsize=(figscale*8, figscale*4))
for i in range(number_of_functions):
    plt.plot(X, ys[i], linestyle='-', marker='o', markersize=3)
plt.xlabel('$t$')
plt.ylabel('$y = f(t)$')
plt.title(
    f'Траектории {number_of_functions} гауссовских процессов,\n\
    построенных по {nb_of_samples} точке каждый'
)
plt.xlim([0, 8])
plt.show()


# + [markdown] slideshow={"slide_type": "subslide"}
# Другой способ визуализировать это &mdash; взять только 2 измерения 41-мерной функции Гаусса и нарисовать некоторые его частные двумерные распределения.
#
# Следующий рисунок слева визуализирует 2D распределение для $X = [0, 0.2]$, где ковариация $k(0, 0.2) = 0.98$. Рисунок справа визуализирует 2D распределение для $X = [0, 2]$, где ковариация $k(0, 2) = 0.14$.

# + [markdown] slideshow={"slide_type": "subslide"}
# Для каждого из 2D гауссовых полей соответствующие примеры реализации функций, приведенные выше, представлены на рисунке цветными точками.
#
# Обратите внимание, что точки, близкие друг к другу во входной области $x$, сильно коррелируют ($y_1$ близко к $y_2$), в то время как точки, находящиеся дальше друг от друга, практически независимы. Это связано с тем, что эти поля являются результатом гауссовского процесса с квадратичной ковариацией, которая добавляет предыдущую информацию о том, что точки в области ввода $X$ должны быть близко друг к другу в области вывода $y$.

# + slideshow={"slide_type": "subslide"}
def generate_surface(mean, covariance):
    """Helper function to generate density surface."""
    nb_of_x = 100 # grid size
    x1s = np.linspace(-5, 5, num=nb_of_x)
    x2s = np.linspace(-5, 5, num=nb_of_x)
    x1, x2 = np.meshgrid(x1s, x2s) # Generate grid
    pdf = np.zeros((nb_of_x, nb_of_x))
    # Fill the cost matrix for each combination of weights
    for i in range(nb_of_x):
        for j in range(nb_of_x):
            pdf[i,j] = scipy.stats.multivariate_normal.pdf(
                np.array([x1[i,j], x2[i,j]]), 
                mean=mean, cov=covariance)
    return x1, x2, pdf  # x1, x2, pdf(x1,x2)


# + slideshow={"slide_type": "subslide"}
# Plot of strong correlation
X_strong = np.array([[0], [0.2]])
mu_strong = np.array([0., 0.])
Sigma_strong = kernel(X_strong, X_strong)

# Plot weak correlation
X_weak = np.array([[0], [2]])
mu_weak = np.array([0., 0.])
Sigma_weak = kernel(X_weak, X_weak)

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
# Show strong and weak correlations
fig = plt.figure(figsize=(figscale*9.0, figscale*4.0)) 
gs = gridspec.GridSpec(1, 2)
ax_p1 = plt.subplot(gs[0,0])
ax_p2 = plt.subplot(gs[0,1], sharex=ax_p1, sharey=ax_p1)

# Plot strong correlation
y1, y2, p = generate_surface(mu_strong, Sigma_strong)
# Plot bivariate distribution
con1 = ax_p1.contourf(y1, y2, p, 100, cmap=cm.magma_r)
ax_p1.set_xlabel(f'$y_1 = f(X={X_strong[0,0]})$')
ax_p1.set_ylabel(f'$y_2 = f(X={X_strong[1,0]})$')
ax_p1.axis([-2.7, 2.7, -2.7, 2.7])
ax_p1.set_aspect('equal')
ax_p1.text(
    -2.3, 2.1, 
    (f'$k({X_strong[0,0]}, {X_strong[1,0]}) = {Sigma_strong[0,1]:.2f}$'))
ax_p1.set_title(f'$X = [{X_strong[0,0]}, {X_strong[1,0]}]$ ')
# Select samples
X_0_index = np.where(np.isclose(X, 0.))
X_02_index = np.where(np.isclose(X, 0.2))
y_strong = ys[:,[X_0_index[0][0], X_02_index[0][0]]]
# Show samples on surface
for i in range(y_strong.shape[0]):
    ax_p1.plot(y_strong[i,0], y_strong[i,1], marker='o')

# Plot weak correlation
y1, y2, p = generate_surface(mu_weak, Sigma_weak)
# Plot bivariate distribution
con2 = ax_p2.contourf(y1, y2, p, 100, cmap=cm.magma_r)
con2.set_cmap(con1.get_cmap())
con2.set_clim(con1.get_clim())
ax_p2.set_xlabel(f'$y_1 = f(X={X_weak[0,0]})$')
ax_p2.set_ylabel(f'$y_2 = f(X={X_weak[1,0]})$')
ax_p2.set_aspect('equal')
ax_p2.text(
    -2.3, 2.1, 
    (f'$k({X_weak[0,0]}, {X_weak[1,0]}) = {Sigma_weak[0,1]:.2f}$'))
ax_p2.set_title(f'$X = [{X_weak[0,0]}, {X_weak[1,0]}]$')
# Add colorbar
divider = make_axes_locatable(ax_p2)
cax = divider.append_axes('right', size='5%', pad=0.1)
cbar = plt.colorbar(con1, ax=ax_p2, cax=cax)
cbar.ax.set_ylabel('density: $p(y_1, y_2)$')
fig.suptitle(
    'Частные 2D распределения: $y \sim \mathcal{N}(0, k(X, X))$',
    x=0.5, y=1.05)
# Select samples
X_0_index = np.where(np.isclose(X, 0.))
X_2_index = np.where(np.isclose(X, 2.))
y_weak = ys[:,[X_0_index[0][0], X_2_index[0][0]]]
# Show samples on surface
for i in range(y_weak.shape[0]):
    ax_p2.plot(y_weak[i,0], y_weak[i,1], marker='o')

plt.tight_layout()
plt.show()


# + [markdown] slideshow={"slide_type": "slide"}
# ### Ещё примеры траекторий ###
#
# Сгенерируем выборку из 1000 гауссовых процессов со средней функцией $\mu(t) = 0$ и квадратичной экспоненциальной функцией ядра, определённой выше $k(t_1, t_2) = \exp{\left( -\frac{1}{2\sigma^2} (t_1 - t_2)^2 \right)}$.
#
# На рисунке ниже приведена траектории первых 50 процессов выборки.

# + slideshow={"slide_type": "subslide"}
def plot_gp(mu, cov, X, X_train=None, Y_train=None, samples=[]):
    X = X.ravel()
    mu = mu.ravel()
    uncertainty = 2 * np.sqrt(np.diag(cov))
    
    plt.figure(figsize=(figscale*8,figscale*5))
    plt.fill_between(
        X, mu + uncertainty, mu - uncertainty,
        color='grey', alpha=0.1, label='$\pm 2\,\sigma$')
    plt.plot(X, samples, '-', lw=1.0)
    plt.plot(X, mu, 'k', label='среднее значение')
    if X_train is not None:
        plt.plot(X_train, Y_train, 'kx', mew=1.5)
    plt.xlim([X.min(), X.max()])
    plt.xlabel('$t$')
    plt.ylabel('$f(t)$')
    plt.legend(loc='upper right')


# + slideshow={"slide_type": "subslide"}
# Test data
x_min, x_max = 0, 10
N_test = 1001
X_test = np.linspace(x_min, x_max, N_test).reshape(-1,1)

# Set mean and covariance
M = np.zeros_like(X_test).reshape(-1,1)
l = 10e-2*(x_max-x_min)
K = kernel(X_test, X_test, l=l)

# Generate samples from the prior
N_gp = 1000
L = np.linalg.cholesky(K + 1e-6*np.eye(N_test))
gp = M + np.dot(L, np.random.normal(size=(N_test,N_gp)))

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
# Draw samples from the prior
x_i = 2.
plot_gp(M, K, X_test, samples=gp[:,:50])
plt.axvline(x_i, c='k', ls=':')
plt.show()

# + [markdown] slideshow={"slide_type": "subslide"}
# Убедимся в правильных статистических характеристиках нашей выборки. Для этого нарисуем гистограмму значений $f(x)$ в каком-либо сечении $x = \mathrm{const}$. По определению гауссовского процесса, распределение $f(x)$ должно быть гауссовым.

# + slideshow={"slide_type": "subslide"}
# Draw section histogram
gp_i = gp[np.flatnonzero(X_test.ravel() == x_i)][0]
fig = plt.figure(figsize=(figscale*8.0, figscale*4.0)) 
plt.hist(gp_i, bins=100, histtype='stepfilled')
plt.title(f'Гистограмма значений $f(x)$ в сечении $x={x_i}$')
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Литература
#
# 1. Roelants P. [Understanding Gaussian processes](https://peterroelants.github.io/posts/gaussian-process-tutorial/).
# 1. Krasser M. [Gaussian_processes.ipynb]().
# 1. Лекции по случайным процессам / под ред. А.В. Гасникова. М.: МФТИ, 2019.
#

# + slideshow={"slide_type": "subslide"}
# Versions used
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))
# -



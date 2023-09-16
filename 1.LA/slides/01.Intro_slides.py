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

# + [markdown] slideshow={"slide_type": "slide"} cell_style="center"
# # Анализ данных, суррогатное моделирование и оптимизация в прикладных задачах
#
# к.т.н. Андрей Александрович Савельев \
# savelyev.aa@mipt.ru

# + [markdown] toc=true slideshow={"slide_type": "skip"}
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Содержание-курса" data-toc-modified-id="Содержание-курса-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Содержание курса</a></span><ul class="toc-item"><li><span><a href="#Осенний-семестр" data-toc-modified-id="Осенний-семестр-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Осенний семестр</a></span></li><li><span><a href="#Материалы-(осень)" data-toc-modified-id="Материалы-(осень)-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Материалы (осень)</a></span></li><li><span><a href="#Весенний-семестр" data-toc-modified-id="Весенний-семестр-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Весенний семестр</a></span></li><li><span><a href="#Материалы-(весна)" data-toc-modified-id="Материалы-(весна)-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Материалы (весна)</a></span></li><li><span><a href="#Организационные-вопросы" data-toc-modified-id="Организационные-вопросы-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>Организационные вопросы</a></span></li><li><span><a href="#Проект" data-toc-modified-id="Проект-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Проект</a></span></li></ul></li><li><span><a href="#Технические-особенности" data-toc-modified-id="Технические-особенности-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Технические особенности</a></span></li><li><span><a href="#Машинное-обучение" data-toc-modified-id="Машинное-обучение-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Машинное обучение</a></span><ul class="toc-item"><li><span><a href="#Определение" data-toc-modified-id="Определение-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Определение</a></span></li><li><span><a href="#Классификация-методов" data-toc-modified-id="Классификация-методов-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Классификация методов</a></span></li></ul></li><li><span><a href="#Восстановление-регрессии" data-toc-modified-id="Восстановление-регрессии-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Восстановление регрессии</a></span><ul class="toc-item"><li><span><a href="#Постановка-задачи" data-toc-modified-id="Постановка-задачи-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Постановка задачи</a></span></li><li><span><a href="#Пример" data-toc-modified-id="Пример-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Пример</a></span></li><li><span><a href="#Полиномы" data-toc-modified-id="Полиномы-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Полиномы</a></span></li><li><span><a href="#Сплайны" data-toc-modified-id="Сплайны-4.4"><span class="toc-item-num">4.4&nbsp;&nbsp;</span>Сплайны</a></span></li><li><span><a href="#Гауссовские-процессы-(GPR)" data-toc-modified-id="Гауссовские-процессы-(GPR)-4.5"><span class="toc-item-num">4.5&nbsp;&nbsp;</span>Гауссовские процессы (GPR)</a></span></li></ul></li><li><span><a href="#Пример-прикладной-задачи" data-toc-modified-id="Пример-прикладной-задачи-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Пример прикладной задачи</a></span><ul class="toc-item"><li><span><a href="#Постановка-задачи" data-toc-modified-id="Постановка-задачи-5.1"><span class="toc-item-num">5.1&nbsp;&nbsp;</span>Постановка задачи</a></span><ul class="toc-item"><li><span><a href="#Процесс-сходимости-однокритериальной-задачи" data-toc-modified-id="Процесс-сходимости-однокритериальной-задачи-5.1.1"><span class="toc-item-num">5.1.1&nbsp;&nbsp;</span>Процесс сходимости однокритериальной задачи</a></span></li><li><span><a href="#Фронт-Парето" data-toc-modified-id="Фронт-Парето-5.1.2"><span class="toc-item-num">5.1.2&nbsp;&nbsp;</span>Фронт Парето</a></span></li><li><span><a href="#Результаты" data-toc-modified-id="Результаты-5.1.3"><span class="toc-item-num">5.1.3&nbsp;&nbsp;</span>Результаты</a></span></li></ul></li></ul></li><li><span><a href="#Литература" data-toc-modified-id="Литература-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Литература</a></span></li></ul></div>

# + [markdown] slideshow={"slide_type": "slide"}
# ## Содержание курса
#
# ### Осенний семестр
#
# 1. Матрицы и действия над ними: $A=CR$, $A=QR$
# 1. Теория систем линейных уравнений: $A\mathbf{x} = \mathbf{b}$, $A=LU$
# 1. Спектральное разложение матриц: $A = X \Lambda X^{-1}$
# 1. Сингулярное разложение матриц: $A = U \Sigma V^\top$
# 1. Малоранговые аппроксимации матриц, метод главных компонент
# 1. Линейная регрессия, метод наименьших квадратов
# 1. Проблема мультиколлинеарности, гребневая регрессия, лассо Тибширани
# 1. Методы оптимизации: BFGS, SLSQP, COBYLA
# 1. Примеры: расстояние Махаланобиса, алгоритм Eigenfaces

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Материалы (осень)
#
# **Основная литература**
# 1. *Воронцов К.В.* [Математические методы обучения по прецедентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). &mdash; 141 c.
# 1. *Strang G.* Linear algebra and learning from data. &mdash; 2019. &mdash; 432 p.
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; 1983. &mdash; 336 с.
#
#
# **Дополнительная литература**
# 1. *Оселедец И.В.* [Numerical Linear Algebra](https://nla.skoltech.ru/index.html). Skoltech term course
# 1. *Martins J.R.R.A. & Ning A.* Engineering Design Optimization. &mdash; 2021. &mdash;  637 с.
# 1. *Гантмахер Ф.Р.* Теория матриц. &mdash;  М.: Наука, 1967. &mdash;  576 с.
# 1. *Стренг Г.* Линейная алгебра и её применения. &mdash; 1980.
#
#
# **Видеокурсы**
# 1. [Машинное обучение](https://www.youtube.com/watch?v=SZkrxWhI5qM&list=PLJOzdkh8T5krxc4HsHbB8g8f0hu7973fK&index=1), д.ф.-м.н. Константин Вячеславович Воронцов, ШАД (Яндекс)
# 1. [Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k), prof. Gilbert Strang, MIT
# 1. [Линейная алгебра](https://www.youtube.com/watch?v=WNl10xl1QT8&list=PLthfp5exSWEqSRXkZgMMzTSXL_WwMV9wK), к.ф.-м.н. Павел Александрович Кожевников, МФТИ

# + [markdown] slideshow={"slide_type": "subslide"} cell_style="center"
# ### Весенний семестр
#
# 1. Данные и методы работы с ними. Числовые характеристики выборки: среднее, среднеквадратичное отклонение, коэффициент корреляции
# 1. Основы теории вероятности: вероятностная модель, условная вероятность, формула Байеса
# 1. Случайные величины и их распределения, числовые характеристики случайных величин
# 1. Многомерное нормальное распределение, ковариационная матрица, условные распределения
# 1. Случайные процессы, гауссовские процессы
# 1. Регрессия на основе гауссовских процессов 
# 1. Оптимизация регрессионной кривой, влияние параметров ядра и амплитуды шума на регрессионную кривую
# 1. Алгоритм эффективной глобальной оптимизации

# + [markdown] slideshow={"slide_type": "skip"}
# ### Материалы (весна)
#
# **Основная литература**
# 1. *Ширяев А.Н.* Вероятность &mdash; 1. &mdash; М.: МЦНМО, 2007. &mdash; 517 с.
# 1. *Rasmussen C.E. & Williams C.K.I.* [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/). The MIT Press, 2006. 248 p.
#
#
# **Дополнительная литература**
# 1. Материалы по гауссовским процессам авторов [P. Roelants](https://peterroelants.github.io/) и [M. Krasser ](http://krasserm.github.io/)
# 1. TBD
#
#
# **Видеокурсы**
# 1. [Теория вероятностей](https://www.youtube.com/watch?v=Q3h9P7lhpNc&list=PLyBWNG-pZKx7kLBRcNW3HXG05BDUrTQVr&index=1), д.ф.-м.н. Максим Евгеньевич Широков, МФТИ
# 1. TBD

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Организационные вопросы
#
# 1. Время и место: среда, 13.45, к. 2**
# 1. Каналы связи
#    - telegram группа: DaOptSm
#    - почта: savelyev.aa@mipt.ru
# 1. Материалы лекций, литература: Яндекс Диск (?)
# 1. Оценка
#    - осень: дифф. зачёт
#    - весна: экзамен

# + [markdown] slideshow={"slide_type": "skip"}
# ### Проект ###
#
# 1. Заявка (15 ноября)
#    - название
#    - аннотация: цель, актуальность, источники
#    - исполнители (3-5 человек)
#    - план работы, планируемое содержание работы каждого исполнителя
#    - планируемые результаты
# 1. Проект в формате `.ipynb` (1 декабря)
# 1. Презентация (8 декабря)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Технические особенности ##

# + slideshow={"slide_type": "subslide"}
# Imports
import numpy as np
import matplotlib.pyplot as plt

# + slideshow={"slide_type": "skip"}
# Styles
import warnings 
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['lines.linewidth'] = 3
matplotlib.rcParams['lines.markersize'] = 10
cm = plt.cm.tab10  # Colormap
figsize=(15, 8)

import seaborn
seaborn.set_style('whitegrid')

# + slideshow={"slide_type": "subslide"}
# Data
X_lims = [np.pi, 3*np.pi]
X = np.linspace(*X_lims, 20)  # Independent variable x
Y = 1/X * np.sin(2*X)         # Dependent variable

# + slideshow={"slide_type": "subslide"}
# Plot
fig, ax = plt.subplots(figsize=figsize)
ax.plot(X, Y, 'ko', label='data')
ax.plot(X, Y, '-', c=cm(0), label='polyline')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='upper right')
plt.show()

# + [markdown] slideshow={"slide_type": "-"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Машинное обучение
#
# ### Определение

# + [markdown] slideshow={"slide_type": "subslide"}
# **Машинное обучение** &mdash; подраздел искусственного интеллекта, изучающий методы построения алгоритмов, способных *обучаться*. \
# Компьютерная программа **учится** на опыте *E* по отношению к некоторому классу задач *T* и измерению производительности *P*, если её производительность *P* в задачах *T*, улучшается с опытом *E*.

# + [markdown] slideshow={"slide_type": "fragment"}
# **Машинное обучение** &mdash; набор методов, которые могут *автоматически обнаруживать закономерности* в данных, а затем использовать обнаруженные закономерности для *прогнозирования* или для выполнения других видов *принятия решений*.

# + [markdown] slideshow={"slide_type": "subslide"}
# **Машинное обучение** &mdash; класс методов искусственного интеллекта, характерной чертой которых является *не прямое решение* задачи, а *обучение* в процессе применения решений множества *сходных задач*.

# + [markdown] slideshow={"slide_type": "fragment"}
# **Машинное обучение** (индуктивное обучение) основано на выявлении общих *закономерностей* по частным эмпирическим данным (*прецедентам*). Термины **машинное обучение** и **обучение по прецедентам** можно считать *синонимами*.

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Классификация методов

# + [markdown] slideshow={"slide_type": "subslide"}
# 1. **Обучение с учителем** (объект, ответ) \
# Найти функциональную зависимость ответов от описаний объектов и построить алгоритм, принимающий на входе описание объекта и выдающий на выходе ответ. \
# Задачи:
#  - классификация: ответ &mdash; метка классов
#  - регрессия: ответ &mdash; число или вектор

# + [markdown] slideshow={"slide_type": "fragment"}
# 2. **Обучение без учителя** (объект, &mdash;) \
# Найти зависимости между объектами \
# Задачи:
#  - кластеризация: сгруппировать объекты в кластеры
#  - сокращение размерности: перейти к меньшему числу новых признаков, потеряв при этом минимум существенной информации об объектах
#  - фильтрация выбросов: обнаружить в обучающей выборке нетипичные объекты

# + slideshow={"slide_type": "subslide"}
from IPython.display import Image
im_width = 1200
display(Image('../pix/01.Intro/ML_scheme_1.jpeg', width=im_width))

# + slideshow={"slide_type": "subslide"}
display(Image('../pix/01.Intro/ML_scheme_2.jpeg', width=im_width))

# + [markdown] slideshow={"slide_type": "slide"}
# ## Восстановление регрессии
#
# ### Постановка задачи

# + [markdown] slideshow={"slide_type": "subslide"}
# Пусть задано множество **объектов** $X$ и множество допустимых **ответов** $Y$. \
# Мы предполагаем существование зависимости $y:X \rightarrow Y$. \
# Значения функции $y_i = y(x_i)$ известны только на конечном подмножестве объектов $\{x_1, \ldots, x_l\} \subset X$.
#
# Пары &laquo;объект&ndash;ответ&raquo; $(x_i, y_i)$ называются *прецендентами*, а совокупность пар $X^l = (x_i, y_i)_{i=1}^l$ &mdash; **обучающей выборкой**.
#
# Требуется построить алгоритм (&laquo;**функцию регрессии**&raquo;) $a: X \rightarrow Y$, аппроксимирующий целевую зависимость $y$.

# + [markdown] slideshow={"slide_type": "subslide"}
# **Признак** $f$ объекта $x$ &mdash; это результат измерения некоторой характеристики объекта.
#
# Пусть имеется набор признаков $f_1, \ldots, f_n$.
#
# Совокупность признаковых описаний всех объектов выборки $X_l$, записанную в виде таблицы размера $l \times n$, называют **матрицей объектов&ndash;признаков**:
# $$
#   \mathbf{F} = 
#   \begin{pmatrix}
#     f_1(x_1) & \ldots & f_n(x_1) \\
#     \ldots   & \ddots & \ldots   \\
#     f_1(x_l) & \ldots & f_n(x_l) \\
#   \end{pmatrix}.
# $$
#
# Матрица объектов–признаков является стандартным и наиболее распространённым способом представления исходных данных в прикладных задачах.

# + [markdown] slideshow={"slide_type": "slide"}
# ### Пример
#
# > Регрессия &mdash; это наука о том, как через точки провести линию.
#
# **Задача**: построить функцию корректно описывающую обучающие данные и **обобщающую их на неизвестные данные**.

# + slideshow={"slide_type": "subslide"}
# Define the data
n = 20
x_lim = [0, 1]  # Limits
x_disp = np.linspace(*x_lim, 1000)  # X array for display
# Underlying relation
x_train = np.linspace(*x_lim, n)  # Independent variable x
y_true = np.sin(3*np.pi*x_train)  # Dependent variable y
# Noise
np.random.seed(42)
e_std = 0.4  # Standard deviation of the noise
err = e_std * np.random.randn(n)  # Noise
# Output
y_train = y_true + err  # Dependent variable with noise

# + slideshow={"slide_type": "subslide"} cell_style="center"
# Show data
fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_disp,  np.sin(3*np.pi*x_disp), 'k:', label='clean data')
ax.plot(x_train, y_train, 'o', c=cm(0), label='noisy data')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Полиномы

# + slideshow={"slide_type": "fragment"}
deg = 1
p1 = np.polyfit(x_train, y_train, deg)

# + slideshow={"slide_type": "subslide"} cell_style="center"
fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p1, x_disp), '-', c=cm(1), label=f'polynomial, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + slideshow={"slide_type": "subslide"} cell_style="center"
deg = 4
p2 = np.polyfit(x_train, y_train, deg)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p2, x_disp), '-', c=cm(2), label=f'polynomial, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + slideshow={"slide_type": "subslide"} cell_style="center"
deg = n-1
p3 = np.polyfit(x_train, y_train, deg)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p3, x_disp), '-', c=cm(3), label=f'polynomial, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Сплайны

# + slideshow={"slide_type": "fragment"}
from scipy import interpolate

tck = interpolate.splrep(x_train, y_train, k=1, s=1e-3)
S = interpolate.splev(x_disp, tck, der=0)

# + slideshow={"slide_type": "subslide"}
tck = interpolate.splrep(x_train, y_train, k=3, s=1e-3)
S = interpolate.splev(x_disp, tck, der=0)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, S, '-', c=cm(3), label='B-spline')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + [markdown] slideshow={"slide_type": "subslide"}
# ### Гауссовские процессы (GPR)

# + slideshow={"slide_type": "fragment"}
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# + slideshow={"slide_type": "fragment"}
X_train = np.atleast_2d(x_train).T

# graph_support.hide_code_in_slideshow()
rbf = ConstantKernel(1.) * RBF(length_scale=0.1)
gpr = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=20)

# Fit GP regressor
gpr.fit(X_train, y_train)

# Compute posterior predictive mean and standard-deviation
mu, std = gpr.predict(x_disp.reshape(-1, 1), return_std=True)
mu = mu.ravel()

# + slideshow={"slide_type": "subslide"}
# Show data
fig, ax = plt.subplots(figsize=figsize)
ax.plot(X_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
# ax.plot(x_disp, S, '--', c=cm(3), label='B-spline')
ax.plot(x_disp, mu, '-', c=cm(4), label='GP')
ax.fill_between(x_disp, mu-3*std, mu+3*std, color=cm(4), alpha=0.2)
plt.xlim((-0.05, 1.05))
plt.ylim((-1.5, 2.0))
plt.legend(loc='upper right')
plt.show()

# + slideshow={"slide_type": "subslide"}
# graph_support.hide_code_in_slideshow(  )
sigma = 10e-2
gpr = GaussianProcessRegressor(kernel=rbf, alpha=sigma, n_restarts_optimizer=20)

# Reuse training data from previous 1D example
gpr.fit(X_train, y_train)

# Compute posterior predictive mean and covariance
mu_2, std_2 = gpr.predict(x_disp.reshape(-1, 1), return_std=True)
mu_2 = mu_2.flatten()

# + slideshow={"slide_type": "subslide"}
# Show data
fig, ax = plt.subplots(figsize=figsize)
ax.plot(X_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, mu_2, '-', c=cm(5), label=f'GP, $\\alpha={sigma}$')
ax.fill_between(x_disp, mu_2-2.*std_2, mu_2+2.*std_2,
                color=cm(5), alpha=0.2, label='$\pm 2\,\sigma$')
plt.xlim((-0.05, 1.05))
plt.ylim((-1.5, 2.0))
plt.legend(loc='upper right')
plt.show()

# + [markdown] slideshow={"slide_type": "slide"}
# ## Пример прикладной задачи ##
#
# **Многорежимное проектирование сопла**

# + [markdown] cell_style="center" slideshow={"slide_type": "subslide"}
# ### Постановка задачи ###
#
# **Объект:** Сопло сверхзвукового пассажирского самолёта
#
# **Цель:** Многорежимная оптимизация модельной геометрии сопла (режимы $M=0.9$ и $M=1.7$)
#
# **Варьируемые параметры:** 8 геометрических параметров
#
# **Метод решения**: Скаляризация целевой функции &mdash; преобразования векторной целевой функции в скалярную посредством взвешенной суммы её компонент: $F_{obj}\left(\vec{f}(\vec{x})\right) = \omega_1 f_1(\vec{x}) + \dots + \omega_n f_n(\vec{x})$.
#
# Многокритериальная задача сводится к множеству однокритериальных, каждая из которых характеризуется набором весовых коэффициентов $\Omega = \{\omega_1, \dots, \omega_n\}$.

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Процесс сходимости однокритериальной задачи ####

# + slideshow={"slide_type": "skip"}
import sys
sys.path.append('../scripts')
import graph_support

from IPython.display import Image
im_width = 1200

# + slideshow={"slide_type": "skip"} language="html"
# <style>
#     .container.slides .celltoolbar, .container.slides .hide-in-slideshow {display: None ! important;}
# </style>

# + cell_style="center" slideshow={"slide_type": "fragment"}
graph_support.hide_code_in_slideshow()
display(Image('../pix/01.Intro/res_sub030sup070.png', width=im_width))

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Фронт Парето ####

# + cell_style="center" slideshow={"slide_type": "fragment"}
graph_support.hide_code_in_slideshow()
display(Image('../pix/01.Intro/res_pareto_1-4.png', width=0.8*im_width))

# + [markdown] slideshow={"slide_type": "subslide"}
# #### Результаты ####
#
# Вариант $\mathbf \Omega = \{0.3, 0.7\}$

# + cell_style="center" slideshow={"slide_type": "fragment"}
graph_support.hide_code_in_slideshow()
print('Режим M=0.9')
display(Image('../pix/01.Intro/i095.r1m090.Mach.png', width=im_width))
print(40*'-')
print('Режим M=1.8')
display(Image('../pix/01.Intro/i095.r2m180.Mach.png', width=im_width))


# + [markdown] slideshow={"slide_type": "-"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Литература ##
#
# 1. *Воронцов К.В.* [Математические методы обучения по прецендентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). &mdash; 141 c.
#

# + slideshow={"slide_type": "skip"}
# Versions used
import sklearn
print('python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))
print('sklearn: {}'.format(sklearn.__version__))
# -


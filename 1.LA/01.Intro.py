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

# # Суррогатное моделирование и оптимизация в прикладных задачах

# ---

# **Лекция 1**
#
# # Информация о курсе. Постановка задачи

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Информация-о-курсе" data-toc-modified-id="Информация-о-курсе-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Информация о курсе</a></span><ul class="toc-item"><li><span><a href="#Аннотация" data-toc-modified-id="Аннотация-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Аннотация</a></span></li><li><span><a href="#Содержание-курса" data-toc-modified-id="Содержание-курса-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Содержание курса</a></span><ul class="toc-item"><li><span><a href="#Осенний-семестр" data-toc-modified-id="Осенний-семестр-1.2.1"><span class="toc-item-num">1.2.1&nbsp;&nbsp;</span>Осенний семестр</a></span></li><li><span><a href="#Весенний-семестр" data-toc-modified-id="Весенний-семестр-1.2.2"><span class="toc-item-num">1.2.2&nbsp;&nbsp;</span>Весенний семестр</a></span></li></ul></li></ul></li><li><span><a href="#Машинное-обучение" data-toc-modified-id="Машинное-обучение-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Машинное обучение</a></span><ul class="toc-item"><li><span><a href="#Определение" data-toc-modified-id="Определение-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Определение</a></span></li><li><span><a href="#Классификация-методов" data-toc-modified-id="Классификация-методов-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Классификация методов</a></span></li></ul></li><li><span><a href="#Постановка-задачи-восстановления-регрессии" data-toc-modified-id="Постановка-задачи-восстановления-регрессии-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Постановка задачи восстановления регрессии</a></span></li><li><span><a href="#Пример" data-toc-modified-id="Пример-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Пример</a></span><ul class="toc-item"><li><span><a href="#Данные" data-toc-modified-id="Данные-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Данные</a></span></li><li><span><a href="#Полиномиальная-регрессия" data-toc-modified-id="Полиномиальная-регрессия-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Полиномиальная регрессия</a></span></li><li><span><a href="#Регрессия-на-основе-гауссовских-процессов" data-toc-modified-id="Регрессия-на-основе-гауссовских-процессов-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Регрессия на основе гауссовских процессов</a></span></li></ul></li><li><span><a href="#Источники" data-toc-modified-id="Источники-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Источники</a></span></li></ul></div>

# + slideshow={"slide_type": "skip"}
# Imports
import numpy as np
import matplotlib.pyplot as plt

# + slideshow={"slide_type": "skip"}
# Styles, fonts
import matplotlib
matplotlib.rcParams['font.size'] = 12
# matplotlib.rcParams['lines.linewidth'] = 1.5
# matplotlib.rcParams['lines.markersize'] = 4
cm = plt.cm.tab10  # Colormap
figsize = (8, 5)

import seaborn
seaborn.set_style('whitegrid')

# +
from IPython.display import Image
im_width = 1000

# # %config InlineBackend.figure_formats = ['pdf']
# # %config Completer.use_jedi = False
# -

# ---

# + [markdown] slideshow={"slide_type": "notes"}
# ## Информация о курсе
#
# ### Аннотация
#
# Прогресс в сфере компьютерной техники и численных методов изменил задачи, с которыми сталкиваются современные учёные и инженеры, занимающиеся проектированием сложных технических систем.
# Например, расчёт аэродинамики самолёта с требуемой для практики точностью в настоящее время может быть проведён менее чем за один час.
# В результате, всё больше усилий исследователей направлено не на получение данных о характеристиках проектируемой системы, а на их анализ, интерпретацию и использование.
#
# Предлагаемый курс посвящён методам интеллектуального анализа данных и машинного обучения, применяемым при проектировании технических систем, а также лежащим в основе этих методов разделам математики: линейной алгебре и теории вероятностей.
# В программе содержится краткое изложение классических курсов по этим разделам, а также дополнительные главы, такие как сингулярное разложение матриц или случайные процессы.
# Среди методов машинного обучения рассматриваются метод главных компонент, метод наименьших квадратов, линейная регрессия, гребневая регрессия и др.
# Отдельный большой блок посвящён гауссовским случайным процессам, а также основанным на них методам суррогатного моделирования и оптимизации.
# Все рассматриваемые методы реализованы на языке Питон, а их применение проиллюстрировано на примере решения задач, взятых из реальной практики аэродинамического проектирования.
#
# Курс ориентирован на студентов 4-го и 5-го курсов, обучающихся по специальностям &laquo;Прикладные математика и физика&raquo; и &laquo;Прикладная математика и информатика&raquo;.
# -

# ### Содержание курса
#
# Цель настоящего курса &mdash; дать краткий обзор методов анализа данных, суррогатного моделирования и оптимизации, применяемых в задачах аэродинамического проектирования, и привести несколько примеров таких задач.
#
# Математическим базисом применяемых методов являются линейная алгебра и теория вероятностей.
# Соответственно с этим фундаментом структурирован материал курса: осенний семестр посвящён линейной алгебре, весенний &mdash; теории вероятностей.
# Каждый семестр начинается с краткого введения в каждую из этих дисциплин.

# #### Осенний семестр
#
# **Содержание**
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
#
#
# **Основная литература**
#
# 1. *Воронцов К.В.* [Математические методы обучения по прецедентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). &mdash; 141 c.
# 1. *Strang G.* Linear algebra and learning from data. &mdash; 2019. &mdash; 432 p.
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; 1983. &mdash; 336 с.
#
#
# **Дополнительная литература**
#
# 1. *Оселедец И.В.* [Numerical Linear Algebra](https://nla.skoltech.ru/index.html). Skoltech term course
# 1. *Martins J.R.R.A. & Ning A.* Engineering Design Optimization. &mdash; 2021. &mdash;  637 с.
# 1. *Гантмахер Ф.Р.* Теория матриц. &mdash;  М.: Наука, 1967. &mdash;  576 с.
# 1. *Стренг Г.* Линейная алгебра и её применения. &mdash; 1980.
#
#
# **Видеокурсы**
#
# 1. [Машинное обучение](https://www.youtube.com/watch?v=SZkrxWhI5qM&list=PLJOzdkh8T5krxc4HsHbB8g8f0hu7973fK&index=1), д.ф.-м.н. Константин Вячеславович Воронцов, ШАД (Яндекс)
# 1. [Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k), prof. Gilbert Strang, MIT
# 1. [Линейная алгебра](https://www.youtube.com/watch?v=WNl10xl1QT8&list=PLthfp5exSWEqSRXkZgMMzTSXL_WwMV9wK), к.ф.-м.н. Павел Александрович Кожевников, МФТИ

# #### Весенний семестр
#
# **Содержание**
#
# 1. Данные и методы работы с ними. Числовые характеристики выборки: среднее, среднеквадратичное отклонение, коэффициент корреляции
# 1. Основы теории вероятности: вероятностная модель, условная вероятность, формула Байеса
# 1. Случайные величины и их распределения, числовые характеристики случайных величин
# 1. Многомерное нормальное распределение, ковариационная матрица, условные распределения
# 1. Случайные процессы, гауссовские процессы
# 1. Регрессия на основе гауссовских процессов 
# 1. Оптимизация регрессионной кривой, влияние параметров ядра и амплитуды шума на регрессионную кривую
# 1. Алгоритм эффективной глобальной оптимизации
#
#
# **Основная литература**
#
# 1. *Ширяев А.Н.* Вероятность &mdash; 1. &mdash; М.: МЦНМО, 2007. &mdash; 517 с.
# 1. *Rasmussen C.E. & Williams C.K.I.* [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/). The MIT Press, 2006. 248 p.
#
#
# **Дополнительная литература**
#
# 1. Материалы по гауссовским процессам авторов [P. Roelants](https://peterroelants.github.io/) и [M. Krasser ](http://krasserm.github.io/)
# 1. TBD
#
#
# **Видеокурсы**
#
# 1. [Теория вероятностей](https://www.youtube.com/watch?v=Q3h9P7lhpNc&list=PLyBWNG-pZKx7kLBRcNW3HXG05BDUrTQVr&index=1), д.ф.-м.н. Максим Евгеньевич Широков, МФТИ
# 1. TBD

# ---

# ## Машинное обучение

# + [markdown] slideshow={"slide_type": "notes"}
# ### Определение
#
# **Машинное обучение** &mdash; подраздел искусственного интеллекта, изучающий методы построения алгоритмов, способных *обучаться*. \
# Компьютерная программа **учится** на опыте *E* по отношению к некоторому классу задач *T* и измерению производительности *P*, если её производительность *P* в задачах *T*, улучшается с опытом *E*.
#
# **Машинное обучение** &mdash; набор методов, которые могут *автоматически обнаруживать закономерности* в данных, а затем использовать обнаруженные закономерности для *прогнозирования* или для выполнения других видов *принятия решений*.
#
# **Машинное обучение** &mdash; класс методов искусственного интеллекта, характерной чертой которых является *не прямое решение* задачи, а *обучение* в процессе применения решений множества *сходных задач*.
#
# **Машинное обучение** (индуктивное обучение) основано на выявлении общих *закономерностей* по частным эмпирическим данным (*прецедентам*). Термины **машинное обучение** и **обучение по прецедентам** можно считать *синонимами*.
# -

# ### Классификация методов
#
# Различают три типа машинного обучения: *обучение с учителем*, *обучение без учителя* и *обучение с подкреплением*.
#
# 1. **Обучение с учителем** \
# Основная задача обучения с учителем состоит в том, чтобы найти функциональную зависимость ответов от описаний объектов и построить алгоритм, принимающий на входе описание объекта и выдающий на выходе ответ. \
# Задачи: *классификация* (ответ &mdash; метка принадлежности к классу) и *регрессия* (ответ &mdash; непрерывная величина).
#
# 1. **Обучение без учителя** \
# В обучении без учителя мы имеем дело с немаркированными данными или данными с неизвестной структурой.
# Используя методы обучения без учителя, мы можем разведать структуру данных с целью выделения содержательной информации без контроля со стороны известной результирующей переменной или функции вознаграждения. \
# Задачи: *кластеризация* (сгруппировать объекты в кластеры), *сокращение размерности* (перейти к меньшему числу новых признаков, потеряв при этом
# минимум существенной информации об объектах), *фильтрация выбросов* (обнаружить
# в обучающей выборке нетипичные объекты).
#
# 1. **Обучение с подкреплением** \
# В отличии от обучения с учителем, когда мы тренируем нашу модель, зная *правильный ответ* заранее, в обучении с подкреплением мы определяем меру *вознаграждения* за выполненные агентом отдельно взятые действия.

display(Image('./pix/01.Intro/ML_scheme_1.jpeg', width=im_width))

display(Image('./pix/01.Intro/ML_scheme_2.jpeg', width=im_width))

# + [markdown] slideshow={"slide_type": "subslide"}
# В рамках данного курса мы будем заниматься, в основном, регрессией и оптимизацией на основе регрессии.
# Но в самом начале будет полезно рассмотреть типичную постановку задачи восстановления регрессии.
# -

# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Постановка задачи восстановления регрессии
#
# Пусть задано множество объектов $X$ и множество допустимы ответов $Y$. Мы предполагаем существование зависимости $y:X \rightarrow Y$. При этом значения функции $y_i = y(x_i)$ известны только на конечном подмножестве объектов $\{x_1, \ldots, x_l\} \subset X$.
# Пары &laquo;объект &mdash; ответ&raquo; $(x_i, y_i)$ называются *прецедентами*, а совокупность пар $X^l = (x_i, y_i)_{i=1}^l$ &mdash; *обучающей выборкой*.
#
# *Признак* $f$ объекта $x$ &mdash; это результат измерения некоторой характеристики объекта.
#
# Пусть имеется набор признаков $f_1, \ldots, f_n$.
# Вектор $(f_1, \ldots, f_n)$ называют признаковым описанием объекта $x \in X$. В дальнейшем мы не будем различать объекты из $X$ и их признаковые описания.
# Совокупность признаковых описаний всех объектов выборки $X_l$, записанную в виде таблицы размера $l \times n$, называют матрицей объектов&ndash;признаков:
# $$
#   \mathbf{F} = 
#   \begin{pmatrix}
#     f_1(x_1) & \ldots & f_n(x_1) \\
#     \vdots   & \ddots & \vdots   \\
#     f_1(x_l) & \ldots & f_n(x_l) \\
#   \end{pmatrix}.
# $$
#
# Матрица объектов&ndash;признаков является стандартным и наиболее распространённым способом представления исходных данных в прикладных задачах.
# -

# **Задача**: построить функцию $a: X \rightarrow Y$, аппроксимирующую неизвестную целевую зависимость $y$. Функция должна корректно описывать обучающие данные и должна быть успешно применима для неизвестных тестовых данных.

# ---

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Пример
#
# > Регрессия &mdash; это наука о том, как провести линию через точки.
#
# **Задача (упрощённо)**: построить функцию корректно описывающую обучающие данные и обобщающую их на неизвестные (тестовые) данные.
# -

# ### Данные

# + slideshow={"slide_type": "skip"}
# Define the data
n = 20
x_lim = [0, 1]  # Limits
x_disp = np.linspace(*x_lim, 1001) # X array for display
# Underlying relation
x_train = np.linspace(*x_lim, n)   # Independent variable x
y_true = np.sin(3*np.pi*x_train)   # Dependent variable y
# Noise
np.random.seed(42)
e_std = 0.4  # Standard deviation of the noise
err = e_std * np.random.randn(n)   # Noise
# Output
y_train = y_true + err  # Dependent variable with noise

# + cell_style="center" slideshow={"slide_type": "subslide"}
# Show data
fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_disp,  np.sin(3*np.pi*x_disp), 'k:', label='clean data')
ax.plot(x_train, y_train, 'o-', markevery=1,c=cm(0), label='noisy data')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()
# -

# ### Полиномиальная регрессия

# + cell_style="center" slideshow={"slide_type": "fragment"}
deg = 1
p1 = np.polyfit(x_train, y_train, deg)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p1, x_disp), '-', c=cm(1), label=f'poly, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + cell_style="center" slideshow={"slide_type": "subslide"}
deg = 4
p2 = np.polyfit(x_train, y_train, deg)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p2, x_disp), '-', c=cm(2), label=f'poly, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + cell_style="center" slideshow={"slide_type": "fragment"}
deg = n-1
p3 = np.polyfit(x_train, y_train, deg)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p3, x_disp), '-', c=cm(3), label=f'poly, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()
# -

# ### Регрессия на основе гауссовских процессов

# + slideshow={"slide_type": "skip"}
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

# +
# Make 2D-array X_train
X_train = np.atleast_2d(x_train).T

# Define GP model
rbf = ConstantKernel(1.0) * RBF(length_scale=1.0)
gpr = GaussianProcessRegressor(kernel=rbf, n_restarts_optimizer=20)

# Train GP model
gpr.fit(X_train, y_train.reshape(-1, 1))

# Compute posterior predictive mean and covariance
mu, std = gpr.predict(x_disp.reshape(-1, 1), return_std=True)

# + slideshow={"slide_type": "subslide"}
# Show data
fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, mu, '-', c=cm(4), label='GP')
plt.xlim((-0.05, 1.05))
plt.ylim((-1.5, 2.0))
plt.legend(loc='upper right')
plt.show()

# +
# Add variance of noise (ridge regression)
sigma = 10e-2
gpr = GaussianProcessRegressor(kernel=rbf, alpha=sigma, n_restarts_optimizer=20)

# Train GP model
gpr.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

# Compute posterior predictive mean and covariance
mu_2, std_2 = gpr.predict(x_disp.reshape(-1, 1), return_std=True)

# + slideshow={"slide_type": "fragment"}
# Show data
fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, mu_2, '-', c=cm(5), label=f'GP, $\\sigma={sigma}$')
mu_2.flatten()
ax.fill_between(x_disp, mu_2.ravel()-2*std_2, mu_2.ravel()+2*std_2,
                color=cm(5), alpha=0.2, label='$\pm 2\,\sigma$')
plt.xlim((-0.05, 1.05))
plt.ylim((-1.5, 2.0))
plt.legend(loc='upper right')
plt.show()

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Источники
#
# 1. *Воронцов К.В.* [Математические методы обучения по прецедентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). &mdash; 141 c.

# + slideshow={"slide_type": "skip"}
# Versions used
import sys, sklearn
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))
print('sklearn: {}'.format(sklearn.__version__))
# -



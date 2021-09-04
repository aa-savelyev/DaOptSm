# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Анализ данных, суррогатное моделирование и птимизция в прикладных задачах #

# ---

# **Лекция 1**
#
# # Информация о курсе. Постановка задачи #

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Информация-о-курсе" data-toc-modified-id="Информация-о-курсе-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Информация о курсе</a></span><ul class="toc-item"><li><span><a href="#Аннотация" data-toc-modified-id="Аннотация-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Аннотация</a></span></li><li><span><a href="#Содержание-курса" data-toc-modified-id="Содержание-курса-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Содержание курса</a></span></li><li><span><a href="#Материалы" data-toc-modified-id="Материалы-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Материалы</a></span></li></ul></li><li><span><a href="#Три-типа-машинного-обучения" data-toc-modified-id="Три-типа-машинного-обучения-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Три типа машинного обучения</a></span></li><li><span><a href="#Постановка-задачи-восстановления-регрессии" data-toc-modified-id="Постановка-задачи-восстановления-регрессии-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Постановка задачи восстановления регрессии</a></span></li><li><span><a href="#Пример" data-toc-modified-id="Пример-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Пример</a></span><ul class="toc-item"><li><span><a href="#Данные" data-toc-modified-id="Данные-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Данные</a></span></li><li><span><a href="#Полиномиальная-регрессия" data-toc-modified-id="Полиномиальная-регрессия-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Полиномиальная регрессия</a></span></li><li><span><a href="#Регрессия-на-основе-гауссовских-процессов" data-toc-modified-id="Регрессия-на-основе-гауссовских-процессов-4.3"><span class="toc-item-num">4.3&nbsp;&nbsp;</span>Регрессия на основе гауссовских процессов</a></span></li></ul></li><li><span><a href="#Источники" data-toc-modified-id="Источники-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Источники</a></span></li></ul></div>

# + slideshow={"slide_type": "skip"}
# Imports
import numpy as np
import matplotlib.pyplot as plt

# + slideshow={"slide_type": "skip"}
# Styles
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 4
cm = plt.cm.tab10  # Colormap
figsize = (7, 4)

import seaborn
seaborn.set_style('whitegrid')
# -

# ---

# + [markdown] slideshow={"slide_type": "notes"}
# ## Информация о курсе ##
#
# ### Аннотация ###
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

# ### Содержание курса ###
#
# Цель настоящего курса &mdash; дать краткий обзор методов анализа данных, суррогатного моделирования и оптимизации, применяемых в задачах аэродинамического проектирования, и привести несколько примеров таких задач.
#
# Математическим базисом применяемых методов являются линейная алгебра и теория вероятностей.
# Соответственно с этим фундаментом структурирован материал курса: осенний семестр посвящён линейной алгебре, весенний &mdash; теории вероятностей.
# Каждый семестр начинается с краткого введения в каждую из этих дисциплин.
#
# **Осенний семестр**
#
# 1. Матрицы и действия над ними: $A=CR$, $A=QR$
# 1. Теория систем линейных уравнений: $A\mathbf{x} = \mathbf{b}$, $A=LU$
# 1. Спектральное разложение матриц: $A = X \Lambda X^{-1}$
# 1. Сингулярное разложение матриц: $A = U \Sigma V^\top$
# 1. Малоранговые аппроксимации матриц, метод главных компонент
# 1. Линейная регрессия, метод наименьших квадратов
# 1. Проблема мультиколлинеарности, гребневая регрессия, лассо Тибширани
# 1. Методы оптимизации: BFGS, SLSQP, COBYLA
#
# **Весенний семестр**
#
# 1. Данные и методы работы с ними. Числовые характеристики выборки: среднее, среднеквадратичное отклонение, коэффициент корреляции
# 1. Основы теории вероятности. Вероятностная модель, условная вероятность, формула Байеса
# 1. Случайные величины и их распределения, числовые характеристики случайных величин
# 1. Многомерное нормальное распределение, ковариационная матрица, условные распределения
# 1. Случайные процессы, гауссовские процессы
# 1. Регрессия на основе гауссовских процессов 
# 1. Оптимизация регрессионной кривой, влияние параметров ядра и амплитуды шума на регрессионную кривую
# 1. Алгоритм эффективной глобальной оптимизации

# + [markdown] slideshow={"slide_type": "slide"}
# ### Материалы ###
#
# В основе этого курса лежит достаточно много различных материалов &mdash; несколько десятков источников.
# Многие части текста взяты из них напрямую или переведены с английского.
# Материалы каждой лекции заканчиваются списком источников, относящихся именно к этой лекции.
# Здесь же приводится список основных, наиболее значимых из них.
#
# **Основная литература**
#
# 1. *Воронцов К.В.* [Математические методы обучения по прецедентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). &mdash; 141 c.
# 1. *Strang G.* Linear algebra and learning from data. &mdash; Wellesley-Cambridge Press, 2019. &mdash; 432 p.
# 1. *Ширяев А.Н.* Вероятность &mdash; 1. &mdash; М.: МЦНМО, 2007. &mdash; 517 с.
#
#
# **Дополнительная литература**
#
# 1. *Оселедец И.В.* [Numerical Linear Algebra](https://nla.skoltech.ru/index.html). Skoltech term course
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; М.: Наука, 1983. &mdash; 336 с.
# 1. Материалы по гауссовским процессам авторов [P. Roelants](https://peterroelants.github.io/) и [M. Krasser ](http://krasserm.github.io/)
# 1. *C.E. Rasmussen & C.K.I. Williams* [Gaussian Processes for Machine Learning](http://www.gaussianprocess.org/gpml/). The MIT Press, 2006. 248 p.
#
#
# **Видеокурсы**
#
# 1. [Машинное обучение](https://www.youtube.com/watch?v=SZkrxWhI5qM&list=PLJOzdkh8T5krxc4HsHbB8g8f0hu7973fK&index=1), д.ф.-м.н. Константин Вячеславович Воронцов, ШАД (Яндекс)
# 1. [Matrix Methods in Data Analysis, Signal Processing, and Machine Learning](https://www.youtube.com/watch?v=Cx5Z-OslNWE&list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k), prof. Gilbert Strang, MIT
# 1. [Линейная алгебра](https://www.youtube.com/watch?v=WNl10xl1QT8&list=PLthfp5exSWEqSRXkZgMMzTSXL_WwMV9wK), к.ф.-м.н. Павел Александрович Кожевников, МФТИ
# 1. [Теория вероятностей](https://www.youtube.com/watch?v=Q3h9P7lhpNc&list=PLyBWNG-pZKx7kLBRcNW3HXG05BDUrTQVr&index=1), д.ф.-м.н. Максим Евгеньевич Широков, МФТИ
# -

# ---

# + [markdown] slideshow={"slide_type": "notes"}
# ## Три типа машинного обучения ##
#
# **Машинное обучение** &mdash; класс методов искусственного интеллекта, характерной чертой которых является не прямое решение задачи, а обучение в процессе применения решений множества сходных задач. Для построения таких методов используются средства математической статистики, численных методов, методов оптимизации, теории вероятностей, теории графов, различные техники работы с данными в цифровой форме.
#
# Различают три типа машинного обучения: *обучение с учителем*, *обучение без учителя* и *обучение с подкреплением*.
#
# 1. **Обучение с учителем** \
# Основная задача обучения с учителем состоит в том, чтобы на маркированных *тренировочных данных* построить модель, которая позволит делать прогнозы для ранее не встречавшихся данных. Выделяют две основные задачи: классификация (ответ &mdash; метка принадлежности к классу) и регрессия (ответ &mdash; непрерывная величина).
#
# 1. **Обучение без учителя** \
# В обучении без учителя мы имеем дело с немаркированными данными или данными с неизвестной структурой.
# Используя методы обучения без учителя, мы можем разведать структуру данных с целью выделения содержательной информации без контроля со стороны известной результирующей переменной или функции вознаграждения.
# Задачи: кластеризация, снижение размерности.
#
# 1. **Обучение с подкреплением** \
# В отличии от обучения с учителем, когда мы тренируем нашу модель, зная *правильный ответ* заранее, в обучении с подкреплением мы определяем меру *вознаграждения* за выполненные агентом отдельно взятые действия.

# + [markdown] slideshow={"slide_type": "subslide"}
# В рамках данного курса мы будем заниматься, в основном, регрессией и оптимизацией на основе регрессии.
# Но в самом начале будет полезно рассмотреть типичную постановку задачи восстановления регрессии.
# -

# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Постановка задачи восстановления регрессии ##
#
# Пусть задано множество объектов $X$ и множество допустимы ответов $Y$. Мы предполагаем существование зависимости $y:X \rightarrow Y$. При этом значения функции $y_i = y(x_i)$ известны только на конечном подмножестве объектов $\{x_1, \ldots, x_l\} \subset X$.
# Пары &laquo;объект &mdash; ответ&raquo; $(x_i, y_i)$ называются *прецедентами*, а совокупность пар $X^l = (x_i, y_i)_{i=1}^l$ &mdash; *обучающей выборкой*.
#
# *Признак* $f$ объекта $x$ &mdash; это результат измерения некоторой характеристики объекта.
#
# Пусть имеется набор признаков $f_1, \ldots, f_n$.
# Вектор $(f_1, \ldots, f_n)$ называют признаковым описанием объекта $x \in X$. В дальнейшем мы не будем различать объекты из $X$ и их признаковые описания.
# Совокупность признаковых описаний всех объектов выборки $X_l$, записанную в виде таблицы размера $l \times n$, называют матрицей объектов &mdash; признаков:
# $$
#   \mathbf{F} = 
#   \begin{pmatrix}
#     f_1(x_1) & \ldots & f_n(x_1) \\
#     \vdots   & \ddots & \vdots   \\
#     f_1(x_l) & \ldots & f_n(x_l) \\
#   \end{pmatrix}.
# $$
#
# Матрица объектов &mdash; признаков является стандартным и наиболее распространённым способом представления исходных данных в прикладных задачах.
#
# **Задача**: построить функцию $a: X \rightarrow Y$, аппроксимирующую неизвестную целевую зависимость $y$. Функция должна корректно описывать обучающие данные и должна быть успешно применима для неизвестных тестовых данных.
# -

# ---

# + [markdown] slideshow={"slide_type": "subslide"}
# ## Пример ##
#
# > Регрессия &mdash; это наука о том, как провести линию через точки.
#
# **Задача (упрощённо)**: построить функцию корректно описывающую обучающие данные и обобщающую их на неизвестные (тестовые) данные.
# -

# ### Данные ###

# + slideshow={"slide_type": "skip"}
# Define the data
n = 20
x_lim = [0, 1]  # Limits
x_disp = np.linspace(*x_lim, 1001)  # X array for display
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
# -

# ### Полиномиальная регрессия ###

# + slideshow={"slide_type": "fragment"} cell_style="center"
deg = 1
p1 = np.polyfit(x_train, y_train, deg)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p1, x_disp), '-', c=cm(1), label=f'poly, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + slideshow={"slide_type": "subslide"} cell_style="center"
deg = 4
p2 = np.polyfit(x_train, y_train, deg)

fig, ax = plt.subplots(figsize=figsize)
ax.plot(x_train, y_train, 'o', c=cm(0), label='data: $(x,y)$')
ax.plot(x_disp, np.polyval(p2, x_disp), '-', c=cm(2), label=f'poly, deg={deg}')
plt.xlim((-0.05, 1.05))
plt.ylim((-2.5, 2.5))
plt.legend(loc='upper right')
plt.show()

# + slideshow={"slide_type": "fragment"} cell_style="center"
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

# ### Регрессия на основе гауссовских процессов ###

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
# ## Источники ##
#
# 1. *Воронцов К.В.* [Математические методы обучения по прецедентам (теория обучения машин)](http://www.machinelearning.ru/wiki/images/6/6d/Voron-ML-1.pdf). &mdash; 141 c.
# 1. *Рашка С.* Python и машинное обучение. &mdash; М.: ДМК Пресс, 2017. &mdash; 418 с.

# + slideshow={"slide_type": "skip"}
# Versions used
import sys, sklearn
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))
print('sklearn: {}'.format(sklearn.__version__))
# -



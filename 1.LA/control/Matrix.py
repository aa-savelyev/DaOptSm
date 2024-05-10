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

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Матрицы" data-toc-modified-id="Матрицы-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Матрицы</a></span></li><li><span><a href="#Сингулярное-разложение" data-toc-modified-id="Сингулярное-разложение-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Сингулярное разложение</a></span></li></ul></div>
# -

# # Вопросы по матрицам

# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Сингулярное-разложение" data-toc-modified-id="Сингулярное-разложение-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Сингулярное разложение</a></span></li><li><span><a href="#Число-обусловленности" data-toc-modified-id="Число-обусловленности-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Число обусловленности</a></span></li><li><span><a href="#Расстояние-Махаланобиса" data-toc-modified-id="Расстояние-Махаланобиса-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Расстояние Махаланобиса</a></span></li><li><span><a href="#Регрессия-L1" data-toc-modified-id="Регрессия-L1-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Регрессия L1</a></span></li></ul></div>

# Imports
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# +
# Styles
import matplotlib
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 4
cm = plt.cm.tab10  # Colormap

import seaborn
seaborn.set_style('whitegrid')

# +
# import warnings
# warnings.filterwarnings('ignore')

# # %config InlineBackend.figure_formats = ['pdf']
# # %config Completer.use_jedi = False
# -

# ## Матрицы
#
# 1. **Вопрос**: Как вычислить норму Фробениуса матрицы через её сингулярные числа? \
#    **Ответ**: Квадрат нормы Фробениуса равен сумме квадратов сингулярных чисел. \
#     Квадрат нормы Фробениуса = следу матрицы $A^\top A$ = сумме квадратов сингулярных чисел.

# ## Сингулярное разложение
#
# 1. **Вопрос**: Как соотносятся собственные и сингулярные числа матрицы? \
#    **Ответ**: В общем случае никак. \
#    Но если $S$ &mdash; симметричная матрица, то $S = Q\Lambda Q^\top = U\Sigma V^\top$. \
#    Если $S$ имеет отрицательные собственные числа ($S x = \lambda x$), то $\sigma = -\lambda$, а $u = -x$ или $v = -x$ (одно из двух). \
#    Сингулярные числа симметричной матрицы равны модулю её собственных чисел: $\sigma_i = |\lambda_i|$. \
#    Также можно показать, что $|\lambda| \le \sigma_1$. \
#    (Strang, p. 61)
#
# 1. **Вопрос**: Привести пример несимметричной матрицы, для которой справедливо $\sigma_i = |\lambda_i|$.\
#    **Ответ**: Кососимметричная матрица. Собственные числа будут комплексными, модуль которых будет равняться сингулярным числам.
#
# 1. **Вопрос**: Чему равны сингулярные числа ортогональной матрицы? \
#    **Ответ**: Все сингулярные числа ортогональной матрицы равны 1 (вспомним геометрический смысл).
#
# 1. **Вопрос**: Рассмотрим матрицу $2 \times 2$. \
#    В общем случае *4 разным элементам* (a, b, c, d) ставится в соответствие *4 геометрических параметра*: угол поворота ($\alpha$), два коэффициента растяжения ($\sigma_1, \sigma_2$), угол обратного поворота ($\beta$). \
#    Но если матрица симметричная, то параметра уже 3 (a, b, b, d). Как в таком случае вычислить четвёрку ($\alpha$, $\sigma_1, \sigma_2$, $\beta$)? \
#    **Ответ**: $\beta = -\alpha$. \
#    (Strang, p. 62)
#    
# 1. **Вопрос**: Какова связь между сингулярным и полярным разложением? \
#    **Ответ**: $A = U \Sigma V^\top = (U V^\top)(V \Sigma V^\top) = Q S$ или $A = U \Sigma V^\top = (U \Sigma U^\top)(U V^\top) = K Q$. \
#    (Strang, p. 67)
#    
# 1. **Вопрос**: Какова связь между сингулярными числами и собственными числами матрицы $S$ в полярном разложении? \
#    **Ответ**: Собственные числа $S$ &mdash; это сингулярные числа исходной матрицы $A$. \
#    (Strang, p. 67)




# Нормальность является удобным тестом приводимости к диагональной форме --- матрица нормальна тогда и только тогда, когда она унитарно подобна диагональной матрице, а потому любая матрица $A$, удовлетворяющая уравнению $A^{*}A=AA^{*}$, допускает приведение к диагональной форме.
#
# In our case the eigenvalues of $A$ are real.
# Then
# $$ A^*=(U^*DU)^*=U^*D^*U=U^*DU=A, $$
# as $D^*=D$, since the eigenvalues are real.

A = np.array([
    [1, 1, 0],
    [0, 1, 1],
    [1, 0, 1],
])

lmbd, U = LA.eig(A)
print('lambda = ', np.round(lmbd, 4))
print('U = ')
np.disp(np.round(U, 4))

LA.norm(lmbd[0])

# +
U, sgm, Vt = LA.svd(A)

print('sigma = ')
np.disp(sgm)
print('U = ')
np.disp(U)
print('Vt = ')
np.disp(Vt)
# -



B = np.array([
    [1, -1],
    [1, 1],
])

lmbd, U = LA.eig(B)
print('lambda = ', np.round(lmbd, 4))
print('U = ')
np.disp(np.round(U, 4))

# +
U, sgm, Vt = LA.svd(B)

print('sigma = ')
np.disp(sgm)
print('U = ')
np.disp(U)
print('Vt = ')
np.disp(Vt)
# -





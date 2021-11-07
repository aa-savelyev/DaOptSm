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

# # Вопросы к первому семестру

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Сингулярное-разложение" data-toc-modified-id="Сингулярное-разложение-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Сингулярное разложение</a></span></li></ul></div>
# -

# ## Сингулярное разложение
#
# 1. **Вопрос**: Как соотносятся собственные и сингулярные числа матрицы? \
#    **Ответ**: В общем случае никак. \
#    Но если $S$ &mdash; симметричная положительно определённая матрица, то $S = Q\Lambda Q^\top = U\Sigma V^\top$. \
#    Если $S$ имеет отрицательные собственные числа ($S x = \lambda x$), то $\sigma = -\lambda$, а $u = -x$ или $v = -x$ (одно из двух). \
#    (Strang, p. 61)
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

# ---

#







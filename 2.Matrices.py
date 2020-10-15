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
# **Лекция 2**
#
# # Матрицы и действия над ними #
# -

# ## Обозначения ##
#
# Для начала введём обозначения. Вектор будем записывать в виде столбца и обозначать стрелочкой $\vec{a}$ или жирной буквой $\mathbf{a}$.
# Строку будем обозначать с помощью звёздочки $\vec{a}^*$ или $\mathbf{a}^*$:
# $$
#   \mathbf{a} =
#   \begin{pmatrix}
#      a_1    \\
#      \cdots \\
#      a_n    \\
#   \end{pmatrix},
#   \mathbf{a}^* = (a_1, \ldots, a_n).
# $$

# ## Умножение матриц ##
#
# Рассмотрим 4 способа умножения матриц $A \cdot B = C$:
#
# 1. Строка на столбец
# 1. Столбец на строку
# 1. Столбец на столбец
# 1. Строка на строку

# ### Способ 1: &laquo;строка на столбец&raquo; ###
#
# Это стандартный способ умножения матриц: $\mathbf{c_{ij}} = \mathbf{a}_i^* \cdot \mathbf{b}_j$.
#
# **Пример 1.** Скалярное произведение векторов $\mathbf{a}$ и $\mathbf{b}$:
# $$ (\mathbf{a}, \mathbf{b}) = \mathbf{a}^\top \cdot \mathbf{b}. $$

# ### Способ 2: &laquo;столбец на строку&raquo; ###
#
# Существует другой способ умножения матриц.
# Произведение $AB$ равно *сумме произведений* всех столбцов матрицы $A$ на соотвествующие строки матрицы $B$.
# $$ A \cdot B = \sum_i \mathbf{a}_i \cdot \mathbf{b}_i^*. $$
#
# **Пример 2.**
# $$
#   \begin{pmatrix}
#      1 & 2 \\
#      0 & 2 \\
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#      0 & 5 \\
#      1 & 1 \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#      1 \\
#      0 \\
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#      0 & 5 \\
#   \end{pmatrix}
#   +
#   \begin{pmatrix}
#      2 \\
#      2 \\
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#      1 & 1 \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#      0 & 5 \\
#      0 & 0 \\
#   \end{pmatrix}
#   +
#   \begin{pmatrix}
#      2 & 2 \\
#      2 & 2 \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#      2 & 7 \\
#      2 & 2 \\
#   \end{pmatrix}.
# $$

# **Пример 3.** Разложение матрицы на сумму матриц ранга 1.
#
# Представим матрицу в виде произведения двух матриц и применим умножение &laquo;столбец на строку&raquo;:
# $$
#   A = 
#   \begin{pmatrix}
#      1 & 2 & 3 \\
#      2 & 1 & 3 \\
#      3 & 1 & 4 \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#      1 & 2 \\
#      2 & 1 \\
#      3 & 1 \\
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#      1 & 0 & 1 \\
#      0 & 1 & 1 \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#      1 & 0 & 1 \\
#      2 & 0 & 2 \\
#      3 & 0 & 3 \\
#   \end{pmatrix}
#   +
#   \begin{pmatrix}
#      0 & 2 & 2 \\
#      0 & 1 & 1 \\
#      0 & 1 & 1 \\
#   \end{pmatrix}.
# $$

# ### Способ 3: &laquo;столбец на столбец&raquo; ###
#
# Посмотрим внимательно на то, как мы получили первый столбец результирующней матрицы.
# Он является суммой столбцов матрицы $A$ с коэффициентами из первого столбца матрицы $B$!
# То же самое можно сказать и про второй столбец.
# Таким образом, можно сформулировать следующее правило: \
# *$\mathbf{i}$-ый столбец результирующей матрицы есть линейная комбинация столбцов левой матрицы с коэффициентами из $\mathbf{i}$-ого столбца правой матрицы.*

# ### Способ 4: &laquo;строка на строку&raquo; ###
#
# Аналогичным образом можно вывести правило и для строк: \
# *$\mathbf{i}$-ая строка результирующей матрицы есть линейная комбинация строк правой матрицы с коэффициентами из $\mathbf{i}$-ой строки левой матрицы.*

# + [markdown] slideshow={"slide_type": "slide"}
# **Пример 4. Умножение матрицы на вектор**
#
# Рассмотрим умножение матрицы $A$ размером $m \times n$ на вектор: $A \mathbf{x}$.
#
# По определению произведение $A \mathbf{x}$ есть вектор, в котором на $i$-ом месте находится скалярное произведение $i$-ой *строки* на столбец $\mathbf{x}$:
# $$
#   A \mathbf{x} = 
#   \begin{pmatrix}
#     -\, \mathbf{a}_1^* \,- \\
#     \cdots \\
#     -\, \mathbf{a}_i^* \,- \\
#     \cdots \\
#     -\, \mathbf{a}_m^* \,- \\
#   \end{pmatrix}
#   \cdot \mathbf{x} = 
#   \begin{pmatrix}
#     \mathbf{a}_1^* \cdot \mathbf{x} \\
#     \cdots \\
#     \mathbf{a}_i^* \cdot \mathbf{x} \\
#     \cdots \\
#     \mathbf{a}_m^* \cdot \mathbf{x} \\
#   \end{pmatrix}.
# $$
#
# Но можно посмотреть на это иначе, как на произведение *столбцов* матрицы $A$ на элементы вектора $\mathbf{x}$:
# $$
#   A \mathbf{x} = 
#   \begin{pmatrix}
#      | & {} & | & {} & | \\
#      \mathbf{a}_1 & \cdots & \mathbf{a}_i & \cdots & \mathbf{a}_n \\
#      | & {} & | & {} & | \\
#   \end{pmatrix}
#   \begin{pmatrix}
#      x_1    \\
#      \cdots \\
#      x_i    \\
#      \cdots \\
#      x_m \\
#   \end{pmatrix}
#   = 
#   x_1 \mathbf{a}_1 + \dots + x_i \mathbf{a}_i + \dots + x_m \mathbf{a}_m.
# $$
# Таким образом, результирующий вектор есть *линейная комбинация* столбцов матрицы $A$ с коэффициентами из вектора $\mathbf{x}$.
# -

# **Пример 5. Умножение столбцов (строк) матрицы на скаляры**
#
# Чтобы каждый вектор матрицы $A$ умножить на скаляр $\lambda_i$, нужно умножить $A$ на матрицу $\Lambda = \mathrm{diag}(\lambda_i)$ *справа*:
# $$
#   \begin{pmatrix}
#     | & {} & | \\
#     \lambda_1 \mathbf{a}_1 & \cdots & \lambda_n \mathbf{a}_n \\
#     | & {} & | \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     | & {} & | \\
#     \mathbf{a}_1 & \cdots & \mathbf{a}_n \\
#     | & {} & | \\
#   \end{pmatrix}
#   \begin{pmatrix}
#     \lambda_{1} & \ldots & 0         \\
#     \vdots      & \ddots & \vdots    \\
#     0           & \ldots & \lambda_n \\
#   \end{pmatrix}
#   = A \cdot \Lambda.
# $$
#
# Чтобы проделать то же самое со строками матрицы, её нужно умножить на $\Lambda$ *слева*:
# $$
#   \begin{pmatrix}
#     -\, \lambda_1 \mathbf{a}_1^* \,- \\
#     \cdots \\
#     -\, \lambda_m \mathbf{a}_m^* \,- \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     \lambda_{1} & \ldots & 0         \\
#     \vdots      & \ddots & \vdots    \\
#     0           & \ldots & \lambda_n \\
#   \end{pmatrix}
#   \begin{pmatrix}
#     -\, \mathbf{a}_1^* \,- \\
#     \cdots \\
#     -\, \mathbf{a}_m^* \,- \\
#   \end{pmatrix}
#   = \Lambda \cdot A.
# $$

# **Пример 6. Умножение блочных матриц**
#
# Рассмотрим блочную матрицу следующего вида: 
# $$ A = \begin{pmatrix} A_1 \\ A_2 \\ \end{pmatrix}. $$
#
# 1. Формулу для $A A^\top$ получим по способу &laquo;строка на столбец&raquo;:
# $$
#   A A^\top = 
#   \begin{pmatrix}
#     A_1 \\
#     A_2 \\
#   \end{pmatrix}
#   \cdot
#   \begin{pmatrix}
#     A_1^\top & A_2^\top \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     A_1 A_1^\top & A_1 A_2^\top \\
#     A_2 A_2^\top & A_2 A_2^\top \\
#   \end{pmatrix}
# $$
# 2. Для $A^\top A$ удобно применить способ &laquo;столбец на строку&raquo;:
# $$
#   A^\top A =
#   \begin{pmatrix}
#     A_1^\top & A_2^\top \\
#   \end{pmatrix}
#   \begin{pmatrix}
#     A_1 \\
#     A_2 \\
#   \end{pmatrix}
#   = A_1^\top A_1 + A_2^\top A_2
# $$

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Литература ##
#
# 1. G. Strang. Linear algebra and learning from data. Wellesley-Cambridge Press, 2019. 432 p.
# -



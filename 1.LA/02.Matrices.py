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

# + [markdown] slideshow={"slide_type": "slide"}
# **Лекция 2**
#
# # Матрицы и действия над ними. Ранг матрицы #

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Введение" data-toc-modified-id="Введение-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Введение</a></span><ul class="toc-item"><li><span><a href="#Основные-задачи" data-toc-modified-id="Основные-задачи-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Основные задачи</a></span></li><li><span><a href="#Обозначения" data-toc-modified-id="Обозначения-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Обозначения</a></span></li></ul></li><li><span><a href="#Умножение-матриц" data-toc-modified-id="Умножение-матриц-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Умножение матриц</a></span><ul class="toc-item"><li><span><a href="#Способ-1:-«строка-на-столбец»" data-toc-modified-id="Способ-1:-«строка-на-столбец»-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Способ 1: «строка на столбец»</a></span></li><li><span><a href="#Способ-2:-«столбец-на-строку»" data-toc-modified-id="Способ-2:-«столбец-на-строку»-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Способ 2: «столбец на строку»</a></span></li><li><span><a href="#Способ-3:-«столбец-на-столбец»" data-toc-modified-id="Способ-3:-«столбец-на-столбец»-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Способ 3: «столбец на столбец»</a></span></li><li><span><a href="#Способ-4:-«строка-на-строку»" data-toc-modified-id="Способ-4:-«строка-на-строку»-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Способ 4: «строка на строку»</a></span></li></ul></li><li><span><a href="#Ранг-матрицы" data-toc-modified-id="Ранг-матрицы-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Ранг матрицы</a></span></li><li><span><a href="#Скелетное-разложение" data-toc-modified-id="Скелетное-разложение-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Скелетное разложение</a></span></li><li><span><a href="#Источники" data-toc-modified-id="Источники-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Источники</a></span></li></ul></div>
# -

# ---

# ## Введение ##
#
# ### Основные задачи ###
#
# Что мы хотим вспомнить из линейной алгебры?
# Мы хотим уметь решать пять базовых задач:
#
# 1. Найти $\mathbf{x}$ в уравнении $A\mathbf{x} = \mathbf{b}$,
# 1. Найти $\mathbf{x}$ и $\lambda$ в уравнении $A\mathbf{x} = \lambda \mathbf{x}$,
# 1. Найти $\mathbf{v}$, $\mathbf{u}$ и $\sigma$ в уравнении $A\mathbf{v} = \sigma \mathbf{u}$.
#
# Подробнее о задачах:
# 1. Можно ли вектор $\mathbf{b}$ представить в виде линейной комбинации векторов матрицы $A$?
# 1. Вектор $A\mathbf{x}$ имеет то же направление, что и вектор $\mathbf{x}$. Вдоль этого направления все сложные взаимодействия с матрицей $A$ чрезвычайно упрощаются.
#    Например, вектор $A^2 \mathbf{x}$ становится просто $\lambda^2 \mathbf{x}$.
#    Упрощается вычисление матричной экспоненты: $e^{A} = e^{X \Lambda X^{-1}} = X e^{\Lambda} X^{-1}$.
#    Короче говоря, многие действия становятся линейными.
# 1. Уравнение $A\mathbf{v} = \sigma \mathbf{u}$ похоже на предыдущее. Но матрица $A$ больше не квадратная. Это сингулярное разложение.

# ### Обозначения ###
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

# ---

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
# Произведение $AB$ равно *сумме произведений* всех столбцов матрицы $A$ на соответствующие строки матрицы $B$.
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
# Посмотрим внимательно на то, как мы получили первый столбец результирующей матрицы.
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
#     A_2 A_1^\top & A_2 A_2^\top \\
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
# -

# ## Ранг матрицы ##
#
# Посмотрим на задачу $A \mathbf{x} = \mathbf{b}$. \
# Её можно сформулировать в виде следующего вопроса: можно ли столбец $\mathbf{b}$ представить в виде линейной комбинации столбцов матрицы $A$?
#
# Поясним на примере.
# Пусть
# $$
#   A = 
#   \begin{pmatrix}
#      1 & 2 & 3 \\
#      2 & 1 & 3 \\
#      3 & 1 & 4 \\
#   \end{pmatrix}.
# $$
# Что мы можем сказать о линейной оболочке её столбцов? Что это за пространство? Какой размерности?

# **Определение 1.** Рангом матрицы $A$ с $m$ строк и $n$ столбцов называется максимальное число линейно независимых столбцов (строк).

# **Свойства ранга:**
#
# 1. $r(AB)  \le r(A), r(B)$,
# 1. $r(A+B) \le r(A) + r(B)$,
# 1. $r(A^\top A) = r(AA^\top) = r(A) = r(A^\top)$,
# 1. Пусть $A: m \times r$, $B: r \times n$ и $r(A) = r(B) = r$, тогда $r(AB) = r$.
#
# **Доказательства:**
#
# 1. При умножении матриц ранг не может увеличиться. Каждый столбец матрицы $AB$ является линейной комбинацией столбцов матрицы $A$, а каждая строка матрицы $AB$ является линейной комбинацией строк матрицы $B$. Поэтому пространство столбцов матрицы $AB$ содержится в пространстве столбцов матрицы $A$, а пространство строк матрицы $AB$ содержится в пространстве строк матрицы $B$.
# 1. Базис пространства столбцов матрицы $A+B$ ($\mathbf{C}(A+B)$) является комбинацией (возможно, с пересечениями) базисов пространств $\mathbf{C}(A)$ и $\mathbf{C}(B)$.
# 1. Матрицы $A$ и $A^\top A$ имеют одно и то же нуль-пространство (доказать), поэтому их ранг одинаков.
# 1. Матрицы $A^\top A$ и $BB^\top$ невырождены, так как $r(A^\top A) = r(BB^\top) = r$. Их произведение, матрица $A^\top A BB^\top$, тоже невырождена и её ранг равен $r$. Отсюда $r = r(A^\top A BB^\top) \le r(AB) \le r(A) = r$.

# >Как доказать, что $A$ и $A^\top A$ имеют одно и то же нуль-пространство? \
# >Если $Ax=0$, то $A^\top Ax = 0$. Поэтому $\mathbf{N}(A) \subset \mathbf{N}(A^\top A)$. \
# >Если $A^\top Ax = 0$, то $x^\top A^\top Ax = \|Ax\|^2 = 0$. Поэтому $\mathbf{N}(A^\top A) \subset \mathbf{N}(A)$.

# *Замечание.* Свойство 4 работает только в случае, когда $A$ имеет *ровно* $r$ столбцов, а $B$ имеет *ровно* $r$ строк. В частности, $r(BA) \le r$ (в соответствии со свойством 1).
#
# Для свойства 3 отметим важный частный случай. Это случай, когда столбцы матрицы $A$ линейно независимы, так что её ранг $r$ равен $n$. Тогда матрица $A^\top A$ является квадратной симметрической обратимой матрицей.

# ---

# ## Скелетное разложение ##
#
# В матрице $A$ первые два вектора линейно независимы.
# Попробуем взять их в качестве базиса и разложить по ним третий.
# Запишем это в матричном виде, пользуясь правилом умножения &laquo;столбец на столбец&raquo;.
#
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
#   = C \cdot R
# $$
#
# Здесь $C$ &mdash; базисные столбцы, $R$ &mdash; базисные строки.
# Мы получили скелетное разложение матрицы.
#
# **Определение 2.** *Скелетным разложением* матрицы $A$ размеров $m \times n$ и ранга $r>0$ называется разложение вида $A = CR$, где матрицы $C$ и $R$ имеют размеры соответственно $m \times r$ и $r \times n$.
# Другое название скелетного разложения &mdash; *ранговая факторизация*.
#
# Это разложение иллюстрирует **теорему**: ранг матрицы по столбцам (количество независимых столбцов) равен рангу матрицы по строкам (количество независимых строк).

# **Дополнительно**
#
# Существует другой вариант скелетного разложения: $A = CMR$.
# В этом случае матрица $С$, как и ранее, состоит из $r$ независимых столбцов матрицы $A$, но матрица $R$ теперь состоит из $r$ независимых строк матрицы $A$, а матрица $M$ размером $r \times r$ называется смешанной матрицей (mixing matrix).
# Для $M$ можно получить следующую формулу:
# $$
#   A = CMR \\
#   C^\top A R^\top = C^\top C M R R^\top \\
#   M = \left[ (C^\top C)^{-1}C^\top \right] A \left[ R^\top (R R^\top)^{-1} \right].
# $$

# > Матрицы $C^+ = (C^\top C)^{-1}C^\top$ и $R^+ = R^\top (R R^\top)^{-1}$ являются *псевдообратными* к матрицам соответственно $C$ и $R$. \
# > Можно показать, что если *столбцы* матрицы линейно независимы (как у матрицы $C$), то $C^+$ является *левой* обратной матрицей для $C$: $C^+ C = I$.
# > Если независимы строки (как у $R$), то $R^+$ &mdash; *правая* обратная матрица для $R$: $R R^+ = I$. \
# > Более подробный материал о псевдообратных матрицах будет на несколько занятий позже.

# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Источники ##
#
# 1. *Strang G.* Linear algebra and learning from data. &mdash; Wellesley-Cambridge Press, 2019. &mdash; 432 p.
# 1. *Гантмахер Ф.Р.* Теория матриц. &mdash; М.: Наука, 1967. &mdash; 576 с.
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; М.: Наука, 1983. &mdash; 336 с.
# -


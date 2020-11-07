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
# # Ортогональное проектирование #
# -

# ## План ##
#
# 1. Метрика, ортогональные векторы и матрицы
# 1. Ортогонализация ряда векторов, $QR$-разложение
# 1. Ортогональное проектирование
# 1. Метод наименьших квадратов

# ---

# ## Метрика пространства ##
#
#
# ### Скалярное произведение ###
#
# Мы уже рассматривали линейный операторы в произвольном $n$-мерном линейном пространстве.
# Все базисы такого пространства равноправны между собой.
# Данному линейному оператору в каждом базисе соответствует некоторая матрица.
# Матрицы, отвечающие одному и тому же оператору в различных базисах, подобны между собой.
#
# Теперь давайте в наше $n$-мерное пространство введём метрику.
# Для этого каждым двум векторам $\mathbf{x}$ и $\mathbf{y}$ мы специальным образом поставим в соответствие некоторое число $(\mathbf{x}, \mathbf{y})$ &mdash; их *скалярное произведение*.
#
# Для любых векторов $\mathbf{x}$, $\mathbf{y}$, $\mathbf{z}$ и любого числа $\alpha$ справедливы следующие **свойства скалярного произведения**:
#
# 1. $(\mathbf{x}, \mathbf{y}) = (\mathbf{y}, \mathbf{x})$ (коммутативность),
# 1. $(\alpha \mathbf{x}, \mathbf{y}) = \alpha (\mathbf{x}, \mathbf{y})$,
# 1. $(\mathbf{x} + \mathbf{y}, \mathbf{z}) = (\mathbf{x}, \mathbf{z}) + (\mathbf{y}, \mathbf{z})$ (дистрибутивность),
# 1. $(\mathbf{x}, \mathbf{x}) > 0$ при $\mathbf{x} \ne 0$.
#
# **Определение 1.** Векторное пространство с положительно определённой евклидовой метрикой называется **евклидовым пространством**.
#
# **Длиной вектора** называется $|\mathbf{x}| = \sqrt{(\mathbf{x}, \mathbf{x})}$.
#
# **Косинус угла** между двумя векторами равен
# $$ cos(\theta) = \frac{(\mathbf{x}, \mathbf{y})}{|\mathbf{x}| |\mathbf{y}|}. $$

# ### Ортогональные векторы ###
#
# Два вектора $\mathbf{x}$ и $\mathbf{y}$ называются **ортогональными** ($\mathbf{x} \perp \mathbf{y}$), если $(\mathbf{x}, \mathbf{y}) = 0$.
#
# Рассмотрим матрицу $Q$, столбцами которой являются ортонормированные векторы.
# Легко видеть, что $Q^\top Q = E$. 
#
# Отсюда следует, что $\forall \mathbf{x}, \mathbf{y}: (Q\mathbf{x}, Q\mathbf{y}) = (Q\mathbf{x})^\top (Q\mathbf{y}) = \mathbf{x}^\top Q^\top Q \mathbf{y} = (\mathbf{x}, \mathbf{y})$.
# В частности, при умножении на $Q$ длина вектора не меняется: $|Q\mathbf{x}| = (Q\mathbf{x}, Q\mathbf{x}) = (\mathbf{x}, \mathbf{x}) = |\mathbf{x}|.$
#
# Что такое матрица $Q Q^\top$?
#
# $$
#   Q Q^\top = \mathbf{q}_1 \mathbf{q}_1^\top + \ldots + \mathbf{q}_n \mathbf{q}_n^\top
# $$
#
# > $QQ^\top$ &mdash; проекционная матрица.
#
# Если $Q$ &mdash; квадратная матрица, то $Q Q^\top$ также является единичной матрицей: $Q Q^\top = E$.

# ### Ортогональные векторы и матрицы ###
#
# **Определение 2.** **Ортогональной матрицей** называется квадратная матрица, столбцами которой являются *ортонормированные* векторы.
#
# Линейные преобразования, соответствующие ортогональным матрицам, представляют собой некоторые &laquo;движения&raquo;.
# В том смысле, что сохраняются углы и длины.
#
# **Пример 1.** Матрица поворота
# $$
#   Q_\mathrm{rotate} = 
#   \begin{pmatrix}
#      \cos\theta & -\sin\theta \\
#      \sin\theta &  \cos\theta \\
#   \end{pmatrix}
# $$
#
# **Пример 2.** Матрица отражения
# $$
#   Q_\mathrm{reflect} = 
#   \begin{pmatrix}
#      \cos\theta &  \sin\theta \\
#      \sin\theta & -\cos\theta \\
#   \end{pmatrix}
# $$
#
# **Пример 3.** Произведение двух ортогональных матриц $Q_1 Q_2$ &mdash; ортогональная матрица.
#
# > Вращение на вращение = вращение, отражение на отражение = отражение, вращение на отражение = отражение.

# ---

# ## Ортогональное проектирование ##
#
# ### Общий случай ###
#
# Пусть заданы вектор $\mathbf{b}$ из $\mathbb{R}_m$ и набор векторов $\mathbf{a}_i$, образующих базис в $\mathbb{R}_n$ ($n<m$).
#
# Требуется найти ортогональную проекцию вектора $\mathbf{b}$ на линейную оболочку векторов $\mathbf{a}_i$.
#
# Искомый вектор является линейной комбинацией базисных векторов $\mathbf{a}_i$ с неизвестными коэффициентами $x_i$.
# Запишем это в виде произведения матрицы $A$, столбцы которой являются векторами $\mathbf{a}_i$, на неизвестный вектор $\mathbf{x}$: $A\mathbf{x}$.
# Мы ищем ортогональную проекцию на пространство столбцов матрицы $A$, поэтому вектор $\mathbf{e} = A\mathbf{x} - \mathbf{b}$ должен быть ортогонален *любому* вектору $A\mathbf{y}$.
# Запишем это через скалярное произведение:
# $$
#   (A \mathbf{y})^\top(A\mathbf{x} - \mathbf{b}) = \mathbf{y}^\top (A^\top A \mathbf{x} - A^\top \mathbf{b}) = 0.
# $$
#
# Это справедливо для произвольного вектора $\mathbf{y}$, откуда следует, что
# $$
#   A^\top A \mathbf{x} = A^\top \mathbf{b}.
# $$
#
# Ранг матрицы $A^\top A$ равен рангу $A$, а столбцы матрицы $A$ линейно независимы, следовательно матрица $A^\top A$ обратима.
# Отсюда находим выражение для вектора коэффициентов проекции и сам вектор проекции $\mathbf{p} = A\mathbf{x}$
# $$
#   \mathbf{x} = (A^\top A)^{-1} A^\top \mathbf{b}, \
#   \mathbf{p} = A (A^\top A)^{-1} A^\top \mathbf{b}.
# $$
#
# Матрица $P = A (A^\top A)^{-1} A^\top$, осуществляющая проекцию, называется *матрицей ортогонального проектирования*.
# Матрица ортопроектирования обладает двумя основными свойствами:
#
# 1. $P^2 = P$ &mdash; характеристическое свойство всех проекторов (*идемпотентность*),
# 1. $P^\top = P$ &mdash; отличительное свойство ортогонального проектора (*симметричность*).

# ### Одномерный случай ###
#
# Матрица проектирования на прямую, определённую вектором $\mathbf{a}$, задаётся формулой
# $$ P = A (A^\top A)^{-1} A^\top = \mathbf{a} (\mathbf{a^\top}\mathbf{a})^{-1}\mathbf{a^\top} = \frac{1}{\mathbf{a^\top}\mathbf{a}} \mathbf{a} \mathbf{a^\top}. $$
#
# А проекция $\mathbf{p}$ точки $\mathbf{b}$ &mdash; формулой
# $$ \mathbf{p} = \frac{\mathbf{a^\top} \mathbf{b}}{\mathbf{a^\top}\mathbf{a}}\mathbf{a}. $$

# ### Проектирование на линейную оболочку ортонормированных векторов ###
#
# Если набор векторов образует ортонормированный базис $\mathbf{q}_i$, то формула для матрицы ортогонального проектирования существенно упрощается:
# $$
#   P = Q (Q^\top Q)^{-1} Q^\top = QQ^\top = \mathbf{q}_1 \mathbf{q}_1^\top + \ldots + \mathbf{q}_n \mathbf{q}_n^\top.
# $$
#
# Каждое слагаемое $\mathbf{q}_i \mathbf{q}_i^\top$ является матрицей ортогональной проекции на вектор $\mathbf{q}_i$.
#
# Таким образом, *когда оси координат взаимно перпендикулярны, проекция на пространство разлагается в сумму проекций на каждую из осей*.

# ### Ортонормированный базис ###
#
# Разложение любого вектора по ортонормированному базису есть сумма ортогональных проекций  на каждый вектор:
#
# $$
#   \mathbf{v} = c_1 \mathbf{q}_1 + \ldots + c_n \mathbf{q}_n = (\mathbf{q}_1^\top \mathbf{v}) \cdot \mathbf{q}_1 + \ldots + (\mathbf{q}_n^\top \mathbf{v}) \cdot \mathbf{q}_n.
# $$

# ---

# ## Ортогонализация векторов ##
#
# ### Алгоритм Грама &mdash; Шмидта ###
#
# Два ряда векторов называются **эквивалентными** если они содержат одинаковое количество векторов и их линейные оболочки совпадают.
#
# Под **ортогонализацией** ряда векторов будем понимать замену этого ряда на *эквивалентный* (порождающий ту же самую линейную оболочку) ортогональный ряд.
#
# **Теорема.** Всякий невырожденный ряд векторов можно проортогонализировать.
#
# Алгоритм Грама &mdash; Шмидта рассмотрим на примере трёхмерного пространства.
#
# Пусть даны три линейно-независимых вектора $\mathbf{a}_1$, $\mathbf{a}_2$, $\mathbf{a}_3$.
#
# 1. $\mathbf{q}_1 = \mathbf{a}_1 / |\mathbf{a}_1|$
# 1. $\mathbf{\hat{a}}_2 = \mathbf{a}_2 - (\mathbf{a}_2^\top \mathbf{q}_1) \cdot \mathbf{q}_1$; $\quad \mathbf{q}_2 = \mathbf{\hat{a}}_2 / |\mathbf{\hat{a}}_2|$
# 1. $\mathbf{\hat{a}}_3 = \mathbf{a}_3 - (\mathbf{a}_3^\top \mathbf{q}_1) \cdot \mathbf{q}_1 - (\mathbf{a}_3^\top \mathbf{q}_2) \cdot \mathbf{q}_2$; $\quad \mathbf{q}_3 = \mathbf{\hat{a}}_3/|\mathbf{\hat{a}}_3|$

# ### $\mathbf{QR}$-разложение ###
#
# #### Квадратные матрицы ####
#
# Большую роль в численных методах играет разложение *квадратной матрицы* $A$ на ортогональную матрицу $Q$ и верхнетреугольную матрицу $R$.
# $$
#   A = QR.
# $$
#
# **Предложение 1.** Любая квадратная матрица $A$ может быть разложена в произведение $QR$.
#
# **Предложение 2.** Если матрица $A$ невырождена, то её $QR$-разложение, в котором диагональные элементы $R$ положительны, единственно.
#
# $QR$-разложение тесно связано с процессом ортогонализации Грама &mdash; Шмидта.
# Действительно, $AU = Q$ равносильно $QR$-разложению с $R = U^{-1}$.
# Причём, $U$ и $R$ &mdash; верхнетреугольная матрица.
#
# > Все неособенные верхние (нижние) треугольные матрицы составляют группу (некоммутативную) относительно умножения.
#
# > *Группой* называется всякая совокупность объектов, в которой установлена операция, относящая любым двум элементам $a$ и $b$ совокупности определённый третий элемент $a \ast b$ той же совокупности, если 1) операция обладает сочетательным свойством, 2) в совокупности существует единичный элемент, 3) для любого элемента совокупности существует обратный элемент.
# > Группа называется *коммутативной*, если групповая операция обладает переместительным свойством.
#
# **Пример.** Рассмотрим систему линейных уравнений $A\mathbf{x} = \mathbf{b}$. Используя $QR$-разложение матрицы $A$, получим
#
# $$
#   QR \mathbf{x} = \mathbf{b}, \\
#   R \mathbf{x} = Q^\top \mathbf{b}.
# $$
#
# И в случае невырожденной $R$
# $$
#   \mathbf{x} = R^{-1}Q^\top \mathbf{b}.
# $$

# #### Прямоугольные матрицы ####
#
# Произвольную матрицу размером $m \times n$ можно представить в виде разложения
# $$
#   A = QR,
# $$
# где $Q$ &mdash; ортогональная матрица порядка $m$, а $R$ &mdash; матрица размеров $m \times n$, элементы $r_{ij}$ которой удовлетворяют условию $r_{ij}=0$ при $i>j$.
# Такое разложение назовём $Qr$-разложением, чтобы подчеркнуть, что матрица $R$, вообще говоря, не является квадратной.
#
# Второе обобщение $QR$-разложения можно получить, применяя метод ортогонализации Грама &mdash; Шмидта.
# Теперь матрица $Q$ размеров $m \times n$ может рассматриваться как совокупность $n$ столбцов из некоторой ортогональной матрицы порядка $m$, а $R$ &mdash; квадратная верхняя треугольная матрица порядка $n$.
# Такое разложение будем называть $qR$-разложением.

# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Литература ##
#
# 1. *Strang G.* Linear algebra and learning from data. &mdash; Wellesley-Cambridge Press, 2019. &mdash; 432 p.
# 1. *Стренг Г.* Линейная алгебра и её применения. &mdash; М.: Мир, 1980. &mdash; 454 с.
# 1. *Гантмахер Ф.Р.* Теория матриц. &mdash; М.: Наука, 1967. &mdash; 576 с.
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; М.: Наука, 1983. &mdash; 336 с.
# -



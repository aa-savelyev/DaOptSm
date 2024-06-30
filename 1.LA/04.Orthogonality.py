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
# # Ортогональное проектирование

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Метрика-пространства" data-toc-modified-id="Метрика-пространства-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Метрика пространства</a></span><ul class="toc-item"><li><span><a href="#Скалярное-произведение" data-toc-modified-id="Скалярное-произведение-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Скалярное произведение</a></span></li><li><span><a href="#Ортогональные-векторы" data-toc-modified-id="Ортогональные-векторы-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Ортогональные векторы</a></span></li><li><span><a href="#Ортогональные-матрицы" data-toc-modified-id="Ортогональные-матрицы-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Ортогональные матрицы</a></span></li></ul></li><li><span><a href="#Ортогональное-проектирование" data-toc-modified-id="Ортогональное-проектирование-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Ортогональное проектирование</a></span><ul class="toc-item"><li><span><a href="#Общий-случай" data-toc-modified-id="Общий-случай-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Общий случай</a></span></li><li><span><a href="#Одномерный-случай" data-toc-modified-id="Одномерный-случай-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Одномерный случай</a></span></li><li><span><a href="#Проектирование-на-линейную-оболочку-ортонормированных-векторов" data-toc-modified-id="Проектирование-на-линейную-оболочку-ортонормированных-векторов-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Проектирование на линейную оболочку ортонормированных векторов</a></span></li><li><span><a href="#Ортонормированный-базис" data-toc-modified-id="Ортонормированный-базис-2.4"><span class="toc-item-num">2.4&nbsp;&nbsp;</span>Ортонормированный базис</a></span></li></ul></li><li><span><a href="#Ортогонализация-векторов" data-toc-modified-id="Ортогонализация-векторов-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Ортогонализация векторов</a></span><ul class="toc-item"><li><span><a href="#Алгоритм-Грама-–-Шмидта" data-toc-modified-id="Алгоритм-Грама-–-Шмидта-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Алгоритм Грама – Шмидта</a></span></li><li><span><a href="#QR-разложение" data-toc-modified-id="QR-разложение-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>QR-разложение</a></span><ul class="toc-item"><li><span><a href="#Квадратные-матрицы" data-toc-modified-id="Квадратные-матрицы-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Квадратные матрицы</a></span></li><li><span><a href="#Прямоугольные-матрицы" data-toc-modified-id="Прямоугольные-матрицы-3.2.2"><span class="toc-item-num">3.2.2&nbsp;&nbsp;</span>Прямоугольные матрицы</a></span></li></ul></li></ul></li><li><span><a href="#Источники" data-toc-modified-id="Источники-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Источники</a></span></li></ul></div>
# -

# ---

# ## Метрика пространства
#
#
# ### Скалярное произведение
#
# Мы уже рассматривали линейный операторы в произвольном $n$-мерном линейном пространстве.
# Все базисы такого пространства равноправны между собой.
# Данному линейному оператору в каждом базисе соответствует некоторая матрица.
# Матрицы, отвечающие одному и тому же оператору в различных базисах, подобны между собой.
#
# Теперь давайте в линейное $n$-мерное пространство введём метрику.
# Для этого каждым двум векторам $\mathbf{x}$ и $\mathbf{y}$ мы специальным образом поставим в соответствие некоторое число $(\mathbf{x}, \mathbf{y})$ &mdash; их *скалярное произведение*.
#
# Для любых векторов $\mathbf{x}$, $\mathbf{y}$, $\mathbf{z}$ и любого числа $\alpha$ справедливы следующие **свойства скалярного произведения**:
#
# 1. $(\mathbf{x}, \mathbf{y}) = (\mathbf{y}, \mathbf{x})$ (коммутативность),
# 1. $(\alpha \mathbf{x}, \mathbf{y}) = \alpha (\mathbf{x}, \mathbf{y})$,
# 1. $(\mathbf{x} + \mathbf{y}, \mathbf{z}) = (\mathbf{x}, \mathbf{z}) + (\mathbf{y}, \mathbf{z})$ (дистрибутивность),
# 1. $(\mathbf{x}, \mathbf{x}) > 0$ при $\mathbf{x} \ne 0$ (положительная определённость).
#
# **Определение.** Векторное пространство с положительно определённым скалярным произведением называется **евклидовым пространством**.
#
# **Длиной вектора** называется $|\mathbf{x}| = \sqrt{(\mathbf{x}, \mathbf{x})}$.
#
# **Косинус угла** между двумя векторами равен
# $$ cos(\theta) = \frac{(\mathbf{x}, \mathbf{y})}{|\mathbf{x}| |\mathbf{y}|}. $$

# ### Ортогональные векторы
#
# Два вектора $\mathbf{x}$ и $\mathbf{y}$ называются **ортогональными** ($\mathbf{x} \perp \mathbf{y}$), если $(\mathbf{x}, \mathbf{y}) = 0$.
#
# Рассмотрим матрицу $Q$, столбцами которой являются ортонормированные векторы.
# Легко видеть, что $Q^\top Q = I$. 
#
# Отсюда следует, что $\forall \mathbf{x}, \mathbf{y}: (Q\mathbf{x}, Q\mathbf{y}) = (Q\mathbf{x})^\top (Q\mathbf{y}) = \mathbf{x}^\top Q^\top Q \mathbf{y} = (\mathbf{x}, \mathbf{y})$.
# В частности, при умножении на $Q$ длина вектора не меняется: $|Q\mathbf{x}| = (Q\mathbf{x}, Q\mathbf{x}) = (\mathbf{x}, \mathbf{x}) = |\mathbf{x}|.$

# Что такое матрица $P = Q Q^\top$?
#
# $$
#   P = Q Q^\top = \mathbf{q}_1 \mathbf{q}_1^\top + \ldots + \mathbf{q}_n \mathbf{q}_n^\top
# $$
#
# Заметим, что $P^2 = (QQ^\top)(QQ^\top) = Q(Q^\top Q)Q^\top = P$.
# А значит, $P$ &mdash; проекционная матрица.
#
#
# Если $Q$ &mdash; квадратная матрица, то $Q Q^\top$ также является единичной матрицей: $Q Q^\top = I$.

# ### Ортогональные матрицы
#
# **Определение.** **Ортогональной матрицей** называется *квадратная* матрица, столбцами которой являются *ортонормированные* векторы.
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
# **Пример 3.** Произведение двух ортогональных матриц $Q_1 Q_2$ &mdash; ортогональная матрица. Вращение на вращение = вращение, отражение на отражение = вращение, вращение на отражение = отражение.
#
# **Пример 4.** Любая матрица перестановки является ортогональной, поскольку её вектор-столбцы имеют единичную длину и ортогональны друг другу.

# ---

# ## Ортогональное проектирование
#
# ### Общий случай
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

# ### Одномерный случай
#
# Матрица проектирования на прямую, определённую вектором $\mathbf{a}$, задаётся формулой
# $$ P = A (A^\top A)^{-1} A^\top = \mathbf{a} (\mathbf{a^\top}\mathbf{a})^{-1}\mathbf{a^\top} = \frac{1}{\mathbf{a^\top}\mathbf{a}} \mathbf{a} \mathbf{a^\top}. $$
#
# А проекция $\mathbf{p}$ точки $\mathbf{b}$ &mdash; формулой
# $$ \mathbf{p} = \frac{\mathbf{a^\top} \mathbf{b}}{\mathbf{a^\top}\mathbf{a}}\mathbf{a}. $$

# ### Проектирование на линейную оболочку ортонормированных векторов
#
# Если набор векторов образует ортонормированный базис $\mathbf{q}_i$, то формула для матрицы ортогонального проектирования существенно упрощается:
# $$
#   P = Q (Q^\top Q)^{-1} Q^\top = QQ^\top = \mathbf{q}_1 \mathbf{q}_1^\top + \ldots + \mathbf{q}_n \mathbf{q}_n^\top.
# $$
#
# Каждое слагаемое $\mathbf{q}_i \mathbf{q}_i^\top$ является матрицей ортогональной проекции на вектор $\mathbf{q}_i$.
#
# Таким образом, *когда оси координат взаимно перпендикулярны, проекция на пространство разлагается в сумму проекций на каждую из осей*.

# ### Ортонормированный базис
#
# Разложение любого вектора по ортонормированному базису есть сумма ортогональных проекций  на каждый вектор:
#
# $$
#   \mathbf{v} = c_1 \mathbf{q}_1 + \ldots + c_n \mathbf{q}_n = (\mathbf{q}_1^\top \mathbf{v}) \cdot \mathbf{q}_1 + \ldots + (\mathbf{q}_n^\top \mathbf{v}) \cdot \mathbf{q}_n.
# $$

# ---

# ## Ортогонализация векторов

# ### Алгоритм Грама &ndash; Шмидта
#
# Два ряда векторов называются **эквивалентными** если они содержат одинаковое количество векторов и их линейные оболочки совпадают.
#
# Под **ортогонализацией** ряда векторов будем понимать замену этого ряда на *эквивалентный* (порождающий ту же самую линейную оболочку) ортогональный ряд.
#
# **Теорема.** Всякий невырожденный ряд векторов можно проортогонализировать.
#
# Алгоритм Грама &ndash; Шмидта рассмотрим на примере трёхмерного пространства.
#
# Пусть даны три линейно-независимых вектора $\mathbf{a}_1$, $\mathbf{a}_2$, $\mathbf{a}_3$.
#
# 1. Первый вектор просто отнормируем:\
# $\mathbf{q}_1 = \mathbf{a}_1 / |\mathbf{a}_1|$
# 1. Второй вектор получим с помощью ортогональной проекции вектора $\mathbf{a}_2$ на $\mathbf{q}_1$:\
# $\mathbf{\hat{a}}_2 = \mathbf{a}_2 - (\mathbf{q}_1^\top \mathbf{a}_2) \cdot \mathbf{q}_1$; $\quad \mathbf{q}_2 = \mathbf{\hat{a}}_2 / |\mathbf{\hat{a}}_2|$
# 1. Третий вектор получим с помощью ортогональной проекции вектора $\mathbf{a}_2$ на плоскость векторов $\mathbf{q}_1$ и $\mathbf{q}_2$:\
# $\mathbf{\hat{a}}_3 = \mathbf{a}_3 - (\mathbf{q}_1^\top \mathbf{a}_3) \cdot \mathbf{q}_1 - (\mathbf{q}_2^\top \mathbf{a}_3) \cdot \mathbf{q}_2$; $\quad \mathbf{q}_3 = \mathbf{\hat{a}}_3/|\mathbf{\hat{a}}_3|$

# ### QR-разложение
#
# #### Квадратные матрицы
#
# Большую роль в численных методах играет разложение *квадратной матрицы* $A$ на ортогональную матрицу $Q$ и верхнетреугольную матрицу $R$.
# $$
#   A = QR.
# $$
#
# **Утверждение 1.** Любая квадратная матрица $A$ может быть разложена в произведение $QR$.
#
# **Утверждение 2.** Если матрица $A$ невырождена, то её $QR$-разложение, в котором диагональные элементы $R$ положительны, единственно.

# $QR$-разложение тесно связано с процессом ортогонализации Грама &ndash; Шмидта.
#
# Действительно, для вектора $\mathbf{q}_i$ в алгоритме Грама &ndash; Шмидта можно записать общее выражение:
# $$ \mathbf{q}_i = u_{1i}\mathbf{a}_1 + \ldots + u_{ii}\mathbf{a}_i.$$
#
# Вводя матрицы $A$ и $Q$ составленные из векторов $\mathbf{a}_i$ и $\mathbf{q}_i$ соотвественно, получим матричное выражение
# $$ AU = Q, $$
# где $U$ &mdash; верхнетреугольная матрица, составленная из векторов $\mathbf{u}_i = (u_{1i}, \ldots, u_{ii}, 0, \ldots , 0)$.
#
# В силу невырожденности $U$ получим
# $$ A = QU^{-1} = QR. $$

# > Альтернативные алгоритмы для вычисления QR-разложения основаны на отражениях Хаусхолдера и вращениях Гивенса

# > Все неособенные верхние (нижние) треугольные матрицы составляют группу (некоммутативную) относительно умножения.
#
# > *Группой* называется всякая совокупность объектов, в которой установлена операция, относящая любым двум элементам $a$ и $b$ совокупности определённый третий элемент $a \ast b$ той же совокупности, если 1) операция обладает сочетательным свойством, 2) в совокупности существует единичный элемент, 3) для любого элемента совокупности существует обратный элемент.
# > Группа называется *коммутативной*, если групповая операция обладает переместительным свойством.

# **Пример 1.**
# Рассмотрим систему линейных уравнений $A\mathbf{x} = \mathbf{b}$. Используя $QR$-разложение матрицы $A$, получим
# $$
#   QR \mathbf{x} = \mathbf{b}, \\
#   R \mathbf{x} = Q^\top \mathbf{b}.
# $$
#
# И в случае невырожденной $R$
# $$
#   \mathbf{x} = R^{-1}Q^\top \mathbf{b}.
# $$

# #### Прямоугольные матрицы
#
# Произвольную матрицу размером $m \times n$ можно представить в виде разложения
#
# $$
#   A = QR,
# $$
# где $Q$ &mdash; ортогональная матрица порядка $m$, а $R$ &mdash; матрица размером $m \times n$, элементы $r_{ij}$ которой удовлетворяют условию $r_{ij}=0$ при $i>j$.

# Рассмотрим два случая.
#
# 1. $m < n$\
#    Тогда $Q$ &mdash; ортогональная матрица порядка $m$, а $R$ &mdash; верхнетреугольная матрица размером $m \times n$.
# 1. $m \ge n$\
#    В этом случае можно получить усечённое QR-разложение ($qR$-разложение):
#    $$
#    A = QR =
#    Q
#    \begin{pmatrix}
#      R_1 \\
#      0 \\
#    \end{pmatrix}
#    =
#    \begin{pmatrix}
#      Q_1 & Q_2 \\
#    \end{pmatrix}
#    \begin{pmatrix}
#      R_1 \\
#      0 \\
#    \end{pmatrix}
#    =
#    Q_1 R_1.
#    $$
#    Если $A$ имеет полный столбцовый ранг, то $R_1$ является верхнетреугольной матрицей в разложении Холецкого матрицы $A^\top A$.

# **Пример 2.**
# Рассмотрим нормальную систему уравнений $A^\top A\mathbf{x} = A^\top \mathbf{b}$. Используя $qR$-разложение матрицы $A$, получим
# $$
#   R^\top q^\top q R \mathbf{x} = R^\top q^\top \mathbf{b}, \\
#   R^\top R \mathbf{x} = R^\top q^\top \mathbf{b}.
# $$
#
# И в случае невырожденной $R$ (матрица $A$ имеет полный столбцовый ранг)
# $$
#   R \mathbf{x} = q^\top \mathbf{b}, \\
#   \mathbf{x} = R^{-1} q^\top \mathbf{b}.
# $$

# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Источники
#
# 1. *Strang G.* Linear algebra and learning from data. &mdash; Wellesley-Cambridge Press, 2019. &mdash; 432 p.
# 1. *Стренг Г.* Линейная алгебра и её применения. &mdash; М.: Мир, 1980. &mdash; 454 с.
# 1. *Гантмахер Ф.Р.* Теория матриц. &mdash; М.: Наука, 1967. &mdash; 576 с.
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; М.: Наука, 1983. &mdash; 336 с.
# -



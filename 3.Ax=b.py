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
# **Лекция 3**
#
# # Теория систем линейных уравнений #
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
# 1. Базис пространства столбцов матрицы $AB$ ($\mathbf{C}(AB)$) является комбинацией (возможно, с пересечениями) базисов пространств $\mathbf{C}(A)$ и $\mathbf{C}(B)$.
# 1. Матрицы $A$ и $A^\top A$ имеют одно и то же нуль-пространство (доказать), поэтому их ранг одинаков.
# 1. Матрицы $A^\top A$ и $BB^\top$ невырождены, так как $r(A^\top A) = r(BB^\top) = r$. Их произведение, матрица $A^\top A BB^\top$, тоже невырождена и её ранг равен $r$. Отсюда $r = r(A^\top A BB^\top) \le r(AB) \le r(A) = r$.

# >Как доказать, что $A$ и $A^\top A$ имеют одно и то же нуль-пространство? \
# >Если $Ax=0$, то $A^\top Ax = 0$. Поэтому $\mathbf{N}(A) \subset \mathbf{N}(A^\top A)$. \
# >Если $A^\top Ax = 0$, то $x^\top A^\top Ax = \|Ax\|^2 = 0$. Поэтому $\mathbf{N}(A^\top A) \subset \mathbf{N}(A)$.

# *Замечание.* Свойство 4 работает только в случае, когда $A$ имеет *ровно* $r$ столбцов, а $B$ имеет *ровно* $r$ строк. В частности, $r(BA) \le r$ (в соответствии со свойством 1).
#
# Для свойства 3 отметим важный частный случай. Это случай, когда столбцы матрицы $A$ линейно независимы, так что её ранг $r$ равен $n$. Тогда матрица $A^\top A$ является квадратной симметрической обратимой матрицей.

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
#   M = (C^\top C)^{-1} (C^\top A R^\top) (R R^\top)^{-1}.
# $$
#
# Обратим внимание на используемый приём &mdash; домножение прямоугольной матрицы на транспонированную.
# Полученную квадратную матрицу можно домножать на обратную (&laquo;делить на матрицу&raquo;).
# Этот приём можно использовать далеко не всегда.

# ---

# ## $\mathbf{LU}$-разложение ##
#
# > Ограничения: матрица $A$ &mdash; квадратная и невырожденная

# ### Определение и критерий существования ###
#
# **Определение.** $LU$-разложением квадратной матрицы $A$ называется разложение матрицы $A$ в произведение невырожденной нижней треугольной матрицы $L$ и верхней треугольной матрицы $U$ с единицами на главной диагонали.
#
# **Теорема (критерий существования).** $LU$-разложение матрицы $A$ существует тогда и только тогда, когда все главные миноры матрицы $A$ отличны от нуля. Если $LU$-разложение существует, то оно единственно.

# ### Метод Гаусса ###
#
# **Замечание.** $LU$-разложение можно рассматривать, как матричную форму записи метода исключения Гаусса.
#
# Пусть дана система линейных уравнений вида
# $$ A \mathbf{x} = \mathbf{b}, $$
# где $A$ &mdash; невырожденная квадратная матрица порядка $n$.
#
# Метод Гаусса состоит в том, что элементарными преобразованиями *строк* матрица $A$ превращается в единичную матрицу.
# Если преобразования производить над расширенной матрицей (включающей столбец свободных членов), то последний столбец превратится в решение системы.

# ### Случай 1: все главные миноры отличны от нуля ###
#
# Отличие от нуля главных миноров позволяет не включать в число производимых элементарных операций перестановки строк.
#
# Метод Гаусса можно разделить на два этапа: прямое исключение и обратная подстановка.
# Первым этапом решения СЛАУ методом Гаусса является процесс превращения матрицы $A$ элементарными преобразованиями в верхнюю треугольную матрицу $U$.
#
# Известно, что выполнение какой-либо элементарной операции со строками матрицы $A$ равносильно умножению $A$ *слева* на некоторую невырожденную матрицу, а последовательное выполнение ряда таких операций &mdash; умножению на матрицу $S$, равную произведению соответствующих матриц.
#
# На этапе прямого исключения кроме умножения строк на числа употребляется  только прибавление строки к нижележащей строке.
# Следовательно матрица $S$ является нижней треугольной.
#
# **Предложение.** Для любой матрицы $A$ с ненулевыми главными минорами найдётся такая невырожденная нижняя треугольная матрица $S$, что $SA$ есть верхняя треугольная матрица $U$ с единицами на главной диагонали:
# $$ SA = U. $$
#
# Матрица $L$, обратная к нижней треугольной матрице $S$, сама является нижней треугольной.
# Тогда получаем:
# $$ A = LU. $$

# ### Случай 2: существуют ненулевые главные миноры ($\mathbf{LUP}$-разложение) ###
#
# Что делать, если не все главные миноры отличны от нуля?
# К используемым элементарным операциям нужно добавить перестановки строк.
#
# **Предложение.** Невырожденную матрицу $A$ перестановкой строк можно перевести в матрицу, главные миноры которой отличны от нуля.
#
# Тогда справедливо
# $$ PA = LU,$$
# где $P$ &mdash; матрица, полученная из единичной перестановками строк.

# ### $\mathbf{LDU}$-разложение ###
#
#
# **Замечание.** Единственным является разложение на такие треугольные множители, что у второго из них на главной диагонали стоят единицы. Вообще же существует много треугольных разложений, в частности такое, в котором единицы находятся на главной диагонали у первого сомножителя.
#
# Матрицу $L$ можно представить как произведение матрицы $L_1$, имеющей единицы на главной диагонали, и диагональной матрицы $D$. Тогда мы получим
#
# **Предложение.** Матрицу $A$, главные миноры которой не равны нулю, можно единственным образом разложить в произведение $L_1 D U$, в котором $D$ &mdash; диагональная, а $L_1$ и $U$ &mdash; нижняя и верхняя треугольные матрицы с единицами на главной диагонали.

# ### Разложение Холецкого ###
#
# Рассмотрим важный частный случай &mdash; $LDU$-разложение симметричной матрицы $S = LDU$. \
# Тогда $S^\top = U^\top D L^\top$, причём $U^\top$ &mdash; нижняя, а $L^\top$ &mdash; верхняя треугольные матрицы. \
# В силу единственности разложения получаем
# $$ S = U^\top D U. $$
#
# Если к тому же $S$ положительно определена, то все диагональные элементы матрицы $D$ положительны и мы можем ввести матрицы $D^{1/2} = \mathrm{diag}\left(\sqrt{d_1}, \dots, \sqrt{d_n}\right)$ и $V = D^{1/2}U$. Тогда мы получаем *разложение Холецкого*
# $$
#   S = V^\top V.
# $$
# Разложение Холецкого играет заметную роль в численных методах, так как существует эффективный алгоритм, позволяющий получить его для положительно определённой симметричной матрицы $S$.

# ---

# ## Основная теорема линейной алгебры ##
#
# ### Четыре основных подпространства ###
#
# Обычно подпространства описывают одним из двух способов.
# Первый способ &ndash; это когда задается множество векторов, порождающих данное подпространство.
# Например, при определении пространства строк или пространства столбцов некоторой матрицы, когда указываются порождающие строки или столбцы.
#
# Второй способ заключается в задании перечня ограничений на подпространство.
# В этом случае указывают не векторы, порождающие это подпространство, а ограничения, которым должны удовлетворять векторы этого подпространства.
# Нуль-пространство, например, состоит из всех векторов, удовлетворяющих системе $Ax=0$, и каждое уравнение этой системы представляет собой такое ограничение.
#
# Пусть матрица A имеет размер $m \times n$. \
# Рассмотрим 4 основных подпространства:
#
# 1. Пространство строк матрицы $A$, $\mathrm{dim} = r$
# 1. Нуль-пространство матрицы $A$ (ядро), $\mathrm{dim} = n-r$
# 1. Пространство столбцов матрицы $A$ (образ), $\mathrm{dim} = r$
# 1. Нуль-пространство матрицы $A^\top$, $\mathrm{dim} = m-r$

# **Определение.** Пусть $V$ &ndash; подпространство пространства $\mathbb{R}^n$. Тогда пространство всех $n$-мерных векторов, ортогональных к подпространству $V$, называется *ортогональным дополнением* к $V$ и обозначается символом $V^\perp$.
#
# **Основная теорема линейно алгебры.** Пространство строк $A$ и нуль-пространство матрицы $A$, а также пространство столбцов $A$ и нуль-пространство матрицы $A^\top$ являются ортогональными дополнениями друг к другу.

# ### Альтернатива Фредгольма ###
#
# Для того чтобы система уравнений $Ax=b$ была совместна, необходимо и достаточно, чтобы каждое решение сопряженной однородной системы удовлетворяло уравнению $y^\top b = 0$.
#
# Либо вектор $b$ 
# уравнение $Ax=b$ имеет решение при любой правой части, либо сопряжённое к нему однородное уравнение $A^\top y = 0$ имеет нетривиальное решение.

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Литература ##
#
# 1. *Strang G.* Linear algebra and learning from data. &mdash; Wellesley-Cambridge Press, 2019. &mdash; 432 p.
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; М.: Наука, 1983. &mdash; 336 с.
# -



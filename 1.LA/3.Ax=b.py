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

# + [markdown] slideshow={"slide_type": "slide"}
# **Лекция 3**
#
# # Системы линейных уравнений. Псевдообратные матрицы

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#$\mathbf{LU}$-разложение" data-toc-modified-id="$\mathbf{LU}$-разложение-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>$\mathbf{LU}$-разложение</a></span><ul class="toc-item"><li><span><a href="#Определение-и-критерий-существования" data-toc-modified-id="Определение-и-критерий-существования-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>Определение и критерий существования</a></span></li><li><span><a href="#Метод-Гаусса" data-toc-modified-id="Метод-Гаусса-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>Метод Гаусса</a></span></li><li><span><a href="#Случай-1:-все-главные-миноры-отличны-от-нуля" data-toc-modified-id="Случай-1:-все-главные-миноры-отличны-от-нуля-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>Случай 1: все главные миноры отличны от нуля</a></span></li><li><span><a href="#Случай-2:-существуют-ненулевые-главные-миноры-($\mathbf{LUP}$-разложение)" data-toc-modified-id="Случай-2:-существуют-ненулевые-главные-миноры-($\mathbf{LUP}$-разложение)-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>Случай 2: существуют ненулевые главные миноры ($\mathbf{LUP}$-разложение)</a></span></li><li><span><a href="#$\mathbf{LDU}$-разложение" data-toc-modified-id="$\mathbf{LDU}$-разложение-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>$\mathbf{LDU}$-разложение</a></span></li><li><span><a href="#Разложение-Холецкого" data-toc-modified-id="Разложение-Холецкого-1.6"><span class="toc-item-num">1.6&nbsp;&nbsp;</span>Разложение Холецкого</a></span></li></ul></li><li><span><a href="#Основная-теорема-линейной-алгебры" data-toc-modified-id="Основная-теорема-линейной-алгебры-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Основная теорема линейной алгебры</a></span><ul class="toc-item"><li><span><a href="#Четыре-основных-подпространства" data-toc-modified-id="Четыре-основных-подпространства-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Четыре основных подпространства</a></span></li><li><span><a href="#Теорема-Фредгольма" data-toc-modified-id="Теорема-Фредгольма-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Теорема Фредгольма</a></span></li></ul></li><li><span><a href="#Псевдорешения-и-псевдообратные-матрицы" data-toc-modified-id="Псевдорешения-и-псевдообратные-матрицы-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Псевдорешения и псевдообратные матрицы</a></span><ul class="toc-item"><li><span><a href="#Постановка-задачи" data-toc-modified-id="Постановка-задачи-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Постановка задачи</a></span></li><li><span><a href="#Псевдорешение" data-toc-modified-id="Псевдорешение-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Псевдорешение</a></span><ul class="toc-item"><li><span><a href="#Примеры" data-toc-modified-id="Примеры-3.2.1"><span class="toc-item-num">3.2.1&nbsp;&nbsp;</span>Примеры</a></span></li></ul></li><li><span><a href="#Псевдообратная-матрица" data-toc-modified-id="Псевдообратная-матрица-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Псевдообратная матрица</a></span></li></ul></li><li><span><a href="#Источники" data-toc-modified-id="Источники-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Источники</a></span></li></ul></div>
# -

# ---

# ## $\mathbf{LU}$-разложение
#
# > Ограничения: матрица $A$ &mdash; квадратная и невырожденная

# ### Определение и критерий существования
#
# **Определение.** $LU$-разложением квадратной матрицы $A$ называется разложение матрицы $A$ в произведение невырожденной нижней треугольной матрицы $L$ и верхней треугольной матрицы $U$ с единицами на главной диагонали.
#
# **Теорема (критерий существования).** $LU$-разложение матрицы $A$ существует тогда и только тогда, когда все главные миноры матрицы $A$ отличны от нуля. Если $LU$-разложение существует, то оно единственно.

# ### Метод Гаусса
#
# **Замечание.** $LU$-разложение можно рассматривать, как матричную форму записи метода исключения Гаусса.
#
# Пусть дана система линейных уравнений вида
# $$ A \mathbf{x} = \mathbf{b}, $$
# где $A$ &mdash; невырожденная квадратная матрица порядка $n$.
#
# Метод Гаусса состоит в том, что элементарными преобразованиями *строк* матрица $A$ превращается в единичную матрицу.
# Если преобразования производить над расширенной матрицей (включающей столбец свободных членов), то последний столбец превратится в решение системы.

# ### Случай 1: все главные миноры отличны от нуля
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

# ### Случай 2: существуют ненулевые главные миноры ($\mathbf{LUP}$-разложение)
#
# Что делать, если не все главные миноры отличны от нуля?
# К используемым элементарным операциям нужно добавить *перестановки* строк или столбцов.
#
# **Предложение.** Невырожденную матрицу $A$ перестановкой строк (или столбцов) можно перевести в матрицу, главные миноры которой отличны от нуля.
#
# Тогда справедливо
# $$ PA = LU,$$
# где $P$ &mdash; матрица, полученная из единичной перестановками строк.

# ### $\mathbf{LDU}$-разложение
#
# **Замечание.** Единственным является разложение на такие треугольные множители, что у второго из них на главной диагонали стоят единицы. Вообще же существует много треугольных разложений, в частности такое, в котором единицы находятся на главной диагонали у первого сомножителя.
#
# Матрицу $L$ можно представить как произведение матрицы $L_1$, имеющей единицы на главной диагонали, и диагональной матрицы $D$. Тогда мы получим
#
# **Предложение.** Матрицу $A$, главные миноры которой не равны нулю, можно единственным образом разложить в произведение $L_1 D U$, в котором $D$ &mdash; диагональная, а $L_1$ и $U$ &mdash; нижняя и верхняя треугольные матрицы с единицами на главной диагонали.

# ### Разложение Холецкого
#
# Рассмотрим важный частный случай &mdash; $LDU$-разложение симметричной матрицы $S = LDU$. \
# Тогда $S^\top = U^\top D L^\top$, причём $U^\top$ &mdash; нижняя, а $L^\top$ &mdash; верхняя треугольные матрицы. \
# В силу единственности разложения получаем
# $$ S = U^\top D U. $$
#
# Если же матрица $S$ является не только симметричной, но и положительно определённой, то все диагональные элементы матрицы $D$ положительны и мы можем ввести матрицы $D^{1/2} = \mathrm{diag}\left(\sqrt{d_1}, \dots, \sqrt{d_n}\right)$ и $V = D^{1/2}U$. Тогда мы получаем *разложение Холецкого*
# $$
#   S = V^\top V.
# $$
# Разложение Холецкого играет заметную роль в численных методах, так как существует эффективный алгоритм, позволяющий получить его для положительно определённой симметричной матрицы $S$.

# ---

# ## Основная теорема линейной алгебры
#
# ### Четыре основных подпространства
#
# Обычно подпространства описывают одним из двух способов.
# В первом способе задаётся множество векторов, порождающих данное подпространство.
# Например, при определении пространства строк или пространства столбцов некоторой матрицы, когда указываются порождающие эти пространства строки или столбцы.
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
# **Основная теорема линейной алгебры.** Пространство строк $A$ и нуль-пространство матрицы $A$, а также пространство столбцов $A$ и нуль-пространство матрицы $A^\top$ являются ортогональными дополнениями друг к другу.

# ### Теорема Фредгольма
#
# **Формулировка 1**: \
# Для того чтобы система уравнений $Ax=b$ была совместна, необходимо и достаточно, чтобы каждое решение сопряжённой однородной системы $A^\top y = 0$ удовлетворяло уравнению $y^\top b = 0$.
#
#
# **Формулировка 2** (*альтернатива Фредгольма*): \
# Для любых $A$ и $b$ одна и только одна из следующих задач имеет решение:
#
# 1. $Ax = b$;
# 1. $A^\top y = 0$, $y^\top b \ne 0$.
#
# Иначе говоря, либо вектор $b$ лежит в пространстве столбцов $A$, либо не лежит в нём.
# В первом случае согласно основной теореме вектор $b$ ортогонален любому вектору из $\mathrm{Ker}(A^\top)$, во-втором же случае в пространстве $\mathrm{Ker}(A^\top)$ найдётся вектор $y$, неортогональный вектору $b$: $y^\top b \ne 0$.
#
#
# **Следствие**: Для того чтобы уравнение $Ax=b$ имело решение при любой правой части, сопряжённое к нему однородное уравнение $A^\top y = 0$ должно иметь только тривиальное решение.

# + [markdown] slideshow={"slide_type": "skip"}
# ---
# -

# ## Псевдорешения и псевдообратные матрицы

# ### Постановка задачи
#
# В практических задачах часто требуется найти решение, удовлетворяющее большому числу возможно противоречивых требований.
# Если такая задача сводится к системе линейных уравнений, то система оказывается, вообще говоря, несовместной.
# В этом случае задача может быть решена только путём выбора некоторого компромисса &mdash; все требования могут быть удовлетворены не полностью, а лишь до некоторой степени.
#
# Рассмотрим систему линейных уравнений
# $$
#   A\mathbf{x} = \mathbf{b} \tag{1}\label{eq:system}
# $$
# с матрицей $A$ размеров $m \times n$ и ранга $r$.
# Поскольку $\mathbf{x}$ &mdash; столбец высоты $n$, а $\mathbf{b}$ &mdash; столбец высоты $m$, для геометрической иллюстрации естественно будет использовать пространства $\mathbb{R}_n$ и $\mathbb{R}_m$.
#
# Под нормой столбца $\mathbf{x}$ мы будем понимать его евклидову норму, т.е. число
# $$
#   \|\mathbf{x}\| = \sqrt{\mathbf{x^\top x}} = \sqrt{x_1^2 + \ldots + x_n^2}.
# $$
#
# Невязкой, которую даёт столбец $\mathbf{x}$ при подстановке в систему $\eqref{eq:system}$, называется столбец
# $$
#   \mathbf{u} = \mathbf{b} - A\mathbf{x}.
# $$
# Решение системы &mdash; это столбец, дающий нулевую невязку.
#
# Если система $\eqref{eq:system}$ несовместна, естественно постараться найти столбец $\mathbf{x}$, который даёт невязку с минимальной нормой, и если такой столбец найдётся, считать его обобщённым решением.

# ### Псевдорешение
#
# Для сравнения невязок воспользуемся евклидовой нормой и, следовательно, будем искать столбец $\mathbf{x}$, для которого минимальная величина
# $$
#   \|\mathbf{u}\|^2 = (\mathbf{b} - A\mathbf{x})^\top (\mathbf{b} - A \mathbf{x}).
# $$
#
# Найдём полный дифференциал $\|\mathbf{u}\|^2$:
# $$
#   d\|\mathbf{u}\|^2 = -d\mathbf{x}^\top A^\top (\mathbf{b}-A\mathbf{x}) - (\mathbf{b}-A\mathbf{x})^\top A d\mathbf{x} = \
#   -2d\mathbf{x}^\top A^\top (\mathbf{b} - A\mathbf{x}).
# $$
#
# Дифференциал равен нулю тогда и только тогда, когда
# $$
#   A^\top A \mathbf{x} = A^\top \mathbf{b}.
# $$
# Эта система линейных уравнений по отношению к системе ([1](#mjx-eqn-eq:system)) называется *нормальной системой*.
# Независимо от совместности системы ([1](#mjx-eqn-eq:system)) справедливо
#
# **Предложение 1.** Нормальная система уравнений всегда совместна. \
# *Доказательство.* Применим критерий Фредгольма: система $A\mathbf{x}=\mathbf{b}$ совместна тогда и только тогда, когда $\mathbf{b}$ ортогонален любому решению $\mathbf{y}$ сопряжённой однородной системы.
# Пусть $\mathbf{y}$ &mdash; решение сопряжённой однородной системы $(A^\top A)^\top \mathbf{y} = 0$.
# Тогда
# $$
#   \mathbf{y}^\top A^\top A \mathbf{y} = (A \mathbf{y})^\top (A \mathbf{y}) = 0 \quad \Rightarrow \quad
#   A \mathbf{y} = 0 \quad \Rightarrow \quad
#   \mathbf{y}^\top (A^\top \mathbf{b}) = (A\mathbf{y})^\top \mathbf{b} = 0.
# $$
#
# **Предложение 2.** Точная нижняя грань квадрата нормы невязки достигается для всех решений нормальной системы и только для них.
#
# **Предложение 3.** Нормальная система имеет единственное решение тогда и только тогда, когда столбцы матрицы $A$ линейно независимы.
#
# **Определение.** *Нормальным псевдорешением* системы линейных уравнений называется столбец с минимальной нормой среди всех столбцов, дающих минимальную по норме невязку при подстановке в эту систему.
#
# **Теорема 1.** Каждая система линейных уравнений имеет одно и только одно нормальное псевдорешение.

# #### Примеры
#
# **Пример 1.** Система из двух уравнений с одной неизвестной:
# $$ x=1, \; x=2. $$
#
# Нормальная система уравнений для этой системы есть
# $$
#   \begin{pmatrix}
#     1 & 1 \\
#   \end{pmatrix}
#   \begin{pmatrix}
#     1 \\
#     1 \\
#   \end{pmatrix}
#   x = 
#   \begin{pmatrix}
#     1 & 1 \\
#   \end{pmatrix}
#   \begin{pmatrix}
#     1 \\
#     2 \\
#   \end{pmatrix}.
# $$
# Отсюда получаем псевдорешене $x = 3/2$.

# **Пример 2.** Система из одного уравнений с двумя неизвестными:
# $$ x + y = 2. $$
#
# Нормальной системой уравнений будет система
# $$
#   \begin{pmatrix}
#     1 \\
#     1 \\
#   \end{pmatrix}
#   \begin{pmatrix}
#     1 & 1 \\
#   \end{pmatrix}
#   \begin{pmatrix}
#     x \\
#     y \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     1 \\
#     1 \\
#   \end{pmatrix}
#   2,
# $$
# содержащая то же уравнение, повторенное дважды.
# Её общее решение
# $$
#   \begin{pmatrix}
#     x \\
#     y \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     2 \\
#     0 \\
#   \end{pmatrix}
#   + \alpha
#   \begin{pmatrix}
#     -1 \\
#     1 \\
#   \end{pmatrix}.
# $$
#
# Действуя согласно определению нормального псевдорешения, среди всех псевдорешений выберем решение с минимальной нормой. \
# Квадрат нормы решения равен
# $$
#   \|\mathbf{x}\|_2^2 = (2-\alpha)^2 + \alpha^2 = 2\alpha^2 - 4\alpha + 4.
# $$
#
# Тогда искомым решением будет столбец
# $$
#   \begin{pmatrix}
#     x \\
#     y \\
#   \end{pmatrix}
#   =
#   \begin{pmatrix}
#     1 \\
#     1 \\
#   \end{pmatrix}.
# $$

# **Пример 3.** Система из одного уравнения с одним неизвестным: $ax = b$.
#
# Если $a \ne 0$, то псевдорешение совпадает с решением $x = b/a$. \
# Если $a = 0$, то любое решение даёт одну и ту же невязку $b$ с нормой $|b|$.
# Выбирая решение с минимальной нормой, получаем $x = 0$.

# **Пример 4.** Система линейных уравнений с нулевой матрицей: $O \mathbf{x} = \mathbf{b}$.
#
# Аналогично примеру 3 находим, что псевдорешением будет нулевой столбец.

# ### Псевдообратная матрица
#
# Для невырожденной квадратной матрицы $A$ порядка $n$ обратную матрицу можно определить как такую, столбы которой являются решениями систем линейных уравнений вида
# $$
#   A\mathbf{x} = \mathbf{e}_i, \tag{2} \label{eq:inv_definition}
# $$
# где $\mathbf{e}_i$ &mdash; $i$-й столбец единичной матрицы порядка $n$.
#
# По аналогии можно дать следующее \
# **Определение.** *Псевдообратной матрицей* для матрицы $A$ размеров $m \times n$ называется матрица $A^+$, столбцы которой &mdash; псевдорешения систем линейных уравнений вида $\eqref{eq:inv_definition}$, где $\mathbf{e}_i$ &mdash; столбцы единичной матрицы порядка $m$.
#
# Из теоремы 1 следует, что каждая матрица имеет одну и только одну псевдообратную. Для невырожденной квадратной матрицы псевдообратная матрица совпадает с обратной.
#
# **Предложение 4.** Если столбцы матрицы $A$ линейно независимы, то
# $$
#   A^+ = (A^\top A)^{-1} A^\top.
# $$
# Если строки матрицы $A$ линейно независимы, то
# $$
#   A^+ = A^\top (A A^\top)^{-1}.
# $$
# В первом случае $A^+$ является левой обратной матрицей для $A$ ($A^+A=I$), во втором &mdash; правой ($A A^+ = I$).
#
# **Предложение 5.** Для любого столбца $\mathbf{y} \in \mathbb{R}_m$ столбец $A A^+ \mathbf{y}$ есть ортогональная проекция $\mathbf{y}$ на линейную оболочку столбцов матрицы $A$.
#
# **Предложение 6.** Если $A = CR$ &mdash; скелетное разложение матрицы $A$, то её псевдообратная равна
# $$
#   A^+ = R^+ C^+ = R^\top (R R^\top)^{-1} (C^\top C)^{-1} C^\top.
# $$
#
# **Предложение 7.** Если $A = U \Sigma V^\top$ &mdash; сингулярное разложение матрицы $A$, то $A^+ = V \Sigma^+ U^\top$. \
# *Примечание.* Для диагональной матрицы псевдообратная получается заменой каждого ненулевого элемента на диагонали на обратный к нему.

# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Источники ##
#
# 1. *Strang G.* Linear algebra and learning from data. &mdash; Wellesley-Cambridge Press, 2019. &mdash; 432 p.
# 1. *Беклемишев Д.В.* Дополнительные главы линейной алгебры. &mdash; М.: Наука, 1983. &mdash; 336 с.
# -



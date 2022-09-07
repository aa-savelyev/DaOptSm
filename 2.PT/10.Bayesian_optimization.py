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

# **Лекция 10**
#
# # Байесовская оптимизация

# + [markdown] toc=true
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Алгоритм-байесовской-оптимизации" data-toc-modified-id="Алгоритм-байесовской-оптимизации-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Алгоритм байесовской оптимизации</a></span></li><li><span><a href="#Функция-продвижения" data-toc-modified-id="Функция-продвижения-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Функция продвижения</a></span><ul class="toc-item"><li><span><a href="#Нижняя-граница-доверительного-интервала" data-toc-modified-id="Нижняя-граница-доверительного-интервала-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Нижняя граница доверительного интервала</a></span></li><li><span><a href="#Вероятность-улучшения" data-toc-modified-id="Вероятность-улучшения-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Вероятность улучшения</a></span></li><li><span><a href="#Ожидаемое-улучшение" data-toc-modified-id="Ожидаемое-улучшение-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Ожидаемое улучшение</a></span></li></ul></li><li><span><a href="#Тест-1.-Бесшумные-данные" data-toc-modified-id="Тест-1.-Бесшумные-данные-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Тест 1. Бесшумные данные</a></span><ul class="toc-item"><li><span><a href="#Данные" data-toc-modified-id="Данные-3.1"><span class="toc-item-num">3.1&nbsp;&nbsp;</span>Данные</a></span></li><li><span><a href="#Стандартная-оптимизация" data-toc-modified-id="Стандартная-оптимизация-3.2"><span class="toc-item-num">3.2&nbsp;&nbsp;</span>Стандартная оптимизация</a></span></li><li><span><a href="#Байесовская-оптимизация" data-toc-modified-id="Байесовская-оптимизация-3.3"><span class="toc-item-num">3.3&nbsp;&nbsp;</span>Байесовская оптимизация</a></span></li></ul></li><li><span><a href="#Тест-2.-Шумные-данные" data-toc-modified-id="Тест-2.-Шумные-данные-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Тест 2. Шумные данные</a></span><ul class="toc-item"><li><span><a href="#Данные" data-toc-modified-id="Данные-4.1"><span class="toc-item-num">4.1&nbsp;&nbsp;</span>Данные</a></span></li><li><span><a href="#Байесовская-оптимизация" data-toc-modified-id="Байесовская-оптимизация-4.2"><span class="toc-item-num">4.2&nbsp;&nbsp;</span>Байесовская оптимизация</a></span></li></ul></li><li><span><a href="#Источники" data-toc-modified-id="Источники-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Источники</a></span></li></ul></div>

# +
# Imports
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize

import sys
sys.path.append('./modules')
import GP_kernels
from GP_utils import plot_GP, GP_predictor

# +
# Styles, fonts
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['lines.markeredgewidth'] = 1.5
import matplotlib.pyplot as plt
from matplotlib import cm # Colormaps

import seaborn
seaborn.set_style('whitegrid')


# +
# # %config InlineBackend.figure_formats = ['pdf']
# # %config Completer.use_jedi = False
# -

# ---

# ## Алгоритм байесовской оптимизации

# При решении практических задач часто приходится иметь дело с оптимизацией &laquo;чёрного ящика&raquo;.
# В таком случае мы не имеем почти никакой информации о целевой функции $f$: мы не знаем её аналитического выражения, значений производных и т. д.
# Всё что мы можем &mdash; это работать с чёрным ящиком по системе &laquo;запрос &ndash; ответ&raquo;, т. е. получать значения функции (отклики) в нужных нам точках.
# Причём эти отклики могут быть шумными, т. е. могут быть подвержены влиянию некоторой случайной ошибки.
#
# Если чёрный ящик работает быстро, можно воспользоваться методом &laquo;грубой силы&raquo;: вычислить отклики на большом массиве точек и выбрать оптимум.
# По такому принципу работают методы поиска по сетке (grid search) или случайного поиска (random search).
# Другой вариант &mdash; воспользоваться градиентным методом, а значения частных производных найти численно.
# Однако в этом случае нужно действовать аккуратно, особенно в случае шумных откликов.
#
# Если чёрный ящик работает долго и оценка целевой функции является вычислительно дорогой, как, например, проведение аэродинамического расчёта, то количество обращений к чёрному ящику лучше свести к минимуму.
# Именно в этом случае наиболее полезны байесовские методы оптимизации, так как они призваны найти глобальный экстремум целевой функции за минимальное количество итераций.
# Байесовская оптимизация начинается с априорной оценки целевой функции $f$ и обновляет её на каждой итерации, используя вновь полученные данные.
#
# Модель, используемая для аппроксимации целевой функции, называется *суррогатной моделью*.
# За поиск новой точки для следующей итерации отвечает *функция продвижения* (acquisition function), направляющая процесс поиска в те области, где наиболее вероятно получить улучшение результата.

# Алгоритм байесовской оптимизации можно формализовать следующим образом:
#
#  1. по точкам обучающей выборки $\{X_{train}, Y_{train}\}$ построить суррогатную модель;
#  1. используя функцию продвижения, найти следующую точку алгоритма $x_{next}$;
#  1. если выполнен критерий останова, закончить;
#  1. получить значение целевой функции $y_{next} = f_{obj}(x_{next})$;
#  1. добавить точку $(x_{next}, y_{next})$ в обучающую выборку и перейти к пункту 1.
#  
# В литературе данный алгоритм встречается также под названием *Efficient Global Optimization (EGO)*.

# Реализуем алгоритм оптимизации с помощью функции `next_point`, возвращающей следующую исследуемую точку.

def next_point(acquisition, X_train, Y_train, bounds):
    '''
    Proposes the next sampling point by optimizing the acquisition function.
    
    Args:
        acquisition: Acquisition function
        X_train: Sample locations (n x d)
        Y_train: Sample values (n x 1)

    Returns:
        Location of the acquisition function minimum
    '''
    dim = X_train.shape[1]
    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), X_train, Y_train)
    
    X = np.linspace(bounds[0], bounds[1], N_test).reshape(-1, 1)
    Y = min_obj(X)
    i_min = np.argmin(Y)
    
    return X[i_min].reshape(-1, 1), Y[i_min]


# ---

# ## Функция продвижения

# Поиск следующей исследуемой точки осуществляет функция продвижения (acquisition function).
# Функция продвижения реализует подход, известный как &laquo;эксплуатация и эксплорация&raquo; (exploitation and exploration).
# Данный подход призван найти компромисс между локальным и глобальным поиском.
# Эксплуатация отвечает за поиск минимума там, где суррогатная модель предсказывает хороший результат, а эксплорация отвечает за поиск в зонах с большой дисперсией прогноза суррогатной модели.

# Далее мы рассмотрим три варианта функции продвижения:
#
#  - нижняя граница доверительного интервала (LB),
#  - вероятность улучшения (PI),
#  - ожидаемое улучшение (EI).

# ### Нижняя граница доверительного интервала
#
# Первый вариант функции продвижения основан на значении нижней границы доверительного интервала $\mu(x) - k \sigma(x)$.
# Функция продвижения возвращает разницу между *располагаемым* минимальным значением $f_\mathrm{min}$ (минимум по обучающей выборке) и *предсказываемым* моделью значением нижней границы доверительного интервала:
# $$
#   \mathrm{LB}(x) = f_\mathrm{min} - \left(\mu(x) - k \sigma(x) \right).
# $$
#
# В качестве величины доверительного интервала используется несколько среднеквадратичных отклонений: $k\sigma$.
# Параметр $k$ влияет на баланс между эксплуатацией и эксплорацией: при $k \rightarrow 0$, $\mathrm{LB}(x) \rightarrow f_\mathrm{min} - \mu(x)$ (чистая эксплуатация), при $k \rightarrow \infty$, $\mathrm{LB}(x) \rightarrow -k \sigma(x)$ (чистая эксплорация).
#

def lower_bound(X, X_train, Y_train, k=0.):
    '''
    Predicted Minimum = mu - k*std
    Computes the predicted minimum at points X based on existing samples
    X_train and Y_train using a Gaussian process surrogate model.
    
    Args:
        X: Points at which function shall be computed (m x d)
        X_train: Sample locations (n x d)
        Y_train: Sample values (n x 1)
        k: Exploitation-exploration trade-off parameter
    
    Returns:
        Predicted minimum at points X
    '''
    mu, cov = GP_predictor(X, X_train, Y_train,
                           kernel_fun, kernel_args, sigma_n)
    mu = mu.flatten()
    std = np.sqrt(np.diag(cov))
    y_min = np.min(Y_train)
    res = y_min - (mu - k*std)

    return res


# ### Вероятность улучшения
#
# Результат работы предыдущей функции сильно зависит от значения параметра $k$.
# Попробуем избавиться от него.
# Для этого посчитаем *вероятность улучшения*.

# Функция улучшения определяется как
# $$
#   I(x) = \max \left( f_\mathrm{min} - y(x), 0 \right),
# $$
# где $f_\mathrm{min}$ &mdash; значение текущего минимума, а $y(x)$ &mdash; предсказываемое суррогатной моделью значение в точке $x$.
# Отметим, что $y(x)$ является сечением гауссовского процесса в точке $x$ и, следовательно, гауссовской случайной величиной с плотностью распределения $p(y) \sim \mathcal{N}\left( \mu(x), \sigma^2(x) \right)$.

# Если в качестве суррогатной модели используется регрессия на гауссовых процессах, вероятность улучшения можно посчитать аналитически:
# $$
#   PI(x) = \mathrm{P}[I(x)>0] = \int \limits_{-\infty}^{f_\mathrm{min}} p(y) dy =
#   \int \limits_{-\infty}^{f_\mathrm{min}} \frac{1}{\sqrt{2\pi}\sigma} \exp{ \left( -\frac{(y - \mu)^2}{2\sigma^2}\right)} dy =
#   \int \limits_{-\infty}^{\frac{f_\mathrm{min}-\mu}{\sigma}}\phi(z) dz = \Phi\left(\frac{f_\mathrm{min} - \mu(x)}{\sigma(x)}\right).
# $$
# Здесь $\Phi(z)$ &mdash; функция стандартного нормального распределения, а $\phi(z)$ &mdash; его плотность.

def probability_of_improvement(X, X_train, Y_train, delta_f=0.1):
    ''''''
    mu, cov = GP_predictor(X, X_train, Y_train,
                           kernel_fun, kernel_args, sigma_n)
    mu = mu.flatten()
    std = np.sqrt(np.diag(cov))
    y_min = np.min(Y_train)

    with np.errstate(divide='warn'):
        imp = y_min - mu - delta_f
        z = imp / std
    res = norm.cdf(z)

    return res


# ### Ожидаемое улучшение
#
# Следующий шаг &mdash; использовать не вероятность улучшения, а его ожидаемую величину.
# Для этого вычислим (тоже аналитически) *математическое ожидание* улучшения.
# $$
# \begin{align*}
#   \mathrm{E}[I]
#   &= \int \limits_{-\infty}^{\infty} I p(y) dy
#   = \int \limits_{-\infty}^{f_\mathrm{min}} (f_\mathrm{min} - y)\,p(y) dy
#   = \int \limits_{-\infty}^{\frac{f_\mathrm{min}-\mu}{\sigma}} \left( f_\mathrm{min} - (\mu+\sigma z) \right) \phi(z) dz \\
#   &= \left( f_\mathrm{min} - \mu(x) \right) \, \Phi\left(\frac{f_\mathrm{min} - \mu(x)}{\sigma(x)}\right) + \sigma(x) \, \phi\left(\frac{f_\mathrm{min} - \mu(x)}{\sigma(x)}\right).
# \end{align*}
# $$
#
# Получившаяся функция продвижения $EI(x)$, называемая *ожидаемым улучшением*, удачно сочетает эксплуатацию и эксплорацию и поэтому используется чаще других.
# Первое слагаемое, с точностью до множителя совпадающее с вероятностью улучшения $PI(x)$, отвечает за эксплуатацию, второе &mdash; за эксплорацию.

def expected_improvement(X, X_train, Y_train, delta_f=0.1):
    '''
    Expected Improvement
    Computes the expected improvement at points X based on existing samples
    X_train and Y_train using a Gaussian process surrogate model.
    
    Args:
        X: Points at which EI shall be computed (m x d)
        X_train: Sample locations (n x d)
        Y_train: Sample values (n x 1)
        xi: Exploitation-exploration trade-off parameter
    
    Returns:
        Expected improvement at points X
    '''
    mu, cov = GP_predictor(X, X_train, Y_train,
                           kernel_fun, kernel_args, sigma_n)
    mu = mu.flatten()
    std = np.sqrt(np.diag(cov))
    mu_sample_opt = np.min(Y_train)

    with np.errstate(divide='warn'):
        imp = mu_sample_opt - mu - delta_f
        z = imp / std
    res = imp * norm.cdf(z) + std * norm.pdf(z)

    return res


# ---

# ## Тест 1. Бесшумные данные

# Проведём тестирование байесовского алгоритма оптимизации.
#
# Будем использовать следующую целевую функцию `f_obj`:
# $$
#   f_{obj} = (6x-2)^2 \cdot \sin\left(12x-3\right).
# $$
#
# На первом этапе рассмотрим бесшумные отклики.
# Исходная обучающая выборка задаётся переменными `X_init` и `Y_init`.

# ### Данные

# Для начала нарисуем исследуемую бесшумную целевую функцию.

# Objective function
def f_obj(X):
    return (6*X-2)**2 * np.sin(12*X-4)


# Dense grid of points within bounds
x_lims = [0., 1.]
N_test = 501
X_test = np.linspace(*x_lims, N_test).reshape(-1, 1)
# Objective function values at X_test 
Y_true = f_obj(X_test)

# Plot optimization objective
plt.figure(figsize=(8, 5))
plt.plot(X_test, Y_true, 'k-', label='Objective function')
plt.legend()
plt.tight_layout()
plt.show()

# ### Стандартная оптимизация

# Далее попробуем найти оптимум стандартными алгоритмами.

# **Метод Бройдена — Флетчера — Гольдфарба — Шанно (BFGS)**

# BFGS
x0 = 0.5
res = minimize(f_obj, x0, bounds=[(0, 1)], tol=1e-2,
               method='L-BFGS-B', options={'disp':True})
print(f'x = {res.x}\n')

# Plot optimization objective
plt.figure(figsize=(8, 5))
plt.plot(X_test, Y_true, 'k-', label='Objective function')
plt.plot(res.x, f_obj(res.x), '*', ms=20, c=cm.tab10(3), label='Minimum')
plt.legend()
plt.tight_layout()
plt.show()

# **Метод Нелдера — Мида (симплекс)**

# Nelder-Mead
# x0 = 0.5
res = minimize(f_obj, x0, bounds=[(0, 1)], tol=1e-2,
               method='Nelder-Mead', options={'disp':True})
print(f'x = {res.x}\n')

# Plot optimization objective
plt.figure(figsize=(8, 5))
plt.plot(X_test, Y_true, 'k-', label='Objective function')
plt.plot(res.x, f_obj(res.x), '*', ms=20, c=cm.tab10(3), label='Minimum')
plt.legend()
plt.tight_layout()
plt.show()

# ### Байесовская оптимизация

# Инициализируем начальную обучающую выборку.

x_range = x_lims[1] - x_lims[0]
X_init = [x_lims[0], 0.5*sum(x_lims), x_lims[1]]
X_init = np.array(X_init).reshape(-1, 1)
# X_init = np.array([0, 1., 0.75]).reshape(-1, 1)
Y_init = f_obj(X_init)

# Plot objective function
plt.figure(figsize=(8, 5))
plt.plot(X_test, Y_true, 'k-', label='Objective function')
plt.plot(X_init, Y_init, 'o', c=cm.tab10(3), ms=8, label='Initial samples')
plt.legend()
plt.tight_layout()
plt.show()

# Построим начальную суррогатную модель.

# +
kernel_fun = GP_kernels.gauss
kernel_args = {'l':.2, 'sigma_k':2.}
sigma_n = 1e-3

mu, cov = GP_predictor(X_test, X_init, Y_init,
                       kernel_fun, kernel_args, sigma_n)
# -

plt.figure(figsize=(8, 5))
plot_GP(X_test, mu, cov, X_init, Y_init, draw_ci=True)
plt.plot(X_test, Y_true, 'k--', label='Objective function')
plt.ylim(1.1*Y_true.min(), 1.1*Y_true.max())
plt.legend(loc=2)
plt.tight_layout()
plt.show()


# Функции `plot_approximation` и `plot_acquisition` выводят графики суррогатной модели и функции продвижения.

# +
def plot_approximation(X, Y, X_train, Y_train, X_next=None, show_legend=False):
    ''''''
    mu, cov = GP_predictor(X, X_train, Y_train,
                           kernel_fun, kernel_args, sigma_n)
    std = np.sqrt(np.diag(cov)).flatten()
    plt.fill_between(X.flatten(), mu.flatten()+2*std, mu.flatten()-2*std,
                     color=cm.tab10(4), alpha=0.1) 
    plt.plot(X, Y, 'k--', label='Objective function')
    plt.plot(X, mu, '-', c=cm.tab10(0), label='Surrogate function')
    plt.plot(X_train, Y_train, 'o', ms=7, c=cm.tab10(3), label='Train samples')
    i_min = np.argmin(Y_train)
    plt.plot(X_train[i_min],Y_train[i_min],'*',ms=15,c=cm.tab10(3))
    if X_next:
        plt.axvline(x=X_next, ls=':', c='k')
    if show_legend: plt.legend(loc=2)

def plot_acquisition(X, Y, X_next, show_legend=False):
    ''''''
    Y[Y<0] = 0
    plt.plot(X, Y, '-', c=cm.tab10(3), label='Acquisition function')
    plt.axvline(X_next, ls=':', c='k', label='Next point')
    if show_legend: plt.legend(loc=1)    


# -

# Запустим алгоритм байесовской оптимизации.
#
# Обучающая выборка для суррогатной модели содержится в переменных `X_train` и `Y_train` и обновляется на каждой итерации.
# Переменная `N_budget` задаёт максимальное количество итераций (бюджет оптимизации).

# +
# Choose acquisition function
acqusition_id = 'EI' # 'LB', 'PI', 'EI'

# Hyperparameters
kernel_args = {'l':.2, 'sigma_k':2.}
sigma_n = 1e-3

# Number of iterations
N_budget = 20

# acquisition function
def acq_function(X, X_train, Y_train):
    if   acqusition_id == 'LB':
        return lower_bound(X, X_train, Y_train, k=5.)
    elif acqusition_id == 'PI':
        return probability_of_improvement(X, X_train, Y_train, delta_f=.5)
    elif acqusition_id == 'EI':
        return expected_improvement(X, X_train, Y_train, delta_f=.5)


# -

# Initialize samples
X_train = X_init
Y_train = f_obj(X_init)

# +
plt.figure(figsize=(16, N_budget * 5))
plt.subplots_adjust(hspace=0.4)

for i in range(N_budget):   
    # Obtain next sampling point from the acquisition function
    X_next, acq = next_point(acq_function, X_train, Y_train, x_lims)
    
    # Obtain next sample from the objective function
    Y_next = f_obj(X_next)
    
    # Plot samples, surrogate function, noise-free objective and next sampling location
    ax = plt.subplot(N_budget, 2, 2 * i + 1)
    plot_approximation(X_test, Y_true, X_train, Y_train, X_next, show_legend=(i==0))
    plt.title(f'Iteration {i+3}, X_next = {X_next[0][0]:.3f}')

    plt.subplot(N_budget, 2, 2 * i + 2)
    Y_acq = acq_function(X_test, X_train, Y_train)
    plot_acquisition(X_test, Y_acq, X_next, show_legend=(i==0))
    
    # Add a new sample to train samples
    X_train = np.vstack((X_train, X_next))
    Y_train = np.vstack((Y_train, Y_next))
    
    print(i+3, *X_next, acq)
    if (-acq < 1e-100) or (abs(X_train[-2]-X_next)[0,0] < 1e-3*x_range):
        break


# -
# ---

# ## Тест 2. Шумные данные

# Теперь добавим к целевой функции шум:
# $$
#   f_{obj} = (6x-2)^2 \cdot \sin\left(12x-3\right) + \sigma_{in} \xi.
# $$
#
# Здесь $\xi$ &mdash; нормальная случайная величина, переменная $\sigma_{in}$ задаёт амплитуду шума.

# ### Данные

# Objective function
def f_obj_noisy(X, sigma_in):
    xi = np.random.randn(*X.shape)
    return (6*X-2)**2 * np.sin(12*X-4) + sigma_in * xi


# Objective function values at X_test
sigma_in = 3.
Y_noisy = f_obj_noisy(X_test, sigma_in)

# Plot optimization objective
plt.figure(figsize=(8, 5))
plt.plot(X_test, Y_true,  'k--', label='True objective')
plt.plot(X_test, Y_noisy, 'kx', alpha=.3, label='Noisy objective')
plt.legend()
plt.tight_layout()
plt.show()


# ### Байесовская оптимизация

# Требуется немного изменить функцию `plot_approximation`.

def plot_approximation(X, Y, X_train, Y_train, X_next=None, show_legend=False):
    ''''''
    mu, cov = GP_predictor(X, X_train, Y_train,
                           kernel_fun, kernel_args, sigma_n)
    std = np.sqrt(np.diag(cov)).flatten()
    plt.fill_between(X.flatten(), mu.flatten()+2*std, mu.flatten()-2*std,
                     color=cm.tab10(4), alpha=0.1)
    plt.plot(X, Y_true, 'k--', label='True objective')
    plt.plot(X, Y, 'kx', alpha=.2, label='Noisy objective')
    plt.plot(X, mu, '-', c=cm.tab10(0), label='Surrogate model')
    plt.plot(X_train, Y_train, 'o', ms=7, c=cm.tab10(3), label='Train samples')
    i_min = np.argmin(Y_train)
    plt.plot(X_train[i_min],Y_train[i_min],'*',ms=15,c=cm.tab10(3))
    if X_next:
        plt.axvline(x=X_next, ls=':', c='k')
    if show_legend: plt.legend(loc=2)


# Запуск алгоритма.

# +
# Choose acquisition function
acqusition_id = 'EI' # 'PM', 'PI', 'EI'

# Hyperparameters
kernel_args = {'l':.2, 'sigma_k':2.}
sigma_n = 1e-1

# Number of iterations
N_budget = 20

# acquisition function
def acq_function(X, X_train, Y_train):
    if   acqusition_id == 'PM':
        return predicted_minimum(X, X_train, Y_train, k=5.)
    elif acqusition_id == 'PI':
        return probability_of_improvement(X, X_train, Y_train, delta_f=.5)
    elif acqusition_id == 'EI':
        return expected_improvement(X, X_train, Y_train, delta_f=.5)


# -

# Initialize samples
np.random.seed(142)
X_train = X_init
Y_train = f_obj_noisy(X_init, sigma_in)

# +
plt.figure(figsize=(14, N_budget * 5))
plt.subplots_adjust(hspace=0.4)

for i in range(N_budget):
    # Obtain next sampling point from the acquisition function
    X_next, acq = next_point(acq_function, X_train, Y_train, x_lims)
    
    # Obtain next sample from the objective function
    Y_next = f_obj_noisy(X_next, sigma_in)
    Y_next_true = f_obj(X_next)
    
    # Plot samples, surrogate function, noise-free objective and next sampling location
    if not (i % 2):
        ax = plt.subplot(N_budget, 2, i + 1)
        plot_approximation(X_test, Y_noisy, X_train, Y_train, X_next, show_legend=(i==0))
        plt.title(f'Iteration {i+3}, X_next = {X_next[0][0]:.3f}')

        plt.subplot(N_budget, 2, i + 2)
        Y_acq = acq_function(X_test, X_train, Y_train)
        plot_acquisition(X_test, Y_acq, X_next, show_legend=(i==0))
    
    # Add a new sample to train samples
    X_train = np.vstack((X_train, X_next))
    Y_train = np.vstack((Y_train, Y_next))
    
    print(i+3, *X_next, acq)
    if (-acq < 1e-200) or (abs(X_train[-2]-X_next)[0,0] < 1e-3*x_range):
        break
# -
# ---

# ## Источники
#
# 1. *Krasser M.* [Gaussian processes](http://krasserm.github.io/2018/03/19/gaussian-processes/).
# 1. *Forrester A. I. J., Sobester A., Keane A. J.* Engineering design via surrogate modelling. &mdash;  John Wiley & Sons Ltd., 2008. &mdash; 210 p. (University of Southampton, UK)

# Versions used
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))




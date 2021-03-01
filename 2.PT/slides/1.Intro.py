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
# # Числовые характеристики выборки #

# + [markdown] toc=true slideshow={"slide_type": "subslide"}
# <h1>Содержание<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Коэффициенты-корреляции" data-toc-modified-id="Коэффициенты-корреляции-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Коэффициенты корреляции</a></span></li><li><span><a href="#Диаграммы-Каиро" data-toc-modified-id="Диаграммы-Каиро-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Диаграммы Каиро</a></span></li><li><span><a href="#Квартет-Энскомба" data-toc-modified-id="Квартет-Энскомба-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Квартет Энскомба</a></span></li><li><span><a href="#Литература" data-toc-modified-id="Литература-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Литература</a></span></li></ul></div>

# + slideshow={"slide_type": "skip"}
# Imports
import sys
sys.path.append('../modules')
import graph_support

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from IPython.display import Image
im_width = 1000

# %config Completer.use_jedi = False

# + slideshow={"slide_type": "skip"}
# Styles
import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['lines.linewidth'] = 1.5
matplotlib.rcParams['lines.markersize'] = 7
cm = plt.cm.tab10  # Colormap

import seaborn
seaborn.set_style('whitegrid')

# + slideshow={"slide_type": "skip"} language="html"
# <style>
#     .container.slides .celltoolbar, .container.slides .hide-in-slideshow {display: None ! important;}
# </style>

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Коэффициенты корреляции ##
#
# **Коэффициент корреляции Пирсона и ранговый коэффициент Спирмена**

# + slideshow={"slide_type": "subslide"}
x = np.linspace(0, 1, 101)
plt.figure(figsize=(8, 8))

for n in range(1,10,2):
    y = x**n
    plt.plot(x, y, label=f'$y=x^{n}$')
    print(stats.pearsonr(x, y)[0], stats.spearmanr(x, x**n)[0])

plt.xlabel('$x$')
plt.ylabel('$y$', rotation=0, ha='right')
plt.legend() 
plt.show()

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Диаграммы Каиро ##
#
# [Альберто Каиро](http://albertocairo.com/)

# + slideshow={"slide_type": "subslide"}
N = 2
data = []
for i in range(1,N+1):
    # df = pd.read_csv('./datasets/dataset-1.csv', header=None, names=['x','y'])
    data.append(np.genfromtxt(f'../pix/1.Intro/dataset-{i}.csv',delimiter=',',unpack=True))

print(np.shape(data))

# + slideshow={"slide_type": "subslide"}
print(f'X_m = {data[0][0].mean():.6}, {data[1][0].mean():.6}')
print(f'Y_m = {data[0][1].mean():.6}, {data[1][1].mean():.6}\n')

print(f'X_std = {data[0][0].std():.6}, {data[1][0].std():.6}')
print(f'Y_std = {data[0][1].std():.6}, {data[1][1].std():.6}\n')

Pr = [stats.pearsonr(item[0], item[1])[0] for item in data]
Sr = [stats.spearmanr(item[0], item[1])[0] for item in data]
print(f'Pearson corr.  = {Pr[0]:.6}, {Pr[1]:.6}')
print(f'Spearman corr. = {Sr[0]:.6}, {Sr[1]:.6}')

# + slideshow={"slide_type": "subslide"}
# Show data
fig, axes = plt.subplots(1, N, figsize=(13,6))

for i, ax in enumerate(axes):
    ax.plot(data[i][0], data[i][1], 'ko', alpha=0.8)
plt.show()

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
# display(Image('../pix/1.Intro/_mat/DinoSequential.gif', width=im_width))
# display(Image('../pix/1.Intro/_mat/AllDinosAnimated.gif', width=im_width))

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
display(Image('../pix/1.Intro/AllDinos.png', width=im_width))

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Квартет Энскомба ##
#
# Четыре набора числовых данных, у которых простые статистические свойства идентичны, но их графики существенно отличаются.
# Каждый набор состоит из 11 пар чисел.
# Квартет был составлен в 1973 году английским математиком Ф. Дж. Энскомбом для иллюстрации важности применения графиков для статистического анализа и влияния выбросов значений на свойства всего набора данных.

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
display(Image('../pix/1.Intro/Unstructured.png', width=im_width))

# + slideshow={"slide_type": "subslide"}
graph_support.hide_code_in_slideshow()
display(Image('../pix/1.Intro/Anscombe.png', width=im_width))

# + [markdown] slideshow={"slide_type": "skip"}
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# ## Литература ##
#
# 1. Шпигельхалтер Д. Искусство статистики. Как находить ответы в данных. &mdash; М.: Манн, Иванов и Фербер, 2021. &mdash; 448 с.

# + slideshow={"slide_type": "skip"}


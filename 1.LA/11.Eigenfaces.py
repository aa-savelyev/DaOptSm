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

# + [markdown] id="yi9SewFVU8lA"
# # Eigenfaces

# + id="6xkhYa-SgUjU"
# Imports
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# + id="EliWxWOVgjNw"
# Styles
import matplotlib
matplotlib.rc('font', size=12)
# matplotlib.rc('lines', lw=1.5, markersize=4)
cm = plt.cm.tab10  # Colormap

import seaborn
seaborn.set_style('white')

# +
import warnings
warnings.filterwarnings('ignore')

# # %config InlineBackend.figure_formats = ['pdf']
# # %config Completer.use_jedi = False
# -

# ---

# ## Данные

# + [markdown] id="xZMrmWMxU8lI"
# ### Исходные данные
#
# Воспользуемся набором данных `fetch_olivetti_faces` из пакета `sklearn`.
# В наборе 400 фотографий размером 64 на 64 &mdash; это портреты 40 людей в 10 ракурсах каждый.

# + colab={"base_uri": "https://localhost:8080/"} id="sHlMkrW1gkeV" outputId="1809f9a7-4949-4f46-dd52-8a10b72b5697"
from sklearn.datasets import fetch_olivetti_faces
data = fetch_olivetti_faces()
Imgs = data.images
print(Imgs.shape)
fW, fH = Imgs.shape[1], Imgs.shape[2]

# + [markdown] id="WDkNI4J3tze5"
# Объединим все портреты в единую матрицу.
# Для этого переформируем (`reshape`) каждую фотографию в столбец.
# Все получившиеся столбцы объединим в матрицу `allFaces`.

# + colab={"base_uri": "https://localhost:8080/"} id="ZY1G0BJ7hTg3" outputId="d9dfd6d3-0e30-46a2-8804-028e78080b04"
# reshaping into one matrix
allFaces = Imgs.reshape(-1, fW*fH).T
np.shape(allFaces)


# -

def display_faces(Faces, pers_idx, view_idx=range(10)): 
    Nf, Nv = len(pers_idx), len(view_idx)
    
    seaborn.set_style('white')
    fig, axes = plt.subplots(Nf, Nv, figsize=(1.5*Nv, 1.5*Nf))
    plt.subplots_adjust(wspace=0., hspace=0.)

    axes = np.atleast_2d(axes)
    for i, pers_i in enumerate(pers_idx):
        for j, view_j in enumerate(view_idx):
            Face = Faces[:,view_j + pers_i*10]
            axes[i,j].imshow(Face.reshape(fW,fH), cmap='gray')
            axes[i,j].axis('off')
    plt.show()


display_faces(allFaces, range(5))


# + [markdown] id="HlJV90yrt2xN"
# Теперь для того, чтобы вывести любую из 400 фотографий, нужно обратиться к соответствующему столбцу матрицы `allFaces` и заново сформировать из столбца матрицу 64 на 64.
# -

def display_face(Face):
    seaborn.set_style('white')
    plt.figure(figsize=(3,3))
    plt.imshow(Face.reshape(fW,fH), cmap='gray')
    plt.axis('off')
    plt.show()


display_face(allFaces[:,121])


def display_face_i(Faces, i):
    display_face(Faces[:,i])


# + colab={"base_uri": "https://localhost:8080/", "height": 273} id="yEOsVRMphbGt" outputId="72e53e12-2c27-4314-fd70-a8bb9956a619"
# access to photo
display_face_i(allFaces, 73)
# -

# ### Обучающая выборка

# Убираем последнюю персону.

trainFaces = allFaces[:,:-10]

# Смотрим на два последних.
# Было:

display_faces(allFaces, [-2, -1])

# Стало:

display_faces(trainFaces, [-2, -1])

# ---

# ## Главные компоненты

# + [markdown] id="iL7Q1pmFr4GO"
# Сделаем сингулярное разложение матрицы с обучающей выборкой. Получим 390 главных компонент.

# + colab={"base_uri": "https://localhost:8080/"} id="EJYaIDGshsFm" outputId="bfcc56cc-12f4-4fe7-cbc3-fe51a74b3c59"
# Singular value decomposition
U, s, Vt = LA.svd(trainFaces, full_matrices=False)
Sigma = np.diag(s)
print(f'Quantity of PC = {len(s)}')

# + [markdown] id="G2MXEFy-umUS"
# Посмотрим на первые 8 главных компонент.

# + colab={"base_uri": "https://localhost:8080/", "height": 376} id="pVGybjR7sqDA" outputId="e338e02d-3c51-4b82-a896-9c8ce7675fa5"
size = 3
fig, axes = plt.subplots(2, 4, figsize=(4*size, 2*size))
plt.subplots_adjust(wspace=0.1, hspace=0.1)
for i in range(8):
    axes[i//4, i%4].imshow(U[:,i].reshape(fW,fH), cmap='gray')
    axes[i//4, i%4].set_title(f'$\mathbf{{u_{i}}}$')
    axes[i//4, i%4].axis('off')
plt.show()

# + [markdown] id="ODpkEWyMvCLr"
# Теперь поиграем с данными.
# Попытаемся восстановить конкретную фотографию по её главным компонентам.
# Интересно сравнить фотографии 60 и 64, они отличаются, в основном, очками.

# + colab={"base_uri": "https://localhost:8080/", "height": 703} id="aHw4wZxehxZM" outputId="1d8ea6d0-d3ea-4188-d94a-1c53d6e065e0"
i_imgs = [60, 64]
k_list = [5, 20, 50]
size = 3
fig, axes = plt.subplots(1*len(i_imgs), 4, figsize=(4*size, len(i_imgs)*size))

# fig.suptitle("Reconstructed image using the first k singular values")
plt.subplots_adjust(wspace=0.1, hspace=0.15)
for j, i_img in enumerate(i_imgs):

    axes[j, 0].imshow(Imgs[i_img], cmap='gray')
    axes[j, 0].set_title("Original image")
    axes[j, 0].axis('off')
    for i, k in enumerate(k_list):
        # Reconstruction of the matrix using the first k singular values
        i += 1
        reconFace = U[:, :k] @ Sigma[:k, :k] @ Vt[:k, i_img]
        axes[j + i//4, i%4].imshow(reconFace.reshape(fW,fH), cmap='gray')
        axes[j + i//4, i%4].set_title("k = {}".format(k))
        axes[j + i//4, i%4].axis('off')
plt.show()
# fig.tight_layout()
# fig.savefig('1.png')

# + [markdown] id="AzO_qI8ZvN1C"
# Сравним аппроксимации 4 первых фотографий, сделанные по $k$ главным компонентам, с оригиналами.

# + colab={"base_uri": "https://localhost:8080/", "height": 400} id="nHZczSCmh1Ek" outputId="4befa13c-523c-46f0-a071-678f13bdad0c"
k = 50
i_img = [0, 10, 50, 70]

size=3
fig, axes = plt.subplots(2, 4, figsize=(4*size, 2*size))
fig.suptitle(f'Reconstructed image using the first {k} PCs', y=0.93)
plt.subplots_adjust(wspace=0.1, hspace=0.1)

for i, img in enumerate(i_img):
    reconFace = U[:, :k] @ Sigma[:k, :k] @ Vt[:k, img]
    axes[0, i].imshow(Imgs[img], cmap='gray')
    axes[1, i].imshow(reconFace.reshape(fW,fH), cmap='gray')

[axis.axis('off') for axis in axes.flatten()]
plt.show()
# -

# ---

# + [markdown] id="Wcry5MK8uaGx"
# ## Коэффициенты

# + [markdown] id="T6I0koEP8T9z"
# Посмотрим взаимное расположение "проекций лиц" (коэффициенты в разложении какого-либо ракурса по компонентам) на i и j главные компоненты.
# Это показывает тенденции в данных.
# -

p = 16
Vt[[6,7]][:,p*10:(p+1)*10]


# + id="LBqhRmTwl8ds"
def plot_pc_coeff(Vt, ps_idx, pc_idx):
    # PC coefficients
    seaborn.set_style('whitegrid')
    plt.figure(figsize=(7, 7))
    
    for p in ps_idx:
        pc = Vt[pc_idx][:,p*10:(p+1)*10]
        plt.plot(*pc, 'o', label=f'Pers. {p}')
    plt.xlabel(f'PC{pc_idx[0]}')
    plt.ylabel(f'PC{pc_idx[1]}')
    plt.legend(loc=0)
    plt.show()


# + id="kDpBpGnanhOh"
# different persons
ps_idx = [5, 7]
pc_idx = [0, 1]

# + colab={"base_uri": "https://localhost:8080/", "height": 290} id="kHtJZSBdAZ4F" outputId="af28ef6d-936e-49c3-c208-95a0956c2038"
display_faces(trainFaces, ps_idx)

# + colab={"base_uri": "https://localhost:8080/", "height": 269} id="b2CYmApTmsj_" outputId="da21c84a-acc4-44a5-fede-7cb0afa25e99"
# pca trends
plot_pc_coeff(Vt, ps_idx, pc_idx)


# -

# ---

# # Аппроксимация

def PC_recon(U, k, testFace):
    F = U[:, :k]
    Face = F @ F.T @ testFace
    return Face


testFace = allFaces[:, -5]
display_face(testFace)

k = 100
reconFace = PC_recon(U, k, testFace)
# display_face(reconFace)

# +
fig, axes = plt.subplots(1, 2, figsize=(2*size, size))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

axes[0].imshow(testFace.reshape(fW,fH), cmap='gray')
axes[1].imshow(reconFace.reshape(fW,fH), cmap='gray')
[axis.axis('off') for axis in axes.flatten()]
plt.show()

# + [markdown] id="aQASig0KJQKo"
# Испортим фотографию
# -

rng = np.random.default_rng(seed=42)
mask = np.ones_like(testFace)
n_bad = int(0.5 * fW*fH)
idx_bad = rng.choice(fW*fH, n_bad, replace=False)
mask[idx_bad] = 0

badFace = mask * testFace.copy()
display_face(badFace)

k = 50
U_mask = mask.reshape(-1,1) * U.copy()
Alpha = U_mask[:, :k].T @ badFace
reconBadFace = U[:, :k] @ Alpha
# display_face(Face)

# +
fig, axes = plt.subplots(1, 4, figsize=(4*size, size))
plt.subplots_adjust(wspace=0.1, hspace=0.1)

axes[0].imshow(testFace.reshape(fW,fH), cmap='gray')
axes[1].imshow(reconFace.reshape(fW,fH), cmap='gray')
axes[2].imshow(reconBadFace.reshape(fW,fH), cmap='gray')
axes[3].imshow(badFace.reshape(fW,fH), cmap='gray')
[axis.axis('off') for axis in axes.flatten()]
plt.show()

# + [markdown] id="7i2HRvldLsf_"
# ---

# +
# from PIL import Image
# img = Image.open("pix/Eigenfaces/Einstein_3.png").convert('L')
# img.save('pix/Eigenfaces/Einstein.png')
# -

# Reading the image
img = plt.imread("pix/Eigenfaces/Einstein.png")
print(np.shape(img))

testFace = img.flatten()
display_face(testFace)

k = 50
Face = PC_recon(U, k, testFace)
display_face(Face)

# ---

# + colab={"base_uri": "https://localhost:8080/"} id="20Zv3BYUJPtS" outputId="72a9f67b-1d57-4421-8a52-4ee69b939f4d"
# Versions used
import sys
print('Python: {}.{}.{}'.format(*sys.version_info[:3]))
print('numpy: {}'.format(np.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('seaborn: {}'.format(seaborn.__version__))

# + id="ce0qeP-f15eq"


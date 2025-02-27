# -*- coding: utf-8 -*-
"""ВКР - ИДЕНТИФИКАЦИЯ ДОРОЖНЫХ ЗНАКОВ

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1dPQvU-ybQFcLwttMiEzdBvrWd3sPVxOf
"""

!pip install ultralytics

import os
import random
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
from IPython.display import Image, display
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

display.clear_output()
!yolo checks

"""## TRAIN YOLOv8 Model on custom dataset"""

!pip install roboflow

from roboflow import Roboflow

import os
import random
import pandas as pd
from PIL import Image
import cv2
from ultralytics import YOLO
from IPython.display import Video
from IPython import display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import pathlib
import glob
from tqdm.notebook import trange, tqdm
import warnings
warnings.filterwarnings('ignore')

#Возьмем сначала предобученную модель
model = YOLO("yolov8n.pt")

#Используем модель для задачи обнаружения
image = "/content/drive/MyDrive/Kaggle dataset (Traffic sign detection)/train/images/00005_00017_00005_png.rf.10de59d74a2a2e7b5821a46bf108f2d1.jpg"
result_predict = model.predict(source = image, imgsz=(416))

#Вывод результатов
plot = result_predict[0].plot()
plot = cv2.cvtColor(plot, cv2.COLOR_BGR2RGB)
a = Image.fromarray(plot)
display.display(a)

"""# ТРЕНИРОВКА МОДЕЛИ"""

import torch
torch.cuda.empty_cache()
#Сборка на основе YAML-файла и перенос весов
model = YOLO('yolov8x.yaml').load('yolov8x.pt')

#Тренировка модели

yolo detect train data=/content/drive/MyDrive/Kaggle dataset (Traffic sign detection)/data.yaml model=yolov8n.pt epochs=100 imgsz=416 batch=12 project=Diplom name=yolov8n_416

"""#АНАЛИЗИРУЕМ ПОЛУЧЕННЫЕ РЕЗУЛЬТАТЫ"""

metrics = ["confusion_matrix.png", "P_curve.png","R_curve.png","PR_curve.png"]

#Загрузка полученных метрик
for i in metrics:
    image = cv2.imread(f"/content/drive/MyDrive/Kaggle dataset (Traffic sign detection)/yolov8n_416"/{i})

    #Создание чуть большей фигуры
    plt.figure(figsize=(16, 12))

    #Вывод изображения
    plt.imshow(image)

    #Вывод графика
    plt.show()

#Выводим данные из csv файла
Result_Final_model = pd.read_csv("/content/drive/MyDrive/Kaggle dataset (Traffic sign detection)/yolov8n_416/results.csv")
Result_Final_model.tail(10)

#Вывод csv файла в виде pandas dataframe'а
Result_Final_model.columns = df.columns.str.strip()

#Создание подграфиков
fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 15))

#Построение столбцов с помощью seaborn
sns.lineplot(x='epoch', y='train/box_loss', data=df, ax=axs[0,0])
sns.lineplot(x='epoch', y='train/cls_loss', data=df, ax=axs[0,1])
sns.lineplot(x='epoch', y='train/dfl_loss', data=df, ax=axs[1,0])
sns.lineplot(x='epoch', y='metrics/precision(B)', data=df, ax=axs[1,1])
sns.lineplot(x='epoch', y='metrics/recall(B)', data=df, ax=axs[2,0])
sns.lineplot(x='epoch', y='metrics/mAP50(B)', data=df, ax=axs[2,1])
sns.lineplot(x='epoch', y='metrics/mAP50-95(B)', data=df, ax=axs[3,0])
sns.lineplot(x='epoch', y='val/box_loss', data=df, ax=axs[3,1])
sns.lineplot(x='epoch', y='val/cls_loss', data=df, ax=axs[4,0])
sns.lineplot(x='epoch', y='val/dfl_loss', data=df, ax=axs[4,1])

#Установка заголовков и меток осей для каждого подзаголовка
axs[0,0].set(title='Train Box Loss')
axs[0,1].set(title='Train Class Loss')
axs[1,0].set(title='Train DFL Loss')
axs[1,1].set(title='Metrics Precision (B)')
axs[2,0].set(title='Metrics Recall (B)')
axs[2,1].set(title='Metrics mAP50 (B)')
axs[3,0].set(title='Metrics mAP50-95 (B)')
axs[3,1].set(title='Validation Box Loss')
axs[4,0].set(title='Validation Class Loss')
axs[4,1].set(title='Validation DFL Loss')


plt.suptitle('Training Metrics and Loss', fontsize=24)
plt.subplots_adjust(top=0.8)
plt.tight_layout()
plt.show()

"""# ПРОВЕРКА НА ТЕСТОВОМ НАБОРЕ ДАННЫХ"""

#Загрузка самый эффективной модели
Valid_model = YOLO('/content/drive/MyDrive/Kaggle dataset (Traffic sign detection)/yolov8n_416/weights/best.pt')

#Оценка модели на тестовом наборе
metrics = Valid_model.val(data="/content/drive/MyDrive/Kaggle dataset (Traffic sign detection)/data.yaml", split = 'test')

print("Results: ", metrics.results_dict)
from tqdm import tqdm
from numba import njit
import numpy as np
import os, sys
import random
import cv2

epochs = 20
image_size = 100
RGB_kernels_size = 3
RGB_kernels_count = 64
convol_chann_size = image_size - RGB_kernels_size + 1
pool_part_size1 = 2
pool_size1 = convol_chann_size // pool_part_size1
kernels_2layer_size = 4
kernels_2layer_count = 192
convol_layer2_size = pool_size1 - kernels_2layer_size + 1
pool_part_size2 = 2
pool_size2 = convol_layer2_size // pool_part_size2
kernels_3layer_size = 3
kernels_3layer_count = 384
convol_layer3_size = pool_size2 - kernels_3layer_size + 1
pool_part_size3 = 2
pool_size3 = convol_layer3_size // pool_part_size3
output_size = 11
flaten_convol_size = kernels_3layer_count * pool_size3 * pool_size3
learning_rate = 0.010

R_chan_weights = np.random.normal(0.0, pow(RGB_kernels_size * RGB_kernels_size, -0.5),
                                   (RGB_kernels_count, RGB_kernels_size, RGB_kernels_size))
G_chan_weights = np.random.normal(0.0, pow(RGB_kernels_size * RGB_kernels_size, -0.5),
                                   (RGB_kernels_count, RGB_kernels_size, RGB_kernels_size))
B_chan_weights = np.random.normal(0.0, pow(RGB_kernels_size * RGB_kernels_size, -0.5),
                                   (RGB_kernels_count, RGB_kernels_size, RGB_kernels_size))
kernel2_weights = np.random.normal(0.0, pow(kernels_2layer_size * kernels_2layer_size, -0.5),
                                    (kernels_2layer_count, kernels_2layer_size, kernels_2layer_size))
kernel3_weights = np.random.normal(0.0, pow(kernels_3layer_size * kernels_3layer_size, -0.5),
                                    (kernels_3layer_count, kernels_3layer_size, kernels_3layer_size))
output_weights = np.random.normal(0.0, pow(flaten_convol_size, -0.5), (output_size, flaten_convol_size))

def RGB_convololution(chann_layer, input_chann, chann_weights):
    for i in range(RGB_kernels_count):
        for h in range(convol_chann_size):
            for w in range(convol_chann_size):
                chann_layer[i, h, w] = np.sum(
                    input_chann[h:h + RGB_kernels_size, w:w + RGB_kernels_size] * chann_weights[i])
    return chann_layer
def next_layers_convololution(convol_layer, convol_layer_size, input_layer, kernel_weights, prev_kernels_count, kernels_size, kernel_per_input):
    l = 0
    r = kernel_per_input
    for k in range(prev_kernels_count):
        for i in range(l, r):
            for h in range(convol_layer_size):
                for w in range(convol_layer_size):
                    convol_layer[i, h, w] = np.sum(
                        input_layer[k, h:h + kernels_size, w:w + kernels_size] * kernel_weights[i])
        l = r
        r += kernel_per_input
    return convol_layer
def test_pooling(pooling_layer, pool_part_size, pool_size, convol_layer, kernels_count):
    for s in range(kernels_count):
        for h in range(pool_size):
            for w in range(pool_size):
                pooling_layer[s, h, w] = convol_layer[s, h * pool_part_size:h * pool_part_size + pool_part_size,
                                                      w * pool_part_size:w * pool_part_size + pool_part_size].max()

def poolmg_eror_convolertation(pooling_layer_eror, pl_eror_wrong_count, prev_kernels, kernel_per_input, pooling_layer):
    l = 0
    r = kernel_per_input
    for k in range(prev_kernels):
        for s in range(l, r):
            pooling_layer_eror[k] += pl_eror_wrong_count[s] * (pooling_layer[k] > 0)
        l = r
        r += kernel_per_input
    return pooling_layer_eror

def train_pooling(pooling_layer, pooling_layer_ind, pool_part_size, pool_size, convol_layer, kernels_count):
    for s in range(kernels_count):
        for h in range(pool_size):
            for w in range(pool_size):
                pool_part = convol_layer[s, h * pool_part_size:h * pool_part_size + pool_part_size,
                                         w * pool_part_size:w * pool_part_size + pool_part_size]
                pooling_layer[s, h, w] = convol_layer[s, h * pool_part_size:h * pool_part_size + pool_part_size,
                                         w * pool_part_size:w * pool_part_size + pool_part_size].max()
                for i in range(pool_part_size):
                    for j in range(pool_part_size):
                        if pool_part[i, j] == pooling_layer[s, h, w]:
                            I = int(i + h * pool_part_size)
                            J = int(j + w * pool_part_size)
                            pooling_layer_ind[s, I, J] = 1
def pooling_eror(pooling_layer_eror, convol_layer_eror, pooling_layer_ind, kernels_count, convol_layer_size):
    for s in range(kernels_count):
        i = 0
        for h in range(convol_layer_size):
            for w in range(convol_layer_size):
                if (pooling_layer_ind[s, h, w] == 1):
                    convol_layer_eror[s, h, w] = pooling_layer_eror[s, i]
                    i += 1
def back_prop(pl_eror_wrong_count, prev_pool_size, convol_layer_size, convol_layer_eror, kernels, weights_rot_180, kernel_size):
    for i in range(kernels):
        pl_eror_pattern[i, kernel_size - 1:convol_layer_size + kernel_size - 1, kernel_size - 1:convol_layer_size + kernel_size - 1] = convol_layer_eror[i]
    for s in range(kernels):
        weights_rot_180[s] = np.fliplr(weights_rot_180[s])
        weights_rot_180[s] = np.flipud(weights_rot_180[s])
    for s in range(kernels):
        for h in range(prev_pool_size):
            for w in range(prev_pool_size):
                pl_eror_wrong_count[s, h, w] = np.sum(pl_eror_pattern[s, h:h + kernel_size, w:w + kernel_size] * weights_rot_180[s])
    return pl_eror_wrong_count
def RGB_weights_updating(kernels_count, kernel_weights, kernels_size, input_layer, convol_layer_eror, convol_layer_size):
    for i in range(kernels_count):
        for h in range(kernels_size):
            for w in range(kernels_size):
                kernel_weights[i, h, w] -= np.sum(convol_layer_eror[i] * input_layer[h:h + convol_layer_size, w:w + convol_layer_size] * learning_rate)
    return kernel_weights

def weights_updating(prev_kernels_count, kernel_weights, kernels_size, kernel_per_input, input_layer, convol_layer_eror, convol_layer_size):
    l = 0
    r = kernel_per_input
    for k in range(prev_kernels_count):
        for i in range(l, r):
            for h in range(kernels_size):
                for w in range(kernels_size):
                    kernel_weights[i, h, w] -= np.sum(convol_layer_eror[i] * input_layer[k, h:h + convol_layer_size, w:w + convol_layer_size] * learning_rate)
        l = r
        r += kernel_per_input
    return kernel_weights
def training(img, targets, R_chann_weights, G_chann_weights, B_chann_weights, output_weights, kernel2_weights, kernel3_weights):
    data = (cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC))
    B, G, R = np.asarray(cv2.split(data)) / 255 * 0.99 + 0.01
    Rconvol_layer = Gconvol_layer = Bconvol_layer = np.zeros((RGB_kernels_count, convol_chann_size, convol_chann_size))
    convol_layer2 = np.zeros((kernels_2layer_count, convol_layer2_size, convol_layer2_size))
    convol_layer3 = np.zeros((kernels_3layer_count, convol_layer3_size, convol_layer3_size))

    # Подготовка пулинговых слоев и массивов, в которых запоминаются индексы максимальных элементов пулинга (понадобится для создания слоя матриц ошибок сверточных слоев)
    pooling_layer1 = np.zeros((RGB_kernels_count, pool_size1, pool_size1))
    pooling_layer1_ind = np.zeros((RGB_kernels_count, convol_chann_size))
    pooling_layer2 = np.zeros((kernels_2layer_count, pool_size2, pool_size2))
    pooling_layer2_ind = np.zeros((kernels_2layer_count, convol_layer2_size, convol_layer2_size))
    pooling_layer3 = np.zeros((kernels_3layer_count, pool_size3, pool_size3))
    pooling_layer3_ind = np.zeros((kernels_3layer_count, convol_layer3_size, convol_layer3_size))

    pl1_eror_pattern = np.zeros(
        (kernels_2layer_count, pool_size1 + kernels_2layer_size - 1, pool_size1 + kernels_2layer_size - 1))
    pl1_eror_wrong_count = np.zeros((kernels_2layer_count, pool_size1, pool_size1))
    pl2_eror_pattern = np.zeros(
        (kernels_3layer_count, pool_size2 + kernels_3layer_size - 1, pool_size2 + kernels_3layer_size - 1))
    pl2_eror_wrong_count = np.zeros((kernels_3layer_count, pool_size2, pool_size2))
    Rconvol_layer = RGB_convololution(Rconvol_layer, R, R_chann_weights)
    Gconvol_layer = RGB_convololution(Gconvol_layer, G, G_chann_weights)
    Bconvol_layer = RGB_convololution(Bconvol_layer, B, B_chann_weights)
    convol_layer1 = np.maximum((Rconvol_layer + Gconvol_layer + Bconvol_layer), 0)
    train_pooling(pooling_layer1, pooling_layer1_ind, pool_part_size1, pool_size1, convol_layer1, RGB_kernels_count)
    convol_layer2 = np.maximum(
        next_layers_convololution(convol_layer2, convol_layer2_size, pooling_layer1, kernel2_weights, RGB_kernels_count,
                                  kernels_2layer_size, (kernels_2layer_count // RGB_kernels_count)), 0)
    train_pooling(pooling_layer2, pooling_layer2_ind, pool_part_size2, pool_size2, convol_layer2, kernels_2layer_count)
    convol_layer3 = np.maximum(
        next_layers_convololution(convol_layer3, convol_layer3_size, pooling_layer2, kernel3_weights,
                                  kernels_2layer_count, kernels_3layer_size,
                                  (kernels_3layer_count // kernels_2layer_count)), 0)
    train_pooling(pooling_layer3, pooling_layer3_ind, pool_size3, convol_layer3, kernels_3layer_count)
    output_values = np.dot(output_weights, np.array(pooling_layer3.flatten(), ndmin=2).T)

    Exit = 1 / (1 + np.exp(-output_values))
    Exit_Eror = -(targets - Exit)

    convol_layer1_eror = np.zeros((RGB_kernels_count, convol_chann_size, convol_chann_size))
    convol_layer2_eror = np.zeros((kernels_2layer_count, convol_layer2_size, convol_layer2_size))
    convol_layer3_eror = np.zeros((kernels_3layer_count, convol_layer3_size, convol_layer3_size))

    pooling_layer1_Eror = np.zeros((RGB_kernels_count, pool_size1, pool_size1))
    pooling_layer2_Eror = np.zeros((kernels_2layer_count, pool_size2, pool_size2))

    pooling_layer3_Eror = np.dot(output_weights.T, Exit_Eror)
    pooling_layer3_Eror = pooling_layer3_Eror.reshape((kernels_3layer_count, pool_size3 * pool_size3))

    pooling_eror_expansion(pooling_layer3_Eror, convol_layer3_eror, pooling_layer3_ind, kernels_3layer_count, convol_layer3_size)

    #	Вычисление матриц ошибок 2 сверточного слоя
    pl2_eror_wrong_count = back_prop(pl2_eror_pattern, pl2_eror_wrong_count, pool_size2, convol_layer3_size, convol_layer3_eror, kernels_3layer_count, kernel3_weights, kernels_3layer_size)
    pooling_layer2_Eror = pooling_eror_convolertation(pooling_layer2_Eror, pl2_eror_wrong_count, kernels_2layer_count, (kernels_3layer_count // kernels_2layer_count), pooling_layer2)

    pooling_layer2_Eror = pooling_layer2_Eror.reshape((kernels_2layer_count, pool_size2 * pool_size2))
    pooling_eror_expansion(pooling_layer2_Eror, convol_layer2_eror, pooling_layer2_ind, kernels_2layer_size, convol_layer2_size)

    #	Вычисление матриц ошибок 1 сверточного слоя
    pl1_eror_wrong_count = back_prop(pl1_eror_pattern, pl1_eror_wrong_count, pool_size1, convol_layer2_size, convol_layer2_eror, kernels_2layer_count, kernel2_weights, kernels_2layer_size)
    pooling_layer1_Eror = pooling_eror_convolertation(pooling_layer1_Eror, pl1_eror_wrong_count, RGB_kernels_count, (kernels_2layer_count // RGB_kernels_count), pooling_layer1)
    pooling_layer1_Eror = pooling_layer1_Eror.reshape((RGB_kernels_count, pool_size1 * pool_size1))
    pooling_eror_expansion(pooling_layer1_Eror, convol_layer1_eror, pooling_layer1_ind, RGB_kernels_count, convol_chann_size)

   #Обновление весов

    output_weights -= learning_rate * np.dot((Exit_Eror * Exit * (1.0 - Exit)),
    np.array(pooling_layer3.flatten(), ndmin=2))
    kernel3_weights = weights_updating(kernels_2layer_count, kernel3_weights, kernels_3layer_size, (kernels_3layer_count // kernels_2layer_count), convol_layer2, convol_layer3_eror, convol_layer3_size)
    kernel2_weights = weights_updating(RGB_kernels_count, kernel2_weights, kernels_2layer_size, (kernels_2layer_count // RGB_kernels_count), convol_layer1, convol_layer2_eror, convol_layer2_size)
    Rconvol_layer = RGB_weights_updating(RGB_kernels_count, R_chann_weights, RGB_kernels_size, R, convol_layer1_eror, convol_chann_size)
    Gconvol_layer = RGB_weights_updating(RGB_kernels_count, G_chann_weights, RGB_kernels_size, G, convol_layer1_eror, convol_chann_size)
    Bconvol_layer = RGB_weights_updating(RGB_kernels_count, B_chann_weights, RGB_kernels_size, B, convol_layer1_eror, convol_chann_size)
    path = ' datas et\\custom\\train'
    inputs_names = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    random_inputs_names = random.sample(inputs_names, len(inputs_names))
    for i in range(epochs):
        for j in tqdm(random_inputs_names, desc=str(i + 1)):
            img = cv2.imread(path + '\\' + j)
            targets = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]).reshape((11, 1))
            if str(j[0]) == '0':
                targets[0] = 0.99
            elif str(j[0]) == '1':
                targets[1] = 0.99
            elif str(j[0]) == '2':
                targets[2] = 0.99
            elif str(j[0]) == '3':
                targets[3] = 0.99
            elif str(j[0]) == '4':
                targets[4] = 0.99
            elif str(j[0]) == '5':
                targets[5] = 0.99
            elif str(j[0]) == '6':
                targets[6] = 0.99
            elif str(j[0]) == '7':
                targets[7] = 0.99
            elif str(j[0]) == '8':
                targets[8] = 0.99
            elif str(j[0]) == '9':
                targets[9] = 0.99
            elif str(j[0]) == '10':
                targets[10] = 0.99
            training(img, targets, R_chann_weights, G_chann_weights, B_chann_weights, output_weights, kernel2_weights,
                     kernel3_weights)
def testing(img):
    data = (cv2.resize(img, dsize=(image_size, image_size), interpolation=cv2.INTER_CUBIC))
    B, G, R = np.asarray(cv2.split(data)) / 255 * 0.99 + 0.01
    Rconvol_layer = Gconvol_layer = Bconvol_layer = np.zeros((RGB_kernels_count, convol_chann_size, convol_chann_size))
    convol_layer1 = np.zeros((kernels_1layer_count, convol_layer1_size, convol_layer1_size))
    convol_layer3 = np.zeros((kernels_3layer_count, convol_layer3_size, convol_layer3_size))
    pooling_layer1 = np.zeros((RGB_kernels_count, pool_size1, pool_size1))
    pooling_layer2 = np.zeros((kernels_2layer_count, pool_size2, pool_size2))
    pooling_layer3 = np.zeros((kernels_3layer_count, pool_size3, pool_size3))
#	Подготовка пулинговых слоев
    Rconvol_layer = RGB_convololution(Rconvol_layer, R, R_chann_weights)
    Gconvol_layer = RGB_convololution(Gconvol_layer, G, G_chann_weights)
    Bconvol_layer = RGB_convololution(Bconvol_layer, B, B_chann_weights)
#	Свертка
    convol_layer1 = np.maximum((Rconvol_layer + Gconvol_layer + Bconvol_layer), 0)
    test_pooling(pooling_layer1, pool_part_size1, pool_size1, convol_layer1, RGB_kernels_count)
    convol_layer2 = np.maximum(next_layers_convololution(convol_layer2, convol_layer2_size, pooling_layer1, kernel2_weights, RGB_kernels_count,
                                  kernels_2layer_size, (kernels_2layer_count // RGB_kernels_count)), 0)
    test_pooling(pooling_layer2, pool_part_size2, pool_size2, convol_layer2, kernels_2layer_count)
    convol_layer3 = np.maximum(
        next_layers_convololution(convol_layer3, convol_layer3_size, pooling_layer2, kernel3_weights, kernels_2layer_count, kernels_3layer_size,
                                  (kernels_3layer_count // kernels_2layer_count)), 0)
    test_pooling(pooling_layer3, pool_part_size3, pool_size3, convol_layer3, kernels_3layer_count)
    output_values = np.dot(output_weights, np.array(pooling_layer3.flatten(), ndmin=2).T)
    Exit = 1 / (1 + np.exp(-output_values))
    return Exit
    path = 'dataset\\custom\\train'
    efficiency = []
    im = []
    for j in tqdm(os.listdir(path)):
        img = cv2.imread(path + ' \\' + j)
        targets_ind = 0
        im.append(str(j))
        if str(j[0]) == '0':
            targets_ind = 0
        elif str(j[0]) == '1' :
            targets_ind = 1
        elif str(j[0]) == '2' :
            targets_ind = 2
        elif str(j[0]) == '3' :
            targets_ind = 3
        elif str(j[0]) == '4' :
            targets_ind = 4
        elif str(j[0]) == '5' :
            targets_ind = 5
        elif str(j[0]) == '6' :
            targets_ind = 6
        elif str(j[0]) == '7' :
            targets_ind = 7
        elif str(j[0]) == '8' :
            targets_ind = 8
        elif str(j[0]) == '9' :
            targets_ind = 9
        elif str(j[0]) == '10' :
            targets_ind = 10

        outputs = testing(img)
        max_output_index = np.argmax(outputs)
        if (max_output_index == targets_ind):
            efficiency.append(1)
        else:
            efficiency.append(0)
        efficiency_array = np.asarray(efficiency)
        print(im)
        performance = (efficiency_array.sum() / efficiency_array.size) * 100
        print('Производительность:', performance, '%')
        np.savetxt("efficiency_arr.txt", efficiency_array, delimiter=',')
        with open('weights/R_convol_layer.csv', 'w') as outfile:
            outfile.write('# Исходный размер R сверточного слоя: {0}\n'.format(R_chann_weights.shape))
            for data_slice in R_chann_weights:
                outfile.write('# Ядро свертокди\n')
                np.savetxt(outfile, data_slice)
        with open('weights/G_convol_layer.csv', 'w') as outfile:
            outfile.write('# Исходный размер G сверточного слоя: {0}\n'.format(G_chann_weights.shape))
            for data_slice in G_chann_weights:
                outfile.write('# Ядро свертокди\n')
                np.savetxt(outfile, data_slice)
        with open('weights/B_convol_layer.csv', 'w') as outfile:
            outfile.write('# Исходный размер B сверточного слоя: {0}\n'.format(B_chann_weights.shape))
            for data_slice in B_chann_weights:
                outfile.write('# Ядро свертокди\n')
                np.savetxt(outfile, data_slice)
        with open('weights/snd_convol_layer.csv', 'w') as outfile:
            outfile.write('# Исходный размер 2-го сверточного слоя: {0}\n'.format(kernel2_weights.shape))
            for data_slice in kernel2_weights:
                outfile.write('# Ядро сверток:\п\n')
                np.savetxt(outfile, data_slice)
        with open('weights/trd_convol_layer.csv', 'w') as outfile:
            outfile.write('# Исходный размер 3-го сверточного слоя: {0}\n'.format(kernel3_weights.shape))
            for data_slice in kernel3_weights:
                outfile.write('# Ядро сверток:\п\n')
                np.savetxt(outfile, data_slice)
        np.savetxt("weights/output_layer.csv", output_weights, delimiter=",")

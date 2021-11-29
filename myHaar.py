from skimage import draw, transform, io, color, exposure
import numpy as np
import matplotlib.pyplot as plt
import os

LIGHT_COLOR = 255
DARK_COLOR = 50
NAME_OF_PROJECT = os.getcwd().split("\\")[-1]

'''
Функция создания признака Хаара
@w_list - список ширин прямоугольников (размер равносилен числу прямоугольников)
@h - высота прямоугольника
@alpha - угол наклона
@show_bit - бит разрешения отобржения
return img - возвращаемое изображение
'''
'''
def create_haar(w_list, h, alpha = 0, show_bit = 0):
    #заполняем нулями прямоугольник со сторонами w_list[n-1] и h        
    img = np.zeros((h, w_list[len(w_list) - 1]), dtype=np.uint8)
    #устанавливаем начальную точку (0;0) - левый верхний угол
    begin_point = (0, 0);

    for i in range(0, len(w_list)):
        #конечня координата (h, w_list[i]) - правый нижний угол 
        end_point = (h, w_list[i])
        rr, cc = draw.rectangle(begin_point, extent=end_point, shape=img.shape)
        if i % 2 == 0:
            img[rr, cc] = LIGHT_COLOR
            print(LIGHT_COLOR, begin_point, end_point)
        else:
            img[rr, cc] = DARK_COLOR
            print(DARK_COLOR, begin_point, end_point)

        begin_point = (0, w_list[i])

    if not show_bit:
        return img
    # показываем полученный признак 
    io.imshow(img)
    io.show()
    # поворачиваем признак для детекции
    img = transform.rotate(img, 360-alpha, resize=True, preserve_range=False)
    io.imshow(img)
    io.show()
    return img
'''
'''
Функция создания признака Хаара
@w1=6 - ширина прямоугольника 1,
@w2=4 - ширина прямоугольника 2,
@w3=9 - ширина прямоугольника 3,
@w4=4 - ширина прямоугольника 4,
@w5=5 - ширина прямоугольника 5,
@h=9 - вытоса прямоугольников,
@alpha=19 - угол наклона признака,
@show_bit=1 - бит разрешения отображения признака
return img - матрица признака
'''


def create_haar_sign(w1, w2, w3, w4, w5, h, alpha, show_bit=0):
    # Заполняем нулями прямоугольную область h*wsum
    img = np.zeros((h, w1 + w2 + w3 + w4 + w5), dtype=np.uint8)
    # создаем белый прямоугольник h*w1
    rr, cc = draw.rectangle((0, 0), extent=(h, w1), shape=img.shape)
    img[rr, cc] = LIGHT_COLOR
    # создаем черный прямоугольник h*w2
    rr, cc = draw.rectangle((0, w1), extent=(h, w2), shape=img.shape)
    img[rr, cc] = DARK_COLOR
    # if w3 != 0, создаем белый прямоугольник h*w3
    if w3:
        rr, cc = draw.rectangle((0, w1 + w2), extent=(h, w3), shape=img.shape)
        img[rr, cc] = LIGHT_COLOR
    # if w4 != 0, создаем черный прямоугольник h*w4
    if w4:
        rr, cc = draw.rectangle((0, w1 + w2 + w3), extent=(h, w4), shape=img.shape)
        img[rr, cc] = DARK_COLOR

    # if w5 != 0, создаем белый прямоугольник h*w5
    if w5:
        rr, cc = draw.rectangle((0, w1 + w2 + w3 + w4), extent=(h, w5), shape=img.shape)
        # rr, cc = draw.rectangle((0, w1+w2+w3), extent=(h, w5+w4), shape=img.shape)
        img[rr, cc] = LIGHT_COLOR

    # возвращаем матрицу признака, если не выставлен бит отображения
    if not show_bit:
        return img
    # показываем полученный признак 
    io.imshow(img)
    io.show()
    # поворачиваем на угол alpha признак для детекции
    img = transform.rotate(img, 360 - alpha, resize=True, preserve_range=False)
    io.imshow(img)
    io.show()
    return img


'''
Функция перевода изображения в полутон и black-white изображение
return dataset - набор из пар (исходное изображение, ЧБ)
'''


def create_dataset():
    dataset = []
    os.chdir("Car")
    img_list = os.listdir()
    for img in img_list:
        image = io.imread(img)
        # перевод в полутон
        img_gray = color.rgb2gray(image)
        # добавляем контраст изображению
        # (изображение, обрезание по сигма-функции, коэф. усиления, инверсия)
        img_contrast = exposure.adjust_sigmoid(img_gray, cutoff=0.5, gain=100, inv=False)
        io.imshow(img_contrast)
        io.show()
        dataset.append((image, img_contrast))

    os.chdir(os.getcwd() + "/../../" + NAME_OF_PROJECT)
    return dataset


'''
Функция определения на картинке элементов, подходящих под признак Хаара
@img - исходное ихображение,
@haar_sing_img - матрица признака.
return coord- координаты объекта,
return max_val - максимальное значение
'''


def pruning(img, haar_sign_img):
    coord = (-1, -1)
    max_value = -1
    size = haar_sign_img.shape
    for x in range(img.shape[0] - size[0]):
        for y in range(img.shape[1] - size[1]):
            # детектируем обьект используя Хаара
            cur_value = detection(img[x:x + size[0], y:y + size[1]], haar_sign_img)
            if cur_value > max_value:
                max_value = cur_value
                coord = x, y

    return coord, max_value


'''
Функция детектирования изображения по признаку Хаара
@img - часть исходного изображения размером с признак
@haar_sign_img - матрица признака
return - разнца между суммами значений пикселей, входящих в белую и черную область
'''


def detection(img, haar_sign_img):
    # проверка размерности
    if img.shape != haar_sign_img.shape:
        print("ERROR: SIZES NOT EQUAL")
        raise IndexError
    # определяем порог изображения, с которым будем сравнивать характерный признак
    # 50*N*M/255
    threshold = 40 * haar_sign_img.shape[0] * haar_sign_img.shape[1] / 255
    light, dark = 0, 0
    # сравниваем с Хааром
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            # если пиксель принадлежит белому прямоугольнику
            if haar_sign_img[x][y] == LIGHT_COLOR / 255:
                light += img[x][y]
            # если приксель принадлежит черному прямоугольнику
            elif haar_sign_img[x][y] == DARK_COLOR / 255:
                dark += img[x][y]
    # если разница привысила границу, то объект найден         
    if light - dark > threshold:
        return light - dark
    return -1


'''
Фуекция обрисовки найденной области
@img - исходное изображение,
@c_img - копия сходного изображения,
@coordinates - координаты объекта,
@size - размер
'''


def show_zone(img, c_img, coordinates, size):
    c_img = color.gray2rgb(c_img)
    if coordinates != (-1, -1):
        rr, cc = draw.rectangle_perimeter(coordinates, extent=size, shape=img.shape)
        c_img[rr, cc] = (0, 1, 0)
        img[rr, cc] = (0, 255, 0)  # зеленая рамка

    io.imshow(c_img)
    io.show();
    io.imshow(img)
    io.show();


    label = "Car is detected" if coordinates != (-1, -1) else "Car is not found"
    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0][0].set(title='Original')
    ax[0][0].imshow(img)
    ax[0][1].set(title='Grayscale (contrast)', xlabel=label)
    ax[0][1].imshow(c_img)
    io.show()

'''
Печать результатов (вывод)
'''


def print2img_in_line(img1, img2, title1, title2, x1_label="", x2_label=""):
    fig, ax = plt.subplots(1, 2, squeeze=False)
    ax[0][0].set(title=title1, xlabel=x1_label)
    ax[0][0].imshow(img1)
    ax[0][1].set(title=title2, xlabel=x2_label)
    ax[0][1].imshow(img2)
    io.show()

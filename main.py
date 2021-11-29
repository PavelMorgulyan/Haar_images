#import os
import myHaar as h

if __name__ == '__main__':

    car_dataset = h.create_dataset()
    '''haar_sign = h.create_haar([12, 20, 80], h=15, alpha=19, show_bit=1)
    for n, image in enumerate(car_dataset):
        if n == 1:
            # ищем координаты в контрастном изображении
            coordinates, value = h.pruning(image[1], haar_sign)
            print(coordinates, value)
            h.show_zone(image[0], image[1], coordinates, haar_sign.shape)
    
    car_dataset = h.create_dataset()
    '''
    #(9,4,40,0,0,12,19,1)
    haar_sign = h.create_haar_sign(w1=4, w2=2, w3=24, w4=0, w5=0, h=4, alpha=19, show_bit=1)
    print("Haar shape: {}".format(haar_sign.shape))
    h.show_zone(car_dataset[0][0], car_dataset[0][1], (25, 50), haar_sign.shape)
    for n, image in enumerate(car_dataset):
        if n == 4:
            # ищем координаты в контрастном изображении
            coordinates, value = h.pruning(image[1], haar_sign)
            print("Coord = {}, diff = {}".format(coordinates, value))
            h.show_zone(image[0], image[1], coordinates, haar_sign.shape)
    
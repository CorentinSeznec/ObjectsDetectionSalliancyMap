import cv2 as cv
import numpy as np
import os 
import time 
# code carbon
# train on BB of salenty map, test with BB, vgg, 
path = '/home/onajib/Analyse vidéos/GITW_light2_TD/train1/Bowl/BowlPlace1Subject1/SaliencyMaps/'

absolute_path = "/home/onajib/Analyse vidéos/"
list_files = os.listdir(path)
bounding_box_saliency = {}

FRAME_PATH = os.path.join(absolute_path, 'Frames')

#create file of BB
def create_bounding_boxes(list_files, path, bounding_box_file):
    for file in list_files:
        img = cv.imread(path + file)
        n_image = file.split('_')
        n_image = n_image[1].strip('.png')
        n_images = list(n_image)
        index =0
        while n_images[0] == "0":
            index+=1
            del n_images[0]
            if index >3:
                break

        n_images = "".join(n_images)

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # print(gray.shape)
        # print(path + file)
        b = np.where(gray > 20)
        if(len(b[0]) != 0 and len(b[1])  !=0):
            x, y = max(b[0]), max(b[1])
            x1, y1 = min(b[0]), min(b[1])
            to = str(n_images) + ' ' + '0' + ' ' + str(x) + ' ' + str(y) + ' ' + str(x1) + ' ' + str(y1)
        else: 
            x = 0
            to = str(n_images) + ' ' + '0'
        bounding_box_file.write(to)
        bounding_box_file.write('\n')


def draw_image(img, bounding_box): 
    x, y = bounding_box[0], bounding_box[1]
    x1, y1 = bounding_box[2], bounding_box[3]
    start_point = (x, y)
    end_point = (x1, y1)
    thickness = 2
    color = (0, 0, 255)
    image = cv.rectangle(img, start_point, end_point, color, thickness)
    cv.imshow("WINDOW",image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# bounding_box_from_saliency = open("/home/onajib/Analyse vidéos/GITW_light2_TD/train1/Bowl/BowlPlace1Subject1/BowlPlace1Subject1_2_saliencymaps.txt", "w+")
# create_bounding_boxex(list_files, path, bounding_box_from_saliency)
def patched_img(img,x,y,x1,y1):
    return img[x:x1, y:y1]


def save_image(file_path, index_img):

    file_path = os.path.join("/home/onajib/Analyse vidéos/GITW_light2_TD/train1/Bowl/BowlPlace1Subject1/BowlPlace1Subject1_2_saliencymaps.txt")
    print("file_path", file_path)
    with open(file_path, "r") as file:
        line = file.readlines()
        file_line = line[index_img]
        file_line =  file_line.split(' ')

        if len(file_line) != 6:
            pass

        else:
            num_img = int(file_line[0])
            x_max, y_max = int(file_line[2]), int(file_line[3])
            x_min, y_min = int(file_line[4]), int(file_line[5])
            

            img_name = 'frame' + str(num_img) + '.jpg'
            print("img_name", img_name)
            img_path = os.path.join(FRAME_PATH, img_name)
            print("img_path", img_path)
            img_to_show = cv.imread(img_path)
            print("shape", img_to_show.shape)
            b_box = [ y_min,x_min , y_max, x_max]
            print("b_box", b_box)
            #draw_image(img_to_show,b_box)
            new_img = img_to_show[x_min:x_max, y_min:y_max,: ]
            cv.imwrite(absolute_path+"Patches/Bowl/"+"patch"+str(num_img)+'.jpg', new_img)


img_name = 'frame' + str(2) + '.jpg'
img_path = os.path.join(FRAME_PATH, img_name)

# img = cv.imread(img_path)
# cv.imshow('image', img )
# cv.waitKey(0) 

file_path = "/home/onajib/Analyse vidéos/GITW_light2_TD/train1/Bowl/BowlPlace1Subject1/BowlPlace1Subject1_2_saliencymaps.txt"
fd = open(file_path, 'r')
n = 0
for i in fd:
    n+=1
for j in range(n):
    save_image(file_path, j)


# entrainer les images sur les bounding box a partir des cartes de salliance
# tester sur les images avec les bounding box du cours
# matrice de confusion pour voir si il y a des classes qui ne sont pas bien prédites

##IMAGE AUGMENTATION CODE


from __future__ import print_function, division
import os 
from imgaug import augmenters as iaa
import cv2
import shutil 


sometimes = lambda aug: iaa.Sometimes(1, aug)
horiz = iaa.Sequential([iaa.Fliplr(1)], random_order=False) # horizontally flip all images
vertical = iaa.Sequential([iaa.Flipud(1)], random_order=False) # vertically flip all images
horiz_ver = iaa.Sequential([iaa.Fliplr(1), iaa.Flipud(1)], random_order=False) # horizontal and vertical flip

contrast_norm = iaa.Sequential([iaa.ContrastNormalization(0.7, 0.7)], random_order=False)



path = 'Input data path'
#
output_folder = 'output path'
#if not os.path.exists(output_folder):
#    os.makedirs(output_folder)


image_data = []
filenames = []
for item in os.listdir(path):
    print(item)
    if ".jpg" in item:
        filenames.append(item.split(".")[0])
        image = cv2.imread(os.path.join(path, item))
        image_data.append(image)
        rows,cols,n = image.shape
        angle = 15
        #Generate rotation matrix
        rot_matrix = cv2.getRotationMatrix2D((rows/2, cols/2), angle, 1)
        transformed_image = cv2.warpAffine(image, rot_matrix, (cols,rows))
        cv2.imwrite(os.path.join(output_folder, filenames[-1]+"_rot_15deg.jpg"), transformed_image)


aug_horiz = horiz.augment_images(image_data)
aug_vertical = vertical.augment_images(image_data)
aug_horvet = horiz_ver.augment_images(image_data)

contrast_normalization = contrast_norm.augment_images(image_data)



for i in range(len(aug_horiz)):
    cv2.imwrite(os.path.join(output_folder, filenames[i]+"_horizontal.jpg"), aug_horiz[i])
    cv2.imwrite(os.path.join(output_folder, filenames[i]+"_vertical.jpg"), aug_vertical[i])
    cv2.imwrite(os.path.join(output_folder, filenames[i]+"_hor_vertical.jpg"), aug_horvet[i])
    cv2.imwrite(os.path.join(output_folder, filenames[i]+"_contrast.jpg"), contrast_normalization[i])


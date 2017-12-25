from glob import glob
import os.path
import re

from tqdm import tqdm
from PIL import Image

root = '/home/cattaneod/CITYSCAPES/'
split = 'test'
file_list = glob(root + 'leftImg8bit/' + split + '/**/*.png', recursive=True)
label_list = {
                os.path.basename(path): re.sub('_leftImg8bit.', '_gtFine_labelTrainIds.', re.sub('leftImg8bit/', 'gtFine/', path))
                for path in file_list}

for i in tqdm(range(len(file_list)), ncols=150):
    img_path = file_list[i]
    cont = 0
    #print('Doing image: %d / %d' % (i,len(file_list)))
    lbl_path = label_list[os.path.basename(img_path)]
    img = Image.open(img_path)
    lbl = Image.open(lbl_path)
    for shift_lateral in [0,416,832,1248]:
        for shift_vertical in [0,112,224]:
            cropped_img = img.crop((shift_lateral, shift_vertical, shift_lateral+800, shift_vertical+800))
            cropped_lbl = lbl.crop((shift_lateral, shift_vertical, shift_lateral+800, shift_vertical+800))
            save_path = re.sub('/CITYSCAPES/', '/CITYSCAPES_crop/', img_path)
            save_path = re.sub('_leftImg8bit.','_%02d_leftImg8bit.' % cont, save_path)
            #print("Saving image to: ",save_path)
            cropped_img.save(save_path)
            save_path = re.sub('/CITYSCAPES/', '/CITYSCAPES_crop/', lbl_path)
            save_path = re.sub('_gtFine_labelTrainIds.','_%02d_gtFine_labelTrainIds.' % cont, save_path)
            #print("Saving lbl to: ",save_path)
            cropped_lbl.save(save_path)
            cont += 1

from __future__ import division
import torch

# from ray import tune
# from ray.tune import CLIReporter
# from ray.tune.schedulers import ASHAScheduler

import os
from operator import itemgetter
import numpy as np
import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from torch._C import wait
from scipy import ndimage

# path = 'dataset/trainsetcopy/final'
# savepath = 'dataset/trainsetcopy2/final'
path = 'dataset/trainset2/final'
savepath = 'test/data'
img_paths = sorted(os.listdir(f'{path}'))

reso = 416
lower_scale = 0.13
upper_scale = 0.87
colorw = (255, 255, 255) #white
colorr = (255, 0, 0) #red
colorg = (0, 255, 0) #green
colorb = (0, 0, 255) #blue
colory = (255, 255, 0) #yellow
threshcolor = 90
threshareamin = 30
threshareamid = 40
threshareamax = 60

for j, raw_img_path in enumerate(img_paths):
    print(f'{((j/len(img_paths))*100):.2f}%\t{raw_img_path}')
    img = Image.open(f'{path}/{raw_img_path}').copy()

    w, h = img.size
    h_ratio = h / reso
    w_ratio = w / reso
    draw = ImageDraw.Draw(img)

    if "[[" in raw_img_path:
        img_path = raw_img_path.split('[[')[1].split(']]')[0].split('],[')
        labels = np.zeros((4, 5))
        for i, l in enumerate(img_path):
            label = np.zeros(5)
            label[:4] = np.array([int(a) for a in l.split(',')]) # [xc, yc, w, h]
    else:
        img_path = raw_img_path.split('[')[1].split(']')[0].split(',') # get the bb info from the filename
        labels = np.zeros((1, 5))
        labels[0, :4] = np.array([int(a) for a in img_path]) # [xc, yc, w, h]
        
    for i in range(len(labels)):
        bbox = labels[i, 0:4]

        xc, yc = bbox[0], bbox[1]
        half_w, half_h = bbox[2] / 2, bbox[3] / 2
        x1, y1, x2, y2 = max([int(xc - half_w),0]), \
            max([int(yc - half_h), 0]), \
            max([int(xc + half_w), 0]), \
            max([int(yc + half_h), 0])

        caption = f'truth'

        scale = int(reso*lower_scale)
        scale_coor = int((y2-y1)*lower_scale)
        colortemp = [np.mean(np.array(img)[y1:y2, x1:x2, 0]), \
            np.mean(np.array(img)[y1:y2, x1:x2, 1]), \
            np.mean(np.array(img)[y1:y2, x1:x2, 2])]
        colortemp[0] = 0 if colortemp[0] == np.nan else int(colortemp[0])
        colortemp[1] = 0 if colortemp[1] == np.nan else int(colortemp[1])
        colortemp[2] = 0 if colortemp[2] == np.nan else int(colortemp[2])

        color = colorg
        if xc <= int(reso*lower_scale) or xc >= int(reso*upper_scale) or yc <= int(reso*lower_scale) or yc >= int(reso*upper_scale):
            color = colorr
        elif int(np.abs(np.mean(colory[0:2])-np.mean(colortemp[0:2]))) >= threshcolor:
            color = colorb
        if bbox[2] >= threshareamax or bbox[3] >= threshareamax:
            color = colory


        temparr = np.zeros((reso, reso, 2))
        temparr[int(reso*lower_scale):int(reso*upper_scale), int(reso*lower_scale):int(reso*upper_scale), :] = \
            np.array(img)[int(reso*lower_scale):int(reso*upper_scale), int(reso*lower_scale):int(reso*upper_scale), 0:2]
        temparr = np.abs(np.mean(temparr, axis=2))  # Sum of R & G channels (which defines yellow)
        tempid = np.array(np.where(np.max(temparr) == temparr))       # Index of highest sum
        tempid = np.array([tempid[1][0]-1, tempid[1][0], tempid[0][0]-1, tempid[0][0]]) #x1, x2, y1, y2
        tempidx = np.asarray(tempid)
        if color != colorg:
            flag = True
            while(flag):
                if (tempidx[1] - tempidx[0]) > threshareamid or (tempidx[3] - tempidx[2]) > threshareamid:
                    if (tempidx[1] - tempidx[0]) < threshareamin:
                        tempidx[:2] += int((threshareamin - (tempidx[1]-tempidx[0]))/2)
                    if (tempidx[3] - tempidx[1]) < threshareamin:
                        tempidx[-2:] += int((threshareamin - (tempidx[3]-tempidx[2]))/2)
                    break

                flag = False
                if tempidx[0]-1 >= 0:
                    tempidx[0] -= 1; flag = True
                if tempidx[1]+1 < reso:
                    tempidx[1] += 1; flag = True
                if tempidx[2]-1 >= 0 and (tempidx[3] - tempidx[2]) <= threshareamin:
                    tempidx[2] -= 1; flag = True
                if tempidx[3]+1 < reso and (tempidx[3] - tempidx[2]) <= threshareamin:
                    tempidx[3] += 1; flag = True

            # draw.rectangle(((tempidx[0], tempidx[2], tempidx[1], tempidx[3])), outline=colorw, width=2)
            # draw.rectangle(((tempidx[0], tempidx[3]+15, tempidx[1], tempidx[3])), fill=colorw)
            # draw.text(((tempidx[0]+2, tempidx[3])), 'new', fill='black')
            tempid = tempidx
        else:
            tempid = np.array([x1, x2, y1, y2])

        # draw.rectangle(((reso*lower_scale, reso*lower_scale, reso*upper_scale, reso*upper_scale)), outline=colorr, width=2)
        # draw.rectangle(((reso*lower_scale, reso*upper_scale+15, reso*upper_scale, reso*upper_scale)), fill=colorr)
        # draw.text(((reso*lower_scale+2, reso*upper_scale)), 'outside', fill='black')

        # draw.rectangle(((x1, y1, x2, y2)), outline=color, width=2)
        # draw.rectangle((x1, y2 + 15, x2, y2), fill=color)
        # draw.text((x1 + 2, y2), caption, fill='black')

        # draw.rectangle(((tempid[0], tempid[2], tempid[1], tempid[3])), outline=colorw, width=2)
        # draw.rectangle(((tempid[0], tempid[3]+15, tempid[1], tempid[3])), fill=colorw)
        # draw.text(((tempid[0]+2, tempid[3])), 'new', fill='black')
    os.makedirs(savepath, exist_ok=True)

    w, h = tempid[1] - tempid[0], tempid[3] - tempid[2]
    xc, yc = tempid[0] + w/2, tempid[2] + h/2
    tempid = [int(xc), int(yc), int(w), int(h)]
    # print(f'{raw_img_path.split("[")[0]}_{list(tempid)}.png')

    # img.show()
    # exit()
    img.save(f'{savepath}/{raw_img_path.split("[")[0]}{list(tempid)}.png')
    img.close()
    # exit()

exit()

fps = 2

gif = []
images = (Image.open(f'{savepath}/{f}').copy() for f in sorted(os.listdir(savepath)) if f.endswith('.png'))
for image in images:
    gif.append(image)

os.makedirs(path, exist_ok=True)
gif[0].save(f'test/sequence.gif', save_all=True, \
    optimize=False, append_images=gif[1:], loop=0, \
    duration=int(1000/fps))
print(f'[LOG] PREDICT | Prediction sequence saved as {path}/sequence.gif')
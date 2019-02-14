"""
This script reand the images created by the Demo_create_adv_images.py script 
and crops them and add the red arrow seen in the paper. 
"""

import cv2 as cv;
import matplotlib.pyplot as plt;
from os.path import join;

runner_id = 4;
samp_type = 'unif';
samp_frac = 0.15;

src = 'plots_adversarial/%s_%g/splits' % (samp_type, samp_frac);

fname1 = join(src, 'adv_%d_x.png' % (runner_id));
fname2 = join(src, 'adv_%d_xr.png' % (runner_id));
fname3 = join(src, 'adv_%d_fx.png' % (runner_id));
fname4 = join(src, 'adv_%d_fxr.png' % (runner_id));

img1 = cv.imread(fname1, cv.IMREAD_COLOR);
img2 = cv.imread(fname2, cv.IMREAD_COLOR);
img3 = cv.imread(fname3, cv.IMREAD_COLOR);
img4 = cv.imread(fname4, cv.IMREAD_COLOR);

N, M = img1.shape[:2];

lbv = int(0.4*N);  # lower bound vertical
ubv = int(0.75*N); # upper bound vertical
lbh = int(0.2*M);  # lower bound horizontal
ubh = int(0.8*M);  # upper bound horizontal 

img1 = img1[lbv:ubv, lbh:ubh, :];
img2 = img2[lbv:ubv, lbh:ubh, :];
img3 = img3[lbv:ubv, lbh:ubh, :];
img4 = img4[lbv:ubv, lbh:ubh, :];
print('img1.shape: ', img1.shape);

pte1 = (180, 100);
pts1 = (130, 130);
#cv.arrowedLine(img3, pte1, pts1, (0,0,255), 2);
cv.arrowedLine(img4, pte1, pts1, (0,0,255), 2);

pte1 = (180, 70);
pts1 = (150, 80);
#cv.arrowedLine(img3, pte1, pts1, (0,0,255), 2);
cv.arrowedLine(img4, pte1, pts1, (0,0,255), 2);

fname1 = 'adv_%d_x_crop.png'  % (runner_id);
fname2 = 'adv_%d_xr_crop.png' % (runner_id);
fname3 = 'adv_%d_fx_crop_arrow.png' % (runner_id);
fname4 = 'adv_%d_fxr_crop_arrow.png' % (runner_id);
cv.imwrite(join(src, fname1), img1);
cv.imwrite(join(src, fname2), img2);
cv.imwrite(join(src, fname3), img3);
cv.imwrite(join(src, fname4), img4);












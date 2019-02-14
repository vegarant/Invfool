"""
This script add the red arrows in the med 50 networks unstable recontruction
"""

import cv2 as cv;
import matplotlib.pyplot as plt;
from os.path import join;

runner_id = 2;
src = 'plots_adversarial_med_50/';

fname1 = join(src, 'r_%d_rec_pert_crop.png'   % (runner_id));

img1 = cv.imread(fname1, cv.IMREAD_COLOR);

N, M = img1.shape[:2];



lbv = int(0.4*N);  # lower bound vertical
ubv = int(0.75*N); # upper bound vertical
lbh = int(0.2*M);  # lower bound horizontal
ubh = int(0.8*M);  # upper bound horizontal 


pte1 = (180, 250);
pts1 = (180, 300);
cv.arrowedLine(img1, pte1, pts1, (0,0,255), 2);


pte1 = (90, 180);
pts1 = (90, 130);
cv.arrowedLine(img1, pte1, pts1, (0,0,255), 2);

fname1 = 'r_%d_rec_crop_arrow.png'  % (runner_id);

cv.imwrite(join(src, fname1), img1);
print('Saved output as \'%s\'' % (join(src, fname1)));











import cv2 as cv;
import matplotlib.pyplot as plt;
from os.path import join;

runner_id = 24;
itr = 2000;

src = 'plots_adv/splits';

fname1 = join(src, 'r_%d_adv_fxpr_itr_%04d.png'   % (runner_id, itr));

img1 = cv.imread(fname1, cv.IMREAD_COLOR);

N, M = img1.shape[:2];


pte1 = (125, 230);
pts1 = (105, 195);
cv.arrowedLine(img1, pte1, pts1, (0,0,255), 2);

pte1 = (235, 240);
pts1 = (235, 210);
cv.arrowedLine(img1, pte1, pts1, (0,0,255), 2);


fname1 = 'r_%d_adv_fxpr_itr_%04d_arrow.png'  % (runner_id, itr);

cv.imwrite(join(src, fname1), img1);
print('Saved output as \'%s\'' % (join(src, fname1)));











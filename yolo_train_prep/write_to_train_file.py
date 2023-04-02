# Place this script in the folder:
# <your-darknet-install-directory>/darknet
# Now as long as your blank train.txt is in the data folder, and your images in data/obj folder, this script will work just fine.
import os
import re
import glob

path_to_imgs = "data/obj/*.jpg"
img_paths = glob.glob(path_to_imgs)
img_paths = sorted(img_paths, key = lambda text: [int(t) if t.isdigit() else t for t in re.split('([0-9]+)', text)])
path_to_txt = "data/train.txt"
with open(path_to_txt, 'w') as file:
    for path in img_paths:
        file.write(path.replace(os.sep, "/") + "\n")
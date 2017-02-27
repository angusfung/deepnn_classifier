
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
from PIL import Image
import hashlib
import random
import shutil
from scipy.misc import toimage
from collections import defaultdict


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.

# List of all actors/actresses required for the assignment
act = ['Steve Carell', 'Bill Hader', 'Alec Baldwin', 'Fran Drescher', 'America Ferrera', 'Kristin Chenoweth']

testfile = urllib.URLopener()            
image_files = defaultdict(lambda: [], {})
hashing = hashlib.sha256()
test_size = 30
validation_size = 15

if not os.path.exists('data'):
    os.makedirs('data')

# Iterate through the act array and download images
for a in act:
    name = a.split()[1].lower()
    if not os.path.exists('data/' + name):
        os.makedirs('data/' + name)
        os.makedirs('data/' + name + '/test')
        os.makedirs('data/' + name + '/training')
        os.makedirs('data/' + name + '/validation')

    i = 0
    for line in open("subset_facescrub.txt"):
        if a in line:
            bbox = line.split()[5].split(',')
            expected_hash = line.split()[6]

            filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
            #A version without timeout (uncomment in case you need to 
            #unsupress exceptions, which timeout() does)
            #testfile.retrieve(line.split()[4], "uncropped/"+filename)
            #timeout is used to stop downloading images which take too long to download
            timeout(testfile.retrieve, (line.split()[4], filename), {}, 10)
            if not os.path.isfile(filename):
                continue
            else:
                # Successful download
                image_hash = hashlib.sha256(open(filename, 'rb').read()).hexdigest()
                if expected_hash == image_hash:
                    print("File " + filename + " was downloaded and matched the sha256 sum")
                    im = imread(filename)
                    im = im[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    im = imresize(im, (32, 32, 3))
                    # Check if the image is not a grayscale already
                    if im.shape != (32, 32):
                        im = rgb2gray(im)
                    # imsave() doesn't save grayscales properly
                    img = toimage(im)
                    new_name = filename.split('.')[0] + '.png'
                    img.save('data/' + name + '/training/' + new_name)
                    image_files[name].append(new_name)
                os.remove(filename)
            i += 1

# Now, randomly select the data for validation and test sets
seed = 34
for name in image_files.keys():
    random.seed(seed)
    seed += 1
    files = random.sample(image_files[name], test_size+validation_size) # 15 images for validation sets, 30 for test sets

    for filename in files[:test_size]:
        shutil.move('data/' + name + '/training/' + filename, 'data/' + name + '/test/' + filename)
    for filename in files[test_size:]:
        shutil.move('data/' + name + '/training/' + filename, 'data/' + name + '/validation/' + filename)

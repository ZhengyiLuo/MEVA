import numpy as np
import os
import shutil
from os import path
from PIL import Image
from meva.khrylib.utils.math import *
import cv2


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../assets'))


def out_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../out'))


def log_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), '../logs'))


def recreate_dirs(*dirs):
    for d in dirs:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)


def load_img(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        I = Image.open(f)
        img = I.resize((224, 224), Image.ANTIALIAS).convert('RGB')
        return img


def save_screen_shots(window, file_name, transparent=False):
    import pyautogui
    import glfw
    xpos, ypos = glfw.get_window_pos(window)
    width, height = glfw.get_window_size(window)
    image = pyautogui.screenshot(region=(xpos*2, ypos*2, width*2, height*2))
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGRA if transparent else cv2.COLOR_RGB2BGR)
    if transparent:
        image[np.all(image >= [240, 240, 240, 240], axis=2)] = [255, 255, 255, 0]
    cv2.imwrite(file_name, image)


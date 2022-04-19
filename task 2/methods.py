import numpy as np
import cv2
from PIL import Image
from scipy.fftpack import dct as sc_dct
import matplotlib.pyplot as plt
import io


def resize_image(width, height, canvas_width, canvas_height):
    if height != canvas_height:
        k = canvas_height / height
        width, height = int(width * k), int(height * k)
        if width > canvas_width:
            k = canvas_width / width
            width, height = int(width * k), int(height * k)
    return width, height


def get_image_by_plot(x, y, is_hist=False):
    if is_hist:
        plt.bar(x, y, align="center", width=160/len(x))
        plt.xlim([0, 255])
        plt.title('Intensity histogram features')
        plt.xlabel('Intensity')
        plt.ylabel('Frequency')
    else:
        plt.plot(x, y)
        plt.title('Gradient features')
        plt.xlabel('Upper border of upper window')
        plt.ylabel('Difference between windows')
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    plt.clf()
    return img


def get_image_to_display(cv2_img, canvas_width, canvas_height):
    cv2_img = cv2.resize(cv2_img, resize_image(cv2_img.shape[1], cv2_img.shape[0], canvas_width, canvas_height),
                         interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(cv2_img)


def scale(image, args, canvas_width=None, canvas_height=None, need_to_display=False):
    displayed_image = None
    new_size = (args[1], args[0])
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    if need_to_display:
        displayed_image = get_image_to_display(resized_image, canvas_width, canvas_height)
    return resized_image.reshape(new_size[0] * new_size[1]), displayed_image


def hist(image, args, canvas_width=None, canvas_height=None, need_to_display=False):
    displayed_image = None
    graphic, bin_edges = np.histogram(image, bins=args[0])
    if need_to_display:
        img = get_image_by_plot(bin_edges[:-1], graphic, is_hist=True)
        img = img.resize(resize_image(img.size[0], img.size[1], canvas_width, canvas_height))
        displayed_image = img
    return graphic, displayed_image


def get_zigzag(arr):
    zigzag = []
    for j in range(arr.shape[0]):
        diag = [arr[i][j - i] for i in range(j)]
        if len(diag) % 2:
            diag.reverse()
        zigzag += diag
    return zigzag


def dft(image, args, canvas_width=None, canvas_height=None, need_to_display=False):
    p = args[0]
    displayed_image = None
    f = np.abs(np.fft.fft2(image))
    if need_to_display:
        min_val = np.min(f)
        max_val = np.max(f)
        cv2_img = np.ubyte(255 * (f - min_val)/(max_val - min_val))[:p, :p]
        displayed_image = get_image_to_display(cv2_img, canvas_width, canvas_height)
    return get_zigzag(f[:p, :p]), displayed_image


def dct(image, args, canvas_width=None, canvas_height=None, need_to_display=False):
    p = args[0]
    displayed_image = None
    c = sc_dct(sc_dct(image, axis=1), axis=0)
    if need_to_display:
        min_val = np.min(c)
        max_val = np.max(c)
        cv2_img = np.ubyte(255 * (c - min_val)/(max_val - min_val))[:p, :p]
        displayed_image = get_image_to_display(cv2_img, canvas_width, canvas_height)
    return get_zigzag(c[:p, :p]), displayed_image


def gradient(image, args, canvas_width=None, canvas_height=None, need_to_display=False):
    w, s = args
    displayed_image = None
    result = []
    x_range = range(0, image.shape[0] // 2 - w, s)
    for i in x_range:
        upper_window = image[i:i + w, :]
        lower_window = np.flip(image[image.shape[0] - i - w:image.shape[0] - i, :], axis=0)
        result.append(np.linalg.norm([[x - y if x >= y else y - x for x, y in zip(u, v)]
                                      for u, v in zip(upper_window, lower_window)]))
    if need_to_display:
        x = list(x_range)
        y = result
        img = get_image_by_plot(x, y)
        img = img.resize(resize_image(img.size[0], img.size[1], canvas_width, canvas_height))
        displayed_image = img
    return result, displayed_image

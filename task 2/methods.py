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
        plt.bar(x, y, align="center", width=160 / len(x))
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


def spectre_functions(p, f, need_to_display, canvas_width, canvas_height):
    displayed_image = None
    if need_to_display:
        min_val = np.min(f)
        max_val = np.max(f)
        cv2_img = np.ubyte(255 * (f - min_val) / (max_val - min_val))[:p, :p]
        displayed_image = get_image_to_display(cv2_img, canvas_width, canvas_height)
    return get_zigzag(f[:p, :p]), displayed_image


def dft(image, args, canvas_width=None, canvas_height=None, need_to_display=False):
    f = np.abs(np.fft.fft2(image))
    return spectre_functions(args[0], f, need_to_display, canvas_width, canvas_height)


def dct(image, args, canvas_width=None, canvas_height=None, need_to_display=False):
    c = sc_dct(sc_dct(image, axis=1), axis=0)
    return spectre_functions(args[0], c, need_to_display, canvas_width, canvas_height)


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


def get_args_scale(progress, width, height):
    min_arg = 0.125
    max_arg = 0.5
    scale_arg = min_arg + progress * (max_arg - min_arg) if progress is not None else 0.35
    return int(height * scale_arg), int(width * scale_arg)


def get_args_hist(progress):
    min_arg = 8
    max_arg = 32
    arg = int(min_arg + progress * (max_arg - min_arg)) if progress is not None else 20
    return arg, None


def get_args_dct(progress):
    min_arg = 4
    max_arg = 24
    arg = int(min_arg + progress * (max_arg - min_arg)) if progress is not None else 10
    return arg, None


def get_args_dft(progress):
    min_arg = 4
    max_arg = 24
    arg = int(min_arg + progress * (max_arg - min_arg)) if progress is not None else 14
    return arg, None


def get_args_gradient(progress, steps):
    min_arg = 1
    max_arg = steps if progress is not None else 1
    arg = int(min_arg + progress * (max_arg - min_arg)) if progress is not None else 5  # (3, 2)

    match arg:
        case 1:
            arg1 = 1
            arg2 = 1
        case 2:
            arg1 = 2
            arg2 = 1
        case 3:
            arg1 = 2
            arg2 = 2
        case _:
            arg1 = ((arg - 1) // 3) + 2
            arg2 = None
            match (arg - 1) % 3:
                case 0:
                    arg2 = arg1 - 2
                case 1:
                    arg2 = arg1 - 1
                case 2:
                    arg2 = arg1
    return arg1, arg2


methods = [
    {
        "name": "Scale",
        "args": [{"name": "Scale coef.", "min": 0.125, "max": 1.0, "default": 0.35}], "get_args": get_args_scale,
        "fun": scale
    }, {
        "name": "Hist",
        "args": [{"name": "BINs number", "min": 8, "max": 64, "default": 32}], "get_args": get_args_hist,
        "fun": hist
    }, {
        "name": "Gradient",
        "args": [{"name": "Window size", "min": 1, "max": 8, "default": 3},
                 {"name": "Step", "min": 1, "max": 8, "default": 2}], "get_args": get_args_gradient,
        "fun": gradient
    }, {
        "name": "DCT",
        "args": [{"name": "Transform length", "min": 4, "max": 24, "default": 10}], "get_args": get_args_dct,
        "fun": dct
    }, {
        "name": "DFT",
        "args": [{"name": "Transform length", "min": 4, "max": 24, "default": 14}], "get_args": get_args_dft,
        "fun": dft
    }
]


def get_args(method_name, progress=None, width=None, height=None, steps=None):
    match method_name:
        case "Scale":
            return get_args_scale(progress, width, height)
        case "Hist":
            return get_args_hist(progress)
        case "DCT":
            return get_args_dct(progress)
        case "DFT":
            return get_args_dft(progress)
        case "Gradient":
            return get_args_gradient(progress, steps)


def get_args_for_print(method_name, args):
    if len(args) != 2:
        raise ValueError("Length of args list which was sent to print is not equal to 2 as it should be")
    arg1, arg2 = args
    match method_name:
        case "Scale":
            return f"{arg1}x{arg2}"
        case "Hist":
            return f"{arg1} BINs"
        case "DCT":
            return f"{arg1}x{arg1}"
        case "DFT":
            return f"{arg1}x{arg1}"
        case "Gradient":
            return f"W={arg1}; S={arg2}"

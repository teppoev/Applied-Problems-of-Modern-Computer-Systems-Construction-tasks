import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import io
from skimage.feature import hog

dict_name_idx = {"Canny": 0, "Color hist": 1, "HOG": 2}


def resize_image(width, height, canvas_width, canvas_height):
    if height != canvas_height:
        k = canvas_height / height
        width, height = int(width * k), int(height * k)
        if width > canvas_width:
            k = canvas_width / width
            width, height = int(width * k), int(height * k)
    return width, height


def get_image_by_plot(x, y, colors=None, args_text=None):
    if colors is not None:
        for i, color in enumerate(colors):
            plt.plot(x, y[i], c=color)
        plt.xlim([0, 255])
        plt.title(f"Color intensities histogram features")
        plt.xlabel('Color intensities')
        plt.ylabel('Frequencies')
    else:
        for i in y:
            plt.plot(x, i)
        plt.ylim([0, 100])
        plt.title(f'Correct answers (%) by number of references. {args_text}')
        plt.xlabel('Number of references')
        plt.ylabel('Correct answers, %')
        ax = plt.gca()
        ax.legend(['Min', 'Max', 'Mean'])

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


def canny(image, args=(None, None), canvas_width=None, canvas_height=None, need_to_display=False):
    displayed_image = None
    arg1, arg2 = args
    if arg1 is None:
        arg1 = methods[0]["args"][0]["default"]
    if arg2 is None:
        arg2 = methods[0]["args"][1]["default"]
    edges = cv2.Canny(image, arg1, arg2)
    features = edges / 255
    features = np.append(features.sum(axis=0), features.sum(axis=1))
    if need_to_display:
        displayed_image = get_image_to_display(edges, canvas_width, canvas_height)
    return features, displayed_image


def color_hist(image, args=(None, None), canvas_width=None, canvas_height=None, need_to_display=False):
    displayed_image = None
    arg1, arg2 = args
    if arg1 is None:
        arg1 = methods[1]["args"][0]["default"]
    color = ('b', 'g', 'r')
    graphics = []
    bin_edges = None
    for i, c in enumerate(color):
        graphic, bin_edges = np.histogram(image[:, :, i], bins=arg1)
        graphics.append(graphic)
    graphics = np.array(graphics)
    if need_to_display:
        img = get_image_by_plot(bin_edges[:-1], graphics, colors=color)
        img = img.resize(resize_image(img.size[0], img.size[1], canvas_width, canvas_height))
        displayed_image = img
    graphic = graphics.flatten()
    return graphic, displayed_image


def method_hog(image, args=(None, None), canvas_width=None, canvas_height=None, need_to_display=False):
    displayed_image = None
    arg1, arg2 = args
    if arg1 is None:
        arg1 = methods[2]["args"][0]["default"]
    if arg2 is None:
        arg2 = methods[2]["args"][1]["default"]
    resized_img = cv2.resize(image, (128, 64))
    fd, hog_image = hog(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), orientations=9, pixels_per_cell=(arg1, arg1),
                        cells_per_block=(arg2, arg2), visualize=True, channel_axis=2)
    if need_to_display:
        displayed_image = get_image_to_display(hog_image, canvas_width, canvas_height)
    return fd, displayed_image


methods = [
    {
        "name": "Canny",
        "args": [{"name": "LB", "min": 0, "max": 255, "default": 100},
                 {"name": "UB", "min": 0, "max": 255, "default": 200}], "get_args": None,
        "fun": canny
    }, {
        "name": "Color hist",
        "args": [{"name": "BINs", "min": 8, "max": 256, "default": 64}], "get_args": None,
        "fun": color_hist
    }, {
        "name": "HOG",
        "args": [{"name": "PpC", "min": 1, "max": 24, "default": 8},
                 {"name": "CpB", "min": 1, "max": 24, "default": 2}], "get_args": None,
        "fun": method_hog
    }
]


def get_args_for_cv(method_name):
    match method_name:
        case "Canny":
            return [(0, 200), (50, 100), (50, 150), (50, 200), (100, 150), (100, 200), (150, 200), (100, 255)]
        case "Color hist":
            return [((16 * i), None) for i in range(1, 9)]
        case "HOG":
            return [(4, 1), (4, 2), (4, 4), (8, 2), (8, 4), (12, 2), (12, 4), (12, 6)]
        case _:
            raise ValueError("Wrong method name!")


def get_args_for_print(method_name, args=(None, None)):
    if method_name == "Voting":
        return ', '.join(args)
    arg1, arg2 = args
    if arg1 is None:
        arg1 = methods[dict_name_idx[method_name]]["args"][0]["default"]
    if arg2 is None and method_name != "Color hist":
        arg2 = methods[dict_name_idx[method_name]]["args"][1]["default"]
    match method_name:
        case "Canny":
            return f"LB={arg1}, UB={arg2}"
        case "Color hist":
            return f"BINs={arg1}"
        case "HOG":
            return f"PpC={arg1}, CpB={arg2}"
        case _:
            raise ValueError("Wrong method name!")


def get_default_args(method_name):
    arg1 = methods[dict_name_idx[method_name]]["args"][0]["default"]
    arg2 = None if method_name == "Color hist" else methods[dict_name_idx[method_name]]["args"][1]["default"]
    return arg1, arg2

import os
import time
import random
import sys
import tkinter as tk
from functools import partial
from threading import Thread
from tkinter.filedialog import askopenfilename

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from tqdm import tqdm
from math import ceil, floor

from GUI import main_menu, cross_validation_gui
from methods import scale, hist, dct, dft, gradient, resize_image

methods = [{"name": "Scale", "arg": {"name": "Scale coef.", "min": 12, "max": None}, "fun": scale},
           {"name": "Hist", "arg": {"name": "BINs number", "min": 8, "max": 64}, "fun": hist},
           {"name": "Gradient", "arg": {"name": "Window size & Step", "min": 1, "max": 16}, "fun": gradient},
           {"name": "DCT", "arg": {"name": "Transform length", "min": 4, "max": 24}, "fun": dct},
           {"name": "DFT", "arg": {"name": "Transform length", "min": 4, "max": 24}, "fun": dft},
           {"name": "Voting", "arg": [], "fun": None}]
images_dir_names = ["default", "high cloaked", "with mask"]
stop_thread = False


def get_images():
    images = []
    for dir_name in images_dir_names:
        images.append([])
        for subdir_name in os.listdir(dir_name):
            images[-1].append([None] * 10)
            for filename in os.listdir(f"{dir_name}/{subdir_name}"):
                idx = int(filename[0]) - 1
                if idx == 0 and filename[1] == '0':
                    idx = 9
                images[-1][-1][idx] = cv2.imread(f"{dir_name}/{subdir_name}/{filename}", 0)
    return images


def get_args_for_gradient(arg):
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


def cross_validation(features):
    cv_results = []
    number_of_images_in_one_group = len(features[0])
    number_of_groups = len(features)
    number_of_combinations = 2
    distances_cache = np.full((number_of_groups * number_of_images_in_one_group,
                               number_of_groups * number_of_images_in_one_group), -1)
    for cv_training_set_size in range(1, number_of_images_in_one_group):
        min_correct = 1.0
        for number_of_combination in range(number_of_combinations):
            cv_training_set_indices = random.sample(range(number_of_images_in_one_group), cv_training_set_size)
            cv_training_set_indices.sort()
            cv_validation_set_indices = [i for i in range(number_of_images_in_one_group)
                                         if i not in cv_training_set_indices]
            cv_training_set = np.take(features, cv_training_set_indices, axis=1)
            cv_validation_set = np.take(features, cv_validation_set_indices, axis=1)
            right_answered = 0
            for i1 in range(number_of_groups):
                for j1 in range(cv_validation_set.shape[1]):
                    min_dist = None
                    min_idx = None
                    for i2 in range(number_of_groups):
                        for j2 in range(cv_training_set.shape[1]):
                            if stop_thread:
                                return
                            idx1 = i1 * number_of_images_in_one_group + cv_validation_set_indices[j1]
                            idx2 = i2 * number_of_images_in_one_group + cv_training_set_indices[j2]
                            if idx1 > idx2:
                                idx1, idx2 = idx2, idx1
                            if distances_cache[idx1][idx2] == -1:
                                distances_cache[idx1][idx2] = sum([x - y if x >= y else y - x
                                                                   for x, y in zip(cv_validation_set[i1][j1],
                                                                                   cv_training_set[i2][j2])])
                            if min_dist is None or distances_cache[idx1][idx2] < min_dist:
                                min_dist = distances_cache[idx1][idx2]
                                min_idx = i2
                    if i1 == min_idx:
                        right_answered += 1
            total = cv_validation_set.shape[0] * cv_validation_set.shape[1]
            if right_answered / total < min_correct - 1e-6:
                min_correct = right_answered / total
        cv_results.append(min_correct)
    return cv_results


def show_in_canvas(img, canvas, images_dict, img_keeper_key, img_container=None):
    width, height = img.size
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    width, height = resize_image(width, height, canvas_width, canvas_height)
    img = img.resize((width, height))
    ph = ImageTk.PhotoImage(img)
    images_dict[img_keeper_key] = ph
    if img_container is None:
        img_container = canvas.create_image(canvas_width // 2, canvas_height // 2, image=ph, anchor='center')
    else:
        canvas.itemconfig(img_container, ph)
    return img_container


def get_args_for_print(method_name, arg1, arg2=None):
    match method_name:
        case "Scale":
            return f"{arg1}x{arg2}"
        case "Hist":
            return f"{arg1} BINs"
        case "dct":
            return f"{arg1}x{arg1}"
        case "dft":
            return f"{arg1}x{arg1}"
        case "Gradient":
            return f"window size = {arg1}; step = {arg2}"


def main():
    window = tk.Tk()
    window.title("FaReS classifier")

    window.rowconfigure(0, minsize=400, weight=1)
    window.columnconfigure(0, minsize=800, weight=1)
    window.resizable(width=False, height=False)

    images = get_images()
    method_names = [method["name"] for method in methods]

    mm = main_menu(window, method_names)
    cvg = cross_validation_gui(window)

    def clear_canvases_cv():
        cvg["images"]["left_images"] = None
        cvg["images"]["right_images"] = None
        cvg["images"]["image00"] = None
        cvg["images"]["image01"] = None
        cvg["images"]["image02"] = None
        cvg["images"]["image10"] = None
        cvg["images"]["image11"] = None
        cvg["images"]["image12"] = None
        cvg["images"]["image00_cv2data"] = None
        cvg["canvases"]["left_canvas"].delete("all")
        cvg["canvases"]["right_canvas"].delete("all")
        cvg["canvases"]["canvas00"].delete("all")
        cvg["canvases"]["canvas01"].delete("all")
        cvg["canvases"]["canvas02"].delete("all")
        cvg["canvases"]["canvas10"].delete("all")
        cvg["canvases"]["canvas11"].delete("all")
        cvg["canvases"]["canvas12"].delete("all")
        text = cvg["upper_label"].cget("text")
        first_entry = text.find(':')
        second_entry = text[first_entry + 1:].find(':')
        if second_entry == -1:
            return
        cvg["upper_label"].config(text=text[0:first_entry + second_entry + 1])

    def display_df(df, column0_width=20, column_width=20):
        cur_main_window_height = window.winfo_height()
        cur_main_window_width = window.winfo_width()

        cvg["table"]["column"] = list(df.columns)
        column0 = list(df.index)
        for column in cvg["table"]["column"]:
            cvg["table"].heading(column, text=column)
            cvg["table"].column(column, width=column_width)
        cvg["table"].column('#0', width=column0_width)
        df_rows = df.to_numpy().tolist()
        # noinspection PyTypeChecker
        for i in range(len(df_rows)):
            cvg["table"].insert("", "end", text=column0[i], values=df_rows[i])

        window.update()
        window.geometry(f"{cur_main_window_width}x{cur_main_window_height}")
        cvg["frm_pandas"].tkraise()

    def cv_computing_thread_fun(method):
        method_arg = method["arg"]
        method_name = method["name"]
        training_images_dataset = images[0]
        min_arg = method_arg["min"]
        img_weight = training_images_dataset[0][0].shape[1]
        img_height = training_images_dataset[0][0].shape[0]
        match method_name:
            case "Scale":
                max_arg = int(img_height / 2)
            case _:
                max_arg = method_arg["max"]
        step_arg = max(floor((max_arg - min_arg) / 9), 1) if method_name != "Gradient" else 1
        cv_results = []

        up_label_text = cvg["upper_label"].cget("text")

        args = range(min_arg, max_arg, step_arg)
        left_container = right_container = None
        arg2 = None if method_name != "Scale" else img_weight * (min_arg / img_height)
        cvg["upper_label"].configure(text=(f"{up_label_text}: {get_args_for_print(method_name, min_arg, arg2)}. "
                                           f"Progress: {0}/{ceil((max_arg - min_arg) / step_arg)}"))
        for arg in tqdm(args):
            i_for_print = random.randint(0, len(training_images_dataset) - 1)
            j_for_print = random.randint(0, len(training_images_dataset[i_for_print]) - 1)
            features = []
            for i in range(len(training_images_dataset)):
                features.append([])
                for j in range(len(training_images_dataset[i])):
                    canvas_for_result_img = cvg["canvases"]["right_canvas"]
                    canvas_width = canvas_for_result_img.winfo_width()
                    canvas_height = canvas_for_result_img.winfo_height()
                    need_to_display_img = i == i_for_print and j == j_for_print
                    if method_name == "Scale":
                        arg1 = arg
                        arg2 = int(img_weight * (arg / img_height))
                    elif method_name == "Gradient":
                        arg1, arg2 = get_args_for_gradient(arg)
                    else:
                        arg1 = arg
                        arg2 = None
                    f, img = method["fun"](training_images_dataset[i][j], [arg1, arg2],
                                           canvas_width, canvas_height, need_to_display_img)
                    features[i].append(f)
                    if i == i_for_print and j == j_for_print:
                        cvg["upper_label"].configure(
                            text=(f"{up_label_text}: {get_args_for_print(method_name, arg1, arg2)}. "
                                  f"Progress: {(arg - min_arg) // step_arg}/{ceil((max_arg - min_arg) / step_arg)}"))
                        show_in_canvas(Image.fromarray(training_images_dataset[i][j]), cvg["canvases"]["left_canvas"],
                                       cvg["images"], "left_images", left_container)
                        show_in_canvas(img, canvas_for_result_img,
                                       cvg["images"], "right_images", right_container)
            if stop_thread:
                clear_canvases_cv()
                return
            cv_results.append(cross_validation(features))

        clear_canvases_cv()
        cvg["upper_label"].configure(text=f"{up_label_text}")

        args = [arg / max_arg for arg in list(args)] if method_name == "Scale" else list(args)
        v_fun = np.vectorize(lambda x: round(x, 3))
        if method_name == "Scale":
            columns = [f"{round(arg, 3)}" for arg in args]
        elif method_name == "Gradient":
            columns = [f"{arg1}; {arg2}" for arg1, arg2 in list(map(get_args_for_gradient, args))]
        else:
            columns = [f"{arg}" for arg in args]
        index = [f"Train sample size: {x + 1}" for x in range(len(cv_results[0]))]
        df = pd.DataFrame(v_fun(np.array(cv_results, dtype=object).transpose()),
                          columns=columns,
                          index=index)
        display_df(df, column0_width=130, column_width=40)

    def p_computing_thread_fun_image(method, training_set_size, arg1, arg2):
        training_images_dataset = images[0]
        number_of_groups = len(training_images_dataset)
        images_in_group = len(training_images_dataset[0])
        training_set_indices = random.sample(range(images_in_group), training_set_size)
        training_set_indices.sort()

        features = []
        for group in training_images_dataset:
            features.append([])
            for i in training_set_indices:
                features[-1].append(method["fun"](group[i], [arg1, arg2])[0])

        canvas_width = cvg["canvases"]["canvas00"].winfo_width()
        canvas_height = cvg["canvases"]["canvas00"].winfo_height()
        my_image_features, mif_img = method["fun"](cvg["images"]["image00_cv2data"], [arg1, arg2],
                                                   canvas_width, canvas_height, need_to_display=True)

        min_dist = None
        min_i = min_j = None
        features = np.array(features)
        for i in range(number_of_groups):
            for j in range(training_set_size):
                if stop_thread:
                    return
                dist = sum([x - y if x >= y else y - x for x, y in zip(my_image_features,
                                                                       features[i][j])])
                if min_dist is None or dist < min_dist:
                    min_dist = dist
                    min_i, min_j = i, training_set_indices[j]

        f_img = method["fun"](training_images_dataset[min_i][min_j], [arg1, arg2],
                              canvas_width, canvas_height, need_to_display=True)[1]

        show_in_canvas(Image.fromarray(training_images_dataset[min_i][min_j]), cvg["canvases"]["canvas01"],
                       cvg["images"], "image01")
        show_in_canvas(mif_img, cvg["canvases"]["canvas10"], cvg["images"], "image10")
        show_in_canvas(f_img, cvg["canvases"]["canvas11"], cvg["images"], "image11")

    def p_computing_thread_fun_set(method, training_set_size, arg1, arg2):
        print(arg1, arg2)
        training_images_dataset = images[0]
        number_of_groups = len(training_images_dataset)
        images_in_group = len(training_images_dataset[0])

        features = []
        for images_set in [training_images_dataset, images[1], images[2]]:
            features.append([])
            for group in images_set:
                features[-1].append([])
                for img in group:
                    if img is None:
                        features[-1][-1].append([-1] * len(features[0][0][0]))
                    else:
                        f, _img = method["fun"](img, [arg1, arg2])
                        features[-1][-1].append(f)
        features = np.array(features)

        training_set_indices = random.sample(range(images_in_group), training_set_size)
        training_set_indices.sort()
        control_set_indices = [i for i in range(images_in_group)
                               if i not in training_set_indices]
        training_set = np.take(features[0], training_set_indices, axis=1)
        control_set = np.take(features[0], control_set_indices, axis=1)

        canvas_width = cvg["canvases"]["canvas00"].winfo_width()
        canvas_height = cvg["canvases"]["canvas00"].winfo_height()

        stats = np.zeros((number_of_groups + 1, features.shape[0] + 1))
        counters = np.zeros((number_of_groups + 1, features.shape[0] + 1))
        external_idx = -1
        for images_set in [control_set, features[1], features[2]]:
            external_idx += 1
            i_print = random.randint(0, images_set.shape[0])
            j_print = random.randint(0, images_set.shape[1])
            for i1 in range(images_set.shape[0]):
                for j1 in range(images_set.shape[1]):
                    if images_set[i1][j1][0] == -1:
                        while i1 == i_print and j1 == j_print:
                            i_print = random.randint(0, images_set.shape[0])
                            j_print = random.randint(0, images_set.shape[1])
                        continue
                    min_dist = None
                    min_i = min_j = None
                    for i2 in range(training_set.shape[0]):
                        for j2 in range(training_set.shape[1]):
                            if stop_thread:
                                return
                            dist = sum([x - y if x >= y else y - x for x, y in zip(images_set[i1][j1],
                                                                                   training_set[i2][j2])])
                            if min_dist is None or dist < min_dist:
                                min_dist = dist
                                min_i, min_j = i2, training_set_indices[j2]

                    counters[i1][external_idx] += 1
                    counters[i1][features.shape[0]] += 1
                    counters[number_of_groups][external_idx] += 1
                    counters[number_of_groups][features.shape[0]] += 1
                    if min_i == i1:
                        stats[i1][external_idx] += 1
                        stats[i1][features.shape[0]] += 1
                        stats[number_of_groups][external_idx] += 1
                        stats[number_of_groups][features.shape[0]] += 1
                    if i1 == i_print and j1 == j_print:
                        i, j = i1, j1 if external_idx > 0 else control_set_indices[j1]
                        img0 = method["fun"](images[external_idx][i][j], [arg1, arg2],
                                             canvas_width, canvas_height, need_to_display=True)[1]
                        show_in_canvas(Image.fromarray(images[external_idx][i][j]),
                                       cvg["canvases"]["canvas00"],
                                       cvg["images"], "image00")
                        show_in_canvas(img0, cvg["canvases"]["canvas10"], cvg["images"], "image10")
                        print(f"i={i}, j={j}, mini={min_i}, minj={min_j}, len(trsi)={len(training_set_indices)}")
                        i, j = min_i, min_j
                        img1 = method["fun"](training_images_dataset[i][j], [arg1, arg2],
                                             canvas_width, canvas_height, need_to_display=True)[1]
                        show_in_canvas(Image.fromarray(training_images_dataset[i][j]),
                                       cvg["canvases"]["canvas01"],
                                       cvg["images"], "image01")
                        show_in_canvas(img1, cvg["canvases"]["canvas11"], cvg["images"], "image11")
                        img2 = method["fun"](training_images_dataset[i1][0], [arg1, arg2],
                                             canvas_width, canvas_height, need_to_display=True)[1]
                        show_in_canvas(Image.fromarray(training_images_dataset[i1][0]),
                                       cvg["canvases"]["canvas02"],
                                       cvg["images"], "image02")
                        show_in_canvas(img2, cvg["canvases"]["canvas12"], cvg["images"], "image12")
                        window.update()
                        time.sleep(8)
        clear_canvases_cv()

        stats = stats / counters
        columns = images_dir_names + ["total"]
        index = [f"s{i}" for i in range(1, number_of_groups + 1)] + ["total"]
        v_fun = np.vectorize(lambda x: round(x, 3))
        df = pd.DataFrame(v_fun(stats), columns=columns, index=index)
        display_df(df)

    def close():
        window.withdraw()
        sys.exit()

    def mm_btn_handler(method):
        cvg["central_area"].tkraise()
        cvg["frm_text"].tkraise()
        cvg["upper_label"].configure(text=f"Method chosen: {method['name']}")
        cvg["frm_buttons"].tkraise()
        cvg["frm_buttons"].focus()

        if method["name"] == "Voting":
            return

        cvg["btn_cv"].configure(command=lambda: cvg_btn_cv_handler(method))
        cvg["btn_start"].configure(command=lambda: cvg_btn_start_handler(method))

    def cvg_btn_back_handler():
        mm["central_area"].tkraise()
        mm["frm_buttons"].tkraise()
        mm["frm_buttons"].focus()

    def cvg_btn_back_cv_handler():
        cvg["frm_text"].tkraise()
        cvg["frm_buttons"].tkraise()
        cvg["frm_buttons"].focus()
        global stop_thread
        stop_thread = True
        clear_canvases_cv()
        cvg["entry_p"].delete(0, 'end')
        cvg["table"].delete(*cvg["table"].get_children())

    def cvg_btn_cv_handler(method):
        cvg["frm_images_cv"].tkraise()
        cvg["frm_buttons_cv"].tkraise()
        cvg["frm_buttons_cv"].focus()

        global stop_thread
        stop_thread = False

        cvg["btn_back_cv"].config(command=cvg_btn_back_cv_handler)
        computing_thread = Thread(daemon=True, target=cv_computing_thread_fun, args=[method])
        computing_thread.start()

    def cvg_btn_start_handler(method):
        cvg["frm_buttons_p"].tkraise()
        cvg["frm_text_p"].tkraise()
        cvg["frm_buttons_p"].focus()

        cvg["btn_back_p"].config(command=cvg_btn_back_p_handler)
        cvg["btn_load_file_p"].config(command=cvg_btn_load_file_p_handler)
        cvg["btn_ok_p"].config(command=lambda: cvg_btn_ok_p_handler(method))

    def cvg_btn_load_file_p_handler():
        file_name = askopenfilename(parent=window,
                                    filetypes=[("Images", (".png .jpg .jpeg .bmp .dib .jpe .jp2 .webp .pbm .pgm .ppm "
                                                           ".pxm .pnm .sr .ras .tiff .tif .exr .hdr .pic"))])
        if file_name != '':
            img = cv2.imread(file_name, 0)
            cvg["images"]["image00_cv2data"] = img
            img = Image.fromarray(img)
            show_in_canvas(img, cvg["canvases"]["canvas00"], cvg["images"], "image00")
            cvg["frm_images_p"].tkraise()

    def cvg_btn_back_p_handler():
        cvg["frm_text"].tkraise()
        cvg["frm_buttons"].tkraise()
        cvg["frm_buttons"].focus()

    def cvg_btn_ok_p_handler(method):
        try:
            text = cvg["entry_p"].get()
            args = text.split(', ')
            if not (len(args) == 2 or method["name"] == "Gradient" and len(args) == 3):
                raise ValueError("Wrong length of args list")
            args = list(map(float, args))
            sample_size = int(args[0])
            if sample_size < 1 or sample_size > 8:
                raise ValueError("Wrong sample size argument")
            arg1 = None
            arg2 = None
            if method["name"] == "Gradient":
                arg1 = int(args[1])
                arg2 = int(args[2])
                if arg1 < 1 or arg1 > images[0][0][0].shape[0] // 2 or arg2 < 1 or arg2 > 2 * arg1:
                    raise ValueError("Wrong method argument")
            elif method["name"] == "Scale":
                arg1 = args[1]
                if arg1 < 14.0 / images[0][0][0].shape[0] or arg1 > 1.0:
                    raise ValueError("Wrong method argument")
                arg1 = int(images[0][0][0].shape[0] * args[1])
                arg2 = int(images[0][0][0].shape[1] * args[1])
            else:
                arg1 = int(args[1])
                if arg1 < method["arg"]["min"] or arg1 > method["arg"]["max"]:
                    raise ValueError("Wrong method argument")
        except ValueError as e:
            print(e)
            return
        cvg["frm_images_p"].tkraise()
        cvg["frm_buttons_cv"].tkraise()
        cvg["frm_buttons_cv"].focus()

        global stop_thread
        stop_thread = False

        cvg["btn_back_cv"].config(command=cvg_btn_back_cv_handler)

        computing_thread = Thread(daemon=True,
                                  target=(p_computing_thread_fun_image if not cvg["images"]["image00"] is None
                                          else p_computing_thread_fun_set),
                                  args=[method, sample_size, arg1, arg2])
        computing_thread.start()

    mm["frm_buttons"].bind('<Escape>', lambda event: close())
    for idx in range(len(method_names)):
        mm["buttons"][idx].configure(command=partial(mm_btn_handler, methods[idx]))
    mm["frm_buttons"].bind('<Return>', lambda event: mm_btn_handler(methods[-1]))

    cvg["btn_back"].configure(command=cvg_btn_back_handler)
    cvg["frm_buttons"].bind('<Escape>', lambda event: cvg_btn_back_handler())

    mm["central_area"].tkraise()
    mm["frm_buttons"].tkraise()
    mm["frm_buttons"].focus()

    window.mainloop()


if __name__ == "__main__":
    main()

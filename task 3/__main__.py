import math
import os
import random
import sys
import time
import tkinter as tk
from functools import partial
from threading import Thread
from tkinter.filedialog import askopenfilename
from itertools import combinations

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageTk
from tqdm import tqdm

from GUI import main_menu, cross_validation_gui, clear_canvases_cv
from methods import methods, resize_image, get_args_for_cv, get_args_for_print, get_default_args, dict_name_idx
from methods import get_image_by_plot

images_dir_names = ["images"]
stop_thread = False
time_to_sleep = 3


def get_images():
    images = []
    for dir_name in images_dir_names:
        images.append([])
        for subdir_name in os.listdir(dir_name):
            images[-1].append([])
            for filename in os.listdir(f"{dir_name}/{subdir_name}"):
                images[-1][-1].append(cv2.imread(f"{dir_name}/{subdir_name}/{filename}"))
    return images


def show_in_canvas(img, canvas, images_dict, img_keeper_key,
                   canvas_width=None, canvas_height=None, img_container=None):
    width, height = img.size
    if canvas_width is None:
        canvas_width = int(canvas.winfo_width())
    else:
        canvas_width = int(canvas_width)
    if canvas_height is None:
        canvas_height = int(canvas.winfo_height())
    else:
        canvas_height = int(canvas_height)
    width, height = resize_image(width, height, canvas_width, canvas_height)
    img = img.resize((width, height))
    if img.mode == "RGB":
        r, g, b = img.split()
        img = Image.merge("RGB", (b, g, r))
    ph = ImageTk.PhotoImage(img)
    images_dict[img_keeper_key] = img
    canvas.image = ph
    canvas.configure(image=ph)
    # if img_container is None:
    #    img_container = canvas.create_image(canvas_width // 2, canvas_height // 2, image=ph, anchor='center')
    # else:
    #    canvas.itemconfig(img_container, ph)
    # return img_container


def get_features_for_set(images_set, method):
    features = []
    for image_group in images_set:
        features.append([])
        for img in image_group:
            if img is None:
                features[-1].append([-1] * len(features[0][0]))
            else:
                features[-1].append(method["fun"](img, method["args"])[0])
    return np.array(features)


# n means how much min_images, min_distances algorithm will use to find best reference
def get_best_reference(img_to_find, references, training_set_indices,
                       distances_cache=None, i1=None, j1=None,
                       validation_set_indices=None, n=1):
    number_of_groups, number_of_images_in_one_group = references.shape[:2]
    min_dist = [None] * n
    min_idx = [None] * n
    min_idx2 = [None] * n
    for i2 in range(number_of_groups):
        for j2 in range(number_of_images_in_one_group):
            if distances_cache is not None:
                idx1 = i1 * (len(training_set_indices) + len(validation_set_indices)) + validation_set_indices[j1]
                idx2 = i2 * (len(training_set_indices) + len(validation_set_indices)) + training_set_indices[j2]
                if idx1 > idx2:
                    idx1, idx2 = idx2, idx1
                if distances_cache[idx1][idx2] == -1:
                    distances_cache[idx1][idx2] = sum([x - y if x >= y else y - x
                                                       for x, y in zip(img_to_find, references[i2][j2])])
                dist = distances_cache[idx1][idx2]
            else:
                dist = sum([x - y if x >= y else y - x for x, y in zip(img_to_find, references[i2][j2])])
            for i in range(n):
                # noinspection PyTypeChecker
                if min_dist[i] is None or dist < min_dist[i]:
                    if min_dist[i] is not None:
                        for j in range(n - 1, i, -1):
                            if min_dist[j - 1] is None:
                                continue
                            min_dist[j] = min_dist[j - 1]
                            min_idx[j] = min_idx[j - 1]
                            min_idx2[j] = min_idx2[j - 1]
                    min_dist[i] = dist
                    # noinspection PyTypeChecker
                    min_idx[i] = i2
                    min_idx2[i] = training_set_indices[j2]
                    break
    return min_idx, min_dist


# noinspection PyUnresolvedReferences,PyTypeChecker
def voting(img, references, chosen_methods, number_of_groups, training_set_indices, dist_cache=None,
           i1=None, j1=None, validation_set_indices=None, n=4):
    min_indices = []
    min_distances = []
    for m_idx in range(len(chosen_methods)):
        min_idx, min_dist = get_best_reference(img[m_idx], references[m_idx], training_set_indices, dist_cache,
                                               i1, j1, validation_set_indices, n)
        min_indices.append(min_idx)
        min_distances.append(min_dist)

    score = [0] * number_of_groups
    for i in range(len(min_distances)):
        max_dist = min_distances[i][-1]
        dists = [(max_dist - x) / max_dist for x in min_distances[i]]
        for j in range(len(min_distances[i])):
            score[min_indices[i][j]] += dists[j]
    return score.index(max(score))


def main():
    window = tk.Tk()
    window.title("Pictures classifier")

    window.rowconfigure(0, minsize=800, weight=1)
    window.columnconfigure(0, minsize=1600, weight=1)
    window.minsize(800, 600)
    # window.resizable(width=False, height=False)

    images = get_images()
    method_names = [method["name"] for method in methods]

    mm = main_menu(window, method_names)
    cvg = cross_validation_gui(window)

    def display_df(df, y_data=None, column0_width=20, column_width=20, cv=True):
        cur_main_window_height = window.winfo_height()
        cur_main_window_width = window.winfo_width()
        table = cvg["table1"] if cv else cvg["table2"]
        table["column"] = list(df.columns)
        column0 = list(df.index)
        df_rows = df.to_numpy().tolist()
        canvas_width = int(cvg["canvases"]["graphic1"].winfo_width() * 1.25)
        canvas_height = int(cvg["canvases"]["graphic1"].winfo_height() * 1.25)

        def graphic_button_handler(y_idx, args_text):
            x = range(len(df_rows))
            # old_table_height = table.winfo_height()
            y = y_data[:, y_idx, :].T
            y = y * 100
            img = get_image_by_plot(x, y, args_text=args_text)
            show_in_canvas(img, cvg["canvases"]["graphic1"], cvg["images"], "graphic1", canvas_width, canvas_height)
            # table.config(height=old_table_height)
            # cvg["canvases"]["graphic1"].config(height=canvas_height)
            window.update()

        for i, column in enumerate(table["column"]):
            if cv:
                text = column[0] + "; " + column[1]
                table.heading(column, text=text, command=partial(graphic_button_handler, i // 2, column[0]))
            else:
                table.heading(column, text=column)
            table.column(column, width=column_width)
        table.column('#0', width=column0_width)
        # noinspection PyTypeChecker
        for i in range(len(df_rows)):
            table.insert("", "end", text=column0[i], values=df_rows[i])

        window.update()
        window.geometry(f"{cur_main_window_width}x{cur_main_window_height}")

        if cv:
            graphic_button_handler(0, table["column"][0][0])
            cvg["frm_pandas1"].tkraise()
        else:
            cvg["frm_pandas2"].tkraise()

    def cv_computing_thread_fun_not_voting(method, number_of_combinations=10):
        training_images_dataset = images[0]
        number_of_images_in_one_group = len(training_images_dataset[0])
        number_of_groups = len(training_images_dataset)
        up_label_text = cvg["upper_label"].cget("text")
        left_container = right_container = None
        canvas_width = cvg["canvases"]["right_canvas"].winfo_width()
        canvas_height = cvg["canvases"]["right_canvas"].winfo_height()

        args = get_args_for_cv(method["name"])
        counter = 0
        inter_args = ["min", "max", "mean", "std"]
        cv_results = np.full((number_of_images_in_one_group - 1, len(args), len(inter_args)), 0.0)
        i_for_print = random.randint(0, len(training_images_dataset) - 1)
        j_for_print = random.randint(0, len(training_images_dataset[i_for_print]) - 1)
        for arg1, arg2 in tqdm(args, position=0, leave=False):
            counter += 1
            features = get_features_for_set(training_images_dataset, {"fun": method["fun"], "args": (arg1, arg2)})
            _, img = method["fun"](training_images_dataset[i_for_print][j_for_print], (arg1, arg2),
                                   canvas_width, canvas_height, True)
            if stop_thread:
                clear_canvases_cv(cvg, up_label_text)
                return
            show_in_canvas(Image.fromarray(training_images_dataset[i_for_print][j_for_print]),
                           cvg["canvases"]["left_canvas"], cvg["images"], "left_images", canvas_width, canvas_height,
                           img_container=left_container)
            show_in_canvas(img,
                           cvg["canvases"]["right_canvas"], cvg["images"], "right_images", canvas_width, canvas_height,
                           img_container=right_container)
            cvg["upper_label"].configure(text=f"{up_label_text}: {get_args_for_print(method['name'], (arg1, arg2))}. "
                                              f"Progress: {counter}/{len(args)}")
            window.update()
            time.sleep(time_to_sleep)

            distances_cache = np.full((number_of_groups * number_of_images_in_one_group,
                                       number_of_groups * number_of_images_in_one_group), -1.0)
            for cv_training_set_size in tqdm(range(1, number_of_images_in_one_group), position=1, leave=False):
                correct = []
                counter1 = 0
                for cv_training_set_indices in tqdm(combinations(range(number_of_images_in_one_group),
                                                                 cv_training_set_size),
                                                    total=math.comb(number_of_images_in_one_group,
                                                                    cv_training_set_size), position=2, leave=False):
                    counter1 += 1
                    if counter1 > number_of_combinations:
                        break
                    cv_validation_set_indices = [i for i in range(number_of_images_in_one_group)
                                                 if i not in cv_training_set_indices]
                    cv_training_set = np.take(features, cv_training_set_indices, axis=1)
                    cv_validation_set = np.take(features, cv_validation_set_indices, axis=1)
                    right_answered = 0
                    for i1 in range(number_of_groups):
                        for j1 in range(cv_validation_set.shape[1]):
                            if stop_thread:
                                return
                            min_idx = get_best_reference(cv_validation_set[i1][j1], cv_training_set,
                                                         cv_training_set_indices, distances_cache,
                                                         i1, j1, cv_validation_set_indices)[0][0]
                            if i1 == min_idx:
                                right_answered += 1
                    total = cv_validation_set.shape[0] * cv_validation_set.shape[1]
                    correct.append(right_answered / total)
                correct = np.array(correct)
                mean_correct = correct.mean()
                std_correct = correct.std()
                min_correct = min(correct)
                max_correct = max(correct)
                cv_results[cv_training_set_size - 1][counter - 1] = (min_correct, max_correct,
                                                                     mean_correct, std_correct)

        clear_canvases_cv(cvg, up_label_text)
        cvg["upper_label"].configure(text=f"{up_label_text}")

        return cv_results[:, :, 2:], args, cv_results[:, :, :3]

    def cv_computing_thread_fun_voting(chosen_methods, up_label_text, number_of_combinations=10):
        training_images_dataset = images[0]
        number_of_images_in_one_group = len(training_images_dataset[0])
        number_of_groups = len(training_images_dataset)
        left_container = right_container = None
        canvas_width = cvg["canvases"]["right_canvas"].winfo_width()
        canvas_height = cvg["canvases"]["right_canvas"].winfo_height()

        i_for_print = random.randint(0, len(training_images_dataset) - 1)
        j_for_print = random.randint(0, len(training_images_dataset[i_for_print]) - 1)
        show_in_canvas(Image.fromarray(training_images_dataset[i_for_print][j_for_print]),
                       cvg["canvases"]["left_canvas"], cvg["images"], "left_images", canvas_width, canvas_height,
                       left_container)

        features = []
        for method in chosen_methods:
            args = get_default_args(method["name"])
            features.append(get_features_for_set(training_images_dataset, {"fun": method["fun"],
                                                                           "args": args}))
            if stop_thread:
                clear_canvases_cv(cvg, up_label_text)
                return
            window.update()
            time.sleep(time_to_sleep)

        inter_args = ["min", "max", "mean", "std"]
        cv_results = np.full((number_of_images_in_one_group - 1, len(inter_args)), 0.0)
        distances_cache = np.full((number_of_groups * number_of_images_in_one_group,
                                   number_of_groups * number_of_images_in_one_group), -1)
        for cv_training_set_size in tqdm(range(1, number_of_images_in_one_group), position=1, leave=False):
            correct = []
            if (cv_training_set_size - 1) % (number_of_images_in_one_group // len(chosen_methods)) == 0:
                part = (cv_training_set_size - 1) // (number_of_images_in_one_group // len(chosen_methods))
                method = methods[part]
                args = get_default_args(method["name"])
                img = method["fun"](training_images_dataset[i_for_print][j_for_print], args,
                                    canvas_width, canvas_height, True)[1]
                show_in_canvas(img,
                               cvg["canvases"]["right_canvas"], cvg["images"], "right_images",
                               canvas_width, canvas_height, right_container)
            counter1 = 0
            for cv_training_set_indices in tqdm(combinations(range(number_of_images_in_one_group),
                                                             cv_training_set_size),
                                                total=math.comb(number_of_images_in_one_group, cv_training_set_size),
                                                position=2, leave=False):
                counter1 += 1
                if counter1 > number_of_combinations:
                    break
                cv_validation_set_indices = [i for i in range(number_of_images_in_one_group)
                                             if i not in cv_training_set_indices]
                cv_training_set = [np.take(features[i], cv_training_set_indices, axis=1)
                                   for i in range(len(features))]
                cv_validation_set = [np.take(features[i], cv_validation_set_indices, axis=1)
                                     for i in range(len(features))]

                number_of_images_in_validation_group = cv_validation_set[0].shape[1]
                right_answered = 0
                for i1 in range(number_of_groups):
                    for j1 in range(number_of_images_in_validation_group):
                        if stop_thread:
                            return
                        min_idx = voting([cv_validation_set[i][i1][j1] for i in range(len(cv_validation_set))],
                                         cv_training_set, chosen_methods,
                                         number_of_groups, cv_training_set_indices, distances_cache,
                                         i1, j1, cv_validation_set_indices, n=4)
                        if i1 == min_idx:
                            right_answered += 1
                total = number_of_groups * number_of_images_in_validation_group
                correct.append(right_answered / total)
            correct = np.array(correct)
            mean_correct = correct.mean()
            std_correct = correct.std()
            min_correct = min(correct)
            max_correct = max(correct)
            cv_results[cv_training_set_size - 1] = (min_correct, max_correct, mean_correct, std_correct)
        return cv_results

    def cv_computing_thread_fun(method):
        method_name = method["name"]
        v_fun = np.vectorize(lambda x: round(x, 3))
        inter_args = ["min", "max", "mean", "std"]

        if method_name == "Voting":
            number_of_images_in_one_group = len(images[0][0])
            args = []
            up_label_text = cvg["upper_label"].cget("text")
            number_of_chosen_methods = [2, 3]
            total = sum([math.comb(len(methods), j) for j in number_of_chosen_methods])
            cv_results = np.full((number_of_images_in_one_group - 1, total, len(inter_args)), 0.0)
            counter = 0
            for i in tqdm(range(1, 2 ** (len(methods)))):
                if stop_thread:
                    return
                bits = [bool(int(x)) for x in bin(i + 2 ** len(methods))[3:]]
                if sum(bits) not in number_of_chosen_methods:
                    continue
                counter += 1
                chosen_methods = [methods[j] for j in range(len(methods)) if bits[j]]
                cvg["upper_label"].configure(
                    text=(f"Chosen methods: {[method['name'] for method in chosen_methods]}. "
                          f"Progress: {counter}/{total}"))
                _data = cv_computing_thread_fun_voting(chosen_methods, up_label_text)
                cv_results[:, counter - 1, :] = _data
                my_str_list = [str(j + 1) for j in range(len(methods)) if bits[j]]
                args.append(', '.join(my_str_list))
            cvg["upper_label"].configure(
                text=f"{'; '.join([f'{i + 1} — {method_names[i]}' for i in range(len(methods))])}")
            data = cv_results[:, :, 2:]
            min_max_data = cv_results[:, :, :3]
            data = data.reshape(data.shape[0], -1)
        else:
            try:
                data, args, min_max_data = cv_computing_thread_fun_not_voting(method)
                args = [get_args_for_print(method_name, _args) for _args in args]
            except TypeError:
                return
            data = data.reshape(data.shape[0], -1)
        columns = pd.MultiIndex.from_product([args, inter_args[2:]])
        if stop_thread:
            return
        index = [f"Train sample size: {x + 1}" for x in range(data.shape[0])]
        df = pd.DataFrame(v_fun(data),
                          columns=columns,
                          index=index)
        display_df(df, y_data=v_fun(min_max_data), column0_width=130, column_width=40, cv=True)

    def p_computing_thread_fun_image(chosen_methods, training_set_size):
        training_images_dataset = images[0]
        number_of_groups = len(training_images_dataset)
        canvas_width = cvg["canvases"]["canvas00"].winfo_width()
        canvas_height = cvg["canvases"]["canvas00"].winfo_height()
        images_in_group = len(training_images_dataset[0])
        training_set_indices = random.sample(range(images_in_group), training_set_size)
        training_set_indices.sort()

        features = []
        my_image_features = []
        for method in chosen_methods:
            args = method["args"]
            features.append(get_features_for_set(training_images_dataset,
                                                 {"fun": method["fun"], "args": args}))
            my_image_features.append(method["fun"](cvg["images"]["image00_cv2data"], args,
                                                   canvas_width, canvas_height, True)[0])
        features = [np.take(features[i], training_set_indices, axis=1) for i in range(len(features))]
        min_i = voting(my_image_features, features, chosen_methods, number_of_groups,
                       training_set_indices, n=4)

        show_in_canvas(Image.fromarray(training_images_dataset[min_i][0]), cvg["canvases"]["canvas01"],
                       cvg["images"], "image01", canvas_width, canvas_height)
        for method in chosen_methods:
            args = method["args"]
            mif_img = method["fun"](cvg["images"]["image00_cv2data"], args, canvas_width, canvas_height, True)[1]
            f_img = method["fun"](training_images_dataset[min_i][0], args, canvas_width, canvas_height, True)[1]
            show_in_canvas(mif_img, cvg["canvases"]["canvas10"], cvg["images"], "image10", canvas_width, canvas_height)
            show_in_canvas(f_img, cvg["canvases"]["canvas11"], cvg["images"], "image11", canvas_width, canvas_height)
            window.update()
            time.sleep(time_to_sleep)

    def p_computing_thread_fun_set(chosen_methods, training_set_size):
        training_images_dataset = images[0]
        number_of_groups = len(training_images_dataset)
        images_in_group = len(training_images_dataset[0])
        up_label_text = cvg["upper_label"].cget("text")
        canvas_width = cvg["canvases"]["canvas00"].winfo_width()
        canvas_height = cvg["canvases"]["canvas00"].winfo_height()

        features = []
        for images_set in [training_images_dataset] + [images[_i] for _i in range(1, len(images))]:
            features.append([])
            for method in chosen_methods:
                features[-1].append(get_features_for_set(images_set, method))

        training_set_indices = random.sample(range(images_in_group), training_set_size)
        training_set_indices.sort()
        control_set_indices = [i for i in range(images_in_group)
                               if i not in training_set_indices]
        training_set = [np.take(features[0][i], training_set_indices, axis=1) for i in range(len(features[0]))]
        control_set = [np.take(features[0][i], control_set_indices, axis=1) for i in range(len(features[0]))]

        stats = np.zeros((number_of_groups + 1, len(features) + 1))
        counters = np.zeros((number_of_groups + 1, len(features) + 1))
        external_idx = -1
        for images_set in [control_set] + [features[_i] for _i in range(1, len(features))]:
            external_idx += 1
            i_print = random.randint(0, images_set[0].shape[0] - 1)
            j_print = random.randint(0, images_set[0].shape[1] - 1)
            while images_set[0][i_print][j_print][0] == -1:
                i_print = random.randint(0, images_set[0].shape[0] - 1)
                j_print = random.randint(0, images_set[0].shape[1] - 1)
            for i1 in range(number_of_groups):
                for j1 in range(images_set[0].shape[1]):
                    if stop_thread:
                        return
                    min_i = voting([images_set[i][i1][j1] for i in range(len(images_set))], training_set,
                                   chosen_methods, number_of_groups, training_set_indices, n=4)
                    counters[i1][external_idx] += 1
                    counters[i1][len(features)] += 1
                    counters[number_of_groups][external_idx] += 1
                    counters[number_of_groups][len(features)] += 1
                    if min_i == i1:
                        stats[i1][external_idx] += 1
                        stats[i1][len(features)] += 1
                        stats[number_of_groups][external_idx] += 1
                        stats[number_of_groups][len(features)] += 1
                    if i1 == i_print and j1 == j_print:
                        j = j1 if external_idx > 0 else control_set_indices[j1]

                        show_in_canvas(Image.fromarray(images[external_idx][i1][j]),
                                       cvg["canvases"]["canvas00"],
                                       cvg["images"], "image00", canvas_width, canvas_height)
                        show_in_canvas(Image.fromarray(training_images_dataset[min_i][0]),
                                       cvg["canvases"]["canvas01"],
                                       cvg["images"], "image01", canvas_width, canvas_height)
                        show_in_canvas(Image.fromarray(training_images_dataset[i1][0]),
                                       cvg["canvases"]["canvas02"],
                                       cvg["images"], "image02", canvas_width, canvas_height)
                        for method in chosen_methods:
                            args = method["args"]
                            img0 = method["fun"](images[external_idx][i1][j], args,
                                                 canvas_width, canvas_height, True)[1]
                            show_in_canvas(img0, cvg["canvases"]["canvas10"], cvg["images"], "image10",
                                           canvas_width, canvas_height)
                            img1 = method["fun"](training_images_dataset[min_i][0], args,
                                                 canvas_width, canvas_height, True)[1]
                            show_in_canvas(img1, cvg["canvases"]["canvas11"], cvg["images"], "image11",
                                           canvas_width, canvas_height)
                            img2 = method["fun"](training_images_dataset[i1][0], args,
                                                 canvas_width, canvas_height, True)[1]
                            show_in_canvas(img2, cvg["canvases"]["canvas12"], cvg["images"], "image12",
                                           canvas_width, canvas_height)
                            window.update()
                            time.sleep(time_to_sleep)
        clear_canvases_cv(cvg, up_label_text)

        stats = stats / counters
        columns = images_dir_names + ["total"]
        index = [f"s{i}" for i in range(1, number_of_groups + 1)] + ["total"]
        v_fun = np.vectorize(lambda x: round(x, 3))
        df = pd.DataFrame(v_fun(stats), columns=columns, index=index)
        display_df(df, cv=False)

    def close():
        window.withdraw()
        sys.exit()

    def mm_btn_handler(method):
        cvg["central_area"].tkraise()
        cvg["frm_text"].tkraise()
        cvg["upper_label"].configure(text=f"Method chosen: {method['name']}")
        cvg["frm_buttons"].tkraise()
        cvg["frm_buttons"].focus()

        cvg["btn_cv"].configure(command=lambda: cvg_btn_cv_handler(method))
        cvg["btn_start"].configure(command=lambda: cvg_btn_start_handler(method))

    def cvg_btn_back_handler():
        mm["central_area"].tkraise()
        mm["frm_buttons"].tkraise()
        mm["frm_buttons"].focus()

    def cvg_btn_back_cv_handler(up_label_text=None):
        cvg["frm_text"].tkraise()
        cvg["frm_buttons"].tkraise()
        cvg["frm_buttons"].focus()
        global stop_thread
        stop_thread = True
        cvg["entry_p"].delete(0, 'end')
        cvg["table1"].delete(*cvg["table1"].get_children())
        cvg["table2"].delete(*cvg["table2"].get_children())
        if up_label_text is None:
            up_label_text = cvg["upper_label"].cget("text")
            first_entry = up_label_text.find(':')
            second_entry = up_label_text[first_entry + 1:].find(':')
            if second_entry != -1:
                up_label_text = up_label_text[0:first_entry + second_entry + 1]
        clear_canvases_cv(cvg, up_label_text)

    def cvg_btn_cv_handler(method):
        cvg["frm_images_cv"].tkraise()
        cvg["frm_buttons_cv"].tkraise()
        cvg["frm_buttons_cv"].focus()

        global stop_thread
        stop_thread = False

        up_label_text = cvg["upper_label"].cget("text")
        cvg["btn_back_cv"].config(command=lambda: cvg_btn_back_cv_handler(up_label_text))
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
            img = cv2.imread(file_name)
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
            if not (len(args) == 3 or len(args) == 2 and (method["name"] == "Voting"
                                                          or method["name"] == "Color hist")):
                raise ValueError("Wrong length of args list")
            args = list(map(float, args))
            sample_size = int(args[0])
            if sample_size < 1 or sample_size > len(images[0][0]) - 2:
                raise ValueError("Wrong sample size argument")
            arg1, arg2 = None, None
            match method["name"]:
                case "Color hist":
                    arg1 = int(args[1])
                    acceptable_args = methods[dict_name_idx[method["name"]]]["args"]
                    if arg1 < acceptable_args[0]["min"] or arg1 > acceptable_args[0]["max"]:
                        raise ValueError("Wrong method argument!")
                    chosen_methods = [{"name": method["name"], "fun": method["fun"], "args": (arg1, arg2)}]
                case "Voting":
                    m_indices = np.array([int(x) for x in str(int(args[1]))])
                    if (np.any(m_indices < 1) or np.any(m_indices > len(methods)) or
                            len(set(m_indices)) != len(m_indices)):
                        raise ValueError("Wrong method indices!")
                    m_indices.sort()
                    chosen_methods = [{"name": methods[i - 1]["name"],
                                       "args": get_default_args(methods[i - 1]["name"]),
                                       "fun": methods[i - 1]["fun"]} for i in m_indices]
                case _:
                    arg1, arg2 = int(args[1]), int(args[2])
                    acceptable_args = methods[dict_name_idx[method["name"]]]["args"]
                    if (arg1 < acceptable_args[0]["min"] or arg1 > acceptable_args[0]["max"]
                            or arg2 < acceptable_args[1]["min"] or arg2 > acceptable_args[1]["max"]):
                        raise ValueError("Wrong method argument!")
                    chosen_methods = [{"name": method["name"], "fun": method["fun"], "args": (arg1, arg2)}]
        except ValueError as e:
            print(e)
            return
        cvg["frm_images_p"].tkraise()
        cvg["frm_buttons_cv"].tkraise()
        cvg["frm_buttons_cv"].focus()

        global stop_thread
        stop_thread = False

        up_label_text = cvg["upper_label"].cget("text")
        cvg["upper_label"].configure(
            text=f"Chosen methods: {[method['name'] for method in chosen_methods]}.")
        cvg["btn_back_cv"].config(command=lambda: cvg_btn_back_cv_handler(up_label_text))
        computing_thread = Thread(daemon=True,
                                  target=(p_computing_thread_fun_image if not cvg["images"]["image00"] is None
                                          else p_computing_thread_fun_set),
                                  args=[chosen_methods, sample_size])
        computing_thread.start()

    mm["frm_buttons"].bind('<Escape>', lambda event: close())
    for idx in range(len(method_names)):
        mm["buttons"][idx].configure(command=partial(mm_btn_handler, methods[idx]))
    mm["button_voting"].configure(command=lambda: mm_btn_handler({"name": "Voting"}))
    mm["frm_buttons"].bind('<Return>', lambda event: mm_btn_handler({"name": "Voting"}))

    cvg["btn_back"].configure(command=cvg_btn_back_handler)
    cvg["frm_buttons"].bind('<Escape>', lambda event: cvg_btn_back_handler())

    mm["central_area"].tkraise()
    mm["frm_buttons"].tkraise()
    mm["frm_buttons"].focus()

    # window.bind('<Configure>', resize)

    window.mainloop()


if __name__ == "__main__":
    main()

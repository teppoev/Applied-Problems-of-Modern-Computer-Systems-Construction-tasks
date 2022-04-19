import tkinter as tk
from tkinter.ttk import Treeview


def main_menu(window, method_names):
    central_area = tk.Frame(window)
    frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
    label = tk.Label(central_area, text=(
        "Welcome to FaReS!\n"
        "This program classifies images of people by their faces.\n"
        "There are 6 different methods. Choose method you wish to use by clicking the corresponding button.\n\n"
        "If you don't know what to choose, pick 'Voting'. This 'method' is actually a compilation of the others and "
        "so it provides the best result possible."
    ))
    buttons = [tk.Button(frm_buttons, text=method_names[i]) for i in range(len(method_names))]

    central_area.columnconfigure(0, weight=1)
    central_area.rowconfigure(0, weight=1)
    frm_buttons.rowconfigure(0, weight=1)
    for i in range(len(method_names)):
        frm_buttons.columnconfigure(i, weight=1)

    central_area.grid(row=0, column=0, sticky="nsew")
    frm_buttons.grid(row=1, column=0, sticky="nsew")
    label.grid(row=0, column=0, sticky="nsew")
    for i in range(len(method_names)):
        buttons[i].grid(row=0, column=i, sticky="nsew", padx=5, pady=5)

    return {
        "central_area": central_area,
        "frm_buttons": frm_buttons,
        "label": label,
        "buttons": buttons
    }


def cross_validation_gui(window):
    central_area = tk.Frame(window)
    frm_text = tk.Frame(central_area)
    frm_text_p = tk.Frame(central_area)
    frm_images_p = tk.Frame(central_area)
    frm_images_cv = tk.Frame(central_area)
    frm_pandas = tk.Frame(central_area)
    frm_buttons = tk.Frame(window, relief=tk.RAISED, bd=2)
    frm_buttons_p = tk.Frame(window, relief=tk.RAISED, bd=2)
    frm_buttons_cv = tk.Frame(window, relief=tk.RAISED, bd=2)

    upper_label = tk.Label(central_area, text="")
    central_label = tk.Label(frm_text, text=(
        "Press 'Start algorithm' to manually enter parameters of the method and "
        "choose picture to which algorithm will be applied.\n"
        "Press 'Cross-validation' to find best parameters for the method.\n"
        "Press 'Back' to choose an other method."
    ))
    central_label_p = tk.Label(frm_text_p, text=(
        "Enter the parameter for the method and desirable size of training set, using comma as separator. "
        "You cannot fill this field empty.\n"
        "After that choose a picture to get answer for her. "
        "This action you can ignore. If you do, algorithm will print percentage of right answers."
    ))
    label0 = tk.Label(frm_images_p, text="Input image:")
    label1 = tk.Label(frm_images_p, text="Nearest image:")
    label2 = tk.Label(frm_images_p, text="Correct reference image:")
    canvas00 = tk.Canvas(frm_images_p)
    canvas01 = tk.Canvas(frm_images_p)
    canvas02 = tk.Canvas(frm_images_p)
    canvas10 = tk.Canvas(frm_images_p)
    canvas11 = tk.Canvas(frm_images_p)
    canvas12 = tk.Canvas(frm_images_p)
    left_canvas = tk.Canvas(frm_images_cv)
    right_canvas = tk.Canvas(frm_images_cv)
    table = Treeview(frm_pandas)
    treescrolly = tk.Scrollbar(frm_pandas, orient="vertical", command=table.yview)
    treescrollx = tk.Scrollbar(frm_pandas, orient="horizontal", command=table.xview)
    table.config(xscrollcommand=treescrollx.set, yscrollcommand=treescrolly.set)
    btn_cv = tk.Button(frm_buttons, text="Cross-validation")
    btn_start = tk.Button(frm_buttons, text="Start algorithm")
    btn_back = tk.Button(frm_buttons, text="Back")
    entry_p = tk.Entry(frm_buttons_p)
    btn_load_file_p = tk.Button(frm_buttons_p, text="Load file")
    btn_ok_p = tk.Button(frm_buttons_p, text="Ok")
    btn_back_p = tk.Button(frm_buttons_p, text="Back")
    btn_back_cv = tk.Button(frm_buttons_cv, text="Back")

    central_area.columnconfigure(0, weight=1)
    central_area.rowconfigure(0, weight=1)
    central_area.rowconfigure(1, weight=11)
    frm_text.rowconfigure(0, weight=1)
    frm_text.columnconfigure(0, weight=1)
    frm_text_p.rowconfigure(0, weight=1)
    frm_text_p.columnconfigure(0, weight=1)
    frm_images_p.rowconfigure(0, weight=1)
    frm_images_p.rowconfigure(1, weight=5)
    frm_images_p.rowconfigure(2, weight=5)
    frm_images_p.columnconfigure(0, weight=1)
    frm_images_p.columnconfigure(1, weight=1)
    frm_images_p.columnconfigure(2, weight=1)
    frm_images_cv.rowconfigure(0, weight=1)
    frm_images_cv.columnconfigure(0, weight=1)
    frm_images_cv.columnconfigure(1, weight=1)
    frm_pandas.rowconfigure(0, weight=1)
    frm_pandas.columnconfigure(0, weight=1)
    frm_buttons.rowconfigure(0, weight=1)
    frm_buttons.columnconfigure(0, weight=1)
    frm_buttons.columnconfigure(1, weight=3)
    frm_buttons.columnconfigure(2, weight=1)
    frm_buttons.columnconfigure(3, weight=1)
    frm_buttons_p.rowconfigure(0, weight=1)
    frm_buttons_p.columnconfigure(0, weight=3)
    frm_buttons_p.columnconfigure(1, weight=1)
    frm_buttons_p.columnconfigure(2, weight=1)
    frm_buttons_p.columnconfigure(3, weight=1)
    frm_buttons_cv.rowconfigure(0, weight=1)
    frm_buttons_cv.columnconfigure(0, weight=5)
    frm_buttons_cv.columnconfigure(1, weight=1)

    central_area.grid(row=0, column=0, sticky="nsew")
    upper_label.grid(row=0, column=0, sticky="nsew")
    frm_text.grid(row=1, column=0, sticky="nsew")
    central_label.grid(row=0, column=0, sticky="nsew")
    frm_text_p.grid(row=1, column=0, sticky="nsew")
    central_label_p.grid(row=0, column=0, sticky="nsew")
    frm_images_cv.grid(row=1, column=0, sticky="nsew")
    left_canvas.grid(row=0, column=0, sticky="nsew")
    right_canvas.grid(row=0, column=1, sticky="nsew")
    frm_images_p.grid(row=1, column=0, sticky="nsew")
    label0.grid(row=0, column=0, sticky="ew")
    label1.grid(row=0, column=1, sticky="ew")
    label2.grid(row=0, column=2, sticky="ew")
    canvas00.grid(row=1, column=0, sticky="nsew")
    canvas01.grid(row=1, column=1, sticky="nsew")
    canvas02.grid(row=1, column=2, sticky="nsew")
    canvas10.grid(row=2, column=0, sticky="nsew")
    canvas11.grid(row=2, column=1, sticky="nsew")
    canvas12.grid(row=2, column=2, sticky="nsew")
    frm_pandas.grid(row=1, column=0, sticky="nsew")
    table.grid(row=0, column=0, sticky="nsew")
    treescrollx.grid(row=1, column=0, sticky="we")
    treescrolly.grid(row=0, column=1, sticky="ns")
    frm_buttons.grid(row=1, column=0, sticky="nsew")
    btn_cv.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    btn_start.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
    btn_back.grid(row=0, column=3, sticky="nsew", padx=5, pady=5)
    frm_buttons_p.grid(row=1, column=0, sticky="nsew")
    entry_p.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    btn_load_file_p.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
    btn_ok_p.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
    btn_back_p.grid(row=0, column=3, sticky="nsew", padx=5, pady=5)
    frm_buttons_cv.grid(row=1, column=0, sticky="nsew")
    btn_back_cv.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

    return {
        "central_area": central_area,
        "frm_text": frm_text,
        "frm_text_p": frm_text_p,
        "frm_images_cv": frm_images_cv,
        "frm_images_p": frm_images_p,
        "frm_pandas": frm_pandas,
        "frm_buttons": frm_buttons,
        "frm_buttons_p": frm_buttons_p,
        "frm_buttons_cv": frm_buttons_cv,
        "upper_label": upper_label,
        "central_label": central_label,
        "central_label_p": central_label_p,
        "labels": {
            "label0": label0,
            "label1": label1,
            "label2": label2
        },
        "canvases": {
            "left_canvas": left_canvas,
            "right_canvas": right_canvas,
            "canvas00": canvas00,
            "canvas01": canvas01,
            "canvas02": canvas02,
            "canvas10": canvas10,
            "canvas11": canvas11,
            "canvas12": canvas12
        },
        "images": {
            "left_image": None,
            "right_image": None,
            "image00": None,
            "image01": None,
            "image02": None,
            "image10": None,
            "image11": None,
            "image12": None,
            "image00_cv2data": None
        },
        "table": table,
        "btn_cv": btn_cv,
        "btn_start": btn_start,
        "btn_back": btn_back,
        "entry_p": entry_p,
        "btn_load_file_p": btn_load_file_p,
        "btn_ok_p": btn_ok_p,
        "btn_back_p": btn_back_p,
        "btn_back_cv": btn_back_cv
    }

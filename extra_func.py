import numpy as np
import cv2
from PIL import Image
import os
import yaml


def normalize_image(image, target_min = 0, target_max = 255, output_size=None, border=0):
    """
    将输入图片的强度归一化到指定区间，并可选地裁剪边缘。
    """
    if border is not None:
        image = image[:, border:]

    image_min = np.min(image)
    image_max = np.max(image)

    normalized_image = (image - image_min) / (image_max - image_min)
    normalized_image = normalized_image * (target_max - target_min) + target_min

    if output_size is not None:
        normalized_image = cv2.resize(normalized_image, (output_size[1], output_size[0]), interpolation=cv2.INTER_AREA)

    return normalized_image


def saveNumpyAsImage(array, path: str):
    array_normalized = array.astype(np.uint8)
    image = Image.fromarray(array_normalized)
    image.save(path)


def find_book_info(book_name, data):
    for shelf in data:
        books = shelf.get('content', {})
        for book in books:
            BOOK = book.get('BOOK')
            if BOOK and BOOK == book_name:
                return book
    return None


def get_nk_data(db_path, book_info):
    page_paths = []
    for i, page in enumerate(book_info['content']):
        if "PAGE" in page:
            page_paths.append(page.get("data"))

    n = []
    k = []
    for i in range(len(page_paths)):
        data_path = os.path.join(db_path, "data-nk", page_paths[i])
        data_path = os.path.normpath(data_path)
        if os.path.exists(data_path):
            tmp_wl_n = []
            tmp_wl_k = []
            tmp_n = []
            tmp_k = []
            datafile = yaml.safe_load(open(data_path, "r", encoding="utf-8").read())
            for data in datafile.get("DATA"):
                datatype = data.get("type").split()
                if datatype[0] == "tabulated":
                    rows = data.get("data").split("\n")
                    splitrows = [c.split() for c in rows]
                    for s in splitrows:
                        if len(s) > 0:
                            if datatype[1] == "n":
                                tmp_wl_n.append(float(s[0]))
                                tmp_n.append(float(s[1]))
                            if datatype[1] == "k":
                                tmp_wl_k.append(float(s[0]))
                                tmp_k.append(float(s[1]))
                            if datatype[1] == "nk":
                                tmp_wl_n.append(float(s[0]))
                                tmp_wl_k.append(float(s[0]))
                                tmp_n.append(float(s[1]))
                                tmp_k.append(float(s[2]))
                elif datatype[0] == "formula":
                    wavelength_range = np.array(data.get("wavelength_range").split()).astype(float)
                    if wavelength_range[1] / wavelength_range[0] > 20:
                        wl = np.logspace(np.log10(wavelength_range[0]), np.log10(wavelength_range[1]), 101)
                    else:
                        wl = np.linspace(wavelength_range[0], wavelength_range[1], 101)
                    tmp_wl_n = wl
                    coefficients = np.array(data.get("coefficients").split()).astype(float)
                    num_coeff = coefficients.size
                    C1 = coefficients[0] if num_coeff > 0 else 0
                    C2 = coefficients[1] if num_coeff > 1 else 0
                    C3 = coefficients[2] if num_coeff > 2 else 0
                    C4 = coefficients[3] if num_coeff > 3 else 0
                    C5 = coefficients[4] if num_coeff > 4 else 0
                    C6 = coefficients[5] if num_coeff > 5 else 0
                    C7 = coefficients[6] if num_coeff > 6 else 0
                    C8 = coefficients[7] if num_coeff > 7 else 0
                    C9 = coefficients[8] if num_coeff > 8 else 0
                    C10 = coefficients[9] if num_coeff > 9 else 0
                    C11 = coefficients[10] if num_coeff > 10 else 0
                    C12 = coefficients[11] if num_coeff > 11 else 0
                    C13 = coefficients[12] if num_coeff > 12 else 0
                    C14 = coefficients[13] if num_coeff > 13 else 0
                    C15 = coefficients[14] if num_coeff > 14 else 0
                    C16 = coefficients[15] if num_coeff > 15 else 0
                    C17 = coefficients[16] if num_coeff > 16 else 0
                    if datatype[1] == "1":
                        tmp_n = (1 + C1 + C2 / (1 - (C3 / wl) ** 2) + C4 / (1 - (C5 / wl) ** 2) + C6 / (
                                    1 - (C7 / wl) ** 2)
                                 + C8 / (1 - (C9 / wl) ** 2) + C10 / (1 - (C11 / wl) ** 2) + C12 / (1 - (C13 / wl) ** 2)
                                 + C14 / (1 - (C15 / wl) ** 2) + C16 / (1 - (C17 / wl) ** 2)) ** 0.5
                    elif datatype[1] == "2":
                        tmp_n = (1 + C1 + C2 / (1 - C3 / wl ** 2) + C4 / (1 - C5 / wl ** 2) + C6 / (1 - C7 / wl ** 2)
                                 + C8 / (1 - C9 / wl ** 2) + C10 / (1 - C11 / wl ** 2) + C12 / (1 - C13 / wl ** 2)
                                 + C14 / (1 - C15 / wl ** 2) + C16 / (1 - C17 / wl ** 2)) ** 0.5
                    elif datatype[1] == "3":
                        tmp_n = (C1 + C2 * wl ** C3 + C4 * wl ** C5 + C6 * wl ** C7 + C8 * wl ** C9 + C10 * wl ** C11
                                 + C12 * wl ** C13 + C14 * wl ** C15 + C16 * wl ** C17) ** 0.5
                    elif datatype[1] == "4":
                        tmp_n = (C1 + C2 * wl ** C3 / (wl ** 2 - C4 ** C5) + C6 * wl ** C7 / (wl ** 2 - C8 ** C9)
                                 + C10 * wl ** C11 + C12 * wl ** C13 + C14 * wl ** C15 + C16 * wl ** C17) ** 0.5
                    elif datatype[1] == "5":
                        tmp_n = C1 + C2 * wl ** C3 + C4 * wl ** C5 + C6 * wl ** C7 + C8 * wl ** C9 + C10 * wl ** C11
                    elif datatype[1] == "6":
                        tmp_n = 1 + C1 + C2 / (C3 - wl ** -2) + C4 / (C5 - wl ** -2) + C6 / (C7 - wl ** -2) + C8 / (
                                    C9 - wl ** -2) + C10 / (C11 - wl ** -2)
                    elif datatype[1] == "7":
                        tmp_n = C1 + C2 / (wl ** 2 - 0.028) + C3 / (
                                    wl ** 2 - 0.028) ** 2 + C4 * wl ** 2 + C5 * wl ** 4 + C6 * wl ** 6
                    elif datatype[1] == "8":
                        tmp = C1 + C2 * wl ** 2 / (wl ** 2 - C3) + C4 * wl ** 2
                        tmp_n = ((2 * tmp + 1) / (1 - tmp)) ** 0.5
                    elif datatype[1] == "9":
                        tmp_n = (C1 + C2 / (wl ** 2 - C3) + C4 * (wl - C5) / ((wl - C5) ** 2 + C6)) ** 0.5
        n.append([[k, v] for k, v in zip(tmp_wl_n, tmp_n)])
        k.append([[k, v] for k, v in zip(tmp_wl_k, tmp_k)])
    return n, k


def set_wavelength(n_all, k_all, select_wl):
    n = 1
    k = 0

    for i in range(len(n_all)):
        n_now = n_all[i]
        wl_min = n_now[0][0]
        wl_max = n_now[-1][0]
        if wl_min < select_wl < wl_max:
            wl_now = np.array([item[0] for item in n_now])
            wl_diff = abs(wl_now - select_wl)
            loc = np.argmin(wl_diff)

            x1, y1 = wl_now[loc], n_now[loc][1]
            x2, y2 = wl_now[loc + 1], n_now[loc + 1][1]
            x = select_wl
            n = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
            break

    for i in range(len(k_all)):
        k_now = k_all[i]
        if k_now:
            wl_min = k_now[0][0]
            wl_max = k_now[-1][0]
            if wl_min < select_wl < wl_max:
                wl_now = np.array([item[0] for item in k_now])
                wl_diff = abs(wl_now - select_wl)
                loc = np.argmin(wl_diff)

                x1, y1 = wl_now[loc], k_now[loc][1]
                x2, y2 = wl_now[loc + 1], k_now[loc + 1][1]
                k = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
                break

    return n, k


def get_eps(material: str, wavelength: int):
    # 输入单位为nm，统一单位至微米
    wavelength = wavelength/1000

    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "material")
    lib_path = os.path.join(db_path, "catalog-nk.yml")
    library = yaml.safe_load(open(lib_path, "r", encoding="utf-8").read())

    book_name = material
    book_info = find_book_info(book_name, library)

    n_total, k_total = get_nk_data(db_path, book_info)

    n, k = set_wavelength(n_total, k_total, wavelength)

    return complex(n, k)**2

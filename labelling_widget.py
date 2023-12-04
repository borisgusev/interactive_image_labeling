import csv

import cv2
import numpy as np
import pandas as pd
import tifffile
from IPython.display import clear_output, display
from ipywidgets import Button, GridBox, HBox, Image, IntProgress, Layout, Output


class LabelingWidget:
    def __init__(
        self, file_df: pd.DataFrame, labels: list[str], start_index: int = 0
    ) -> None:
        self._file_df = file_df.copy()
        if "label" not in self._file_df.columns:
            self._file_df["label"] = None

        self.index = start_index
        self._labels = labels

        # self.frame = Output(layout=Layout(max_width="2400px"))
        self._frame = Output()

        self._navigation_buttons = self._init_navigation_buttons()
        self._label_buttons = self._init_label_buttons()
        self._progress = self._init_progress_bar()

    def start(self) -> None:
        # display contents
        display(self._frame)
        display(HBox(self._navigation_buttons))
        display(HBox(self._label_buttons))
        display(self._progress)

        self._refresh_display()

    @property
    def result(self):
        return self._file_df

    @classmethod
    def from_csv(klass, csv_file, labels: list[str], start_index: int = 0):
        file_df = pd.read_csv(csv_file)
        return klass(file_df, labels, start_index)

    def save_to_csv(self, file_path):
        self.result.to_csv(file_path, index=False, quoting=csv.QUOTE_NONNUMERIC)

    def _select_label(self, label_button: Button) -> None:
        self._file_df.at[self.index, "label"] = label_button.description
        self._load_next()

    def _load_next(self, *args) -> None:
        index = self.index
        length = len(self._file_df)
        if index + 1 >= length:
            # print('looped back to first image')
            pass
        self.index = (index + 1) % length

        self._refresh_display()

    def _load_prev(self, *args) -> None:
        index = self.index
        length = len(self._file_df)
        if index - 1 < 0:
            # print('looped back to last image')
            pass
        self.index = (index - 1) % length

        self._refresh_display()

    def _refresh_display(self) -> None:
        images = [Image(value=img) for img in self._load_images()]
        # img_widget = HBox(img_widget)
        img_widget = GridBox(
            images, layout=Layout(grid_template_columns="repeat(3, 300px)")
        )

        with self._frame:
            clear_output(wait=False)
            display(img_widget)

        self._progress.value = self.index + 1
        self._progress.description = f"{self.index + 1}/{len(self._file_df)}"

    def _load_images(self):
        row = self._file_df.iloc[self.index]
        row = row.drop("label")
        imgs = []
        for file in row:
            img = tifffile.imread(file)
            if img.ndim == 2:
                imgs.append(img)
            elif img.ndim == 3:
                # move channel axis to axis=0
                channel_axis = np.argmin(img.shape)
                img = np.moveaxis(img, source=channel_axis, destination=0)
                # add each channel individually
                imgs.extend(img)

        imgs = [_preprocess_image(img) for img in imgs]
        return imgs

    def _init_navigation_buttons(self):
        button_prev = Button(description="< previous")
        button_prev.on_click(self._load_prev)

        button_next = Button(description="next >")
        button_next.on_click(self._load_next)

        return [button_prev, button_next]

    def _init_label_buttons(self):
        class_buttons = []
        for label in self._labels:
            label_button = Button(description=str(label))
            label_button.on_click(self._select_label)
            class_buttons.append(label_button)
        return class_buttons

    def _init_progress_bar(self):
        current = self.index + 1
        length = len(self._file_df)

        progress_bar = IntProgress(
            value=current,
            min=0,
            max=length,
            description=f"{current} / {length}",
        )
        return progress_bar


def _preprocess_image(img):
    img = _normalize_image(img)
    _, img = cv2.imencode(".jpg", img)
    return img


def _normalize_image(image, dest_dtype=np.uint8):
    # return rescale_intensity(image)
    dtype_mapping = {
        np.uint16: cv2.CV_16U,
        np.uint8: cv2.CV_8U,
        np.float32: cv2.CV_32F,
        np.float64: cv2.CV_64F,
    }

    if dest_dtype not in dtype_mapping:
        raise ValueError(
            "dest_dtype must be one of np.uint16, np.uint8, np.float32, np.float64"
        )

    dest_dtype_cv2 = dtype_mapping[dest_dtype]
    max_value = np.iinfo(dest_dtype).max if dest_dtype in (np.uint16, np.uint8) else 1

    return cv2.normalize(image, None, 0, max_value, cv2.NORM_MINMAX, dtype=dest_dtype_cv2)  # type: ignore

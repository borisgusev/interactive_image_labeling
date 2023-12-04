# ImageLabeling
small Jupyter widget to manually label images, for example to label good and bad segmentation

## How to use

1. Get a CSV or pandas.DataFrame with a row per set of images to be displayed (e.g. a column with raw images and a column with masks)
2. Create the widget: `w = LabelingWidget.from_csv('images.csv', labels=['good', 'bad'])` or `w = LabelingWidget(df, labels=['good', 'bad'])`
3. Run the widget: `w.start()`
4. Label accordingly
5. When finished, either save results to csv `w.save_to_csv("new_file.csv")` or read the df directly with `w.result`
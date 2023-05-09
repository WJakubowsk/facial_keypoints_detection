import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

def get_image_and_keypoints_from_observation(series: pd.Series, preprocess: bool = False) -> Tuple[np.array, np.array]:
    """
    Returns tuple of two np.arrays from observation:
        - 96x96 pixels image
        - array of facial keypoints' coordinates
    """
    image = series['Image']
    if preprocess:
        image = np.fromstring(image, sep=' ').reshape([96, 96]) / 255.0
    keypoints = pd.DataFrame(series).drop(['Image'], axis=0).values.reshape([15, 2])
    return image, keypoints

def show_image_with_keypoints(image: np.array, keypoints: np.array):
    """
    Plots the image with marked facial keypoints given preprocessed image and keypoints from get_image_and_keypoints_from_raw_observation function
    """
    plt.imshow(image, cmap='gray')
    plt.plot(keypoints[:, 0], keypoints[:, 1], 'ro')
    plt.axis("off")
    plt.show()

def visualize_duplicates(df: pd.DataFrame, n_pictures: int = 5):
    """
    Visualizes the first n_pictures duplicated images in the provided dataframe
    """
    df_duplicates_list = [grouped for row, grouped in df.groupby("Image") if len(grouped) > 1]
    for df_duplicates in df_duplicates_list[:n_pictures]:
        n_rows = len(df_duplicates)
        fig = plt.figure(figsize=(5, 5*n_rows))
        i = 1
        for index, row in df_duplicates.iterrows():        
            image, keypoints = get_image_and_keypoints_from_observation(row, preprocess = True)
            fig.add_subplot(1, n_rows, i)
            plt.imshow(image, cmap='gray')
            plt.plot(keypoints[:,0], keypoints[:,1], 'ro')
            plt.axis("off")
            i+=1 
        plt.show()


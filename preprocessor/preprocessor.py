import numpy as np
import pandas as pd

class Preprocessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df


    def preprocess_dataset(self):
        print('Handling duplicates...', end='')
        self._handle_duplicates()
        print('done.')
        print('Transforming image column...', end='')
        self._preprocess_image_col()
        print('done.')
    
    def _preprocess_image_col(self):
        """
        Transforms Image column to the format of 96x96 pixel map
        """
        self.df['Image'] = self.df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=" ").reshape((96, 96)) / 255)

    def _handle_duplicates(self):
        """
        Transforms every duplicate group aggregating every duplicated feature coordinate into the mean of all values
        """
        df_duplicates = pd.concat([grouped for row, grouped in self.df.groupby("Image") if len(grouped) > 1])
        df_duplicates_transformed = df_duplicates.groupby("Image").mean().reset_index()
        self.df = pd.concat([self.df.drop(index = df_duplicates.index.to_list()), df_duplicates_transformed])
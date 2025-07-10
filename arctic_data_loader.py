import numpy as np
import pandas as pd
from skimage.transform import resize
import math


class ArcticDataLoader:
    def __init__(self, filespath):
        self.filespath = filespath
        self.names_dict = {
            'kara': ((0, 140), (130, 250)),
            'laptev': ((50, 160), (210, 340)),
            'barents': ((20, 180), (50, 200)),
            'chukchi': ((180, 265), (260, 406)),
            'eastsib': ((100, 200), (260, 385)),
        }

    def load_sea(self, sea_name, period, time_discret=7, resize_shape=None, season=None):
        if season is None:
            season = list(range(1, 13))

        print(f'Loading files for {sea_name} from {period[0]} to {period[1]} with step {time_discret} days.')

        region_slice = self.names_dict[sea_name]

        dates_range = pd.date_range(period[0], period[1], freq='1D')
        years = sorted(list(set([year.year for year in dates_range])))
        dates = []
        for year in years:
            year_dates = dates_range[dates_range.year == year]
            dates += list(pd.date_range(year_dates[0], year_dates[-1], freq=f'{time_discret}D'))

        dates = pd.DatetimeIndex(dates)
        dates = dates[dates.month.isin(season)]

        filenames = [t.strftime('osi_%Y%m%d.npy') for t in dates]

        data = []
        for file in filenames:
            try:
                matrix = np.load(f'{self.filespath}\\{file}')
                cropped = matrix[region_slice[0][0]:region_slice[0][1], region_slice[1][0]:region_slice[1][1]]
                if resize_shape:
                    cropped = resize(cropped, resize_shape, anti_aliasing=True)
                data.append(cropped)
            except FileNotFoundError:
                print(f"Warning: File {file} not found. Skipping.")

        data = np.array(data)
        if len(data) == 0:
            raise ValueError("No data loaded. Check file paths and dates.")

        data_result = []
        full_years = []

        for i in years:
            mask = dates.year == i
            if sum(mask) == math.ceil(365 / time_discret / (12 / len(season))):
                selected_data = data[mask, :, :]
                data_result.append(selected_data)
                full_years.append(i)

        data_result = np.array(data_result)
        dates = dates[dates.year.isin(full_years)]

        return data_result, dates

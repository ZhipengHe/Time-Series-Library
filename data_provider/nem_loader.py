import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from utils.augmentation import run_augmentation_single
import warnings

warnings.filterwarnings('ignore')

class Dataset_NEM(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='QLD1_hourly.csv',
                 target='RRP', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4  # 4 days
            self.label_len = 24 * 4    # 1 day
            self.pred_len = 24 * 4     # 1 day
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
            
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
            
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        # Read the CSV file
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # Convert SETTLEMENTDATE to datetime and set as index
        df_raw['SETTLEMENTDATE'] = pd.to_datetime(df_raw['SETTLEMENTDATE'])
        df_raw.set_index('SETTLEMENTDATE', inplace=True)
        
        # Select only RRP column for univariate forecasting
        if self.features == 'S':
            df_data = df_raw[[self.target]]
        # For multivariate, you can add more features like:
        # - RAISE6SECRRP, LOWER6SECRRP (ancillary services)
        # - PRE_AP_ENERGY_PRICE (pre-dispatch prices)
        elif self.features == 'M' or self.features == 'MS':
            # Add more features for multivariate forecasting
            feature_cols = [
                self.target,  # RRP
                # 'RAISE6SECRRP',  # 6-second raise price
                # 'LOWER6SECRRP',  # 6-second lower price
                # 'RAISE60SECRRP',  # 60-second raise price
                # 'LOWER60SECRRP',  # 60-second lower price
                # 'PRE_AP_ENERGY_PRICE'  # Pre-dispatch price
            ]
            df_data = df_raw[feature_cols]
        
        # Add time-based features
        df_data['hour'] = df_data.index.hour
        df_data['day_of_week'] = df_data.index.dayofweek
        df_data['month'] = df_data.index.month
        df_data['day_of_year'] = df_data.index.dayofyear
        
        # Add lag features (previous day, week)
        df_data['price_lag_24'] = df_data[self.target].shift(24)
        df_data['price_lag_168'] = df_data[self.target].shift(168)  # weekly lag
        
        # Add rolling statistics
        df_data['price_rolling_mean_24'] = df_data[self.target].rolling(window=24).mean()
        df_data['price_rolling_std_24'] = df_data[self.target].rolling(window=24).std()
        
        # Remove rows with NaN after feature creation
        df_data = df_data.dropna()
        
        # Split data into train/val/test
        num_train = int(len(df_data) * 0.7)
        num_test = int(len(df_data) * 0.2)
        num_vali = len(df_data) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_data) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_data)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # Scale the data
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        # Prepare time features
        df_stamp = df_data.index[border1:border2]
        if self.timeenc == 0:
            data_stamp = np.zeros((len(df_stamp), 4))
            data_stamp[:, 0] = df_stamp.month
            data_stamp[:, 1] = df_stamp.day
            data_stamp[:, 2] = df_stamp.weekday
            data_stamp[:, 3] = df_stamp.hour
        elif self.timeenc == 1:
            data_stamp = time_features(df_stamp, freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        
        # Apply augmentation if in training mode
        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)
            
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data) 
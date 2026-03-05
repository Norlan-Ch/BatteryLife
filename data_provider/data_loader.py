import os
import random
import re
import numpy as np
import shutil
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import copy
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from utils.timefeatures import time_features
import warnings
import pickle
from sklearn.cluster import k_means
import torch
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json
from torch.nn.utils.rnn import pad_sequence
from batteryml.data.battery_data import BatteryData
from utils.augmentation import BatchAugmentation_battery_revised
from data_provider.data_split_recorder import split_recorder
import accelerate
from denseweight import DenseWeight
warnings.filterwarnings('ignore')
datasetName2ids = {
    'CALCE':0,
    'HNEI':1,
    'HUST':2,
    'MATR':3,
    'RWTH':4,
    'SNL':5,
    'MICH':6,
    'MICH_EXP':7,
    'Tongji1':8,
    'Stanford':9,
    'ISU-ILCC':11,
    'XJTU':12,
    'ZN-coin':13,
    'UL-PUR':14,
    'Tongji2':15,
    'Tongji3':16,
    'CALB':17,
    'ZN42':22,
    'ZN2024':23,
    'CALB42':24,
    'CALB2024':25,
    'NA-ion':27,
    'NA-ion42':28,
    'NA-ion2024':29,
}
def my_collate_fn_withId(samples):
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples]) # [B, early_cycle, num_var, resample_len]
    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples]) # [B, early_cycle]
    # input_ids = pad_sequence([i['input_ids'] for i in samples], batch_first=True, padding_value=2)
    # attention_mask = pad_sequence([i['attention_mask'] for i in samples], batch_first=True, padding_value=0)
    life_class = torch.Tensor([i['life_class'] for i in samples])
    labels = torch.Tensor([i['labels'] for i in samples])
    scaled_life_class = torch.Tensor([i['scaled_life_class'] for i in samples])
    weights = torch.Tensor([i['weight'] for i in samples])
    dataset_ids = torch.Tensor([i['dataset_id'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])

    tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data) # 生成掩码，[B, early_cycle, num_var, resample_len]
    cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros 应用掩码
    return cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights, dataset_ids, seen_unseen_ids

def my_collate_fn_baseline(samples):
    cycle_curve_data = torch.vstack([i['cycle_curve_data'].unsqueeze(0) for i in samples])
    curve_attn_mask = torch.vstack([i['curve_attn_mask'].unsqueeze(0) for i in samples])
    life_class = torch.Tensor([i['life_class'] for i in samples])
    labels = torch.Tensor([i['labels'] for i in samples])
    scaled_life_class = torch.Tensor([i['scaled_life_class'] for i in samples])
    weights = torch.Tensor([i['weight'] for i in samples])
    seen_unseen_ids = torch.Tensor([i['seen_unseen_id'] for i in samples])

    tmp_curve_attn_mask = curve_attn_mask.unsqueeze(-1).unsqueeze(-1) * torch.ones_like(cycle_curve_data)
    cycle_curve_data[tmp_curve_attn_mask==0] = 0 # set the unseen data as zeros
    return cycle_curve_data, curve_attn_mask,  labels, life_class, scaled_life_class, weights, seen_unseen_ids

class Dataset_original(Dataset):
    def __init__(self, args, flag='train', label_scaler=None, tokenizer=None, eval_cycle_max=None, eval_cycle_min=None, total_prompts=None, 
                 total_charge_discharge_curves=None, total_curve_attn_masks=None, total_labels=None, unique_labels=None,
                 class_labels=None, life_class_scaler=None, use_target_dataset=False):
        '''
        init the Dataset_BatteryFormer class
        :param args:model parameters
        :param flag:including train, val, test
        :param scaler:scaler or not
        '''
        self.ZN_coin_charge_first_file_names = ['ZN-coin_402-1_20231209225636_01_1.pkl', 'ZN-coin_402-2_20231209225727_01_2.pkl', 'ZN-coin_402-3_20231209225844_01_3.pkl', 'ZN-coin_403-1_20231209225922_01_4.pkl', 'ZN-coin_428-1_20231212185048_01_2.pkl', 'ZN-coin_428-2_20231212185058_01_4.pkl', 'ZN-coin_429-1_20231212185129_01_5.pkl', 'ZN-coin_429-2_20231212185157_01_8.pkl', 'ZN-coin_430-1_20231212185250_02_6.pkl', 'ZN-coin_430-2_20231212185305_02_7.pkl', 'ZN-coin_430-3_20231212185323_03_2.pkl']
        self.life_classes = json.load(open('data_provider/life_classes.json'))
        self.eval_cycle_max = eval_cycle_max
        self.eval_cycle_min = eval_cycle_min
        self.args = args
        self.root_path = args.root_path
        self.seq_len = args.seq_len
        self.charge_discharge_len = args.charge_discharge_length  # The resampled length for charge and discharge curves
        self.flag = flag
        self.dataset = args.dataset if not use_target_dataset else args.target_dataset
        self.early_cycle_threshold = args.early_cycle_threshold
        self.KDE_samples = []

        self.need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']
        self.aug_helper = BatchAugmentation_battery_revised()
        assert flag in ['train', 'test', 'val']
        if self.dataset == 'exp':
            self.train_files = split_recorder.Stanford_train_files[:3]
            self.val_files = split_recorder.Tongji_val_files[:2] + split_recorder.HUST_val_files[:2]
            self.test_files =  split_recorder.Tongji_test_files[:2] + split_recorder.HUST_test_files[:2]
        elif self.dataset == 'Tongji':
            self.train_files = split_recorder.Tongji_train_files
            self.val_files = split_recorder.Tongji_val_files
            self.test_files = split_recorder.Tongji_test_files
        elif self.dataset == 'HUST':
            self.train_files = split_recorder.HUST_train_files
            self.val_files = split_recorder.HUST_val_files
            self.test_files = split_recorder.HUST_test_files
        elif self.dataset == 'MATR':
            self.train_files = split_recorder.MATR_train_files
            self.val_files = split_recorder.MATR_val_files
            self.test_files = split_recorder.MATR_test_files
        elif self.dataset == 'SNL':
            self.train_files = split_recorder.SNL_train_files
            self.val_files = split_recorder.SNL_val_files
            self.test_files = split_recorder.SNL_test_files
        elif self.dataset == 'MICH':
            self.train_files = split_recorder.MICH_train_files
            self.val_files = split_recorder.MICH_val_files
            self.test_files = split_recorder.MICH_test_files
        elif self.dataset == 'MICH_EXP':
            self.train_files = split_recorder.MICH_EXP_train_files
            self.val_files = split_recorder.MICH_EXP_val_files
            self.test_files = split_recorder.MICH_EXP_test_files
        elif self.dataset == 'UL_PUR':
            self.train_files = split_recorder.UL_PUR_train_files
            self.val_files = split_recorder.UL_PUR_val_files
            self.test_files = split_recorder.UL_PUR_test_files
        elif self.dataset == 'RWTH':
            self.train_files = split_recorder.RWTH_train_files
            self.val_files = split_recorder.RWTH_val_files
            self.test_files = split_recorder.RWTH_test_files
        elif self.dataset == 'HNEI':
            self.train_files = split_recorder.HNEI_train_files
            self.val_files = split_recorder.HNEI_val_files
            self.test_files = split_recorder.HNEI_test_files
        elif self.dataset == 'CALCE':
            self.train_files = split_recorder.CALCE_train_files
            self.val_files = split_recorder.CALCE_val_files
            self.test_files = split_recorder.CALCE_test_files
        elif self.dataset == 'Stanford':
            self.train_files = split_recorder.Stanford_train_files
            self.val_files = split_recorder.Stanford_val_files
            self.test_files = split_recorder.Stanford_test_files
        elif self.dataset == 'ISU_ILCC':
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == 'XJTU':
            self.train_files = split_recorder.XJTU_train_files
            self.val_files = split_recorder.XJTU_val_files
            self.test_files = split_recorder.XJTU_test_files
        elif self.dataset == 'MIX_large':
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files 
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'ZN-coin':
            self.train_files = split_recorder.ZNcoin_train_files
            self.val_files = split_recorder.ZNcoin_val_files 
            self.test_files = split_recorder.ZNcoin_test_files  
        elif self.dataset == 'CALB':
            self.train_files = split_recorder.CALB_train_files
            self.val_files = split_recorder.CALB_val_files 
            self.test_files = split_recorder.CALB_test_files
        elif self.dataset == 'ZN-coin42':
            self.train_files = split_recorder.ZN_42_train_files
            self.val_files = split_recorder.ZN_42_val_files
            self.test_files = split_recorder.ZN_42_test_files
        elif self.dataset == 'ZN-coin2024':
            self.train_files = split_recorder.ZN_2024_train_files
            self.val_files = split_recorder.ZN_2024_val_files
            self.test_files = split_recorder.ZN_2024_test_files
        elif self.dataset == 'CALB42':
            self.train_files = split_recorder.CALB_42_train_files
            self.val_files = split_recorder.CALB_42_val_files
            self.test_files = split_recorder.CALB_42_test_files
        elif self.dataset == 'CALB2024':
            self.train_files = split_recorder.CALB_2024_train_files
            self.val_files = split_recorder.CALB_2024_val_files
            self.test_files = split_recorder.CALB_2024_test_files
        elif self.dataset == 'NAion':
            self.train_files = split_recorder.NAion_2021_train_files
            self.val_files = split_recorder.NAion_2021_val_files
            self.test_files = split_recorder.NAion_2021_test_files
        elif self.dataset == 'NAion42':
            self.train_files = split_recorder.NAion_42_train_files
            self.val_files = split_recorder.NAion_42_val_files
            self.test_files = split_recorder.NAion_42_test_files
        elif self.dataset == 'NAion2024':
            self.train_files = split_recorder.NAion_2024_train_files
            self.val_files = split_recorder.NAion_2024_val_files
            self.test_files = split_recorder.NAion_2024_test_files
        
        if flag == 'train':
            self.files = [i for i in self.train_files]
        elif flag == 'val':
            self.files = [i for i in self.val_files]
        elif flag == 'test':
            self.files = [i for i in self.test_files]
            if self.dataset == 'ZN-coin42':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_ZN42.json'))
            elif self.dataset == 'ZN-coin2024':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_ZN2024.json'))
            elif self.dataset == 'CALB42':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_CALB42.json'))
            elif self.dataset == 'CALB2024':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_CALB2024.json'))
            elif self.dataset == 'NAion':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA2021.json'))
            elif self.dataset == 'NAion42':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA42.json'))
            elif self.dataset == 'NAion2024':
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test_NA2024.json'))
            else:
                self.unseen_seen_record = json.load(open(f'{self.root_path}/seen_unseen_labels/cal_for_test.json'))
            # self.unseen_seen_record = json.load(open(f'{self.root_path}/cal_for_test.json'))
        
        self.total_charge_discharge_curves, self.total_curve_attn_masks, self.total_labels, self.unique_labels, self.class_labels, self.total_dataset_ids, self.total_cj_aug_charge_discharge_curves, self.total_seen_unseen_IDs = self.read_data()
        
        self.KDE_samples = copy.deepcopy(self.total_labels) if flag == 'train' else []

        self.weights = self.get_loss_weight()
        if np.any(np.isnan(self.total_charge_discharge_curves)):
            raise Exception('Nan in the data')
        if np.any(np.isnan(self.unique_labels)):
            raise Exception('Nan in the labels')
        # K-means to classify the battery life
        
        self.raw_labels = copy.deepcopy(self.total_labels)
        if flag == 'train' and label_scaler is None:
            self.label_scaler = StandardScaler()
            self.life_class_scaler = StandardScaler()
            self.label_scaler.fit(np.array(self.unique_labels).reshape(-1, 1))
            self.life_class_scaler.fit(np.array(self.class_labels).reshape(-1, 1))
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1, 1))
            self.scaled_life_classes = np.array(self.class_labels) - 1
            #self.scaled_life_classes = self.life_class_scaler.transform(np.array(self.class_labels).reshape(-1, 1))
        else:
            # validation set or testing set
            assert label_scaler is not None
            self.label_scaler = label_scaler
            self.life_class_scaler = life_class_scaler
            self.total_labels = self.label_scaler.transform(np.array(self.total_labels).reshape(-1,1))
            self.scaled_life_classes = np.array(self.class_labels) - 1
            #self.scaled_life_classes = self.life_class_scaler.transform(np.array(self.class_labels).reshape(-1,1))

    def get_loss_weight(self, method='KDE'):
        '''
        Get the weight for weighted loss
        method can be ['1/n', '1/log(x+1)', 'KDE']
        '''
        if self.args.weighted_loss and self.flag == 'train':
            if method == '1/n':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                weights = 1.0 / label_to_count[df["label"]].values
            elif method == '1/log(x+1)':
                indices = list(range(self.__len__()))
                df = pd.DataFrame()
                df["label"] = self.total_dataset_ids
                df.index = indices
                df = df.sort_index()

                label_to_count = df["label"].value_counts()

                x = label_to_count[df["label"]].values
                normalized_x = np.log(x / np.min(x)+1)
                weights = 1 / normalized_x
            elif method == 'KDE':
                # Define DenseWeight
                dw = DenseWeight(alpha=1.0)
                # Fit DenseWeight and get the weights for the 1000 samples
                dw.fit(self.KDE_samples)
                # Calculate the weight for an arbitrary target value
                weights = []
                for label in self.KDE_samples:
                    single_sample_weight = dw(label)[0]
                    weights.append(single_sample_weight)
            else:
                raise Exception('Not implemented')
            return weights
        else:
            return np.ones(len(self.total_charge_discharge_curves))

    
    def get_center_vector_index(self, file_name):
        prefix = file_name.split('_')[0]
        if prefix in ['MATR', 'HUST'] or 'LFP' in file_name:
            return 0
        else:
            return 1 
        
    def return_label_scaler(self):
        return self.label_scaler
    
    def return_life_class_scaler(self):
        return self.life_class_scaler
    
    def __len__(self):
        return len(self.total_labels)
        
    def read_data(self):
        '''
        read all data from files
        :return: x_enc, x_cycle_numbers, prompts, charge_data, discharge_data, RPT_masks, labels
        '''
    
        total_charge_discharge_curves = []
        total_curve_attn_masks = []
        total_labels = [] # RUL
        unique_labels = []
        class_labels = [] # the pseudo class for samples
        total_dataset_ids = []
        total_cj_aug_charge_discharge_curves = []
        total_seen_unseen_IDs = []

        for file_name in tqdm(self.files):
            if file_name not in split_recorder.MICH_EXP_test_files and file_name not in split_recorder.MICH_EXP_train_files and file_name not in split_recorder.MICH_EXP_val_files:
                dataset_id = datasetName2ids[file_name.split('_')[0]]
            else:
                dataset_id = datasetName2ids['MICH_EXP']

            charge_discharge_curves, attn_masks, labels, eol, cj_aug_charge_discharge_curves = self.read_samples_from_one_cell(
                file_name)
            if eol is None:
                # This battery has not reached end of life
                continue
            
            for class_label, life_range in self.life_classes.items():
                if eol >= life_range[0] and eol < life_range[1]:
                    class_label = int(class_label)
                    class_labels += [class_label for _ in range(len(charge_discharge_curves))]
                    break
            

            total_charge_discharge_curves += charge_discharge_curves
            total_cj_aug_charge_discharge_curves += cj_aug_charge_discharge_curves
            total_curve_attn_masks += attn_masks
            total_labels += labels 
            total_dataset_ids += [dataset_id for _ in range(len(labels))]
            # total_center_vector_indices += [center_vector_index for _ in range(len(labels))]
            unique_labels.append(eol)
            if self.flag == 'test':
                seen_unseen_id = self.unseen_seen_record[file_name]
                if seen_unseen_id == 'unseen':
                    total_seen_unseen_IDs += [0 for _ in range(len(labels))]
                elif seen_unseen_id == 'seen':
                    total_seen_unseen_IDs += [1 for _ in range(len(labels))]
                else:
                    raise Exception('Check the bug!')
            else:
                total_seen_unseen_IDs += [1 for _ in range(len(labels))] # 1 indicates seen. This is not used on training or evaluation set

        return total_charge_discharge_curves, total_curve_attn_masks, np.array(total_labels), unique_labels, class_labels, total_dataset_ids, total_cj_aug_charge_discharge_curves, total_seen_unseen_IDs

    
    def read_cell_data_according_to_prefix(self, file_name):
        '''
        Read the battery data and eol according to the file_name
        The dataset is indicated by the prefix of the file_name
        '''
        prefix = file_name.split('_')[0]
        if prefix.startswith('MATR'):
            data =  pickle.load(open(f'{self.root_path}/MATR/{file_name}', 'rb'))
        elif prefix.startswith('HUST'):
            data =  pickle.load(open(f'{self.root_path}/HUST/{file_name}', 'rb'))
        elif prefix.startswith('SNL'):
            data =  pickle.load(open(f'{self.root_path}/SNL/{file_name}', 'rb'))
        elif prefix.startswith('CALCE'):
            data =  pickle.load(open(f'{self.root_path}/CALCE/{file_name}', 'rb'))
        elif prefix.startswith('HNEI'):
            data =  pickle.load(open(f'{self.root_path}/HNEI/{file_name}', 'rb'))
        elif prefix.startswith('MICH'):
            if not os.path.isdir(f'{self.root_path}/total_MICH/'):
                self.merge_MICH(f'{self.root_path}/total_MICH/')
            data =  pickle.load(open(f'{self.root_path}/total_MICH/{file_name}', 'rb'))
        elif prefix.startswith('OX'):
            data =  pickle.load(open(f'{self.root_path}/OX/{file_name}', 'rb'))
        elif prefix.startswith('RWTH'):
            data =  pickle.load(open(f'{self.root_path}/RWTH/{file_name}', 'rb'))  
        elif prefix.startswith('UL-PUR'):
            data =  pickle.load(open(f'{self.root_path}/UL_PUR/{file_name}', 'rb'))  
        elif prefix.startswith('SMICH'):
            data =  pickle.load(open(f'{self.root_path}/MICH_EXP/{file_name[1:]}', 'rb')) 
        elif prefix.startswith('BIT2'):
            data =  pickle.load(open(f'{self.root_path}/BIT2/{file_name}', 'rb')) 
        elif prefix.startswith('Tongji'):
            data =  pickle.load(open(f'{self.root_path}/Tongji/{file_name}', 'rb'))
        elif prefix.startswith('Stanford'):
            data =  pickle.load(open(f'{self.root_path}/Stanford/{file_name}', 'rb'))
        elif prefix.startswith('ISU-ILCC'):
            data =  pickle.load(open(f'{self.root_path}/ISU_ILCC/{file_name}', 'rb'))
        elif prefix.startswith('XJTU'):
            data =  pickle.load(open(f'{self.root_path}/XJTU/{file_name}', 'rb'))
        elif prefix.startswith('ZN-coin'):
            data =  pickle.load(open(f'{self.root_path}/ZN-coin/{file_name}', 'rb'))
        elif prefix.startswith('CALB'):
            data =  pickle.load(open(f'{self.root_path}/CALB/{file_name}', 'rb'))
        elif prefix.startswith('NA-ion'):
            data =  pickle.load(open(f'{self.root_path}/NA-ion/{file_name}', 'rb'))
        
        if prefix == 'MICH':
            with open(f'{self.root_path}/Life labels/total_MICH_labels.json') as f:
                life_labels = json.load(f)
        elif prefix.startswith('Tongji'):
            file_name = file_name.replace('--', '-#')
            with open(f'{self.root_path}/Life labels/Tongji_labels.json') as f:
                life_labels = json.load(f)
        else:
            with open(f'{self.root_path}/Life labels/{prefix}_labels.json') as f:
                life_labels = json.load(f)
        if file_name in life_labels:
            eol = life_labels[file_name]
        else:
            eol = None
        return data, eol
    
    def read_cell_df(self, file_name):
        '''
        read the dataframe of one cell, and drop its formation cycles.
        In addition, we will resample its charge and discharge curves
        :param file_name: which file needs to be read
        :return: df, charge_discharge_curves, basic_prompt, eol
        '''
        data, eol = self.read_cell_data_according_to_prefix(file_name)
        if eol is None:
            # This battery has not reached the end of life
            return None, None, None, None, None
        cell_name = file_name.split('.pkl')[0]
        
        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']
            
        cycle_data = data['cycle_data'] # list of cycle data dict
        valid_cycle_number = len(cycle_data)

        total_cycle_dfs = []
        for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
            cycle_df = pd.DataFrame()
            for key in self.need_keys:
                cycle_df[key] = sub_cycle_data[key]
            cycle_df['cycle_number'] = correct_cycle_index + 1
            cycle_df.loc[cycle_df['charge_capacity_in_Ah']<0] = np.nan # deal with outliers in capacity
            cycle_df.loc[cycle_df['discharge_capacity_in_Ah']<0] = np.nan
            cycle_df.bfill(inplace=True) # deal with NaN
            total_cycle_dfs.append(cycle_df)
            
            correct_cycle_number = correct_cycle_index + 1
            if correct_cycle_number > self.early_cycle_threshold or correct_cycle_number > eol:
                break
            
        df = pd.concat(total_cycle_dfs)
        
        # obtain the charge and discahrge curves
        charge_discharge_curves = self.get_charge_discharge_curves(file_name, df, self.early_cycle_threshold, nominal_capacity)
        cj_aug_charge_discharge_curves, fm_aug_charge_discharge_curves  = self.aug_helper.batch_aug(charge_discharge_curves)

        return df, charge_discharge_curves, eol, nominal_capacity, cj_aug_charge_discharge_curves, valid_cycle_number
    
        
    def read_samples_from_one_cell(self, file_name):
        '''
        read all samples using this function
        :param file_name: which file needs to be read
        :return: history_sohs, future_sohs, masks, cycles, prompts, charge_data, discharge_data and RPT_masks in each sample
        '''

        df, charge_discharge_curves_data, eol, nominal_capacity, cj_aug_charge_discharge_curves, valid_cycle_number = self.read_cell_df(file_name)
        if df is None or eol<=self.early_cycle_threshold:
            return None, None, None, None, None

        # the charge and discharge data
        charge_discharge_curves = []  # [N, seq_len, fix_charge_resample_len]
        total_cj_aug_charge_discharge_curves = []
        attn_masks = []
        labels = []
        # get the early-life data
        early_charge_discharge_curves_data = charge_discharge_curves_data[:self.early_cycle_threshold]
        early_cj_aug_charge_discharge_curves = cj_aug_charge_discharge_curves[:self.early_cycle_threshold]
        if np.any(np.isnan(early_charge_discharge_curves_data)):
            raise Exception(f'Failure in {file_name} | Early data contains NaN! Cycle life is {eol}!')
        for i in range(self.seq_len, self.early_cycle_threshold+1):
            if i >= eol:
                # If we encounter a battery whose cycle life is even smaller than early_cycle_threhold
                # We should not include the eol cycle data
                break

            if i > valid_cycle_number:
                # only effective for some CALB batteries that have only cycling data of 99 cycles available for modeling.
                break
            
            tmp_attn_mask = np.zeros(self.early_cycle_threshold)
            tmp_attn_mask[:i] = 1 # set 1 not to mask
            
            if self.eval_cycle_max is not None and self.eval_cycle_min is not None:
                if i <= self.eval_cycle_max and i >= self.eval_cycle_min:
                    # Only keep the val and test samples that satisfy the eval_cycle
                    pass
                else:
                    continue
            

            # tmp_prompt = basic_prompt
            labels.append(eol)
            charge_discharge_curves.append(early_charge_discharge_curves_data)
            total_cj_aug_charge_discharge_curves.append(early_cj_aug_charge_discharge_curves)
            attn_masks.append(tmp_attn_mask)

        return charge_discharge_curves, attn_masks, labels, eol, total_cj_aug_charge_discharge_curves

    def get_charge_discharge_curves(self, file_name, df, early_cycle_threshold, nominal_capacity):
        '''
        Get the resampled charge and discharge curves from the dataframe
        file_name: the file name
        df: the dataframe for a cell
        early_cycle_threshold: obtain the charge and discharge curves from the required early cycles
        '''
        curves = []
        unique_cycles = df['cycle_number'].unique()
        prefix = file_name.split('_')[0]
        if prefix == 'CALB':
            prefix = file_name.split('_')[:2]
            prefix = '_'.join(prefix)

        for cycle in range(1, early_cycle_threshold+1):
            if cycle in df['cycle_number'].unique():
                cycle_df = df.loc[df['cycle_number'] == cycle]
                
                voltage_records = cycle_df['voltage_in_V'].values
                current_records = cycle_df['current_in_A'].values
                current_records_in_C = current_records/nominal_capacity
                charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
                discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values
                time_in_s_records = cycle_df['time_in_s'].values

                cutoff_voltage_indices = np.nonzero(current_records_in_C>=0.01) # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
                charge_end_index = cutoff_voltage_indices[0][-1] # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

                cutoff_voltage_indices = np.nonzero(current_records_in_C<=-0.01) 
                discharge_end_index = cutoff_voltage_indices[0][-1]
                
                # tmp_discharge_capacity_records = max(charge_capacity_records) - discharge_capacity_records
                if prefix in ['RWTH', 'OX', 'ZN-coin', 'CALB_0', 'CALB_25', 'CALB_45'] or (file_name not in self.ZN_coin_charge_first_file_names and prefix=='ZN-coin'):
                    # Every cycle first discharge and then charge
                    #capacity_in_battery = np.where(charge_capacity_records==0, discharge_capacity_records, charge_capacity_records)
                    discharge_voltages = voltage_records[:discharge_end_index]
                    discharge_capacities = discharge_capacity_records[:discharge_end_index]
                    discharge_currents = current_records[:discharge_end_index]
                    discharge_times = time_in_s_records[:discharge_end_index]
                    
                    charge_voltages = voltage_records[discharge_end_index:]
                    charge_capacities = charge_capacity_records[discharge_end_index:]
                    charge_currents = current_records[discharge_end_index:]
                    charge_times = time_in_s_records[discharge_end_index:]
                    charge_current_in_C = charge_currents / nominal_capacity
                    
                    charge_voltages = charge_voltages[np.abs(charge_current_in_C)>0.01]
                    charge_capacities = charge_capacities[np.abs(charge_current_in_C)>0.01]
                    charge_currents = charge_currents[np.abs(charge_current_in_C)>0.01]
                    charge_times = charge_times[np.abs(charge_current_in_C)>0.01]
                else:
                    # Every cycle first charge and then discharge
                    #capacity_in_battery = np.where(np.logical_and(current_records>=-(nominal_capacity*0.01), discharge_capacity_records<=nominal_capacity*0.01), charge_capacity_records, discharge_capacity_records)
                    discharge_voltages = voltage_records[charge_end_index:]
                    discharge_capacities = discharge_capacity_records[charge_end_index:]
                    discharge_currents = current_records[charge_end_index:]
                    discharge_times = time_in_s_records[charge_end_index:]
                    discharge_current_in_C = discharge_currents / nominal_capacity
                    
                    discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C)>0.01]
                    discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C)>0.01]
                    discharge_currents = discharge_currents[np.abs(discharge_current_in_C)>0.01]
                    discharge_times = discharge_times[np.abs(discharge_current_in_C)>0.01]
                    
                    charge_voltages = voltage_records[:charge_end_index]
                    charge_capacities = charge_capacity_records[:charge_end_index]
                    charge_currents = current_records[:charge_end_index]
                    charge_times = time_in_s_records[:charge_end_index]
                
                # try:
                #     discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curvesv2(discharge_voltages, discharge_currents, discharge_capacities)
                #     charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curvesv2(charge_voltages, charge_currents, charge_capacities)
                # except:
                #     print('file_name', file_name, cycle)

                discharge_voltages, discharge_currents, discharge_capacities = self.resample_charge_discharge_curves(discharge_voltages, discharge_currents, discharge_capacities)
                charge_voltages, charge_currents, charge_capacities = self.resample_charge_discharge_curves(charge_voltages, charge_currents, charge_capacities)


                
                voltage_records = np.concatenate([charge_voltages, discharge_voltages], axis=0)
                current_records = np.concatenate([charge_currents, discharge_currents], axis=0)
                capacity_in_battery = np.concatenate([charge_capacities, discharge_capacities], axis=0)
                
                voltage_records = voltage_records.reshape(1, self.charge_discharge_len) / max(voltage_records) # normalize using the cutoff voltage
                current_records = current_records.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the current to C rate
                capacity_in_battery = capacity_in_battery.reshape(1, self.charge_discharge_len) / nominal_capacity # normalize the capacity
                
                curve_data = np.concatenate([voltage_records, current_records, capacity_in_battery], axis=0)
                # curve_data = np.concatenate([voltage_records, current_records], axis=0)
            else:
                # fill zeros when the cell doesn't have enough cycles
                curve_data = np.zeros((3, self.charge_discharge_len))

            curves.append(curve_data.reshape(1, curve_data.shape[0], self.charge_discharge_len))
              
        curves = np.concatenate(curves, axis=0) # [L, 3, fixed_len]
        return curves

    def resample_charge_discharge_curves(self, voltages, currents, capacity_in_battery):
        '''
        resample the charge and discharge curves based on the natural records
        :param voltages:charge or dicharge voltages
        :param currents: charge or discharge current
        :param capacity_in_battery: remaining capacities in the battery
        :return:interploted records
        '''
        charge_discharge_len = self.charge_discharge_len // 2
        raw_bases = np.arange(1, len(voltages)+1)
        interp_bases = np.linspace(1, len(voltages)+1, num=charge_discharge_len,
                                        endpoint=True)
        interp_voltages = np.interp(interp_bases, raw_bases, voltages)
        interp_currents = np.interp(interp_bases, raw_bases, currents)
        interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
        return interp_voltages, interp_currents, interp_capacity_in_battery
    

    def __getitem__(self, index):
        sample = {
                'cycle_curve_data': torch.Tensor(self.total_charge_discharge_curves[index]),
                'curve_attn_mask': torch.Tensor(self.total_curve_attn_masks[index]),
                'labels': self.total_labels[index],
                'life_class': self.class_labels[index],
                'scaled_life_class': self.scaled_life_classes[index],
                'weight': self.weights[index],
                'dataset_id': self.total_dataset_ids[index],
                'cj_cycle_curve_data': self.total_cj_aug_charge_discharge_curves[index],
                'seen_unseen_id': self.total_seen_unseen_IDs[index]
            }
        return sample
    
    def read_train_labels(self, train_files):
        train_labels = []
        for file_name in train_files:
            prefix = file_name.split('_')[0]
            if prefix == 'MICH':
                with open(f'{self.root_path}/total_MICH_labels.json') as f:
                    life_labels = json.load(f)
            elif prefix.startswith('Tongji'):
                with open(f'{self.root_path}/Tongji_labels.json') as f:
                    life_labels = json.load(f)
            else:
                with open(f'{self.root_path}/{prefix}_labels.json') as f:
                    life_labels = json.load(f)
            if file_name in life_labels:
                eol = life_labels[file_name]
            else:
                continue
            train_labels.append(eol)
        return train_labels

    def get_RPT_str(self, RPT_masks, cycle_numbers):
        RPT_masks = np.array(RPT_masks)
        cycle_numbers = np.array(cycle_numbers)
        
        if np.all(RPT_masks==1):
            prompt = 'Described operating condition is used in all cycles.'
        else:
            tmp_RPT_cycles = []
            tmp_normal_cycles = []
            sample_RPT_masks = RPT_masks
            for index, RPT_mask in enumerate(sample_RPT_masks):
                if RPT_mask == 0:
                    tmp_RPT_cycles.append(cycle_numbers[index])
                elif RPT_mask == 1:
                    tmp_normal_cycles.append(cycle_numbers[index])
            prompt = f'Describned operating condition is used in {tmp_normal_cycles} cycles, wheras cycles {tmp_RPT_cycles} are conducted using other operating conditions. '       
        return prompt
    
    def merge_MICH(self, merge_path):
        os.makedirs(merge_path)
        source_path1 = f'{self.root_path}/MICH/'
        source_path2 = f'{self.root_path}/MICH_EXP/'
        source1_files = [i for i in os.listdir(source_path1) if i.endswith('.pkl')]
        source2_files = [i for i in os.listdir(source_path2) if i.endswith('.pkl')]
        target_path = f'{self.root_path}/total_MICH/'

        for file in source1_files:
            shutil.copy(source_path1 + file, target_path)
        for file in source2_files:
            shutil.copy(source_path2 + file, target_path)


class Dataset_AE(Dataset_original):
    """
    Dataset for AutoEncoder in Latent Diffusion Model
    Extracts full discharge capacity sequences across entire battery life cycle
    
    Provides both:
    1. Normalized SOH sequences (discharge_capacity / nominal_capacity)
    2. Original discharge capacity sequences (in Ah)
    
    NOTE: This class inherits from Dataset_original but implements its own
    initialization to avoid loading unnecessary charge/discharge curve data.
    The dataset split assignment code is copied from parent class to maintain
    independence from upstream changes.
    """
    def __init__(self, args, flag='train', soh_len=3000, padding_mode='zero'):
        """
        Initialize Dataset_AE for AutoEncoder training
        
        Args:
            args: model parameters containing dataset configuration
            flag: 'train', 'val', or 'test'
            soh_len: target sequence length for padding (None means no padding)
            padding_mode: 'zero' (pad with zeros) or 'last' (pad with last value)
        """
        self.soh_len = soh_len
        self.padding_mode = padding_mode

        # 添加必要的父类属性
        self.life_classes = json.load(open('data_provider/life_classes.json'))
        
        assert flag in ['train', 'test', 'val']
        assert padding_mode in ['zero', 'last'], "padding_mode must be 'zero' or 'last'"
        
        # Initialize basic attributes needed by parent class methods
        self.args = args
        self.root_path = args.root_path
        self.flag = flag
        self.dataset = args.dataset
        
        # Reuse parent's dataset split assignment logic (copy from parent __init__)
        # ==================== Dataset Split Assignment ====================
        # NOTE: The following code is copied from Dataset_original.__init__()
        # to avoid modifying the parent class (which is forked from upstream).
        # This ensures independence and easier maintenance.
        # 
        # Source: Dataset_original.__init__() lines 82-186
        # Last synced: 2024-11-04
        # ================================================================

        if self.dataset == 'exp':
            self.train_files = split_recorder.Stanford_train_files[:3]
            self.val_files = split_recorder.Tongji_val_files[:2] + split_recorder.HUST_val_files[:2]
            self.test_files = split_recorder.Tongji_test_files[:2] + split_recorder.HUST_test_files[:2]
        elif self.dataset == 'Tongji':
            self.train_files = split_recorder.Tongji_train_files
            self.val_files = split_recorder.Tongji_val_files
            self.test_files = split_recorder.Tongji_test_files
        elif self.dataset == 'HUST':
            self.train_files = split_recorder.HUST_train_files
            self.val_files = split_recorder.HUST_val_files
            self.test_files = split_recorder.HUST_test_files
        elif self.dataset == 'MATR':
            self.train_files = split_recorder.MATR_train_files
            self.val_files = split_recorder.MATR_val_files
            self.test_files = split_recorder.MATR_test_files
        elif self.dataset == 'SNL':
            self.train_files = split_recorder.SNL_train_files
            self.val_files = split_recorder.SNL_val_files
            self.test_files = split_recorder.SNL_test_files
        elif self.dataset == 'MICH':
            self.train_files = split_recorder.MICH_train_files
            self.val_files = split_recorder.MICH_val_files
            self.test_files = split_recorder.MICH_test_files
        elif self.dataset == 'MICH_EXP':
            self.train_files = split_recorder.MICH_EXP_train_files
            self.val_files = split_recorder.MICH_EXP_val_files
            self.test_files = split_recorder.MICH_EXP_test_files
        elif self.dataset == 'UL_PUR':
            self.train_files = split_recorder.UL_PUR_train_files
            self.val_files = split_recorder.UL_PUR_val_files
            self.test_files = split_recorder.UL_PUR_test_files
        elif self.dataset == 'RWTH':
            self.train_files = split_recorder.RWTH_train_files
            self.val_files = split_recorder.RWTH_val_files
            self.test_files = split_recorder.RWTH_test_files
        elif self.dataset == 'HNEI':
            self.train_files = split_recorder.HNEI_train_files
            self.val_files = split_recorder.HNEI_val_files
            self.test_files = split_recorder.HNEI_test_files
        elif self.dataset == 'CALCE':
            self.train_files = split_recorder.CALCE_train_files
            self.val_files = split_recorder.CALCE_val_files
            self.test_files = split_recorder.CALCE_test_files
        elif self.dataset == 'Stanford':
            self.train_files = split_recorder.Stanford_train_files
            self.val_files = split_recorder.Stanford_val_files
            self.test_files = split_recorder.Stanford_test_files
        elif self.dataset == 'ISU_ILCC':
            self.train_files = split_recorder.ISU_ILCC_train_files
            self.val_files = split_recorder.ISU_ILCC_val_files
            self.test_files = split_recorder.ISU_ILCC_test_files
        elif self.dataset == 'XJTU':
            self.train_files = split_recorder.XJTU_train_files
            self.val_files = split_recorder.XJTU_val_files
            self.test_files = split_recorder.XJTU_test_files
        elif self.dataset == 'MIX_large':
            self.train_files = split_recorder.MIX_large_train_files
            self.val_files = split_recorder.MIX_large_val_files
            self.test_files = split_recorder.MIX_large_test_files
        elif self.dataset == 'ZN-coin':
            self.train_files = split_recorder.ZNcoin_train_files
            self.val_files = split_recorder.ZNcoin_val_files
            self.test_files = split_recorder.ZNcoin_test_files
        elif self.dataset == 'CALB':
            self.train_files = split_recorder.CALB_train_files
            self.val_files = split_recorder.CALB_val_files
            self.test_files = split_recorder.CALB_test_files
        elif self.dataset == 'ZN-coin42':
            self.train_files = split_recorder.ZN_42_train_files
            self.val_files = split_recorder.ZN_42_val_files
            self.test_files = split_recorder.ZN_42_test_files
        elif self.dataset == 'ZN-coin2024':
            self.train_files = split_recorder.ZN_2024_train_files
            self.val_files = split_recorder.ZN_2024_val_files
            self.test_files = split_recorder.ZN_2024_test_files
        elif self.dataset == 'CALB42':
            self.train_files = split_recorder.CALB_42_train_files
            self.val_files = split_recorder.CALB_42_val_files
            self.test_files = split_recorder.CALB_42_test_files
        elif self.dataset == 'CALB2024':
            self.train_files = split_recorder.CALB_2024_train_files
            self.val_files = split_recorder.CALB_2024_val_files
            self.test_files = split_recorder.CALB_2024_test_files
        elif self.dataset == 'NAion':
            self.train_files = split_recorder.NAion_2021_train_files
            self.val_files = split_recorder.NAion_2021_val_files
            self.test_files = split_recorder.NAion_2021_test_files
        elif self.dataset == 'NAion42':
            self.train_files = split_recorder.NAion_42_train_files
            self.val_files = split_recorder.NAion_42_val_files
            self.test_files = split_recorder.NAion_42_test_files
        elif self.dataset == 'NAion2024':
            self.train_files = split_recorder.NAion_2024_train_files
            self.val_files = split_recorder.NAion_2024_val_files
            self.test_files = split_recorder.NAion_2024_test_files
        
        # Select files based on flag
        if flag == 'train':
            self.files = [i for i in self.train_files]
        elif flag == 'val':
            self.files = [i for i in self.val_files]
        elif flag == 'test':
            self.files = [i for i in self.test_files]
            
        # ==================== End of Copied Code ====================

        # Read all discharge capacity sequences (AE-specific data loading)
        # Returns both SOH (normalized) and original discharge capacity sequences
        self.total_soh_seqs, self.total_discharge_capacity_seqs, self.eols, self.total_dataset_ids, self.total_padding_masks = self.read_data_AE()
        
        # Validate data
        self._validate_data()
        
        # Print statistics (training only)
        if flag == 'train':
            self._print_statistics()
    
    def extract_discharge_capacity_sequence(self, file_name):
        """
        Extract full discharge capacity sequence from a battery cell
        Reuses parent's read_cell_data_according_to_prefix method
        
        Returns:
            soh_seq: np.array of SOH values (normalized: discharge_capacity / nominal_capacity)
            discharge_capacity_seq: np.array of original discharge capacities (in Ah)
            eol: end of life (cycle count)
        """
        # Reuse parent's method to read data and eol
        data, eol = super().read_cell_data_according_to_prefix(file_name)
        
        if eol is None:
            # Battery has not reached end of life
            return None, None, None
        
        # Get nominal capacity for normalization (same logic as parent class)
        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']
        
        cycle_data = data['cycle_data']
        soh_values = []  # Normalized SOH
        discharge_capacity_values = []  # Original discharge capacity in Ah
        
        # Extract discharge capacity from each cycle
        for cycle_idx, sub_cycle_data in enumerate(cycle_data):
            if cycle_idx >= eol:
                break
            
            # Get discharge capacity for this cycle
            discharge_capacity_in_Ah = sub_cycle_data.get('discharge_capacity_in_Ah', None)
            
            if discharge_capacity_in_Ah is not None and len(discharge_capacity_in_Ah) > 0:
                # Take the maximum discharge capacity in this cycle
                max_discharge_capacity = np.max(discharge_capacity_in_Ah)  # Unit: Ah
                
                # Store original discharge capacity
                discharge_capacity_values.append(max_discharge_capacity)
                
                # Normalize by nominal capacity to get SOH
                soh_values.append(max_discharge_capacity / nominal_capacity)
            else:
                # Handle missing data
                if len(discharge_capacity_values) > 0:
                    # Use last valid values
                    discharge_capacity_values.append(discharge_capacity_values[-1])
                    soh_values.append(soh_values[-1])
                else:
                    # Use default values for new battery
                    discharge_capacity_values.append(nominal_capacity)
                    soh_values.append(1.0) 
        
        soh_seq = np.array(soh_values, dtype=np.float32)
        discharge_capacity_seq = np.array(discharge_capacity_values, dtype=np.float32)
        
        return soh_seq, discharge_capacity_seq, eol
    
    def read_data_AE(self):
        """
        Read all discharge capacity sequences from files
        
        Returns:
            total_soh_seqs: list of SOH sequences (normalized, possibly padded)
            total_discharge_capacity_seqs: list of discharge capacity sequences (in Ah, possibly padded)
            eols: list of end-of-life values
            total_dataset_ids: list of dataset identifiers
            padding_masks: Valid values in total_soh_seqs and total_discharge_capacity_seqs
        """
        total_soh_seqs = []
        total_discharge_capacity_seqs = []
        eols = []
        total_dataset_ids = []
        total_padding_masks = []
        
        for file_name in tqdm(self.files, desc=f'Loading {self.flag} data for AE'):
            soh_seq, discharge_capacity_seq, eol = self.extract_discharge_capacity_sequence(file_name)
            
            # Get dataset ID
            if file_name not in split_recorder.MICH_EXP_test_files and file_name not in split_recorder.MICH_EXP_train_files and file_name not in split_recorder.MICH_EXP_val_files:
                dataset_id = datasetName2ids[file_name.split('_')[0]]
            else:
                dataset_id = datasetName2ids['MICH_EXP']

            if soh_seq is None or discharge_capacity_seq is None or eol is None:
                # Skip batteries that haven't reached EOL
                continue
            
            # Apply Padding/Truncation if soh_len is specified
            if self.soh_len is not None:
                if len(soh_seq) < self.soh_len:
                    # Padding
                    if self.padding_mode == 'zero':
                        # Pad with zeros
                        soh_seq = np.pad(soh_seq, (0, self.soh_len - len(soh_seq)), mode='constant', constant_values=0)
                        discharge_capacity_seq = np.pad(discharge_capacity_seq, (0, self.soh_len - len(discharge_capacity_seq)), mode='constant', constant_values=0)
                    elif self.padding_mode == 'last':
                        # Pad with last value
                        soh_seq = np.pad(soh_seq, (0, self.soh_len - len(soh_seq)), mode='edge')
                        discharge_capacity_seq = np.pad(discharge_capacity_seq, (0, self.soh_len - len(discharge_capacity_seq)), mode='edge')

                elif len(soh_seq) > self.soh_len:
                    # Truncate sequences (keep first soh_len cycles)
                    soh_seq = soh_seq[:self.soh_len]
                    discharge_capacity_seq = discharge_capacity_seq[:self.soh_len]
                    eol = min(eol, self.soh_len)

            # Padding mask
            padding_mask = np.zeros(self.soh_len, dtype=np.bool_)
            padding_mask[:eol] = True
            
            total_soh_seqs.append(soh_seq)
            total_discharge_capacity_seqs.append(discharge_capacity_seq)
            eols.append(eol)
            total_dataset_ids.append(dataset_id)
            total_padding_masks.append(padding_mask)

        return total_soh_seqs, total_discharge_capacity_seqs, eols, total_dataset_ids, total_padding_masks

    def _validate_data(self):
        """Validate data integrity"""
        n = len(self.total_soh_seqs)
        assert len(self.total_discharge_capacity_seqs) == n
        assert len(self.eols) == n
        assert len(self.total_dataset_ids) == n
        assert len(self.total_padding_masks) == n
        
        # Check for NaN in valid regions
        for idx, (soh_seq, eol) in enumerate(zip(self.total_soh_seqs, self.eols)):
            if np.any(np.isnan(soh_seq[:eol])):
                raise ValueError(f"Sample {idx}: NaN in SOH sequence")
            if np.any(np.isnan(self.total_discharge_capacity_seqs[idx][:eol])):
                raise ValueError(f"Sample {idx}: NaN in discharge capacity")
        
        print(f"✓ Data validation passed: {n} samples loaded")
    
    def _print_statistics(self):
        """Print dataset statistics (training only)"""
        eols_array = np.array(self.eols)
        print(f"\n{'='*60}")
        print(f"Dataset_AE Statistics ({self.flag} set, dataset={self.dataset}):")
        print(f"  Total samples: {len(self.total_soh_seqs)}")
        print(f"  EOL range: [{eols_array.min()}, {eols_array.max()}] cycles")
        print(f"  EOL mean ± std: {eols_array.mean():.1f} ± {eols_array.std():.1f}")
        print(f"  EOL median: {np.median(eols_array):.0f}")
        
        if self.soh_len is not None:
            n_padded = np.sum(eols_array < self.soh_len)
            n_truncated = np.sum(eols_array > self.soh_len)
            n_exact = np.sum(eols_array == self.soh_len) 
            total = len(eols_array)
            
            print(f"  Target seq_len: {self.soh_len}")
            print(f"  Padded samples: {n_padded} ({n_padded/total*100:.1f}%)")
            print(f"  Truncated samples: {n_truncated} ({n_truncated/total*100:.1f}%)")
            print(f"  Exact match: {n_exact} ({n_exact/total*100:.1f}%)")  # ✅ 显示精确匹配
            print(f"  Padding mode: '{self.padding_mode}'")
            
            # ✅ 如果有截断，显示截断样本的 EOL 范围
            if n_truncated > 0:
                truncated_eols = eols_array[eols_array > self.soh_len]
                print(f"  Truncated EOL range: [{truncated_eols.min()}, {truncated_eols.max()}]")
        
        print(f"{'='*60}\n")

    def __len__(self):
        return len(self.total_soh_seqs)
    
    def __getitem__(self, index):
        """
        Get a single sample
        
        Returns:
            dict containing:
                - soh_seq: SOH sequence (normalized: discharge_capacity / nominal_capacity)
                - discharge_capacity_seq: original discharge capacity sequence (in Ah)
                - eol: end of life value
                - dataset_id: dataset identifier
        """
        return {
            'soh_seq': torch.tensor(self.total_soh_seqs[index], dtype=torch.float32),
            'discharge_capacity_seq': torch.tensor(self.total_discharge_capacity_seqs[index], dtype=torch.float32),
            'eol': self.eols[index],
            'dataset_id': self.total_dataset_ids[index],
            'padding_mask': torch.tensor(self.total_padding_masks[index], dtype=torch.bool)
        }


# Collate function for Dataset_AE
def collate_fn_AE(samples):
    """
    Collate function for Dataset_AE
    
    Args:
        samples: list of dict from __getitem__, each containing:
            - soh_seq: SOH sequence (normalized)
            - discharge_capacity_seq: discharge capacity sequence (Ah)
            - eol: end of life value
            - padding_mask: valid value mask
    
    Returns:
        batched dict with tensors:
            - soh_seq: (batch_size, seq_len)
            - discharge_capacity_seq: (batch_size, seq_len)
            - eol: (batch_size,)
            - padding_mask: (batch_size, seq_len)
    TODO: 更改dataset，不统一padding，以及在collate_fn_AE中添加
            (1)自动padding到最长序列功能,可选的padding模式（zero/last）
            (2)将padding_mask放到collate_fn中生成
            (3)相应的，transformer tokenizer应该有可学习token，以实现压缩功能
    """
    soh_seqs = torch.stack([s['soh_seq'] for s in samples])
    discharge_capacity_seqs = torch.stack([s['discharge_capacity_seq'] for s in samples])
    eols = torch.tensor([s['eol'] for s in samples], dtype=torch.long)
    padding_masks = torch.stack([s['padding_mask'] for s in samples])

    return {
        'soh_seq': soh_seqs,
        'discharge_capacity_seq': discharge_capacity_seqs,
        'eol': eols,
        'padding_mask': padding_masks
    }

def collate_fn_AE_withID(samples):
    """
    Collate function for Dataset_AE
    
    Args:
        samples: list of dict from __getitem__, each containing:
            - soh_seq: SOH sequence (normalized)
            - discharge_capacity_seq: discharge capacity sequence (Ah)
            - eol: end of life value
            - padding_mask: valid value mask
            - dataset_id: dataset identifier
    
    Returns:
        batched dict with tensors:
            - soh_seq: (batch_size, seq_len)
            - discharge_capacity_seq: (batch_size, seq_len)
            - eol: (batch_size,)
            - padding_mask: (batch_size, seq_len)
            - dataset_id: (batch_size,)
    """
    soh_seqs = torch.stack([s['soh_seq'] for s in samples])
    discharge_capacity_seqs = torch.stack([s['discharge_capacity_seq'] for s in samples])
    eols = torch.tensor([s['eol'] for s in samples], dtype=torch.long)
    padding_masks = torch.stack([s['padding_mask'] for s in samples])
    dataset_ids = torch.tensor([s['dataset_id'] for s in samples], dtype=torch.long)
    
    return {
        'soh_seq': soh_seqs,
        'discharge_capacity_seq': discharge_capacity_seqs,
        'eol': eols,
        'padding_mask': padding_masks,
        'dataset_id': dataset_ids
    }

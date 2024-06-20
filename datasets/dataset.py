import glob
import torch
import os
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import cv2
from PIL import Image
import pandas as pd
import random
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedGroupKFold
import csv
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __len__(self) -> int:
        return len(self.pair_list)

    def __getitem__(self, index):
        img_path, label = self.pair_list[index]
        if self.args.type != 'single_encoder':
            caption = combine_hard_prompt_with_label(self.hard_text_prompt, label)
        image = Image.open(img_path)
        if self.args.encoder_type == 'phikon':
            img_tensor = self.transfrom(image, return_tensors="pt").pixel_values.squeeze()
        else:
            img_tensor = self.transfrom(image)

        if self.args.type == 'single_encoder':
            return img_path, img_tensor, 'no_hard_prompt', label
        else:
            return img_path, img_tensor, self.hard_text_prompt, caption

    def __init__(self, pair_list, args, transform=None, train=True):
        self.args = args
        self.pair_list = pair_list
        self.resize = args.encoder_resize
        self.hard_text_prompt = get_hard_prompt(args.dataset)
        self.mean = args.encoder_mean
        self.std = args.encoder_std
        self.train = train
        self.transfrom = transform

class WholeSlideDataset(Dataset):
    def __len__(self):
        return len(self.pair_list)
    
    def __init__(self, pair_list, args) -> None:
        # self.file_list= os.listdir(patch_imgs_path)
        self.pair_list = pair_list
        self.args = args
        self.hard_text_prompt = get_hard_prompt(args.dataset)

    def __getitem__(self, index):
        intances = torch.load(self.pair_list[index][0])
        if self.args.label_type == 'caption':
            label = self.hard_text_prompt + ' ' + self.pair_list[index][1]
        else:
            label = self.pair_list[index][1]

        return intances, label
        
def prepare_camelyon_wsi(fold=2, type='caption'):
    pt_folder = '/data4/doanhbc/camelyon_patches_20x_bwh_biopsy/features_ctranspath/pt_files'
    split_file = f'/data4/doanhbc/camelyon/fold{fold}.csv'
    split_data = pd.read_csv(split_file)

    train_list = []
    val_list = []
    test_list = []

    def mapping_label(label):
        if type == 'caption':
            return 'normal.' if label == 0.0 else 'tumor.'
        else:
            return int(label)

    for file in os.listdir(pt_folder):
        if file[:-3] in split_data['train'].values:
            idx = split_data.index[split_data['train'] == file[:-3]].tolist()[0]
            label = split_data.at[idx, 'train_label']
            train_list.append((os.path.join(pt_folder, file), mapping_label(label)))
        elif file[:-3] in split_data['val'].values:
            idx = split_data.index[split_data['val'] == file[:-3]].tolist()[0]
            label = split_data.at[idx, 'val_label']
            val_list.append((os.path.join(pt_folder, file), mapping_label(label)))
        elif file[:-3] in split_data['test'].values:
            idx = split_data.index[split_data['test'] == file[:-3]].tolist()[0]
            label = split_data.at[idx, 'test_label']
            test_list.append((os.path.join(pt_folder, file), mapping_label(label)))
    
    return train_list, val_list, test_list
    
def prepare_panda_wsi(type='caption'):
    pt_folder = '/data4/anhnguyen/datasets/PANDA_slide/ctp_feature/pt_files'
    data_csv = '/data4/anhnguyen/datasets/PANDA_slide/train.csv'
    data_csv = pd.read_csv(data_csv)

    data_list = []

    def mapping_label(label):
        if type == 'caption':
            mapping = {
                0: 'grade 0.',
                1: 'grade 1.',
                2: 'grade 2.',
                3: 'grade 3.',
                4: 'grade 4.',
                5: 'grade 5.',
            }
            return mapping[label]
        else:
            return label

    for file in os.listdir(pt_folder):
        path = (os.path.join(pt_folder, file))
        idx = data_csv.index[data_csv['image_id'] == file[:-3]].tolist()[0]
        label = data_csv.at[idx, 'isup_grade']
        label = mapping_label(label)
        data_list.append((path, label))
    
    x_values = [x for x, _ in data_list]
    y_values = [y for _, y in data_list]

    x_train, x_remaining, y_train, y_remaining = train_test_split(x_values, y_values, 
                                                                  train_size=0.8, 
                                                                  stratify=y_values,
                                                                  random_state=2010
                                                                  )
    x_val, x_test, y_val, y_test = train_test_split(x_remaining, y_remaining, 
                                                    train_size=0.5, stratify=y_remaining,
                                                    random_state=2010)

    train_list = []
    val_list = []
    test_list = []

    for i in range(len(x_train)):
        train_list.append((x_train[i], y_train[i]))
    for i in range(len(x_val)):
        val_list.append((x_val[i], y_val[i]))
    for i in range(len(x_test)):
        test_list.append((x_test[i], y_test[i]))

    return train_list, val_list, test_list

def prepare_huncrc_wsi(type='caption'):
    pt_folder = '/data4/anhnguyen/datasets/huncrc_patch_20x_512x512/ctp_feature'
    # split_data = pd.read_csv(split_file)

    data_csv = '/data4/anhnguyen/datasets/huncrc_patch_20x_512x512/slide_level_annotations.csv'
    data_csv = pd.read_csv(data_csv)
    # test_csv = '/data4/anhnguyen/datasets/PANDA_slide/test.csv'

    data_list = []

    def mapping_label(label):
        if type == 'caption':
            mapping = {
                'adenoma': 'adenoma.',
                'CRC': 'colorectal cancer.',
                'non_neoplastic_lesion': 'non-neoplastic lesion.',
                'negative': 'negative.'
            }
        else:
            mapping = {
                'adenoma': 0,
                'CRC': 1,
                'non_neoplastic_lesion': 2,
                'negative': 3
            }
        return mapping[label]

    for file in os.listdir(pt_folder):
        if file[-2:] == 'pt':
            path = (os.path.join(pt_folder, file))
            idx = data_csv.index[data_csv['slideID'] == int(file[:-3][-3:])].tolist()[0]
            label = data_csv.at[idx, 'CATEGORY']
            label = mapping_label(label)
            data_list.append((path, label))
    
    x_values = [x for x, _ in data_list]
    y_values = [y for _, y in data_list]

    x_train, x_remaining, y_train, y_remaining = train_test_split(x_values, y_values, 
                                                                  train_size=0.5, 
                                                                  stratify=y_values,
                                                                  random_state=2010
                                                                  )
    x_val, x_test, y_val, y_test = train_test_split(x_remaining, y_remaining, 
                                                    train_size=0.5, stratify=y_remaining,
                                                    random_state=2010)

    train_list = []
    val_list = []
    test_list = []

    for i in range(len(x_train)):
        train_list.append((x_train[i], y_train[i]))
    for i in range(len(x_val)):
        val_list.append((x_val[i], y_val[i]))
    for i in range(len(x_test)):
        test_list.append((x_test[i], y_test[i]))

    return train_list, val_list, test_list

def prepare_dhmc_wsi(type='caption'):
    pt_folder = '/data4/anhnguyen/datasets/DHMC_wsi_kidney_subtyping/ctp_feature'

    data_csv = '/data4/anhnguyen/datasets/DHMC_wsi_kidney_subtyping/MetaData_Release_1.1.csv'
    data_csv = pd.read_csv(data_csv)
    # test_csv = '/data4/anhnguyen/datasets/PANDA_slide/test.csv'

    train_list = []
    val_list = []
    test_list = []

    def mapping_label(label):
        if type == 'caption':
            mapping = {
                'Benign': 'benign.',
                'Chromophobe': 'chromophobe.',
                'Clearcell': 'clear cell.',
                'Papillary': 'papillary.',
                'Oncocytoma': 'oncocytoma.',
            }
        else:
            mapping = {
                'Benign': 0,
                'Chromophobe': 1,
                'Clearcell': 2,
                'Papillary': 3,
                'Oncocytoma': 4,
            }
        return mapping[label]

    for file in os.listdir(pt_folder):
        if file[-2:] == 'pt':
            path = (os.path.join(pt_folder, file))
            idx = data_csv.index[data_csv['File Name'] == (file[:-3])].tolist()[0]
            label = mapping_label(data_csv.at[idx, 'Diagnosis'])
            split = data_csv.at[idx, 'Data Split']
            if split == 'Train':
                train_list.append((path, label))
            elif split == 'Val':
                val_list.append((path, label))
            elif split == 'Test':
                test_list.append((path, label))

    return train_list, val_list, test_list

def find_files(root_folder, pattern):
		for foldername, subfolders, filenames in os.walk(root_folder):
			for filename in filenames:
				if pattern in filename:
					return os.path.join(foldername, filename)
		return None




def prepare_bracs_wsi(type='caption', num_class=3):
    pt_folder = '/data4/anhnguyen/datasets/BRACS_WSI/ctp_feature/pt_files'
    csv_file_path = "/data4/anhnguyen/datasets/BRACS_WSI/dataset_info.csv"

    train_list = []
    val_list = []
    test_list = []

    def mapping_label(slide_path):
        if num_class == 3:
            label = slide_path.split('/')[-3].split('_')[-1]
            if type == 'caption':
                mapping = {
                    'AT': 'atypical tumor.',
                    'BT': 'benign tumor.',
                    'MT': 'malignant tumor.'
                }
            else:
                mapping = {
                    'AT': 0,
                    'BT': 1,
                    'MT': 2
                }
            print(3)
        else:
            label = slide_path.split('/')[-2].split('_')[-1]
            if type == 'caption':
                mapping = {
                    'ADH': 'atypical ductal hyperplasia',
                    'FEA': 'flat epithelial atypia',
                    'N': 'normal',
                    'PB': 'pathlogical benign',
                    'UDH': 'usual ductal hyperplasia',
                    'DCIS': 'ductal carcinoma in situ',
                    'IC': 'invasive carcinoma',
                }
            else:
                mapping = {
                    'ADH': 0,
                    'FEA': 1,
                    'N': 2,
                    'PB': 3,
                    'UDH': 4,
                    'DCIS': 5,
                    'IC': 6,
                }
            print(7)
        return mapping[label]
    count = {}
    for file in os.listdir(pt_folder):
        if file[-2:] == 'pt':
            path = (os.path.join(pt_folder, file))
            slide_path = find_files('/data4/anhnguyen/datasets/BRACS_WSI/slides', file[:-3])
            label = mapping_label(slide_path)
            if label in count:
                count[label] += 1
            else:
                count[label] = 1
            if 'train' in slide_path:
                train_list.append((path, label))
            elif 'val' in slide_path:
                val_list.append((path, label))
            elif 'test' in slide_path:
                test_list.append((path, label))
        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [file[:-3], file[:-3], label]
            writer.writerow(row)
    return train_list, val_list, test_list

def prepare_huncrc_patch(type='caption'):
    data_csv = '/data4/anhnguyen/datasets/huncrc_patch_10x_512x512/label.csv'
    data_csv = pd.read_csv(data_csv)

    def mapping_label(label):
        if type == 'caption':
            return label.replace('_', ' ') + '.'
        else:
            mapping = {
                'highgrade_dysplasia': 0,
                'adenocarcinoma': 1,
                'suspicious_for_invasion': 2,
                'inflammation': 3,
                'resection_edge': 4,
                'tumor_necrosis': 5,
                'artifact': 6,
                'normal': 7,
                'lowgrade_dysplasia': 8
            }
            return mapping[label]

    X = data_csv['fname'].tolist()
    y = data_csv['label'].tolist()
    y = list(map(mapping_label, y))

    groups = data_csv['group'].tolist()
    
    sgkf = StratifiedGroupKFold(n_splits=4)
    split = sgkf.split(X, y, groups=groups)

    train_folds = []
    test_folds = []
    for train, test in split:
        train_folds.append(train)
        test_folds.append(test)

    train_x = list(map(X.__getitem__, train_folds[0]))
    train_y = list(map(y.__getitem__, train_folds[0]))

    test_x = list(map(X.__getitem__, test_folds[0]))
    test_y = list(map(y.__getitem__, test_folds[0]))
    
    train_list = list(zip(train_x, train_y))
    test_list = list(zip(test_x, test_y))

    return train_list[:] , None, test_list[:]

def prepare_panda_512_data(label_type='caption'):
    def map_label_caption(path):
        mapping_dict = {
            '2': 'benign.',
            '3': 'grade 3 cancer.',
            '4': 'grade 4 cancer.',
            '5': 'grade 5 cancer.',
        }
        label = path.split('_')[-3]

        return mapping_dict[label]


    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        if label_type == 'caption':
            label_list = [map_label_caption(file_path) for file_path in file_list]
        else:
            label_list = [int(file_path.split('_')[-3])-2 for file_path in file_list]
        return list(zip(file_list, label_list))

    # 1000 ~ 6158
    data_root_dir = '/home/compu/anhnguyen/dataset/PANDA/PANDA_512'
    train_set_1 = load_data_info('%s/1*/*.png' % data_root_dir)
    train_set_2 = load_data_info('%s/2*/*.png' % data_root_dir)
    train_set_3 = load_data_info('%s/3*/*.png' % data_root_dir)
    train_set_4 = load_data_info('%s/4*/*.png' % data_root_dir)
    train_set_5 = load_data_info('%s/5*/*.png' % data_root_dir)
    train_set_6 = load_data_info('%s/6*/*.png' % data_root_dir)

    train_set = train_set_1 + train_set_2 + train_set_4 + train_set_6
    valid_set = train_set_3
    test_set = train_set_5

    return train_set, valid_set, test_set

def prepare_colon(label_type='caption'):
    def map_label_caption(path):
        mapping_dict = {
            '0': 'benign.',
            '1': 'well differentiated cancer.',
            '2': 'moderately differentiated cancer.',
            '3': 'poorly differentiated cancer.',
        }
        label = path.split('_')[-1].split('.')[0]
        if label_type == 'caption':
            return mapping_dict[label]
        else:
            return int(path.split('_')[-1].split('.')[0])
    
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [map_label_caption(file_path) for file_path in file_list]

        return list(zip(file_list, label_list))

    data_root_dir = '/home/compu/anhnguyen/dataset/KBSMC_512'
    set_tma01 = load_data_info('%s/tma_01/*.jpg' % data_root_dir)
    set_tma02 = load_data_info('%s/tma_02/*.jpg' % data_root_dir)
    set_tma03 = load_data_info('%s/tma_03/*.jpg' % data_root_dir)
    set_tma04 = load_data_info('%s/tma_04/*.jpg' % data_root_dir)
    set_tma05 = load_data_info('%s/tma_05/*.jpg' % data_root_dir)
    set_tma06 = load_data_info('%s/tma_06/*.jpg' % data_root_dir)
    set_wsi01 = load_data_info('%s/wsi_01/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi02 = load_data_info('%s/wsi_02/*.jpg' % data_root_dir)  # benign exclusively
    set_wsi03 = load_data_info('%s/wsi_03/*.jpg' % data_root_dir)  # benign exclusively

    train_set = set_tma01 + set_tma02 + set_tma03 + set_tma05 + set_wsi01
    valid_set = set_tma06 + set_wsi03
    test_set = set_tma04 + set_wsi02

    return train_set, valid_set, test_set

def prepare_colon_test_2(label_type='caption'):
    def map_label_caption(path):
        mapping_dict = {
            '1': 'benign.',
            '2': 'well differentiated cancer.',
            '3': 'moderately differentiated cancer.',
            '4': 'poorly differentiated cancer.',
        }
        label = path.split('_')[-1].split('.')[0]

        if label_type == 'caption':
            return mapping_dict[label]
        else:
            return int(label)-1

    def load_data_info_from_list(data_dir, path_list):
        file_list = []
        for WSI_name in path_list:
            pathname = glob.glob(f'{data_dir}/{WSI_name}/*/*.png')
            file_list.extend(pathname)
            label_list = [map_label_caption(file_path) for file_path in file_list]
        list_out = list(zip(file_list, label_list))

        return list_out

    data_root_dir = '/home/compu/anhnguyen/dataset/KBSMC_512_test2/KBSMC_test_2'
    wsi_list = ['wsi_001', 'wsi_002', 'wsi_003', 'wsi_004', 'wsi_005', 'wsi_006', 'wsi_007', 'wsi_008', 'wsi_009',
                'wsi_010', 'wsi_011', 'wsi_012', 'wsi_013', 'wsi_014', 'wsi_015', 'wsi_016', 'wsi_017', 'wsi_018',
                'wsi_019', 'wsi_020', 'wsi_021', 'wsi_022', 'wsi_023', 'wsi_024', 'wsi_025', 'wsi_026', 'wsi_027',
                'wsi_028', 'wsi_029', 'wsi_030', 'wsi_031', 'wsi_032', 'wsi_033', 'wsi_034', 'wsi_035', 'wsi_090',
                'wsi_092', 'wsi_093', 'wsi_094', 'wsi_095', 'wsi_096', 'wsi_097', 'wsi_098', 'wsi_099', 'wsi_100']

    test_set = load_data_info_from_list(data_root_dir, wsi_list)

    return test_set

def prepare_prostate_uhu_data(label_type='caption'):
    def map_label_caption(path):
        mapping_dict = {
            '0': 'benign.',
            '1': 'grade 3 cancer.',
            '2': 'grade 4 cancer.',
            '3': 'grade 5 cancer.',
        }
        mapping_dict_2 = {
            0:0,
            1:4,
            2:5,
            3:6
        }
        label = path.split('_')[-1].split('.')[0]
        if label_type == 'caption':
            return mapping_dict[label]
        elif label_type == 'combine_dataset':
            temp = int(path.split('_')[-1].split('.')[0])
            return mapping_dict_2[temp]
        else:
            return int(label)

    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [map_label_caption(file_path) for file_path in file_list]
        return list(zip(file_list, label_list))

    data_root_dir = '/data5/anhnguyen/datasets/prostate_harvard'
    data_root_dir_train = f'{data_root_dir}/patches_train_750_v0'
    data_root_dir_valid = f'{data_root_dir}/patches_validation_750_v0'
    data_root_dir_test = f'{data_root_dir}/patches_test_750_v0'

    train_set_111 = load_data_info('%s/ZT111*/*.jpg' % data_root_dir_train)
    train_set_199 = load_data_info('%s/ZT199*/*.jpg' % data_root_dir_train)
    train_set_204 = load_data_info('%s/ZT204*/*.jpg' % data_root_dir_train)
    valid_set = load_data_info('%s/ZT76*/*.jpg' % data_root_dir_valid)
    test_set = load_data_info('%s/patho_1/*/*.jpg' % data_root_dir_test)

    train_set = train_set_111 + train_set_199 + train_set_204
    return train_set, valid_set, test_set

def prepare_prostate_ubc_data(label_type='caption'):
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        label_dict = {
            0: 'benign.', 
            2: 'grade 3 cancer.', 
            3: 'grade 4 cancer.', 
            4: 'grade 5 cancer.'
        }
        mapping_dict_2 = {
            0:0,
            2:4,
            3:5,
            4:6
        }
        if label_type == 'caption':
            label_list = [label_dict[k] for k in label_list]
        elif label_type == 'combine_dataset':
            for i in range(len(label_list)):
                label_list[i] = mapping_dict_2[label_list[i]]
        else:
            for i in range(len(label_list)):
                if label_list[i] != 0:
                    label_list[i] = label_list[i] - 1

        return list(zip(file_list, label_list))
    
    data_root_dir = '/data5/anhnguyen/datasets'
    data_root_dir_train_ubc = f'{data_root_dir}/prostate_miccai_2019_patches_690_80_step05_test/'
    test_set_ubc = load_data_info('%s/*/*.jpg' % data_root_dir_train_ubc)
    return test_set_ubc

def prepare_gastric(nr_classes=4, label_type='caption'):
    def load_data_info_from_list(path_list, gt_list, data_root_dir, label_type='caption'):
        mapping_dict = {
            0: 'benign.',
            1: 'tubular well differentiated cancer.',
            2: 'tubular moderately differentiated cancer.',
            3: 'tubular poorly differentiated cancer.',
            4: 'other'
        }

        mapping_dict_2 = {
            0:0,
            1:7,
            2:8,
            3:9,
            4:2
        }

        file_list = []
        for tma_name in path_list:
            pathname = glob.glob(f'{data_root_dir}/{tma_name}/*.jpg')
            file_list.extend(pathname)
        
        label_list = [int(file_path.split('_')[-1].split('.')[0]) for file_path in file_list]
        if label_type == 'caption':
            label_list = [mapping_dict[gt_list[i]] for i in label_list]
        elif label_type == 'combine_dataset':
            label_list = [mapping_dict_2[gt_list[i]] for i in label_list]
        else:
            label_list = [gt_list[i] for i in label_list]
        list_out = list(zip(file_list, label_list))
        if label_type == 'caption':
            list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] != 'other']
        elif label_type == 'combine_dataset':
            list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] != 2]
        else:
            list_out = [list_out[i] for i in range(len(list_out)) if list_out[i][1] < 4]

        return list_out

    def load_a_dataset(csv_path, gt_list, data_root_dir, data_root_dir_2, down_sample=True, label_type='caption'):
        df = pd.read_csv(csv_path).iloc[:, :3]
        train_list = list(df.query('Task == "train"')['WSI'])
        valid_list = list(df.query('Task == "val"')['WSI'])
        test_list = list(df.query('Task == "test"')['WSI'])
        train_set = load_data_info_from_list(train_list, gt_list, data_root_dir, label_type)

        if down_sample:
            train_normal = [train_set[i] for i in range(len(train_set)) if train_set[i][1] == 0]
            train_tumor = [train_set[i] for i in range(len(train_set)) if train_set[i][1] != 0]

            random.shuffle(train_normal)
            train_normal = train_normal[: len(train_tumor) // 3]
            train_set = train_normal + train_tumor

        valid_set = load_data_info_from_list(valid_list, gt_list, data_root_dir_2, label_type)
        test_set = load_data_info_from_list(test_list, gt_list, data_root_dir_2, label_type)
        return train_set, valid_set, test_set

    if nr_classes == 3:
        gt_train_local = {1: 4,  # "BN", #0
                          2: 4,  # "BN", #0
                          3: 0,  # "TW", #2
                          4: 1,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 4:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 4,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 4  # "LVI", #ignore
                          }
    elif nr_classes == 5:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 8,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 6:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 2,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 3,  # "papillary", #5
                          8: 4,  # "Mucinous", #6
                          9: 5,  # "signet", #7
                          10: 5,  # "poorly", #7
                          11: 6  # "LVI", #ignore
                          }
    elif nr_classes == 8:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 2,  # "TW", #2
                          4: 3,  # "TM", #3
                          5: 4,  # "TP", #4
                          6: 1,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 7,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }
    elif nr_classes == 10:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 4,  # "TLS", #1
                          7: 5,  # "papillary", #5
                          8: 6,  # "Mucinous", #6
                          9: 7,  # "signet", #7
                          10: 8,  # "poorly", #7
                          11: 9  # "LVI", #ignore
                          }
    else:
        gt_train_local = {1: 0,  # "BN", #0
                          2: 0,  # "BN", #0
                          3: 1,  # "TW", #2
                          4: 2,  # "TM", #3
                          5: 3,  # "TP", #4
                          6: 8,  # "TLS", #1
                          7: 8,  # "papillary", #5
                          8: 5,  # "Mucinous", #6
                          9: 4,  # "signet", #7
                          10: 4,  # "poorly", #7
                          11: 8  # "LVI", #ignore
                          }

    csv_her02 = '/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_cancer_wsi_1024_80_her01_split.csv'
    csv_addition = '/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi_addition_PS1024_ano08_split.csv'

    data_her_root_dir = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step05_bright230_resize05'
    data_her_root_dir_2 = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi/gastric_cancer_wsi_1024_80_her01_step10_bright230_resize05'
    data_add_root_dir = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step05_bright230_resize05'
    data_add_root_dir_2 = f'/home/compu/anhnguyen/dataset/data2/lju/gastric/gastric_wsi_addition/gastric_wsi_addition_PS1024_ano08_step10_bright230_resize05'

    train_set, valid_set, test_set = load_a_dataset(csv_her02, gt_train_local,data_her_root_dir, data_her_root_dir_2, label_type=label_type)
    train_set_add, valid_set_add, test_set_add = load_a_dataset(csv_addition, gt_train_local, data_add_root_dir, data_add_root_dir_2, down_sample=False, label_type=label_type)
    
    train_set += train_set_add
    valid_set += valid_set_add
    test_set += test_set_add

    print(len(train_set))
    print(len(valid_set))
    print(len(test_set))

    return train_set, valid_set, test_set

def prepare_k19(label_type='caption'):
    data_root_dir = '/data5/anhnguyen/datasets/kather_2019/'
    json_dir = '/data5/anhnguyen/datasets/kather_2019/K19_9class_split.json'
    with open(json_dir) as json_file:
        data = json.load(json_file)

    train_set = data['train_set']
    valid_set = data['valid_set']
    test_set = data['test_set']
    train_set = [[data_root_dir + train_set[i][0], train_set[i][1]] for i in range(len(train_set))]
    valid_set = [[data_root_dir + valid_set[i][0], valid_set[i][1]] for i in range(len(valid_set))]
    test_set = [[data_root_dir + test_set[i][0], test_set[i][1]] for i in range(len(test_set))]

    mapping_dict = {
        0: 'adipole.',
        1: 'background.',
        2: 'debris.',
        3: 'lymphocyte.',
        4: 'debris.',   # mucus -> debris (MUC->DEB)
        5: 'stroma.',   # muscle -> stroma (MUS->STR)
        6: 'normal.',
        7: 'stroma.',
        8: 'tumor.'
    }
    # ADI: 0, BACK: 1, DEB: 2, LYMP: 3, STM: 4, NORM: 5, TUM: 6
    mapping_dict_idx = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 2,   # mucus -> debris (MUC->DEB)
        5: 4,   # muscle -> stroma (MUS->STR)
        6: 5,
        7: 4,
        8: 6
    }
    if label_type == 'caption':
        for i in range(len(train_set)):
            train_set[i][1] = mapping_dict[train_set[i][1]]
        
        for i in range(len(valid_set)):
            valid_set[i][1] = mapping_dict[valid_set[i][1]]
        
        for i in range(len(test_set)):
            test_set[i][1] = mapping_dict[test_set[i][1]]
    else:
        for i in range(len(train_set)):
            train_set[i][1] = mapping_dict_idx[train_set[i][1]]
        
        for i in range(len(valid_set)):
            valid_set[i][1] = mapping_dict_idx[valid_set[i][1]]
        
        for i in range(len(test_set)):
            test_set[i][1] = mapping_dict_idx[test_set[i][1]]

    return train_set, valid_set, test_set

def prepare_k16(label_type='caption'):
    def load_data_info(covert_dict):
        data_root_dir_k16 = '/data5/anhnguyen/datasets/kather_2016'
        pathname = f'{data_root_dir_k16}/*/*.tif'
        file_list = glob.glob(pathname)
        COMPLEX_list = glob.glob(f'{data_root_dir_k16}/03_COMPLEX/*.tif')
        file_list = [elem for elem in file_list if elem not in COMPLEX_list]
        label_list = [covert_dict[file_path.split('/')[-2]] for file_path in file_list]
        return list(zip(file_list, label_list))

    const_kather16 = {
        '07_ADIPOSE': 'adipole.', 
        '08_EMPTY': 'background.', 
        '05_DEBRIS': 'debris.',
        '04_LYMPHO': 'lymphocyte.', 
        '06_MUCOSA': 'normal.', 
        '02_STROMA': 'stroma.',
        '01_TUMOR': 'tumor.'
    }

    const_kather16_2 = {
        '07_ADIPOSE': 0, 
        '08_EMPTY': 1, 
        '05_DEBRIS': 2,
        '04_LYMPHO': 3, 
        '06_MUCOSA': 5, 
        '02_STROMA': 4,
        '01_TUMOR': 6
    }

    # ADI: 0, BACK: 1, DEB: 2, LYMP: 3, STM: 4, NORM: 5, TUM: 6

    if label_type == 'caption':
        k16_set = load_data_info(covert_dict=const_kather16)
    else:
        k16_set = load_data_info(covert_dict=const_kather16_2)

    test_set = k16_set

    return test_set

def prepare_aggc2022_data(label_type='caption'):
    mapping_dict = {
        '2': 'benign.',
        '3': 'grade 3 cancer.',
        '4': 'grade 4 cancer.',
        '5': 'grade 5 cancer.',
    }
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        file_list = [file_path for file_path in file_list if int(file_path.split('_')[-1][0]) > 1]
        if label_type != 'caption':
            label_list = [int(file_path.split('_')[-1][0]) - 2 for file_path in file_list if int(file_path.split('_')[-1][0]) > 1]
        else:
            label_list = [mapping_dict[file_path.split('_')[-1][0]] for file_path in file_list if int(file_path.split('_')[-1][0]) > 1]
        return list(zip(file_list, label_list))

    data_root_dir = '/data5/anhnguyen/datasets/AGGC22_patch_512_c08'
    train_set_1 = load_data_info('%s/Subset1_Train_image/*/*' % data_root_dir)
    train_set_2 = load_data_info('%s/Subset2_Train_image/*/*' % data_root_dir)
    train_set_3 = load_data_info('%s/Subset3_Train_image/*/*/*' % data_root_dir)

    return train_set_1 + train_set_2 + train_set_3

def prepare_kidney(label_type='caption'):
    mapping_dict = {
        '0': 'normal.',
        '1': 'grade 1 cancer.',
        '2': 'grade 2 cancer.',
        '3': 'grade 3 cancer.',
        '4': 'grade 4 cancer.',
    }
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        if label_type != 'caption':
            label_list = [int(file_path.split('/')[-2][-1]) for file_path in file_list]
        else:
            label_list = [mapping_dict[file_path.split('/')[-2][-1]] for file_path in file_list]
            # label_list = ['' for file_path in file_list]
            #/home/compu/anhnguyen/prompt_works/TissueSamples_KEK/kidney/G0/Image_2775.jpg a.jpg
        return list(zip(file_list, label_list))
    data_root_dir = '/data4/anhnguyen/kidney_grading'
    train_set = load_data_info('%s/Training/*/*' % data_root_dir)
    valid_set = load_data_info('%s/Validation/*/*' % data_root_dir)
    test_set = load_data_info('%s/Test/*/*' % data_root_dir)
    # test_set = load_data_info('%s/*/*' % data_root_dir)
    return train_set, valid_set, test_set

def prepare_liver(label_type='caption'):
    mapping_dict = {
        '0': 'normal.',
        '1': 'grade 1 cancer.',
        '2': 'grade 2 cancer.',
        '3': 'grade 3 cancer.'
    }
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        if label_type != 'caption':
            label_list = [int(file_path.split('/')[-2][-1]) for file_path in file_list]
        else:
            label_list = [mapping_dict[file_path.split('/')[-2][-1]] for file_path in file_list]
        return list(zip(file_list, label_list))
    data_root_dir = '/data4/anhnguyen/liver_grading'
    train_set = load_data_info('%s/Training/*/*' % data_root_dir)
    valid_set = load_data_info('%s/Validation/*/*' % data_root_dir)
    test_set = load_data_info('%s/Test/*/*' % data_root_dir)
    return train_set, valid_set, test_set

def prepare_bladder(label_type='caption'):
    mapping_dict = {
        '1': 'low grade cancer.',
        '2': 'high grade cancer.',
        '3': 'normal.',
    }
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        if label_type != 'caption':
            label_list = []
            for file_path in file_list:
                idx = int(file_path.split('/')[-2][-1]) - 1
                if idx != 3:
                    label_list.append(int(file_path.split('/')[-2][-1]) - 1)
                else:
                    label_list.append(0)
        else:
            label_list = [mapping_dict[file_path.split('/')[-2][-1]] for file_path in file_list]
        return list(zip(file_list, label_list))
    data_root_dir = '/data2/doanhbc/prosessed_bladder_data_1024_2'
    train_set = load_data_info('%s/train/*/*/*' % data_root_dir)
    valid_set = load_data_info('%s/val/*/*/*' % data_root_dir)
    test_set = load_data_info('%s/test/*/*/*' % data_root_dir)

    return train_set, valid_set, test_set

def prepare_pcam(label_type='caption'):
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = []
        if label_type != 'caption':
            for file_path in file_list:
                if 'normal' in file_path:
                    label_list.append(0)  # normal: 0
                else:
                    label_list.append(1)  # tumor: 1
        else:
            # /data3/anhnguyen/wsss4luad/training/436219-7159-48057-[1, 0, 0].png
            for file_path in file_list:
                if 'normal' in file_path:
                    label_list.append('normal.')
                else:
                    label_list.append('tumor.')
            
        return list(zip(file_list, label_list))
    data_root_dir = '/data3/anhnguyen/pcam/images'
    train_set = load_data_info('%s/train/*' % data_root_dir)
    valid_set = load_data_info('%s/valid/*' % data_root_dir)
    test_set = load_data_info('%s/test/*' % data_root_dir)
    print(len(train_set), len(valid_set), len(test_set))
    return train_set, valid_set, test_set

def prepape_bach(label_type='caption'):
    mapping_dict = {
        '0': 'normal.',
        '1': 'benign.',
        '2': 'in situ carcinoma.',
        '3': 'invasive carcinoma.',
    }
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        if label_type != 'caption':
            label_list = []
            for file_path in file_list:
                idx = file_path.split('/')[-1][-5]
                label_list.append(int(idx))
        else:
            label_list = [mapping_dict[file_path.split('/')[-1][-5]] for file_path in file_list]
        return list(zip(file_list, label_list))
    data_root_dir = '/data3/anhnguyen/BACH_512_v3'

    train_set = []
    for i in [9,1,7,8,10,4]:
        train_set_t = load_data_info(f'%s/A0{i}/*' % data_root_dir)
        train_set += train_set_t
    
    valid_set = []
    for i in [2,6]:
        valid_set_t = load_data_info(f'%s/A0{i}/*' % data_root_dir)
        valid_set += valid_set_t
    
    test_set = []
    for i in [3,5]:
        test_set_t = load_data_info(f'%s/A0{i}/*' % data_root_dir)
        test_set += test_set_t
    print(len(train_set), len(valid_set), len(test_set))
    return train_set, valid_set, test_set

def prepare_medfm(label_type='caption'):
    mapping_dict = {
        '0': 'non-tumor.',
        '1': 'tumor.'
    }
    train_csv = '/data3/anhnguyen/medfm2023/colon_train/colon_train.csv'
    val_csv = '/data3/anhnguyen/medfm2023/colon_valid/colon.csv'

    def load_csv_to_dict(csv_path):
        import csv
        result_dict = {}
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    result_dict[row[-2]] = row[-1]
                line_count += 1
        return result_dict
    
    train_dict = load_csv_to_dict(train_csv)
    valid_dict = load_csv_to_dict(val_csv)
    test_dict = load_csv_to_dict(val_csv)

    
    def load_data_info(pathname, check_dict):
        i = 0
        j = 0
        file_list = glob.glob(pathname)
        label_list = []
        if label_type == 'caption':
            for file_path in file_list:
                file_name = file_path.split('/')[-1]
                try:
                    label_list.append(mapping_dict[check_dict[file_name]])
                    i += 1
                except:
                    # print(file_path)
                    j += 1
        else:
            for file_path in file_list:
                file_name = file_path.split('/')[-1]
                label_list.append(int(check_dict[file_name]))
        print(i,j)
        return list(zip(file_list, label_list))
    
    data_root_dir = '/data3/anhnguyen/medfm2023'
    train_set = load_data_info('%s/colon_train/images/*' % data_root_dir, train_dict)
    valid_set = load_data_info('%s/colon_valid/images/*' % data_root_dir, valid_dict)
    test_set = load_data_info('%s/colon_test/images/*' % data_root_dir, valid_dict)

    
    return train_set, valid_set, test_set

def prepare_unitopath(label_type='caption'):
    import PIL.Image
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    train_list = []
    valid_list = []
    test_list = []
    def mapping(label):
        if label_type == 'caption':
            mapping_dict = {
                'HP': 'hyperplastic.',
                'NORM': 'normal.',
                'TA.HG': 'tubular adenoma, high-grade dysplasia.',
                'TA.LG': 'tubular adenoma, low-grade dysplasia.',
                'TVA.LG': 'tubular-villous adenoma, low-grade dysplasia.',
                'TVA.HG': 'tubular-villous adenoma, high-grade dysplasia.',
            }
        else:
            mapping_dict = {
                'HP': 0,
                'NORM': 1,
                'TA.HG': 2,
                'TA.LG': 3,
                'TVA.LG': 4,
                'TVA.HG': 5,
            }
        return mapping_dict[label]

    def process_data(yaml_file, root_dir, train_list, valid_list, test_list):
        import yaml
        with open(yaml_file, 'r') as yaml_file:
            data = yaml.safe_load(yaml_file)
        file = data['images']

        idx = data['split']['training']
        file_path = list(map(file.__getitem__, idx))
        train_list += list(map(lambda x: (f'{root_dir}/{x['location']}', mapping(x['label'])), file_path))

        idx = data['split']['validation']
        file_path = list(map(file.__getitem__, idx))
        valid_list += list(map(lambda x: (f'{root_dir}/{x['location']}', mapping(x['label'])), file_path))

        idx = data['split']['test']
        file_path = list(map(file.__getitem__, idx))
        test_list += list(map(lambda x: (f'{root_dir}/{x['location']}', mapping(x['label'])), file_path))

    process_data('/data4/anhnguyen/unitopath-public/800_224x224/unitopath-public-800.yml', '/data4/anhnguyen/unitopath-public/800_224x224', train_list, valid_list, test_list)
    process_data('/data4/anhnguyen/unitopath-public/7000_224x224/unitopath-public-7000.yml', '/data4/anhnguyen/unitopath-public/7000_224x224', train_list, valid_list, test_list)

    print(len(train_list), len(valid_list),len(test_list))
    return train_list, valid_list, test_list

def prepare_luad(label_type='caption'):
    def load_data_info(pathname):
        file_list = glob.glob(pathname)
        label_list = []
        if label_type != 'caption':
            for file_path in file_list:
                if 'training' in file_path:
                    if '[1' in file_path:
                        label_list.append(0)
                    elif '[0' in file_path:
                        label_list.append(1)
                else:
                    if 'non-tumor' in file_path:
                        label_list.append(1)
                    else:
                        label_list.append(0)
        else:
            # /data3/anhnguyen/wsss4luad/training/436219-7159-48057-[1, 0, 0].png
            for file_path in file_list:
                if 'training' in file_path:
                    if '[1' in file_path:
                        label_list.append('tumor.')
                    elif '[0' in file_path:
                        label_list.append('normal.')
                else:
                    if 'non-tumor' in file_path:
                        label_list.append('normal.')
                    else:
                        label_list.append('tumor.')
            
        return list(zip(file_list, label_list))
    data_root_dir = '/data3/anhnguyen/wsss4luad'
    train_set = load_data_info('%s/training/*' % data_root_dir)
    valid_set = load_data_info('%s/validation/img/*' % data_root_dir)
    test_set = load_data_info('%s/testing/img/*' % data_root_dir)
    print(len(train_set), len(valid_set), len(test_set))
    return train_set, valid_set, test_set

def prepare_data(args): 
    # if args.label_type == 'index':
    #     dataset_type = 'class_index'
    # else:
    dataset_type = 'caption'
    if args.dataset == 'colon-1':
        return prepare_colon(dataset_type)
    elif args.dataset == 'colon-2':
        return prepare_colon_test_2(dataset_type)
    elif args.dataset == 'luad':
        return prepare_luad(dataset_type)
    elif args.dataset == 'medfm':
        return prepare_medfm(dataset_type)
    elif args.dataset == 'pcam':
        return prepare_pcam(dataset_type)
    elif args.dataset == 'prostate-1':
        return prepare_prostate_uhu_data(dataset_type)
    elif args.dataset == 'bach':
        return prepape_bach(dataset_type)
    elif args.dataset == 'prostate-2':
        return prepare_prostate_ubc_data(dataset_type)
    elif args.dataset == 'prostate-3':
        return prepare_aggc2022_data(dataset_type)
    elif args.dataset == 'gastric':
        return prepare_gastric(nr_classes=4, label_type=dataset_type)
    elif args.dataset == 'k19':
        return prepare_k19(dataset_type)
    elif args.dataset == 'panda':
        return prepare_panda_512_data(dataset_type)
    elif args.dataset == 'k16':
        return prepare_k16(dataset_type)
    elif args.dataset == 'kidney':
        return prepare_kidney(dataset_type)
    elif args.dataset == 'unitopath':
        return prepare_unitopath(dataset_type)
    elif args.dataset == 'liver':
        return prepare_liver(dataset_type)
    elif args.dataset == 'bladder':
        return prepare_bladder(dataset_type)
    elif args.dataset == 'c16':
        return prepare_camelyon_wsi(type = dataset_type)
    elif args.dataset == 'panda_wsi':
        return prepare_panda_wsi(type = dataset_type)
    elif args.dataset == 'huncrc_wsi':
        return prepare_huncrc_wsi(type = dataset_type)
    elif args.dataset == 'dhmc_wsi':
        return prepare_dhmc_wsi(type = dataset_type)
    elif args.dataset == 'bracs_wsi_3':
        return prepare_bracs_wsi(type = dataset_type, num_class=3)
    elif args.dataset == 'bracs_wsi_7':
        return prepare_bracs_wsi(type = dataset_type, num_class=7)
    elif args.dataset == 'huncrc_patch':
        return prepare_huncrc_patch(type = dataset_type)
    else:
        raise ValueError(f'Not support {args.dataset}')

# get the hint aka hard prompt text
def get_hard_prompt(dataset_name):
    if dataset_name in ['colon-1', 'colon-2']:
        return "the cancer grading of this colorectal patch is"
    elif dataset_name in ['kidney']:
        return "the cancer grading of this kidney patch is"
    elif dataset_name in ['medfm']:
        return "this colon patch is tumor or non-tumor?"
    elif dataset_name in ['breakhis']:
        return "the tumor type of this breast patch is"
    elif dataset_name in ['unitopath']:
        return "the polyps type of this colon patch is"
    elif dataset_name in ['pcam']:
        return "the type of this breast patch is"
    elif dataset_name in ['luad']:
        return "the type of this lung patch is"
    elif dataset_name in ['liver']:
        return "the cancer grading of this liver patch is"
    elif dataset_name in ['bach']:
        return "the cancer type of this breast patch is"
    elif dataset_name in ['bladder']:
        return "the tumor type of this bladder patch is"
    elif dataset_name in ['prostate-1', 'prostate-2', 'prostate-3', 'panda']:
        return "the cancer grading of this prostate patch is"
    elif dataset_name in ['gastric']:
        return "the cancer grading of this gastric patch is"
    elif dataset_name in ['k19','k16']:
        return "the tissue type of this colorectal patch is"
    elif dataset_name in ['c16']:
        return "this breast slide is"
    elif dataset_name in ['panda_wsi']:
        return "the cancar grading of this slide is"
    elif dataset_name in ['huncrc_wsi']:
        return "the colorectal cancer screening result of this slide is"
    elif dataset_name in ['dhmc_wsi']:
        return "the subtype of renal cell carcinoma is"
    elif dataset_name in ['bracs_wsi_3', 'bracs_wsi_7']:
        return "the breast lesion type is"
    elif dataset_name in ['huncrc_patch']:
        return "the colorectal cancer screening result of this patch is"
    else:
        raise ValueError(f'Not support dataset {dataset_name}')

# prepend hard prompt to label
def combine_hard_prompt_with_label(hard_prompt_text, label):
    try:
        if label.split(' ')[-1] == 'cancer.':               # eliminate "duplicated" cancer word at the end
            label = " ".join(label.split(' ')[:-1]) + '.'
    except:
        print(label)
    if hard_prompt_text[-1] == ' ':                     # make sure to seperate by a space
        hard_prompt_text += label
    else:
        hard_prompt_text += " " + label
    return hard_prompt_text

def get_caption(dataset_name, type='caption'):
    if dataset_name in ['colon-1', 'colon-2']:
        label = ['benign.',
                 'well differentiated cancer.',
                 'moderately differentiated cancer.',
                 'poorly differentiated cancer.'
        ]
        if type != 'caption':
            label = [0,1,2,3]
    elif dataset_name == 'liver':
        label = ['normal.',
                 'grade 1 cancer.',
                 'grade 2 cancer.',
                 'grade 3 cancer.'
        ]
        if type != 'caption':
            label = [0,1,2,3]
    elif dataset_name == 'kidney':
        label = ['normal.',
                 'grade 1 cancer.',
                 'grade 2 cancer.',
                 'grade 3 cancer.',
                 'grade 4 cancer.',
        ]
        if type != 'caption':
            label = [0,1,2,3,4]
    elif dataset_name == 'bladder':
        label = ['normal.',
                 'low grade cancer.',
                 'high grade cancer.'
        ]
        if type != 'caption':
            label = [2,0,1]
    elif dataset_name in ['prostate-1', 'prostate-2', 'prostate-3', 'panda']:
        label = ['benign.',
                 'grade 3 cancer.',
                 'grade 4 cancer.',
                 'grade 5 cancer.'
        ]
        if type != 'caption':
            label = [0,1,2,3]
    elif dataset_name in ['panda_wsi']:
        label = ['grade 0.',
                 'grade 1.',
                 'grade 2.',
                 'grade 3.',
                 'grade 4.',
                 'grade 5.'
        ]
        if type != 'caption':
            label = [0,1,2,3,4,5]
    elif dataset_name in ['gastric']:
        label = ['benign.',
                 'tubular well differentiated cancer.',
                 'tubular moderately differentiated cancer.',
                 'tubular poorly differentiated cancer.'
        ]
        if type != 'caption':
            label = [0,1,2,3]
    elif dataset_name in ['bach']:
        label = ['normal.',
                 'benign.',
                 'in situ carcinoma.',
                 'invasive carcinoma.'
        ]
        if type != 'caption':
            label = [0,1,2,3]
    else:
        raise ValueError(f'Not support dataset {dataset_name}')
    result = []
    if type != 'caption':
        return label
    for l in label:
        hard_prompt = get_hard_prompt(dataset_name)
        result.append(combine_hard_prompt_with_label(hard_prompt, l))
    return result

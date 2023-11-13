import glob
import torch
import os
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import random
import json
from torchvision.transforms import Normalize

class ImageDataset(Dataset):
    def __len__(self) -> int:
        return len(self.pair_list)

    def train_augmentors(self):
        sometimes = lambda aug: iaa.Sometimes(0.2, aug)
        input_augs = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5),  # horizontally flip 50% of all images
                iaa.Flipud(0.5),  # vertically flip 50% of all images
                sometimes(iaa.Affine(
                    rotate=(-45, 45),  # rotate by -45 to +45 degrees
                    shear=(-16, 16),  # shear by -16 to +16 degrees
                    order=[0, 1],  # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255),  # if mode is constant, use a cval between 0 and 255
                    mode='symmetric'
                    # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                           [
                               iaa.OneOf([
                                   iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                                   iaa.AverageBlur(k=(2, 7)),
                                   # blur image using local means with kernel sizes between 2 and 7
                                   iaa.MedianBlur(k=(3, 11)),
                                   # blur image using local medians with kernel sizes between 2 and 7
                               ]),
                               iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                               # add gaussian noise to images
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                               # change brightness of images (by -10 to 10 of original value)
                               iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                               iaa.LinearContrast((0.5, 2.0), per_channel=0.5),  # improve or worsen the contrast
                           ],
                           random_order=True
                           )
            ],
            random_order=True
        )
        return input_augs

    def __getitem__(self, index):
        img_path, label = self.pair_list[index]
        if self.args.type != 'single_encoder':
            caption = combine_hard_prompt_with_label(self.hard_text_prompt, label)
        image = cv2.imread(img_path)
        image = cv2.resize(image, (self.resize,self.resize))
        if self.train == True:
            train_augmentors = self.train_augmentors()
            image = train_augmentors.augment_image(image)
        img_tensor = torch.tensor(image.copy(), dtype=torch.float32).permute(2,0,1) # C,H,W
        if self.args.encoder_type == 'ctranspath':
            img_tensor = Normalize(mean=self.mean, std=self.std)(img_tensor)

        if self.args.type == 'single_encoder':
            return img_path, img_tensor, 'no_hard_prompt', label
        else:
            return img_path, img_tensor, self.hard_text_prompt, caption

    def __init__(self, pair_list, args, train=True):
        self.args = args
        self.pair_list = pair_list
        self.resize = args.encoder_resize
        self.hard_text_prompt = get_hard_prompt(args.dataset)
        self.mean = args.encoder_mean
        self.std = args.encoder_std
        self.train = train

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

    data_root_dir = '/home/compu/doanhbc/datasets/prostate_harvard'
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
    
    data_root_dir = '/home/compu/doanhbc/datasets'
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

    return train_set, valid_set, test_set

def prepare_k19(label_type='caption'):
    data_root_dir = '/data1/trinh/data/raw_data/Domain_Invariance/colon_class/NCT-CRC-HE-100K/'
    json_dir = '/data1/trinh/code/DoIn/pycontrast/datasets/K19_9class_split.json'
    with open(json_dir) as json_file:
        data = json.load(json_file)

    train_set = data['train_set']
    valid_set = data['valid_set']
    test_set = data['test_set']
    train_set = [[data_root_dir + train_set[i][0], train_set[i][1]] for i in range(len(train_set))]
    valid_set = [[data_root_dir + valid_set[i][0], valid_set[i][1]] for i in range(len(valid_set))]
    test_set = [[data_root_dir + test_set[i][0], test_set[i][1]] for i in range(len(test_set))]

    # mapping_dict = {
    #     0: 'tissue adipole.',
    #     1: 'tissue background.',
    #     2: 'tissue debris.',
    #     3: 'tissue lymphocyte.',
    #     4: 'tissue mucus.',
    #     5: 'tissue muscle.',
    #     6: 'tissue normal.',
    #     7: 'tissue stroma.',
    #     8: 'tissue tumor.'
    # }
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
        data_root_dir_k16 = '/data1/trinh/data/raw_data/Domain_Invariance/colon_class/Kather_texture_2016_image_tiles_5000'
        pathname = f'{data_root_dir_k16}/*/*.tif'
        file_list = glob.glob(pathname)
        COMPLEX_list = glob.glob(f'{data_root_dir_k16}/03_COMPLEX/*.tif')
        file_list = [elem for elem in file_list if elem not in COMPLEX_list]
        label_list = [covert_dict[file_path.split('/')[-2]] for file_path in file_list]
        return list(zip(file_list, label_list))

    # const_kather16 = {
    #         '07_ADIPOSE': 'tissue adipole.', 
    #         '08_EMPTY': 'tissue background.', 
    #         '05_DEBRIS': 'tissue debris.',
    #         '04_LYMPHO': 'tissue lymphocyte.', 
    #         '06_MUCOSA': 'tissue normal.', 
    #         '02_STROMA': 'tissue stroma.',
    #         '01_TUMOR': 'tissue tumor.'
    #     }
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

    data_root_dir = '/home/compu/doanhbc/datasets/AGGC22_patch_512_c08'
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
        return list(zip(file_list, label_list))
    data_root_dir = '/data4/anhnguyen/kidney_grading'
    train_set = load_data_info('%s/Training/*/*' % data_root_dir)
    valid_set = load_data_info('%s/Validation/*/*' % data_root_dir)
    test_set = load_data_info('%s/Test/*/*' % data_root_dir)

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
    # train_set = load_data_info('%s/colon_train/images/*' % data_root_dir, train_dict)
    # valid_set = load_data_info('%s/colon_valid/images/*' % data_root_dir, valid_dict)
    test_set = load_data_info('%s/colon_test/images/*' % data_root_dir, valid_dict)

    
    return train_set, valid_set, test_set

def prepare_unitopath(label_type='caption'):
    def load_csv_to_list(csv_path):
        import csv
        result_list = []
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count != 0:
                    result_list.append(row[0])
                line_count += 1
        return result_list
    
    def load_data_info(pathname): # /data4/anhnguyen/unitopath-public/800/NORM/57-B2-NORM.ndpi_ROI__mpp0.44_reg000_crop_sk00000_(67531,14365,1812,1812).png
        file_list = glob.glob(pathname)
        if label_type != 'caption':
            label_list = [mapping_dict_2[file_path.split('/')[5]] for file_path in file_list]
        else:
            label_list = [mapping_dict[file_path.split('/')[5]] for file_path in file_list]
        return list(zip(file_list, label_list))
    
    mapping_dict = {
        'HP': 'normal.',
        'NORM': 'hyperplastic polyp.',
        'TA.HG': 'tubular adenoma, high-grade dysplasia.',
        'TA.LG': 'tubular adenoma, low-grade dysplasia.',
        'TVA.LG': 'tubular-villous adenoma, low-grade dysplasia.',
        'TVA.HG': 'tubular-villous adenoma, high-grade dysplasia.',
    }

    mapping_dict_2 = {
        'HP': 0,
        'NORM': 1,
        'TA.HG': 2,
        'TA.LG': 3,
        'TVA.LG': 4,
        'TVA.HG': 5,
    }

    train_csv = '/data4/anhnguyen/unitopath-public/800/train.csv'
    test_csv = '/data4/anhnguyen/unitopath-public/800/test.csv'
    train_val_list = load_csv_to_list(train_csv)
    test_list = load_csv_to_list(test_csv)
    
    # /data4/anhnguyen/unitopath-public/800/NORM/57-B2-NORM.ndpi_ROI__mpp0.44_reg000_crop_sk00000_(67531,14365,1812,1812).png
    data_root_dir = '/data4/anhnguyen/unitopath-public/800_resize_224/'
    data = load_data_info('%s/*/*' % data_root_dir)

    # valid_filter = {
    #     'TVA.LG': ['100-B2-TVALG', '109-B3-TVALG', '191-B4-TVALG', 'TVA.LG CASO 10'],
    #     'TVA.HG': ['108-B3-TVAHG', '181-B4-TVAHG', '249-B5-TVAHG', 'TVA.HG CASO 2 - 2018-12-04 12.53.37'],
    #     'TA.HG': ['221-B5-TAHG','223-B5-TAHG','226-B5-TAHG','TA.HG CASO 16'],
    #     'TA.LG': ['101-B2-TALG', '105-B3-TALG', '125-B3-TALG', '236-B5-TALG', 'TA.LG CASO 11 - 2018-12-04 13.46.00', 
    #               'TA.LG CASO 64 - 2019-03-04 17.42.27', 'TA.LG CASO 88 B1', 'TA.LG CASO 91', 'TA.LG CASO 92 B1'],
    #     'NORM': ['131-B3-NORM', '185-B4-NORM', '208-B5-NORM'],
    #     'HP': ['103-B3-HP', '149-B3-HP', '158-B4-HP', '224-B5-HP', 'HP CASO 24 - 2019-03-04 08.59.32']
    # }
    valid_filter = ['100-B2-TVALG', '109-B3-TVALG', '191-B4-TVALG', 'TVA.LG CASO 10', '108-B3-TVAHG', 
                    '181-B4-TVAHG', '249-B5-TVAHG', 'TVA.HG CASO 2 - 2018-12-04 12.53.37',
                    '221-B5-TAHG','223-B5-TAHG','226-B5-TAHG','TA.HG CASO 16',
                    '101-B2-TALG', '105-B3-TALG', '125-B3-TALG', '236-B5-TALG', 'TA.LG CASO 11 - 2018-12-04 13.46.00', 
                  'TA.LG CASO 64 - 2019-03-04 17.42.27', 'TA.LG CASO 88 B1', 'TA.LG CASO 91', 'TA.LG CASO 92 B1',
                  '131-B3-NORM', '185-B4-NORM', '208-B5-NORM',
                  '103-B3-HP', '149-B3-HP', '158-B4-HP', '224-B5-HP', 'HP CASO 24 - 2019-03-04 08.59.32']

    train_list = []
    valid_list = []
    test_list = []
    for sample in data:
        file_path = sample[0].split('/')[-1]
        if file_path in train_val_list:
            if file_path.split('/')[-1].split('.')[0] in valid_filter:
                valid_list.append(sample)
            else:
                train_list.append(sample)
        else:
            test_list.append(sample)

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

    return train_set, valid_set, test_set

prepare_luad()

def prepare_data(args):
    if args.type != 'single_encoder':
        dataset_type = 'caption'
    else:
        dataset_type = 'class_index'
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
    elif args.dataset == 'breakhis':
        return prepare_breakhis(dataset_type, args.breakhis_fold)
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
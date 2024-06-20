from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, cohen_kappa_score
from datasets.dataset import get_caption

def calculate_metrics(dataset, ground_truth_list, prediction_list):
    metrics = {}
    if dataset in ['colon-1', 'colon-2', 'prostate-1', 'prostate-2', 'prostate-3',
                   'gastric','kidney','liver','bladder','bach','panda']:
        cancer_ground_truth_list = []
        cancer_prediction_list = []
        for i in range(len(ground_truth_list)):
            try:
                if ground_truth_list[i].split(' ')[-1] not in ['normal.', 'benign.', 
                                                               'non-tumor.', 'non - tumor.']:
                    cancer_ground_truth_list.append(ground_truth_list[i])
                    cancer_prediction_list.append(prediction_list[i])
            except:
                if int(ground_truth_list[i]) != 0 and dataset != 'bladder':
                    cancer_ground_truth_list.append(ground_truth_list[i])
                    cancer_prediction_list.append(prediction_list[i])
                elif int(ground_truth_list[i]) != 2 and dataset == 'bladder':
                    cancer_ground_truth_list.append(ground_truth_list[i])
                    cancer_prediction_list.append(prediction_list[i])
        metrics['valid_acc'] = accuracy_score(ground_truth_list, prediction_list)
        metrics['valid_cancer_acc'] = accuracy_score(cancer_ground_truth_list, cancer_prediction_list)
        if isinstance(ground_truth_list[i], str):
            label_type = 'caption'
        else:
            label_type = 'not_caption'
        metrics['valid_f1'] = f1_score(ground_truth_list, prediction_list, average='macro', labels=get_caption(dataset, label_type))
        metrics['valid_kappa'] = cohen_kappa_score(ground_truth_list, prediction_list, labels=get_caption(dataset, label_type), weights='quadratic')
        try:
            metrics['valid_avg'] = (metrics['valid_acc']+metrics['valid_cancer_acc']+metrics['valid_f1']+metrics['valid_kappa'])/4
        except:
            metrics['valid_avg'] = 0

    elif dataset in ['k19', 'k16', 'breakhis','pcam', 'medfm', 'unitopath', 'luad', 'c16', 'panda_wsi', \
                     'huncrc_wsi', 'dhmc_wsi','bracs_wsi_7', 'bracs_wsi_3', 'huncrc_patch']:
        metrics['valid_acc'] = accuracy_score(ground_truth_list, prediction_list)
        metrics['valid_f1'] = f1_score(ground_truth_list, prediction_list, average='macro', labels=list(set(ground_truth_list)))
        metrics['valid_pre'] = precision_score(ground_truth_list, prediction_list, labels=list(set(ground_truth_list)), average='macro')
        metrics['valid_rec'] = recall_score(ground_truth_list, prediction_list, labels=list(set(ground_truth_list)), average='macro')
        try:
            metrics['valid_avg'] = (metrics['valid_acc']+metrics['valid_pre']+metrics['valid_f1']+metrics['valid_rec'])/4
        except:
            metrics['valid_avg'] = 0
    else:
        raise ValueError(f'Not support dataset {dataset}')
    return metrics

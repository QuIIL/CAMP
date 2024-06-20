import argparse
from argparse import Namespace
import os 
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.model_with_prompt import PromptModel
from model.single_encoder import SingleEncoder
from datasets import ImageDataset, prepare_data
from utils import generate, calculate_metrics, save_config_and_metric, get_num_class


def test(args, test_dataset, model):
    #print(args)
    batch_size = args.bs
    device = args.device
    model = model.to(device)
    
    model.eval()
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=40)

    # TESTING LOOP 
    ground_truth_list = []
    prediction_list = []

    with torch.no_grad():
        print(f">>> Testing")
        progress = tqdm(total=len(test_dataloader))
        for idx, (img_path, img_tensor, hard_text_prompt, label) in enumerate(test_dataloader):
            img_tensor = img_tensor.to(device, dtype=torch.float32)  # bs x 3 x 512 x 512               
            if args.type != 'single_encoder':                    
                gen_cap = generate(model, img_tensor, hard_text_prompt, args)
                ground_truth_list += label
                prediction_list += gen_cap
            else:
                outputs = model(img_tensor)
                ground_truth_list += list(label)
                prediction_list += torch.argmax(outputs, dim=1).tolist()
            progress.update()
        progress.close()
    
        assert len(ground_truth_list) == len(prediction_list)
        metrics = calculate_metrics(args.dataset, ground_truth_list, prediction_list)
        
        print(args.model_pth)
        print(metrics)
        save_config_and_metric(args, metrics, best_epoch=None, run_type='test')
    return model


def main():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', choices=['colon-1','colon-2',
                                              'prostate-1','prostate-2','prostate-3',
                                              'gastric', 'k19', 'kidney', 'liver', 
                                              'bach', 'breakhis', 'pcam', 'panda',
                                              'k16', 'unitopath', 'bladder', 'luad', 'unitopatho']
                        default='prostate-1')

    # Testing configuaration
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--generate_length', type=int, default=12)
    parser.add_argument('--model_pth', type=str, default=None)
    
    # Saving configuration
    parser.add_argument('--out_dir', default='/data4/anhnguyen/experiments/prompt_work/testing/')

    overwrite_args = parser.parse_args()

    last_ext = '-' + overwrite_args.model_pth.split('-')[-1]
    training_config_file = overwrite_args.model_pth.replace(last_ext, '.json')
    with open(training_config_file) as file:
        args = json.load(file)
        args['model_pth'] = None
        args = Namespace(**args)
    
    file.close()

    args.dataset = overwrite_args.dataset
    # args.device = overwrite_args.device
    args.bs = overwrite_args.bs
    args.generate_length = overwrite_args.generate_length
    args.model_pth = overwrite_args.model_pth
    args.out_dir = overwrite_args.out_dir
    args.device = torch.device(f'cuda:{args.device}')
    if 'type' not in args:
        args.type = args.prompt_type
    
    data = prepare_data(args)
    if isinstance(data,tuple):
        test_set = data[2]
    else:
        test_set = data
    
    if args.type == 'single_encoder':
        model = SingleEncoder(args, get_num_class(args.dataset))
        td = torch.load(args.model_pth, map_location=args.device)
        model.load_state_dict(td['model_state_dict'], strict=True)
    else:
        model = PromptModel(args)
        td = torch.load(args.model_pth, map_location=args.device)
        model.load_state_dict(td['model_state_dict'], strict=True)

    print(args.model_pth)
    test_dataset = ImageDataset(test_set, args, train=False)
    test(args, test_dataset, model)

if __name__ == '__main__':
    main()

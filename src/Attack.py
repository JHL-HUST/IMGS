import numpy as np
import json
import os
import torch
import torchvision
from torchvision import models
from data import preprocess
from data import loader
from utils import common
from model import *
from tqdm import tqdm
from data import mstar
from Attack import MI_FGSM
from Attack import SI_MI_FGSM
from Attack import TI_MI_FGSM
from Attack import Admix_MI_FGSM
from Attack import DI_MI_FGSM
from Attack import IMGS_MI_FGSM
import pickle
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_dataset(path, is_train, name, batch_size):
    transform = [preprocess.CenterCrop(88),torchvision.transforms.ToTensor()]
    if is_train:
        transform = [preprocess.RandomCrop(88), torchvision.transforms.ToTensor()]
    _dataset = loader.Dataset(
        path, name=name, is_train=is_train,
        transform=torchvision.transforms.Compose(transform)
    )
    data_loader = torch.utils.data.DataLoader(
        _dataset, batch_size=batch_size, shuffle=is_train, num_workers = 0
    )
    return data_loader

def Attack(_m, pre_model_name, ds, epsilon, attak_method):    
    num_data = 0
    corrects = 0

    # Address to save the adversarial examples
    output_root = './IMGS/dataset/soc'

    _m.net.eval()
    _softmax = torch.nn.Softmax(dim=1)
    finally_data = []
    add_Perturbation = []
    name = pre_model_name.split('.')[0]
    for i, data in enumerate(tqdm(ds)):
        images, labels = data
        images = images.clone().detach().to(_m.device)
        labels = torch.tensor([label.tolist() for label in labels]).to(_m.device)
        labels = torch.transpose(labels, 0, 1)
        labels = labels.squeeze(1)

        # Chiose attack method
        attack_methods = {
            'MI_FGSM': MI_FGSM.mi_fgsm_attack,
            'DI_MI_FGSM': DI_MI_FGSM.di_mi_fgsm_attack,
            'TI_MI_FGSM': TI_MI_FGSM.ti_mi_fgsm_attack,
            'SI_MI_FGSM': SI_MI_FGSM.si_mi_fgsm_attack,
            'Admix_MI_FGSM': Admix_MI_FGSM.admix_mi_attack,
            'IMGS_MI_FGSM': IMGS_MI_FGSM.imgs_mi_attack,
        }

        try:
            attack_function = attack_methods[attak_method]
            perturbed_data = attack_function(_m, images, labels, epsilon)
        except KeyError:
            raise ValueError('Invalid attack method: {}'.format(attak_method))
        
        predictions = _m.inference(perturbed_data)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)

        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)

        corrects += (predictions == labels.to(_m.device)).sum().item()

        # Save the adversarial examples
        Per_img = perturbed_data.cpu().detach().numpy()

        for j in range(Per_img.shape[0]):
            finally_data.append((Per_img[j],[labels[j]]))
            target_name = mstar.target_name_soc[labels[j]]
            out_path = os.path.join(output_root, f'per_img/', name, attak_method, target_name)
            if not os.path.exists(out_path):
                os.makedirs(out_path, exist_ok=True)
            img = Per_img[j].transpose(1,2,0)
            np.save(os.path.join(out_path, f'{target_name}-{attak_method}-{epsilon}-{i}-{j}.npy'), img)   
    
    # name = pre_model_name.split('.')[0]
    out_path_pickle = os.path.join(output_root, name)
    if not os.path.exists(out_path_pickle):
        os.makedirs(out_path_pickle, exist_ok=True)

    pkl_path = os.path.join(out_path_pickle, f'{attak_method}-{epsilon}.pkl')
    pickle.dump(finally_data, open(pkl_path, 'wb'))
    accuracy = 100 * corrects / num_data
    return accuracy 

if __name__ == "__main__":

    batch_size = 100
    input_data = './IMGS/dataset/soc/eval_data_1.pkl'
    test_set = pickle.load(open(input_data, 'rb'))
    dataloder = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, num_workers=0)

    pre_model_path = './IMGS/experiments/net'
    pre_model_name = 'MstarSOC_SOC_ResNet_18.pth'
    pre_model = torch.load(os.path.join(pre_model_path, pre_model_name))

    for eps in [0.01]:
        print("eps=",eps)
        acc = Attack(pre_model, pre_model_name, dataloder, eps, 'TI_MI_FGSM')
        print("Accuracy: {:.2f}%".format(acc))
        print("Attack success rate: {:.2f}%".format(100 - acc))
import numpy as np
import torch
from utils import common
from model import *
from tqdm import tqdm
import numpy as np
import torch
import pickle
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def Valuation(net, path):
    num_data = 0
    corrects = 0
    net.net.eval()
    _softmax = torch.nn.Softmax(dim=1)

    count = 0
    data_finally = pickle.load(open(path, 'rb'))
    for i, data in enumerate(tqdm(data_finally)):
        images, labels = data
        #print(images)
        images = torch.tensor([images])
        labels = torch.tensor(labels)
        predictions = net.inference(images)
        predictions = _softmax(predictions)

        _, predictions = torch.max(predictions.data, 1)

        labels = labels.type(torch.LongTensor)
        num_data += labels.size(0)

        corrects += (predictions == labels.to(net.device)).sum().item()
    acc = 100 * corrects / num_data
    return acc

def main():

    # Read the data
    input_path = './IMGS/dataset/soc/MstarSOC_SOC_ResNet_18/SI_MI_FGSM-0.01.pkl'

    # Define a dictionary of models
    models = {
        'ResNet_18': ('./IMGS/experiments/net/MstarSOC_SOC_ResNet_18.pth', 0.01),
        'ResNet_50': ('./IMGS/experiments/net/MstarSOC_SOC_ResNet_50.pth', 0.01),
        'InceptionV3': ('./IMGS/experiments/net/MstarSOC_SOC_InceptionV3.pth', 0.01),
        'InceptionResNetV2': ('./IMGS/experiments/net/MstarSOC_SOC_InceptionResNetV2.pth', 0.01)
    }

    # Loop through the models and print the results
    for name, (path, epsilon) in models.items():
        # Load the model
        net = torch.load(path)
        # Evaluate the model
        acc = Valuation(net, input_path)
        # Print the accuracy and attack success rate
        print(f"epsilon={epsilon}")
        print(f"{name} Accuracy: {acc:.2f}%")
        print(f"{name} Attack success rate: {100 - acc:.2f}%")


if __name__ == "__main__":
    main()
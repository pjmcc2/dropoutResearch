from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import torch
from torch import nn
from scripts import Testing
from BaseDropConvModel import BaseDropConvModel
from BaseNoDropConvModel import BaseNoDropConvModel
from TsetlinConvModel import TsetConvModel
from CurrDropConvModel import CurrDropConvModel
from HybridConvModel import HybridConvModel
from tqdm import tqdm
import pandas as pd
import time

if __name__ == "__main__":

    # Change to CIFAR10 if using cifar10
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_data = datasets.CIFAR100(
        root=r"C:\Users\pjmcc\PycharmProjects\dropoutResearch\data\CIFAR100",
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.CIFAR100(
        root=r"C:\Users\pjmcc\PycharmProjects\dropoutResearch\data\CIFAR100",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64, num_workers=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, num_workers=1, shuffle=True)

    loss_fn = nn.CrossEntropyLoss()
    num_iters = 2
    epochs = 2
    lr = 0.0001
    momentum = 0.95
    num_batches = len(train_dataloader)
    grads = num_batches * epochs

    data_dict = {"Epoch": [],
                 "Mean Train Loss": [],
                 "Mean Test Loss": [],
                 "Test Acc(%)": [],
                 "Time Per Epoch(s)": []
                 }

    for j in tqdm(range(num_iters)):
        # Create new instance to reset parameters (and probabilities)
        curr_model = BaseDropConvModel(100)
        curr_model.to(device)
        optimizer = torch.optim.SGD(curr_model.parameters(), lr=lr, momentum=momentum)
        for i in tqdm(range(epochs)):
            # print(f"epoch {i} \n")
            data_dict["Epoch"].append(i + 1)
            starttime = time.time()
            avg_t_loss, probs_l = Testing.train_loop(train_dataloader, curr_model, optimizer, loss_fn, device, i,
                                                     mode=None, record=True,
                                                     return_probs=False)
            avg_v_loss, v_acc = Testing.test_loop(test_dataloader, curr_model, loss_fn, device, i, mode=None,
                                                  record=True)
            end_time = time.time()
            time_per_train_test = end_time - starttime
            data_dict["Mean Train Loss"].append(avg_t_loss)
            data_dict["Mean Test Loss"].append(avg_v_loss)
            data_dict["Test Acc(%)"].append(v_acc)
            data_dict["Time Per Epoch(s)"].append(time_per_train_test)

    data_frame = pd.DataFrame(data_dict)
    data_frame.to_csv(r"CC:\Users\pjmcc\PycharmProjects\dropoutResearch\results\CIFAR100\NoDrop10x34.csv",
                      mode='w')

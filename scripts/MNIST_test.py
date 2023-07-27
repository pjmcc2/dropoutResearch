import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from BaselineDropoutModel import BaselineDropoutModel
from TSModel import TSModel
from BaselineNoDropoutModel import BaselineNoDropoutModel
from CDropModel import CurriculumDropoutModel
from HybridModel import HybridCurrDropModel
import time
from tqdm import tqdm


def train_loop(dataloader, model, optimizer, lossfn, device, update_num, mode=None, return_probs=False, record=False):
    model.train()
    batches = len(dataloader)
    train_loss = 0
    prob_list = []
    for batch_num, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        if mode == "curr":
            guess = model(X, update_num)
        else:
            guess = model(X)
        loss = lossfn(guess, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if mode == "adapt" or mode == "hybrid":
            flag = True # TODO MAKE MODLAR
            #flag = 0
            corr = guess.argmax(dim=1) == y
            for l in model.modules():
                try:
                    if hasattr(l, "update_probs"):
                        l.update_probs(corr)
                        #flag += 1
                        #if flag == 2:
                        if flag:
                            #flag = 0
                            if mode == "adapt":
                                prob_list.append(l.probabilities[0, 0].item())
                            else:
                                prob_list.append(1 - l.get_prob(l.p_out, l.t))
                            flag = False
                except AttributeError:
                    pass
        if record:
            train_loss += loss.item()
        if batch_num % 100 == 0:
            loss = loss.item()

    if record:
        if return_probs:
            return train_loss / batches, prob_list
        else:
            return train_loss / batches, None


# also HEAVILY borrowed from pytorch tutorial
def test_loop(dataloader, model, lossfn, device, i, mode=None, curr=False, record=False):
    model.eval()
    batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            if mode == "curr":
                guess = model(X, i)
            else:
                guess = model(X)

            test_loss += lossfn(guess, y).item()
            correct += (guess.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= batches
    correct /= len(dataloader.dataset)

    if record:
        return test_loss, 100 * correct


################################

def init_weights(layer):  # initialization scheme derived from Morerio et al
    if isinstance(layer, nn.Linear):
        torch.nn.init.trunc_normal_(layer.weight, std=0.01)
        layer.bias.data.fill_(0.01)


if __name__ == "__main__":
    # Initial setup graciously copied from pytorch website:
    # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'

    training_data = datasets.MNIST(
        root=r"C:\Users\pjmcc\PycharmProjects\dropoutResearch\data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root=r"C:\Users\pjmcc\PycharmProjects\dropoutResearch\data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True, drop_last=False)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True, drop_last=False)

    lr = 0.0001
    epochs = 150
    momentum = 0.95
    num_iters = 10
    grad_updates = (len(train_dataloader)*epochs)
    loss_fn = nn.CrossEntropyLoss()
    data_dict = {"Epoch": [],
                 "Mean Train Loss": [],
                 "Mean Test Loss": [],
                 "Test Acc(%)": [],
                 "Time Per Epoch(s)": []}
    probs = []
    epoch_list = []
    runs = []
    for j in range(num_iters):
        curr_model = HybridCurrDropModel(keep_p_hidden=0.5,gamma=10/grad_updates)  # Create new instance to reset parameters (and probabilites)
        curr_model.apply(init_weights)
        curr_model.to(device)
        optimizer = torch.optim.SGD(curr_model.parameters(), lr, momentum=momentum)
        for i in tqdm(range(epochs)):
            data_dict["Epoch"].append(i + 1)
            starttime = time.time()
            avg_t_loss, prob_l = train_loop(train_dataloader, curr_model, optimizer, loss_fn, device, i, record=True, mode="hybrid", return_probs=True)
            avg_v_loss, v_acc = test_loop(test_dataloader, curr_model, loss_fn, device, i, mode="hybrid", record=True)
            end_time = time.time()
            time_per_train_test = end_time - starttime
            data_dict["Mean Train Loss"].append(avg_t_loss)
            data_dict["Mean Test Loss"].append(avg_v_loss)
            data_dict["Test Acc(%)"].append(v_acc)
            data_dict["Time Per Epoch(s)"].append(time_per_train_test)
            probs += prob_l
            count = len(prob_l)
            epoch_list += [i for q in range(count)]
        runs += [j for k in range(epochs * count)]
    probs_frame = pd.DataFrame({"Run": runs, "Epoch": epoch_list, "Keep Probability": probs})
    probs_frame.to_csv(
        r"C:\Users\pjmcc\PycharmProjects\dropoutResearch\results\MNIST\2000 Model\MNISTHybrid10x150_probs.csv",
        mode='w')
    data_frame = pd.DataFrame((data_dict))
    data_frame.to_csv(r"C:\Users\pjmcc\PycharmProjects\dropoutResearch\results\MNIST\2000 Model\MNISTHybrid10x150.csv",
                      mode='w')

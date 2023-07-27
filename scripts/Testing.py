import torch
from torch import nn

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
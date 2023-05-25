import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

torch.manual_seed(1066)
# TODO ADD GPU STUFF
# TODO ADD LOG FILE
# TODO ADD VISUALIZATION
# TODO FIGURE OUT WHERE TF THE DATA IS AND PUT IT IN THE RIGHT PLACE
# TODO TRIPLE CHECK THE DOCS FOR EACH STEP AND WRITE IT OUT. I WANT...
# TODO ...TO KNOW WHAT EVERYTHING DOES FOR MY WRITE UP.

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(  # three layers before output
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(),
            nn.LazyLinear(10)
        )

    def forward(self, X):
        x = self.flatten(X)
        logits = self.net(x)
        return logits

    # heavily borrows from pytorch tutorial


def train_loop(dataloader, model, optimizer, lossfn):
    for batch_num, (X, y) in enumerate(dataloader):
        guess = model(X)
        loss = lossfn(guess, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_num % 100 == 0:
            loss = loss.item()
            print(f"loss: {loss:>5f}   Batch: {batch_num + 1}")


# also HEAVILY borrowed from pytorch tutorial
def test_loop(dataloader, model, lossfn):
    batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            guess = model(X)
            test_loss += lossfn(guess, y).item()
            correct += (guess.argmax(dim=1) == y).type(torch.float).sum().item()

    test_loss /= batches
    correct /= len(dataloader.dataset)

    print(f"Average loss: {test_loss:>5f}    Test Accuracy: {100 * correct}%")


if __name__ == "__main__":
    # Initial setup graciously copied from pytorch website:
    # https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(test_data, batch_size=64)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    # Create baseline model from above class.
    base_model = BaselineModel().to(device)

    lr = 0.001
    epochs = 10
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(base_model.parameters(),lr)

    for i in range(epochs):
        print(f"epoch {i} \n")
        train_loop(train_dataloader, base_model, optimizer, loss_fn)
        test_loop(test_dataloader, base_model, loss_fn)
import torch
from tqdm import tqdm
import torchvision

class ResNet:
    def __init__(self, device):
        self.net = None
        self.device = device

    def init_model(self, num_classes):
        self.net = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        self.net.to(self.device)

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def fit(self, train_set, test_set, epochs, loss_fn, optimizer, verbose, save_path):
        if self.net is None:
            print('You haven\'t initialized a model')
            return

        best_acc = 0

        for epoch in range(epochs):
            train_loss, train_accuracy = self.train(train_set, loss_fn, optimizer)
            test_loss, test_accuracy = self.test(test_set, loss_fn)
            
            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}]:')
                print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
                print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                self.save_model(save_path)

    def train(self, train_set, loss_fn, optimizer):
        self.net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_set, leave=False, dynamic_ncols=True) as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                preds = self.net(imgs)

                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                target_index = torch.argmax(labels, dim=1)
                correct += torch.sum(predicted_classes == target_index).item()
                total += labels.size(0)


                pbar.set_description(f'Training Loss: {total_loss:.4f}, Training Accuracy: {100 * correct / total:.2f}%')

        accuracy = 100 * correct / total
        return total_loss, accuracy

    def test(self, test_set, loss_fn):
        self.net.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(test_set, leave=False, dynamic_ncols=True) as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self.net(imgs)

                loss = loss_fn(preds, labels)
                total_loss += loss.item()
                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)
                target_index = torch.argmax(labels, dim=1)
                correct += torch.sum(predicted_classes == target_index).item()
                total += labels.size(0)

                pbar.set_description(f'Test Loss: {total_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%')

        accuracy = 100 * correct / total
        return total_loss, accuracy

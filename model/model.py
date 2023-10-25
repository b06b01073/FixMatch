import torch
from tqdm import tqdm
import torch.nn.functional as F
from . import resnet
import torch.nn as nn
from torchvision.utils import save_image

class ResNet: 
    def __init__(self, device):
        self.device = device
        self.net = resnet.ResNet50()
        self.net.to(self.device)

    def save_model(self, save_dir, file_name):
        print('saving model')
        torch.save(self.net.state_dict(), f'{save_dir}/{file_name}')

    def fit(self, train_set, test_set, epochs, loss_fn, optimizer, scheduler, verbose, save_dir, file_name, test_interval):
        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train(train_set, loss_fn, optimizer)
            scheduler.step()
            
            if verbose and ((epoch + 1) % test_interval == 0):
                test_loss, test_accuracy = self.test(test_set, loss_fn)
                print(f'Epoch [{epoch+1}/{epochs}]:')
                print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
                print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    self.save_model(save_dir, file_name)

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
                correct += torch.sum(predicted_classes == labels).item()
                total += labels.size(0)

                pbar.set_description(f'Test Loss: {total_loss:.4f}, Test Accuracy: {100 * correct / total:.2f}%')

        accuracy = 100 * correct / total
        return total_loss, accuracy


class FixMatch:
    def __init__(self, labeled_set, unlabeled_set, test_set, num_classes, threshold, lamb, device):
        self.device = device
        self.net = resnet.ResNet50()
        self.net.to(self.device)
        self.labeled_set = labeled_set
        self.unlabeled_set = unlabeled_set

        self.threshold = threshold # confidence threshold
        self.lamb = lamb # unlabeled loss weight
        self.num_classes = num_classes

        self.test_set = test_set



    def save_model(self, path, file_name):
        print('saving model')
        torch.save(self.net.state_dict(), f'{path}/{file_name}')

    
    def load_model(self, pretrained):
        print(f'loading pretrained model from {pretrained}')
        self.net.load_state_dict(torch.load(pretrained))


    def fit(self, steps, optimizer, scheduler, verbose, save_dir, file_name, log_interval):
        best_acc = 0
        train_loss = 0
        epochs = steps // min(len(self.labeled_set), len(self.unlabeled_set))

        for epoch in tqdm(range(epochs), leave=False, dynamic_ncols=True):
            train_loss += self.train(optimizer)
            scheduler.step()
            
            if verbose and (epoch + 1) % log_interval == 0:
                test_loss, test_accuracy = self.test()
                print(f'Epoch [{epoch+1}/{epochs}]:')
                print(f'Training Loss: {train_loss:.4f}')
                print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

                if test_accuracy > best_acc:
                    best_acc = test_accuracy
                    self.save_model(save_dir, file_name)

    def train(self, optimizer):
        self.net.train()
        # correct = 0
        # total = 0

        total_loss = 0

        for (labeled_imgs, labels), (weak_imgs, strong_imgs, _) in zip(self.labeled_set, self.unlabeled_set):
            loss = self.fixmatch_loss(
                labeled_imgs,
                labels,
                weak_imgs,
                strong_imgs,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss


    def test(self):
        self.net.eval()

        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(), tqdm(self.test_set, leave=False, dynamic_ncols=True) as pbar:
            for imgs, labels in pbar:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                preds = self.net(imgs)

                loss = loss_fn(preds, labels)
                total_loss += loss.item()
                predicted_classes = torch.argmax(torch.softmax(preds, dim=1), dim=1)


                correct += torch.sum(predicted_classes == labels).item()
                total += labels.size(0)


        accuracy = 100 * correct / total
        return total_loss, accuracy
            

    

    def fixmatch_loss(self, labeled_imgs, labels, weak_imgs, strong_imgs):
        # Algorithm 1 of https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf

        labeled_imgs = labeled_imgs.to(self.device)
        weak_imgs = weak_imgs.to(self.device)
        strong_imgs = strong_imgs.to(self.device)
        labels = labels.to(self.device)

        # supervised loss
        labels = self.one_hot_encode(labels)

        preds = self.net(labeled_imgs)
        supervised_loss = nn.CrossEntropyLoss()(preds, labels)


        # unsupervised loss
        unlabeled_batch_size = weak_imgs.shape[0]
        weak_preds = F.softmax(self.net(weak_imgs), dim=1) # q_b
        confidence, _ = torch.max(weak_preds, dim=1) # max(q_b)
        confidence = (confidence > self.threshold).type(torch.int32) # max(q_b) > threshold
        confidence_indices = torch.nonzero(confidence == 1) # get the indices of those who pass the threshold

        pseudo_labels = self.one_hot_encode(torch.argmax(weak_preds, dim=1))[confidence_indices].squeeze(dim=1).detach() # argmax(q_b) and filtered them by confidence_indices

        strong_preds = self.net(strong_imgs)[confidence_indices].squeeze(dim=1)

    
        unsupervised_loss = nn.CrossEntropyLoss(reduction='sum')(strong_preds, pseudo_labels) / unlabeled_batch_size
        

        return supervised_loss + self.lamb * unsupervised_loss 


    def one_hot_encode(self, labels):
        return F.one_hot(labels, self.num_classes).type(torch.float32) 
    

        
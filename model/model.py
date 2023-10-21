import torch
from tqdm import tqdm
import torchvision
import torch.nn.functional as F


class ResNet:
    def __init__(self, num_classes, device):
        self.device = device
        self.net = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        self.net.to(self.device)

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def fit(self, train_set, test_set, epochs, loss_fn, optimizer, verbose, save_path):
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


class FixMatch:
    def __init__(self, labeled_set, unlabeled_set, test_set, num_classes, threshold, lamb, device):
        self.device = device
        self.net = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
        self.net.to(self.device)
        self.labeled_set = iter(labeled_set)
        self.unlabeled_set = iter(unlabeled_set)
        self.threshold = threshold # confidence threshold
        self.lamb = lamb # unlabeled loss weight

        self.test_set = test_set

    def save_model(self, path):
        torch.save(self.net.state_dict(), path)

    def fit(self, updates, epochs, optimizer, verbose, save_path):
        best_acc = 0
        for epoch in range(epochs):
            train_loss, train_accuracy = self.train(updates, optimizer)
            test_loss, test_accuracy = self.test(loss_fn)
            
            if verbose:
                print(f'Epoch [{epoch+1}/{epochs}]:')
                print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
                print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                self.save_model(save_path)

    def train(self, updates, optimizer):
        self.net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for _ in tqdm(range(updates), leave=False, dynamic_ncols=True):
            labeled_imgs, labels = next(self.labeled_set)
            unlabeled_imgs = next(self.unlabeled_imgs)

            optimizer.zero_grad()
            loss = self.fixmatch_loss(
                labeled_imgs.to(device), 
                labels.to(device),
                unlabeled_imgs.to(device),
                loss_fn
            )
            loss.backward()
            optimizer.step()

        accuracy = 100 * correct / total
        return total_loss, accuracy


    def fixmatch_loss(self, labeled_imgs, labels, unlabeled_imgs):
        # Algorithm 1 of https://arxiv.org/ftp/arxiv/papers/2001/2001.07685.pdf

        # supervised loss
        cross_entropy_loss = nn.CrossEntropyLoss()
        labeled_imgs = self.weak_augment(labeled_imgs)
        labels = self.one_hot_encode(labels)
        preds = self.net(labeled_imgs)
        supervised_loss = cross_entropy_loss(preds, labels)



        # unsupervised loss
        weak_unlabeled_imgs = self.weak_augment(unlabeled_imgs)
        weak_preds = F.softmax(self.net(weak_unlabeled_imgs), dim=-1) # q_b
        confidence, _ = torch.max(weak_preds, dim=1) # max(q_b)
        confidence = (confidence > self.threshold).type(torch.float32).detach() # max(q_b) > threshold
        pseudo_labels = self.one_hot_encode(torch.argmax(weak_preds, dim=1)).detach() # argmax(q_b)

        strong_unlabed_imgs = self.strong_augment(unlabeled_imgs)
        strong_preds = self.net(strong_unlabed_imgs) 

        unsupervised_cross_entropy = cross_entropy_loss(strong_preds, pseudo_labels)

        unsupervised_loss = torch.mul(confidence, unsupervised_cross_entropy)

        return supervised_loss + self.lamb * unsupervised_loss


    def strong_augment(self, imgs):
        return  transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip
            transforms.RandomVerticalFlip(p=0.2),    # Random vertical flip
            transforms.RandomRotation(degrees=30),    # Random rotation up to 30 degrees
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jitter
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),  # Random perspective distortion
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.5),  # Random erasing
            transforms.ToTensor(),  # Convert the image to a tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize the image
        ])

    def weak_augment(self, imgs):
        return transforms.Compose([
            transforms.ToTensor(), 
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        ])



    def one_hot_encode(self, labels):
        return F.one_hot(labels, self.num_classes).type(torch.float32) 
        
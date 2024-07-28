import torch.nn as nn
import torch
from torchvision import transforms
import numpy as np
from torch.nn import functional as F
from PIL import Image
import torch.optim as optim
from myNetwork import network
from iCIFAR100 import iCIFAR100
from torch.utils.data import DataLoader
from logging import info, warn
from utils import Examplear_Dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_one_hot(target,num_class):
    one_hot=torch.zeros(target.shape[0],num_class).to(device)
    one_hot=one_hot.scatter(dim=1,index=target.long().view(-1,1),value=1.)
    return one_hot

class iCaRLmodel:

    def __init__(self,numclass,feature_extractor,batch_size,task_size,memory_size,epochs,learning_rate, 
                 beta:float,
                 output:str):
        super(iCaRLmodel, self).__init__()
        self.beta = beta
        self.output = output
        self.epochs=epochs
        self.learning_rate=learning_rate
        self.model = network(numclass,feature_extractor)
        self.exemplar_set = []
        self.class_mean_set = []
        self.numclass = numclass
        self.transform = transforms.Compose([#transforms.Resize(img_size),
                                             transforms.ToTensor(),
                                            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        self.old_model = None

        self.train_transform = transforms.Compose([#transforms.Resize(img_size),
                                                  transforms.RandomCrop((32,32),padding=4),
                                                  transforms.RandomHorizontalFlip(p=0.5),
                                                  transforms.ColorJitter(brightness=0.24705882352941178),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.test_transform = transforms.Compose([#transforms.Resize(img_size),
                                                   transforms.ToTensor(),
                                                 transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

        self.classify_transform=transforms.Compose([transforms.RandomHorizontalFlip(p=1.),
                                                    #transforms.Resize(img_size),
                                                    transforms.ToTensor(),
                                                   transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        
        self.train_dataset = iCIFAR100('dataset', transform=self.train_transform, download=True)
        self.test_dataset = iCIFAR100('dataset', test_transform=self.test_transform, train=False, download=True)

        self.batchsize = batch_size
        self.memory_size=memory_size
        self.task_size=task_size

        self.train_loader=None
        self.test_loader=None
        
        # TODO
        self.acc_budget = 72.95
        self.knn_acc_budget = 74.15

    # get incremental train data
    # incremental
    def beforeTrain(self):
        self.model.eval()
        classes=[self.numclass-self.task_size,self.numclass]
        warn(f'Class Boundary: {classes}')
        
        self.train_loader,self.test_loader,self.exemplar_loader=self._get_train_and_test_dataloader(classes)
        
        if self.numclass>self.task_size:
            self.model.Incremental_learning(self.numclass)
        self.model.train()
        self.model.to(device)

    def _get_train_and_test_dataloader(self, classes):
        self.train_dataset.getTrainData(classes, list())
        self.test_dataset.getTestData(classes)
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
        g = torch.Generator()
        g.manual_seed(0)
        
        self.exemplar_dataset = None
        exemplar_loader = None
        if len(self.exemplar_set) != 0:
            self.exemplar_dataset = Examplear_Dataset(self.exemplar_set, self.train_transform, self.test_transform)
            ratio = len(self.exemplar_dataset) / (len(self.train_dataset) + len(self.exemplar_dataset))
            exemplar_batch_size = round(self.batchsize * ratio)
            train_batch_size = self.batchsize - exemplar_batch_size

            exemplar_loader = DataLoader(dataset=self.exemplar_dataset,
                                         shuffle=True,
                                         batch_size=exemplar_batch_size,
                                         num_workers=2,
                                         worker_init_fn=seed_worker,
                                         generator=g)
            train_loader = DataLoader(dataset=self.train_dataset,
                                    shuffle=True,
                                    batch_size=train_batch_size,
                                    num_workers=2,
                                    worker_init_fn=seed_worker,
                                    generator=g)
        else:
            train_loader = DataLoader(dataset=self.train_dataset,
                                    shuffle=True,
                                    batch_size=self.batchsize,
                                    num_workers=4,
                                    worker_init_fn=seed_worker,
                                    generator=g)

        test_loader = DataLoader(dataset=self.test_dataset,
                                shuffle=True,
                                batch_size=self.batchsize,
                                num_workers=4,
                                worker_init_fn=seed_worker,
                                generator=g)

        return train_loader, test_loader, exemplar_loader

    # train model
    # compute loss
    # evaluate model
    def train(self):
        accuracy = 0
        opt = optim.SGD(self.model.parameters(), lr=self.learning_rate, weight_decay=0.00001)
        for epoch in range(self.epochs):
            if epoch == 48:
                if self.numclass==self.task_size:
                     print(1)
                     opt = optim.SGD(self.model.parameters(), lr=1.0/5, weight_decay=0.00001)
                else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 5
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 5,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                print("change learning rate:%.3f" % (self.learning_rate / 5))
            elif epoch == 62:
                if self.numclass>self.task_size:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 25
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate/ 25,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                else:
                     opt = optim.SGD(self.model.parameters(), lr=1.0/25, weight_decay=0.00001)
                print("change learning rate:%.3f" % (self.learning_rate / 25))
            elif epoch == 80:
                  if self.numclass==self.task_size:
                     opt = optim.SGD(self.model.parameters(), lr=1.0 / 125,weight_decay=0.00001)
                  else:
                     for p in opt.param_groups:
                         p['lr'] =self.learning_rate/ 125
                     #opt = optim.SGD(self.model.parameters(), lr=self.learning_rate / 125,weight_decay=0.00001,momentum=0.9,nesterov=True,)
                  print("change learning rate:%.3f" % (self.learning_rate / 100))
            
            if self.exemplar_loader == None:
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    images, target = images.to(device), target.to(device)
                    #output = self.model(images)
                    loss_value = self._compute_loss(indexs, images, target)
                    opt.zero_grad()
                    loss_value.backward()
                    opt.step()
                    print('epoch:%d,step:%d,loss:%.3f' % (epoch, step, loss_value.item()))
            else:
                # TODO
                exemplar_iter = iter(self.exemplar_loader)
                for step, (indexs, images, target) in enumerate(self.train_loader):
                    try:
                        ex_aug_imgs, ex_ori_imgs, ex_slcs, ex_targets = next(exemplar_iter)
                    except:
                        exemplar_iter = iter(self.exemplar_loader)
                        ex_aug_imgs, ex_ori_imgs, ex_slcs, ex_targets = next(exemplar_iter)
                    images = torch.cat((images, ex_aug_imgs), dim=0)
                    target = torch.cat((target, ex_targets), dim=0)
                    images, target = images.to(device), target.to(device)
                    nll_loss = self._compute_loss(indexs, images, target)
                    
                    # Gradient Matching Loss
                    ex_ori_imgs = ex_ori_imgs.to(device)
                    ex_ori_imgs.requires_grad = True
                    preds = self.model(ex_ori_imgs)
                    preds = F.log_softmax(preds, dim=-1)
                    scores = preds[torch.arange(preds.shape[0]), ex_targets]
                    scores = torch.sum(scores)
                    slcs = torch.autograd.grad(scores, ex_ori_imgs, create_graph=True)[0]
                    slcs = torch.mean(torch.abs(slcs), dim=1)
                    slcs = (slcs - slcs.min())/(slcs.max()-slcs.min() + 1e-8)
                    
                    ex_slcs = ex_slcs.clone().detach()
                    ex_slcs = ex_slcs.view(ex_slcs.shape[0], -1).to(device)
                    slcs = slcs.view(slcs.shape[0], -1)
                    mse = F.mse_loss(slcs, ex_slcs)

                    loss_value = nll_loss + self.beta*mse
                    opt.zero_grad()
                    loss_value.backward()
                    opt.step()
                    warn('epoch:%d, step:%d, loss:%.3f, nll_loss:%.3f, mse:%.3f' % (epoch, step, loss_value.item(), nll_loss.item(), mse.item()))

            accuracy = self._test(self.test_loader, 1)
            warn('epoch:%d,accuracy:%.3f' % (epoch, accuracy))
        return accuracy

    def _test(self, testloader, mode):
        if mode==0:
            print("compute NMS")
        self.model.eval()
        correct, total = 0, 0
        for setp, (indexs, imgs, labels) in enumerate(testloader):
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                outputs = self.model(imgs) if mode == 1 else self.classify(imgs)
            predicts = torch.max(outputs, dim=1)[1] if mode == 1 else outputs
            correct += (predicts.cpu() == labels.cpu()).sum()
            total += len(labels)
        accuracy = 100 * correct / total
        self.model.train()
        return accuracy


    def _compute_loss(self, indexs, imgs, target):
        output=self.model(imgs)
        target = get_one_hot(target, self.numclass)
        output, target = output.to(device), target.to(device)
        if self.old_model == None:
            return F.binary_cross_entropy_with_logits(output, target)
        else:
            #old_target = torch.tensor(np.array([self.old_model_output[index.item()] for index in indexs]))
            old_target=torch.sigmoid(self.old_model(imgs))
            old_task_size = old_target.shape[1]
            target[..., :old_task_size] = old_target
            return F.binary_cross_entropy_with_logits(output, target)


    # change the size of examplar
    def afterTrain(self, accuracy):
        self.model.eval()
        m=int(self.memory_size/self.numclass)
        self._reduce_exemplar_sets(m)
        for i in range(self.numclass-self.task_size,self.numclass):
            print('construct class %s examplar:'%(i),end='')
            images=self.train_dataset.get_image_class(i)
            self._construct_exemplar_set(images,m,i)
        self.numclass+=self.task_size
        self.compute_exemplar_class_mean()
        self.model.train()
        KNN_accuracy=self._test(self.test_loader,0)
        warn("NMS accuracy："+str(KNN_accuracy.item()))
        
        from os import makedirs
        from os import path as osp
        output_dir = osp.join('Custom_model', self.output)
        makedirs(output_dir, exist_ok=True)
        filename='accuracy:%.3f_KNN_accuracy:%.3f_increment:%d_net.pkl' % (accuracy, KNN_accuracy, i)
        torch.save(self.model,osp.join(output_dir, filename))
        self.old_model=torch.load(osp.join(output_dir, filename))
        self.old_model.to(device)
        self.old_model.eval()
        
        # TODO
        # if accuracy < self.acc_budget and KNN_accuracy < self.knn_acc_budget:
        #     exit()
        
        
    def _construct_exemplar_set(self, images, m, label):
        class_mean, feature_extractor_output = self.compute_class_mean(images, self.transform)
        exemplar = []
        now_class_mean = np.zeros((1, 512))
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        
        imgs = list()
        for i in range(m):
            # shape：batch_size*512
            x = class_mean - (now_class_mean + feature_extractor_output) / (i + 1)
            # shape：batch_size
            x = np.linalg.norm(x, axis=1)
            index = np.argmin(x)
            now_class_mean += feature_extractor_output[index]
            imgs.append(self.test_transform(images[index]).unsqueeze(0))
            exemplar.append([images[index]])

        # Make Saliency Map for each Image
        imgs = torch.cat(imgs, dim=0) # [200, 3, 32, 32]
        imgs = imgs.to(device)
        imgs.requires_grad = True
        preds = self.model(imgs) # [200, 10]
        preds = F.log_softmax(preds, dim=-1) # [200]
        scores = preds[torch.arange(preds.shape[0]), torch.full((preds.shape[0],), label)]
        scores = torch.sum(scores)
        slcs = torch.autograd.grad(scores, imgs, create_graph=True)[0]
        slcs = torch.mean(torch.abs(slcs), dim=1)
        slcs = (slcs - slcs.min())/(slcs.max()-slcs.min() + 1e-8)
        
        slcs = slcs.detach().cpu()  # torch.Size([200, 32, 32])
        assert slcs.shape[0] == len(exemplar)
        
        for i in range(slcs.shape[0]):
            slc = slcs[i]
            exemplar[i].extend([slc, label])
        
        print("the size of exemplar :%s" % (str(len(exemplar))))
        self.exemplar_set.append(exemplar)
        
        for param in self.model.parameters():
            param.requires_grad = True

    def _reduce_exemplar_sets(self, m):
        for index in range(len(self.exemplar_set)):
            self.exemplar_set[index] = self.exemplar_set[index][:m]
            print('Size of class %d examplar: %s' % (index, str(len(self.exemplar_set[index]))))


    def Image_transform(self, images, transform):
        data = transform(Image.fromarray(images[0])).unsqueeze(0)
        for index in range(1, len(images)):
            data = torch.cat((data, self.transform(Image.fromarray(images[index])).unsqueeze(0)), dim=0)
        return data

    def compute_class_mean(self, images, transform):
        x = self.Image_transform(images, transform).to(device)
        feature_extractor_output = F.normalize(self.model.feature_extractor(x).detach()).cpu().numpy()
        class_mean = np.mean(feature_extractor_output, axis=0)
        return class_mean, feature_extractor_output

    def compute_exemplar_class_mean(self):
        self.class_mean_set = []
        for index in range(len(self.exemplar_set)):
            print("compute the class mean of %s"%(str(index)))
            exemplar=[x[0] for x in self.exemplar_set[index]]
            class_mean, _ = self.compute_class_mean(exemplar, self.transform)
            class_mean_,_=self.compute_class_mean(exemplar,self.classify_transform)
            class_mean=(class_mean/np.linalg.norm(class_mean)+class_mean_/np.linalg.norm(class_mean_))/2
            self.class_mean_set.append(class_mean)

    def classify(self, test):
        result = []
        test = F.normalize(self.model.feature_extractor(test).detach()).cpu().numpy()
        #test = self.model.feature_extractor(test).detach().cpu().numpy()
        class_mean_set = np.array(self.class_mean_set)
        for target in test:
            x = target - class_mean_set
            x = np.linalg.norm(x, ord=2, axis=1)
            x = np.argmin(x)
            result.append(x)
        return torch.tensor(result)

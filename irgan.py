import argparse
import os
import torch
import torch.nn as nn
from   torch.utils.data import Dataset
import ujson
from skimage import io, transform

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--niter', type=int, default=25, help='total number of epochs to train for')
parser.add_argument('--g_epochs', type=int, default=1, help='number of epochs to train Generator for')
parser.add_argument('--d_epocs', type=int, default=1, help='number of epochs to train Discriminator for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

def load_vocabulary():
    with open("recipeqa/vocab_clean.txt", "r") as f:
        vocab = {}
        for i, w in enumerate(f):
            vocab[str(w).strip()] = i
        return vocab

class Generator(nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass

class Discriminator(nn.Module):
    def __init__(self):
        pass

    def forward(self, input):
        pass

class RecipeQADataset(Dataset):
    
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_list = ujson.load(open(csv_file, "r"))
        self.root_dir = root_dir
        self.transform = transform
        self.vocab = load_vocabulary()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ret = {}
        data_item = self.data_list[idx]

        ret["context"] = [
            [
                self.vocab[word] for word in c_item["cleaned_body"].split()
            ] for c_item in data_item["context"]
        ]

        ret["choice_list"] = [
            io.imread(os.path.join(self.root_dir, img)) for img in data_item["choice_list"]
        ]

        ret["question"] = [
            io.imread(os.path.join(self.root_dir, img)) for img in data_item["question"]
        ]

        if self.transform:
            ret = self.transform(ret)

        return ret


train_dataset = RecipeQADataset("recipeqa/new_train_cleaned.json", "recipeqa/images/train/images-qa")
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers)
)

val_dataset = RecipeQADataset("recipeqa/new_val_cleaned.json", "recipeqa/images/val/images-qa")
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers)
)

netG = Generator()
netD = Discriminator()

def weights_init(m):
    pass

criterion = nn.BCEWithLogitsLoss()

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def gen_expected_answers(batch):

    one_hot = [[] for i in range(batch.size(0))]

    return one_hot

def pre_train(train_netG, train_netD):
    for epoch in range(opt.niter):
        for i, batch in enumerate(dataloader, 0):
            netD.zero_grad()
            outputs = netD(batch)
            labels = torch.full((batch.size(0),), 1, device=device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizerD.step()

    for epoch in range(opt.niter):
        for i, batch in enumerate(dataloader, 0):
            netG.zero_grad()
            probs = netG(batch)
            expected_outputs = gen_expected_answers(batch)
            loss = criterion(probs, expected_outputs)
            loss.backward()
            optimizerG.step()

def generate_samples(num_samples, batch):
    samples = []

    probs = netG(batch) # context + question + choice_list ==> probability distribution over choice_list

    for i, data in enumerate(batch):
        sample = np.random.choice(data.choice_list, p=probs[i], replace=True, size=num_samples)
        samples.append(sample)

    return samples

def training():
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))

    print(netG)

    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    print(netD)

    pre_train(opt.netG, opt.netD)

    for epoch in range(opt.niter):
        for g in range(opt.g_epochs):
            for i, batch in enumerate(dataloader, 0):
                # [(question, [samples from the relevance prob dist for this question])]
                samples = generate_samples(num_samples, batch)
                loss = torch.mean(torch.log(netG(samples)) * torch.log(1 + torch.exp(netD(samples))))
                loss.backward()
                optimizerG.step()

        for d in range(opt.d_epochs):
            for batch in enumerate(dataloader, 0):
                samples = generate_samples(num_samples, batch)
                neg_labels = torch.full((batch.size(0),), 0, device=device)
                outputs = netD(samples)
                neg_loss = criterion(outputs, neg_labels)
                pos_labels = torch.full((batch.size(0),), 1, device=device)
                outputs = netD(batch)
                pos_loss = criterion(outputs, pos_labels)

                total_loss = neg_loss + pos_loss
                total_loss.backward()
                optimizerD.step()

        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

if __name__ == "__main__":
    training()

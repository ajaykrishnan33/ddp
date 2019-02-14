import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

import dataservices
import networks

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method("spawn")

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
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
parser.add_argument('--vgg_pretrained', action='store_true', help='vgg pretrained?', default=False)

opt = parser.parse_args()
print(opt)

device = torch.device("cuda:0" if opt.cuda else "cpu")

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

train_embeddings = dataservices.Doc2Vec(
    "paragraph-vectors/data/sentences_train_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.100_loss.0.781092.csv"
)
train_dataset = dataservices.RecipeQADataset(
    "recipeqa/new_train_cleaned.json", 
    "recipeqa/images/train/images-qa",
    train_embeddings,
    transform=dataservices.RescaleToTensorAndNormalize(224)
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=False, num_workers=int(opt.workers),
    collate_fn=dataservices.batch_collator(device=device)
)

# val_embeddings = dataservices.Doc2Vec(
#     "paragraph-vectors/data/sentences_train_model.dbow_numnoisewords\
#     .2_vecdim.100_batchsize.32_lr.0.001000_epoch.100_loss.0.781092.csv"
# )  # change this to val
# val_dataset = dataservices.RecipeQADataset(
#     "recipeqa/new_val_cleaned.json", 
#     "recipeqa/images/val/images-qa", 
#     val_embeddings,
#     transform=dataservices.RescaleToTensorAndNormalize(224)
# )
# val_dataloader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=opt.batchSize,
#     shuffle=True, num_workers=int(opt.workers),
#     collate_fn=dataservices.batch_collator(device=device)
# )

netG = networks.Generator(opt).to(device)
netD = networks.Discriminator(opt).to(device)

def weights_init(m):
    pass

criterionD = nn.BCEWithLogitsLoss()  # logsigmoid + binary cross entropy
criterionG = nn.CrossEntropyLoss()   # logsoftmax + multi-class cross entropy

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def pre_train(train_netG, train_netD):

    if train_netD:
        for epoch in range(opt.niter):
            for i, batch in enumerate(train_dataloader, 0):
                netD.zero_grad()
                logits, probabilities = netD(batch)
                labels = batch["answers"]
                loss = criterionD(logits, labels)
                loss.backward()
                optimizerD.step()

    if train_netG:
        for epoch in range(opt.niter):
            for i, batch in enumerate(train_dataloader, 0):
                netG.zero_grad()
                logits, distributions = netG(batch)
                expected_outputs = batch["answers"]
                loss = criterionG(logits, expected_outputs)
                loss.backward()
                optimizerG.step()

def generate_samples(batch, distributions, num_samples):
    samples = []
    sample_indices = []

    for i, data in enumerate(batch["choices"]):
        local_sample_indices = torch.multinomial(distributions[i], num_samples, replacement=True)
        samples.append(data[local_sample_indices])
        sample_indices.append(local_sample_indices)

    samples = torch.tensor(samples).to(device)

    sample_batch = {
        "questions":batch["questions"],
        "contexts":batch["contexts"],
        "choices":samples,
        "size": batch["size"]
    }

    return sample_batch, sample_indices

def training():
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))

    print(netG)

    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    print(netD)

    pre_train(opt.netG == '', opt.netD == '')

    for epoch in range(opt.niter):
        for g in range(opt.g_epochs):
            for i, batch in enumerate(dataloader, 0):
                print("Pretraining Generator Epoch:{}, batch:{}".format(epoch, i))
                netG.zero_grad()
                
                g_logits, distributions = netG(batch) # context + question + choice_list ==> probability distribution over choice_list
                
                sample_batch, sample_indices = generate_samples(batch, distributions, num_samples)

                d_sample_logits, _ = netD(sample_batch)

                g_sample_probs = []

                for i, local_sample_indices in enumerate(sample_indices):
                    g_sample_probs.append(distributions[i, local_sample_indices])

                """
                Explanation regarding generate_samples, d_sample_logits, g_sample_probs:
                    generate_samples generates K samples per question from the choiceset for the question
                    and sets the K samples as the new choiceset for the question. Since the output of the 
                    discriminator for each choice is independent of the other choices, we can use this trick
                    of setting all K samples as the choices of a single question. If the generator were 
                    perfect, then all the K samples will be identical and will be equal to the right answer
                    and hence the discriminator will output 1 for every choice of the question.        

                    d_sample_logits has the shape: batch_size X K
                    Each K-sized vector in d_sample_logits is a list of likelihoods for each of the K 
                    samples for the question being the right answer to the question. 
                    d_sample_logits[question_num][j] --> probability of sample j being the right answer to
                    the question at index "question_num" in the batch (computed by the discriminator)

                    g_sample_probs has the shape: batch_size X K
                    Each K-sized vector in g_sample_probs is a list of probability values (of relevance) 
                    for the K samples computed over the original choices for the question 
                    (and not just the chosen samples).
                    g_sample_probs[question_num][j] --> probability of sample j being the right answer
                    to the question at index "question_num" in the batch (as computed by the generator 
                    from the original choice set).

                    For every question q, for every sample j, we need to compute:
                    logsigmoid(d_sample_logits[q][j]) * log(g_sample_probs[q][j])

                    For every question q, we need to find:
                    mean_over_all_j(logsigmoid(d_sample_logits[q][j]) * log(g_sample_probs[q][j]))
                    
                """                

                loss = torch.mean(
                    torch.log_softmax(g_sample_logits) * torch.logsigmoid(d_sample_logits)
                )
                loss.backward()
                optimizerG.step()

        for d in range(opt.d_epochs):
            for batch in enumerate(dataloader, 0):
                print("Pretraining Discriminator Epoch:{}, batch:{}".format(epoch, i))
                netD.zero_grad()
                
                """
                Here, we need to generate N samples from the choice list for every question and pass
                them through the discriminator. We want the discriminator to output 0 for these images.
                We will also pass the true answers through the discriminator which must output 1 for them.
                """
                g_logits, distributions = netG(batch)
                sample_batch, sample_indices = generate_samples(num_samples, distributions, batch)
                
                neg_labels = torch.full((batch["size"], num_samples), 0).to(device)
                sample_logits, _ = netD(sample_batch)
                neg_loss = criterionD(sample_logits, neg_labels)
                
                batch_labels = batch["answers"]
                batch_logits = netD(batch)
                batch_loss = criterionD(batch_logits, batch_labels)

                total_loss = neg_loss + batch_loss
                total_loss.backward()
                optimizerD.step()

        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))

if __name__ == "__main__":
    training()

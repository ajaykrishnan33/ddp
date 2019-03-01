import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from tqdm import tqdm

import dataservices
import networks

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
# torch.multiprocessing.set_start_method("spawn")

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--num_samples', type=int, default=10, help='number of samples per question')
parser.add_argument('--pre_niter', type=int, default=10, help='total number of epochs to train for')
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
parser.add_argument('--pretrain', action='store_true', help='pretrain?', default=False)
parser.add_argument('--train', action='store_true', help='train?', default=False)

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
    "recipeqa/features/train",
    train_embeddings
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers),
    collate_fn=dataservices.batch_collator(device=device)
)

val_embeddings = dataservices.Doc2Vec(
    "paragraph-vectors/data/sentences_val_model.dbow_numnoisewords.2_vecdim.100_batchsize.32_lr.0.001000_epoch.100_loss.0.557238.csv",
)
val_dataset = dataservices.RecipeQADataset(
    "recipeqa/new_val_cleaned.json", 
    "recipeqa/features/val", 
    val_embeddings
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers),
    collate_fn=dataservices.batch_collator(device=device)
)

netG = networks.Generator(opt).to(device)
netD = networks.Discriminator(opt).to(device)

def weights_init(m):
    pass

criterionD = nn.BCEWithLogitsLoss()  # logsigmoid + binary cross entropy
criterionG = nn.CrossEntropyLoss()   # logsoftmax + multi-class cross entropy

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

def fscore(probabilities, labels):
    true_positives = float(((torch.round(probabilities)==1.0)*(labels==1.0)).sum().item())
    false_positives = float(((torch.round(probabilities)==1.0)*(labels==0.0)).sum().item())
    false_negatives = float(((torch.round(probabilities)==0.0)*(labels==1.0)).sum().item())

    epsilon = 0.01

    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)
    fscore = (2 * precision * recall) / (precision + recall + epsilon)

    print("True positives: {}, False positives: {}, False negatives: {}".format(true_positives, false_positives, false_negatives))

    return precision, recall, fscore

def score_gen(distributions, expected_outputs):
    correct_answers = (distributions.argmax(dim=1)==expected_outputs).sum().item()

    return correct_answers

def eval_netD(epoch):
    # validation booyeah!
    netD.eval()
    print("\nEvaluation:")
    for i, batch in tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader)):
        with torch.no_grad():
            logits, probabilities = netD(batch)
            labels = torch.full((batch["size"],), 1).to(device)
            loss = criterionD(logits, labels)
            print(
                "Eval:: Pretraining discriminator Epoch: {}, batch_num: {}, loss: {}, precision: {}, recall: {}, fscore: {}"
                .format(
                    epoch, i, loss,
                    *fscore(probabilities, labels)
                )
            )

def eval_netG(epoch):
    # validation booyeah!
    netG.eval()
    print("\nEvaluation:")
    for i, batch in tqdm(enumerate(val_dataloader, 0), total=len(val_dataloader)):
        with torch.no_grad():
            logits, distributions = netG(batch)
            
            # expected_outputs = torch.full((batch["size"], batch["choices"].size(1)), 0).to(device)
            # expected_outputs[range(batch["size"]), batch["answers"]] = 1
            expected_outputs = torch.tensor(batch["answers"]).to(device)

            loss = criterionG(logits, expected_outputs)
            print(
                "Eval:: Pretraining generator Epoch: {}, batch_num: {}, loss: {}, correct_answers: {}"
                .format(
                    epoch, i, loss,
                    score_gen(distributions, expected_outputs)
                )
            )

def pre_train(train_netD, train_netG):

    if train_netD:
        for epoch in tqdm(range(opt.pre_niter)):
            netD.train()
            for i, batch in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
                netD.zero_grad()
                logits, probabilities = netD(batch)
                labels = torch.full((batch["size"],), 1).to(device)

                loss = criterionD(logits, labels)
                loss.backward()
                optimizerD.step()

                print(
                    "Train:: Pretraining discriminator Epoch: {}, batch_num: {}, loss: {}, precision: {}, recall: {}, fscore: {}"
                    .format(
                        epoch, i, loss,
                        *fscore(probabilities, labels)
                    )
                )

            # validation booyeah!
            eval_netD(epoch)

            torch.save(netD.state_dict(), '%s/netD_pretrain_epoch_%d.pth' % (opt.outf, epoch))



    if train_netG:
        for epoch in tqdm(range(opt.pre_niter)):
            netG.train()
            for i, batch in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
                netG.zero_grad()
                logits, distributions = netG(batch)

                # expected_outputs = torch.full((batch["size"], batch["choices"].size(1)), 0).to(device)
                # expected_outputs[range(batch["size"]), batch["answers"]] = 1  
                expected_outputs = torch.tensor(batch["answers"]).to(device)

                loss = criterionG(logits, expected_outputs)
                loss.backward()
                optimizerG.step()

                print(
                    "Train:: Pretraining generator Epoch: {}, batch_num: {}, loss: {}, correct_answers: {}"
                    .format(
                        epoch, i, loss,
                        score_gen(distributions, expected_outputs)
                    )
                )

            eval_netG(epoch)

            torch.save(netG.state_dict(), '%s/netG_pretrain_epoch_%d.pth' % (opt.outf, epoch))

def generate_samples(batch, distributions, num_samples):
    samples = []
    sample_indices = []

    sample_batch = {
        "questions":[],
        "contexts":[],
        "choices":[],
        "answers":[],
        "probabilities":[],
        "wrongs":[],
        "size": batch["size"]*num_samples
    }

    for i, data in enumerate(batch["choices"]):
        sample_batch["questions"].extend([batch["questions"][i]]*num_samples)
        sample_batch["contexts"].extend([batch["contexts"][i]]*num_samples)
        sample_batch["choices"].extend([batch["choices"][i]]*num_samples)
        answers = [a.item() for a in list(torch.multinomial(distributions[i], num_samples, replacement=True))]
        # print("distributions:", distributions.shape, "answers:", answers)
        probabilities = distributions[i][answers]
        sample_batch["answers"].extend(answers)
        sample_batch["probabilities"].extend(probabilities)
        sample_batch["wrongs"].extend([batch["wrongs"][i]]*num_samples)

        # samples.append(torch.tensor(data[local_sample_indices]))
        # sample_indices.append(local_sample_indices)

    # samples = torch.stack((*samples,)).to(device)

    sample_batch["questions"] = torch.stack((*sample_batch["questions"],)).to(device)
    sample_batch["contexts"] = torch.stack((*sample_batch["contexts"],)).to(device)
    sample_batch["choices"] = torch.stack((*sample_batch["choices"],)).to(device)
    sample_batch["probabilities"] = torch.tensor(sample_batch["probabilities"]).to(device)

    return sample_batch

def training():
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))

    print(netG)

    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))

    print(netD)


    for epoch in tqdm(range(opt.niter)):
        netG.train()
        for g in range(opt.g_epochs):
            for i, batch in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
                netG.zero_grad()
                
                g_logits, distributions = netG(batch) # context + question + choice_list ==> probability distribution over choice_list
                
                sample_batch = generate_samples(batch, distributions, opt.num_samples)

                d_sample_logits, d_probabilities = netD(sample_batch)

                g_sample_probs = sample_batch["probabilities"]
                
                loss = torch.mean(
                    torch.log(g_sample_probs) * nn.functional.logsigmoid(d_sample_logits)
                )
                loss.backward()
                optimizerG.step()

                neg_labels = torch.full((sample_batch["size"],), 0).to(device)
                expected_outputs = torch.tensor(batch["answers"]).to(device)

                print(
                    "\nTraining generator Epoch: {}, batch_num: {}, generator_loss: {} \
                    \nd_precision: {}, d_recall: {}, d_fscore: {} \
                    \ng_correct_answers: {}/{}"
                    .format(
                        epoch, i, loss,
                        *fscore(d_probabilities, neg_labels)
                        score_gen(distributions, expected_outputs), batch["size"]
                    )
                )

        eval_netG(epoch)

        torch.save(netG.state_dict(), '%s/netG_train_epoch_%d.pth' % (opt.outf, epoch))

        netD.train()
        for d in range(opt.d_epochs):
            for batch in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
                netD.zero_grad()
                
                """
                Here, we need to generate N samples from the choice list for every question and pass
                them through the discriminator. We want the discriminator to output 0 for these images.
                We will also pass the true answers through the discriminator which must output 1 for them.
                """
                g_logits, distributions = netG(batch)
                sample_batch = generate_samples(batch, distributions, opt.num_samples)
                
                neg_labels = torch.full((sample_batch["size"],), 0).to(device)
                sample_logits, sample_probs = netD(sample_batch)
                neg_loss = criterionD(sample_logits, neg_labels)
                
                batch_labels = torch.full((batch["size"],), 1).to(device)
                batch_logits, batch_probs = netD(batch)
                batch_loss = criterionD(batch_logits, batch_labels)

                total_loss = neg_loss + batch_loss
                total_loss.backward()
                optimizerD.step()

                all_labels = torch.stack((neg_labels, batch_labels))
                d_probabilities = torch.stack((sample_probs, batch_probs))
                expected_outputs = torch.tensor(batch["answers"]).to(device)

                print(
                    "\nTraining discriminator Epoch: {}, batch_num: {}, discriminator_loss: {} \
                    \nd_precision: {}, d_recall: {}, d_fscore: {} \
                    \ng_correct_answers: {}/{}"
                    .format(
                        epoch, i, loss,
                        *fscore(d_probabilities, all_labels)
                        score_gen(distributions, expected_outputs), batch["size"]
                    )
                )

        eval_netD(epoch)

        torch.save(netD.state_dict(), '%s/netD_train_epoch_%d.pth' % (opt.outf, epoch))

if __name__ == "__main__":
    
    if opt.pretrain:
        pre_train(True, True)
    
    if opt.train:
        training()

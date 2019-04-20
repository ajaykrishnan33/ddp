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
parser.add_argument('--start_iter', type=int, default=0, help='epoch to start from')
parser.add_argument('--g_epochs', type=int, default=1, help='number of epochs to train Generator for')
parser.add_argument('--d_epochs', type=int, default=1, help='number of epochs to train Discriminator for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--vgg_pretrained', action='store_true', help='vgg pretrained?', default=False)
# parser.add_argument('--pretrain', action='store_true', help='pretrain?', default=False)
# parser.add_argument('--train', action='store_true', help='train?', default=False)
parser.add_argument("--mode", required=True, choices=["pretrain", "train", "test"])

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

train_dataset = dataservices.RecipeQADataset(
    "recipeqa/new_train_cleaned.json", 
    "recipeqa/features/train"
)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=opt.batchSize,
    shuffle=True, num_workers=int(opt.workers),
    collate_fn=dataservices.batch_collator(device=device)
)

val_dataset = dataservices.RecipeQADataset(
    "recipeqa/new_val_cleaned.json", 
    "recipeqa/features/val",
    "recipeqa/features/train"   
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=opt.batchSize,
    shuffle=False, num_workers=int(opt.workers),
    collate_fn=dataservices.batch_collator(device=device)
)

testG_dataset = dataservices.RecipeQATestGDataset(
    "recipeqa/new_val_cleaned.json", 
    "recipeqa/features/val",
    "recipeqa/features/train"   
)
testG_dataloader = torch.utils.data.DataLoader(
    testG_dataset, batch_size=opt.batchSize,
    shuffle=False, num_workers=int(opt.workers),
    collate_fn=dataservices.batch_collator(device=device)
)

testD_dataset = dataservices.RecipeQATestDDataset(
    "recipeqa/new_val_cleaned.json", 
    "recipeqa/features/val",
    "recipeqa/features/train"   
)
testD_dataloader = torch.utils.data.DataLoader(
    testD_dataset, batch_size=dataservices.NUM_CHOICES*dataservices.NUM_CHOICES,
    shuffle=False, num_workers=int(opt.workers),
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

    print("\nTrue positives: {}, False positives: {}, False negatives: {}".format(true_positives, false_positives, false_negatives))

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
                "\nEval:: training discriminator Epoch: {}, batch_num: {}, loss: {}, precision: {}, recall: {}, fscore: {}"
                .format(
                    epoch, i, loss,
                    *fscore(probabilities, labels)
                )
            )

    netD.train()

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
                "\nEval:: training generator Epoch: {}, batch_num: {}, loss: {}, correct_answers: {}"
                .format(
                    epoch, i, loss,
                    score_gen(distributions, expected_outputs)
                )
            )
    netG.train()

def pre_train(train_netD, train_netG):

    if train_netD:
        for epoch in tqdm(range(opt.pre_niter)):
            for i, batch in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
                netD.zero_grad()
                logits, probabilities = netD(batch)
                # labels = torch.full((batch["size"],), 1).to(device)
                labels = batch["d_answers"].to(device)

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


    for epoch in tqdm(range(opt.start_iter, opt.niter)):
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
                    \nd_correct_answers: {}/{} \
                    \ng_correct_answers: {}/{}"
                    .format(
                        epoch, i, loss,
                        (torch.round(d_probabilities)==neg_labels).sum().item(), sample_batch["size"],
                        score_gen(distributions, expected_outputs), batch["size"]
                    )
                )

        eval_netG(epoch)

        torch.save(netG.state_dict(), '%s/netG_train_epoch_%d.pth' % (opt.outf, epoch))

        for d in range(opt.d_epochs):
            for i, batch in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader)):
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

                all_labels = torch.cat((neg_labels, batch_labels))
                d_probabilities = torch.cat((sample_probs, batch_probs))
                expected_outputs = torch.tensor(batch["answers"]).to(device)

                print(
                    "\nTraining discriminator Epoch: {}, batch_num: {}, neg_loss: {}, batch_loss: {}, total_loss: {} \
                    \nd_precision: {}, d_recall: {}, d_fscore: {} \
                    \ng_correct_answers: {}/{}"
                    .format(
                        epoch, i, neg_loss, batch_loss, total_loss,
                        *fscore(d_probabilities, all_labels),
                        score_gen(distributions, expected_outputs), batch["size"]
                    )
                )

        eval_netD(epoch)

        torch.save(netD.state_dict(), '%s/netD_train_epoch_%d.pth' % (opt.outf, epoch))

def test_netG():
    netG.apply(weights_init)
    netG.load_state_dict(torch.load(opt.netG))

    print(netG)

    netG.eval()
    print("\nTesting netG:")
    correct_answers = 0
    for i, batch in tqdm(enumerate(testG_dataloader, 0), total=len(testG_dataloader)):
        with torch.no_grad():
            logits, distributions = netG(batch)
            expected_outputs = torch.tensor(batch["answers"]).to(device)
            correct_answers += score_gen(distributions, expected_outputs)

    print("Correctly answered/Total Questions:{}/{}".format(correct_answers, len(testG_dataloader)))
    print("Percentage:{}".format(correct_answers/len(testG_dataloader)*100.0))

    

def test_netD():
    netD.apply(weights_init)
    netD.load_state_dict(torch.load(opt.netD))

    print(netD)

    netD.eval()
    print("\nTesting netD:")
    correct_answers = 0

    total_qs = len(testD_dataloader)/(dataservices.NUM_CHOICES**2)

    for i, batch in tqdm(enumerate(testD_dataloader, 0), total=len(testD_dataloader)):
        votes = np.array([0.0]*dataservices.NUM_CHOICES)
        actual_answer = batch["real_answer"]        
        with torch.no_grad():
            logits, probabilities = netD(batch)

            for j, p in enumerate(probabilities):
                a_id = j//dataservices.NUM_CHOICES
                votes[a_id] += p

        answer = votes.argmax()

        if answer==actual_answer:
            correct_answers+=1
    
    print("Correctly answered/Total Questions:{}/{}".format(correct_answers, total_qs))
    print("Percentage:{}".format(correct_answers/total_qs*100.0))


if __name__ == "__main__":
    
    if opt.mode == "pretrain":
        pre_train(True, True)
    elif opt.mode == "train":
        training()
    elif opt.mode == "test":
        test_netG()
        test_netD()
    else:
        print("Error!")


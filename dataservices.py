import os
import torch
import torchvision
import torch.nn as nn
from   torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import ujson
from skimage import io, transform
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def load_vocabulary():
    with open("recipeqa/vocab_clean.txt", "r") as f:
        vocab = {}
        for i, w in enumerate(f):
            vocab[str(w).strip()] = i
        return vocab

class Doc2Vec:
    def __init__(self, embeddings_file_path):
        self.embeddings = np.loadtxt(embeddings_file_path, delimiter=",", skiprows=1)

    def get_vectors(self, indices):
        return torch.from_numpy(self.embeddings[indices]).to(torch.float)

class RecipeQADataset(Dataset):
    
    def __init__(self, csv_file, root_dir, embeddings, transform=None):
        self.data_list = ujson.load(open(csv_file, "r"))
        self.root_dir = root_dir
        self.transform = transform
        self.embeddings = embeddings
        # self.vocab = load_vocabulary()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ret = {}
        data_item = self.data_list[idx]

        # ret["context"] = [
        #     [
        #         self.vocab[word] for word in c_item["cleaned_body"].split()
        #     ] for c_item in data_item["context"]
        # ]

        document_ids = [
            data_item["context_base_id"] + i for i, _ in enumerate(data_item["context"])
        ]

        ret["context"] = self.embeddings.get_vectors(document_ids)

        ret["choice_list"] = [
            io.imread(os.path.join(self.root_dir, img)) for img in data_item["choice_list"]
        ]

        ret["question"] = [
            io.imread(os.path.join(self.root_dir, img)) for img in data_item["question"]
        ]

        ret["answer"] = data_item["answer"]

        if self.transform:
            ret = self.transform(ret)

        return ret

class RescaleToTensorAndNormalize(object):

    def __init__(self, output_size):
        self.output_size = output_size

    def process_img_list(self, img_list):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        final_img_list = []
        for img in img_list:
            img = transform.resize(img, (self.output_size, self.output_size))
            try:
                img = img.transpose((2,0,1))
            except Exception as e:
                if img.shape==(self.output_size, self.output_size):
                    print("Greyscale instead of color")
                    img = np.stack((img,)*3, axis=0)
                else:
                    print("Error:", img.shape)
                    raise e
            img = torch.from_numpy(img)
            img = normalize(img)
            final_img_list.append(img)

        return torch.stack((*final_img_list,)).to(torch.float)

    def __call__(self, sample):

        sample["choice_list"] = self.process_img_list(sample["choice_list"])

        sample["question"] = self.process_img_list(sample["question"])

        return sample

def batch_collator(device):
    
    def _internal(batch):

        questions = nn.utils.rnn.pad_sequence(
            [ x["question"] for x in batch ],
            batch_first = True
        ).to(device)  # batch of question arrays

        contexts = nn.utils.rnn.pad_sequence(
            [ x["context"] for x in batch ],
            batch_first = True
        ).to(device) # batch of context vector sets

        choices = nn.utils.rnn.pad_sequence(
            [ x["choice_list"] for x in batch ],
            batch_first = True
        ).to(device)  # batch of "choice_list"s

        answers = torch.full((len(batch), choices.size(1)), 0).to(device)
        answer_indices = [ x["answer"] for x in batch ]  # batch of "answers"s
        answers[range(len(batch)), answer_indices] = 1  

        final_batch = {
            "questions": questions,
            "contexts": contexts,
            "choices": choices,
            "answers": answers,
            "size": len(batch),
        }

        return final_batch

    return _internal


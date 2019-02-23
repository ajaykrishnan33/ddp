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

NUM_CHOICES = 4

def load_vocabulary():
    with open("recipeqa/vocab_clean.txt", "r") as f:
        vocab = {}
        for i, w in enumerate(f):
            vocab[str(w).strip()] = i
        return vocab

class Doc2Vec:
    def __init__(self, embeddings_file_path):
        self.embeddings = np.loadtxt(embeddings_file_path, delimiter=",")

    def get_vectors(self, indices):
        return torch.from_numpy(self.embeddings[indices]).to(torch.float)

class RecipeQADataset(Dataset):
    
    def __init__(self, csv_file, root_dir, embeddings):
        self.data_list = ujson.load(open(csv_file, "r"))
        self.root_dir = root_dir
        self.embeddings = embeddings
        # self.vocab = load_vocabulary()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        ret = {}

        q_id = idx/(NUM_CHOICES-1)
        a_id = idx%(NUM_CHOICES-1)

        data_item = self.data_list[q_id]

        # ret["context"] = [
        #     [
        #         self.vocab[word] for word in c_item["cleaned_body"].split()
        #     ] for c_item in data_item["context"]
        # ]

        document_ids = [
            data_item["context_base_id"] + i for i, _ in enumerate(data_item["context"])
        ]

        ret["context"] = self.embeddings.get_vectors(document_ids)

        ret["choice_list"] = torch.stack((*[
            # io.imread(os.path.join(self.root_dir, img)) for img in data_item["choice_list"]
            torch.load(os.path.join(self.root_dir, img)) for img in data_item["choice_list"]
        ],))

        ret["question"] = torch.stack((*[
            # io.imread(os.path.join(self.root_dir, img)) for img in data_item["question"]
            torch.load(os.path.join(self.root_dir, img)) for img in data_item["question"]
        ],))

        ret["answer"] = data_item["answer"]
        ret["wrong"] = a_id

        return ret

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

        answers = [x["answer"] for x in batch]
        wrongs = [x["wrong"] for x in batch]

        final_batch = {
            "questions": questions,
            "contexts": contexts,
            "choices": choices,
            "answers": answers,
            "wrongs": wrongs,
            "size": len(batch),
        }

        return final_batch

    return _internal


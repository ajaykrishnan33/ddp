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

NUM_CHOICES = 20

class Vocabulary:
    def __init__():
        with open("recipeqa/vocab_clean.txt", "r") as f:
            vocab = {}
            for i, w in enumerate(f):
                vocab[str(w).strip()] = i
            self._vocab = vocab

    def get_index(word):
        if word in self._vocab:
            return self._vocab[word]
        else:
            return self.vocab["<UNKNOWN>"]

vocabulary = Vocabulary()

class RecipeQADataset(Dataset):
    
    def __init__(self, csv_file, root_dir):
        self.data_list = ujson.load(open(csv_file, "r"))
        self.root_dir = root_dir
        # self.vocab = load_vocabulary()

    def __len__(self):
        return len(self.data_list)*(NUM_CHOICES-1)

    def __getitem__(self, idx):
        ret = {}

        q_id = idx//(NUM_CHOICES-1)
        a_id = idx%(NUM_CHOICES-1)

        data_item = self.data_list[q_id]

        # ret["context"] = [
        #     [
        #         self.vocab[word] for word in c_item["cleaned_body"].split()
        #     ] for c_item in data_item["context"]
        # ]

        sentences = [
            [ vocabulary.get_index(word) for word in sentence ]
            for sentence in data_item["context"]            
        ]

        ret["context"] = sentences

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

        # contexts_temp = [ x["context"] for x in batch ]

        contexts_temp = []
        batch_size = len(batch)
        reverse_map = {}
        ct = 0

        for i, data_item in enumerate(batch):
            for sentence in data_item["context"]:  # list of index lists
                reverse_map[ct] = i
                contexts_temp.append(sentence)
                ct += 1

        contexts_temp = nn.utils.rnn.pad_sequence(
            contexts_temp,
            batch_first = True
        ).to(device)

        singly_padded_contexts = [[] for i in range(batch_size)]
        for i, sentence in enumerate(contexts_temp):
            singly_padded_contexts[reverse_map[i]].append(sentence)

        for i in range(len(singly_padded_contexts)):
            singly_padded_contexts[i] = torch.stack((*singly_padded_contexts[i],)).to(device)

        doubly_padded_contexts = nn.utils.rnn.pad_sequence(
            singly_padded_contexts,
            batch_first = True
        ).to(device)

        choices = nn.utils.rnn.pad_sequence(
            [ x["choice_list"] for x in batch ],
            batch_first = True
        ).to(device)  # batch of "choice_list"s

        answers = [x["answer"] for x in batch]
        wrongs = [x["wrong"] for x in batch]

        final_batch = {
            "questions": questions,
            "contexts": doubly_padded_contexts,
            "choices": choices,
            "answers": answers,
            "wrongs": wrongs,
            "size": len(batch),
        }

        return final_batch

    return _internal


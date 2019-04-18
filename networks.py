import torch
import torchvision
import torch.nn as nn

VOCAB_SIZE = 9273

class BaseNetwork(nn.Module):
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()

        ## IMAGE AUTO ENCODER BEGINS ##
        self.img_compressor = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Linear(1024, 300)
        )

        self.img_expander = nn.Sequential(
            nn.Linear(300, 1024),
            nn.Linear(1024, 4096)
        )

        ## IMAGE AUTO ENCODER ENDS ##

        self.question_encoder = nn.GRU(  # for encoding the 4 question images together
            input_size=300,   # img_compressor output size 
            hidden_size=300,
            num_layers=1,
            batch_first=True
        )

        self.word_encoder = nn.Embedding(VOCAB_SIZE, 300)

        self.sentence_encoder = nn.GRU( # for encoding the context v
            input_size=300,
            hidden_size=300,
            num_layers=1,
            batch_first=True
        )

        self.context_encoder = nn.GRU(   # for encoding the context vectors together
            input_size=300,    # sentence encoder embedding size
            hidden_size=300,
            num_layers=1,
            batch_first=True
        )

        # the concatenation of the question and context vectors is transformed to size 100.
        self.combined_encoder = nn.Linear(600, 100) 


    def encode_questions_and_contexts(self, input_data):
        questions = input_data["questions"]  # batch of question arrays

        questions_temp = questions.view(-1, *questions.shape[2:])

        # encoded_questions_temp = self.question_img_encoder(questions_temp)

        encoded_questions_temp_compressed = self.img_compressor(questions_temp)

        # will use this for autoencoder loss by comparing against encoded_questions_temp
        # encoded_questions_temp_expanded = self.img_expander(encoded_questions_temp_compressed)

        encoded_questions_single = encoded_questions_temp_compressed.view(
            *questions.shape[:2], 
            *encoded_questions_temp_compressed.shape[1:]
        )

        _, encoded_questions_seq = self.question_encoder(encoded_questions_single)  # finally a list of vectors

        encoded_questions_seq = torch.squeeze(encoded_questions_seq.transpose(0,1), dim=1)

        contexts = input_data["contexts"]  # (batch_size, max_ct, max_length, vocab_size)

        indices_temp = contexts.view(-1, *indices_temp.shape[2:]) # (batch_size*max_ct, max_length, vocab_size)

        sentences_temp = self.word_encoder(indices_temp)

        _, encoded_sentences_temp = self.sentence_encoder(sentences_temp)   # finally a list of vectors

        encoded_sentences_temp = torch.squeeze(encoded_sentences_temp.transpose(0,1), dim=1) 
        # (batch_size*max_ct, 300)

        encoded_sentences_temp = encoded_sentences_temp.view(
            contexts.shape[0],
            -1, encoded_sentences_temp.shape[-1]
        ) # (batch_size, max_ct, 300)

        _, encoded_contexts_temp = self.context_encoder(encoded_sentences_temp) # (1, batch_size, 300)

        encoded_contexts = torch.squeeze(encoded_contexts_temp.transpose(0,1), dim=1) # (batch_size, 300)

        encoded_questions_and_contexts_temp = torch.cat((encoded_questions_seq, encoded_contexts), 1)

        encoded_questions_and_contexts = self.combined_encoder(encoded_questions_and_contexts_temp)

        return encoded_questions_and_contexts


class Generator(BaseNetwork):

    def encode_choices(self, input_data, encoded_questions_and_contexts):
        choices = input_data["choices"]  # batch of "choice_list"s

        # choices_temp = choices.view(-1, *choices.shape[2:])

        # encoded_choices_temp = []
        # for choice_list in choices:
        #     encoded_choices_temp.append(self.choice_img_encoder(choice_list))

        # encoded_choices_temp = torch.stack((*encoded_choices_temp,))

        # using a set of weights different from question images for choice images
        # encoded_choices_temp = self.choice_img_encoder(choices_temp)  # a list with (batch_size * choice_list_size) number of vectors

        # re-using same compressor as used for images.
        # encoded_choices_temp_compressed = self.img_compressor(encoded_choices_temp)

        relevance_logit_list = []
        for i, choice_list in enumerate(choices):
            relevance_logits = []
            for choice in choice_list:
                encoded_choice = self.img_compressor(
                    # self.choice_img_encoder(
                    #     choice.unsqueeze(dim=0)
                    # )
                    choice.unsqueeze(dim=0)
                )

                relevance = nn.functional.cosine_similarity(
                    encoded_questions_and_contexts[i].unsqueeze(dim=0), 
                    encoded_choice
                ).squeeze(dim=-1)

                relevance_logits.append(relevance)
            relevance_logits = torch.stack((*relevance_logits,), dim=0)
            relevance_logit_list.append(relevance_logits)

        relevance_logits = torch.stack((*relevance_logit_list,), dim=0)


        # will be used for autoencoder loss by comparing against encoded_choices_temp
        # encoded_choices_temp_expanded = self.img_expander(encoded_choices_temp_compressed) 

        # encoded_choices_temp_compressed = torch.stack((*encoded_choices_temp_compressed,))

        return relevance_logits

    def forward(self, input_data):
        
        encoded_questions_and_contexts = self.encode_questions_and_contexts(input_data)

        relevance_logits = self.encode_choices(input_data, encoded_questions_and_contexts)

        relevance_distributions = nn.functional.softmax(relevance_logits, dim=1)

        return relevance_logits, relevance_distributions



class Discriminator(BaseNetwork):

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def compute_score(self, img_list, encoded_questions_and_contexts):

        encoded_imgs = self.img_compressor(img_list)

        score = nn.functional.cosine_similarity(
            encoded_questions_and_contexts, 
            encoded_imgs
        )

        return score

    def forward(self, input_data):

        encoded_questions_and_contexts = self.encode_questions_and_contexts(input_data)        

        score_right = self.compute_score(
            input_data["choices"][range(input_data["size"]), input_data["answers"]], 
            encoded_questions_and_contexts
        )

        score_wrong = self.compute_score(
            input_data["choices"][range(input_data["size"]), input_data["wrongs"]], 
            encoded_questions_and_contexts
        )

        logits = score_right - score_wrong

        probabilities = self.sigmoid(logits)
        # loss = ((1-z)>0)*(1-z)

        return logits, probabilities

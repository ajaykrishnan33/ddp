import torch
import torchvision
import torch.nn as nn

class BaseNetwork(nn.Module):
    def __init__(self, opt):
        super(BaseNetwork, self).__init__()

        img_encoder = torchvision.models.vgg16_bn(pretrained=opt.vgg_pretrained)

        img_encoder.classifier = nn.Sequential(*list(img_encoder.classifier)[:4]) 

        self.question_img_encoder = img_encoder   # final size will be 4096

        ## IMAGE AUTO ENCODER BEGINS ##
        self.img_compressor = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.Linear(1024, 100)
        )

        self.img_expander = nn.Sequential(
            nn.Linear(100, 1024),
            nn.Linear(1024, 4096)
        )

        ## IMAGE AUTO ENCODER ENDS ##

        self.question_encoder = nn.GRU(  # for encoding the 4 question images together
            input_size=100,   # img_compressor output size 
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        self.context_encoder = nn.GRU(   # for encoding the context vectors together
            input_size=100,    # doc2vec embedding size
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        # the concatenation of the question and context vectors is transformed to size 100.
        self.combined_encoder = nn.Linear(200, 100) 

        # the outputs of the two encoders above will be concatenated together

        # img_encoder = torchvision.models.vgg16_bn(pretrained=opt.vgg_pretrained)

        # img_encoder.classifier = nn.Sequential(*list(img_encoder.classifier)[:4]) 

        self.choice_img_encoder = img_encoder

    def encode_questions_and_contexts(self, input_data):
        questions = input_data["questions"]  # batch of question arrays

        questions_temp = questions.view(-1, *questions.shape[2:])

        encoded_questions_temp = self.question_img_encoder(questions_temp)

        encoded_questions_temp_compressed = self.img_compressor(encoded_questions_temp)

        # will use this for autoencoder loss by comparing against encoded_questions_temp
        encoded_questions_temp_expanded = self.img_expander(encoded_questions_temp_compressed)

        encoded_questions_single = encoded_questions_temp_compressed.view(
            *questions.shape[:2], 
            *encoded_questions_temp_compressed.shape[1:]
        )

        _, encoded_questions_seq = self.question_encoder(encoded_questions_single)  # finally a list of vectors

        encoded_questions_seq = torch.squeeze(encoded_questions_seq.transpose(0,1), dim=1)

        contexts = input_data["contexts"]  # batch of context vector sets

        _, encoded_contexts = self.context_encoder(contexts)   # finally a list of vectors

        encoded_contexts = torch.squeeze(encoded_contexts.transpose(0,1), dim=1)

        encoded_questions_and_contexts_temp = torch.cat((encoded_questions_seq, encoded_contexts), 1)

        encoded_questions_and_contexts = self.combined_encoder(encoded_questions_and_contexts_temp)

        return encoded_questions_and_contexts

    def encode_choices(self, input_data):
        choices = input_data["choices"]  # batch of "choice_list"s

        choices_temp = choices.view(-1, *choices.shape[2:])

        # encoded_choices_temp = []
        # for choice_list in choices:
        #     encoded_choices_temp.append(self.choice_img_encoder(choice_list))

        # encoded_choices_temp = torch.stack((*encoded_choices_temp,))

        # using a set of weights different from question images for choice images
        # encoded_choices_temp = self.choice_img_encoder(choices_temp)  # a list with (batch_size * choice_list_size) number of vectors

        # re-using same compressor as used for images.
        # encoded_choices_temp_compressed = self.img_compressor(encoded_choices_temp)

        encoded_choices_temp_compressed = []
        for choice in choices_temp:
            encoded_choices_temp_compressed.append(
                self.img_compressor(
                    self.choice_img_encoder(
                        choice.unsqueeze(dim=0)
                    )
                )
            )

        # will be used for autoencoder loss by comparing against encoded_choices_temp
        # encoded_choices_temp_expanded = self.img_expander(encoded_choices_temp_compressed) 

        encoded_choices_temp_compressed = torch.stack((*encoded_choices_temp_compressed,))
        print("encoded_choices_temp_compressed", encoded_choices_temp_compressed.shape)

        return torch.stack((*encoded_choices_temp_compressed,))


class Generator(BaseNetwork):

    def forward(self, input_data):
        
        encoded_questions_and_contexts = self.encode_questions_and_contexts(input_data)

        print("encoded_questions_and_contexts", encoded_questions_and_contexts.shape)

        encoded_choices_temp_compressed = self.encode_choices(input_data)

        # encoded_choices = encoded_choices_temp.view(
        #     *choices.shape[:2],
        #     *encoded_choices_temp.shape[1:]
        # )

        relevance_temp = nn.functional.cosine_similarity(
            encoded_questions_and_contexts, 
            encoded_choices_temp_compressed, 
            dim=1
        )

        relevance_logits = relevance_temp.view(
            *input_data["choices"].shape[:2],
            *relevance_temp.shape[1:]
        )           # list of vectors - one for each entry in the batch

        relevance_distributions = nn.functional.softmax(relevance_logits, dim=1)

        return relevance_logits, relevance_distributions



class Discriminator(BaseNetwork):

    def __init__(self, *args, **kwargs):
        super(Discriminator, self).__init__(*args, **kwargs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_data):

        encoded_questions_and_contexts = self.encode_questions_and_contexts(input_data)
        
        encoded_choices_temp_compressed = self.encode_choices(input_data)

        # encoded_choices = encoded_choices_temp.view(
        #     *choices.shape[:2],
        #     *encoded_choices_temp.shape[1:]
        # )

        # print()

        relevance_temp = nn.functional.cosine_similarity(
            encoded_questions_and_contexts, 
            encoded_choices_temp_compressed, 
            dim=1
        )

        relevance_logits = relevance_temp.view(
            *input_data["choices"].shape[:2],
            *relevance_temp.shape[1:]
        )           # list of vectors - one for each entry in the batch

        relevance_probabilities = self.sigmoid(relevance_logits)

        return relevance_logits, relevance_probabilities

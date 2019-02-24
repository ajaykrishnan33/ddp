import ujson
import numpy as np

RANDOM_SEED = 1234

np.random.seed(RANDOM_SEED)

MAX_CHOICES = 20

train_data = ujson.load(open("recipeqa/train.json", "rb"))
val_data = ujson.load(open("recipeqa/val.json", "rb"))

vcloze_train_data = [x for x in train_data['data'] if x['task']=="visual_cloze"]
vcloze_val_data = [x for x in val_data['data'] if x['task']=="visual_cloze"]

# print(ujson.dumps(vcloze_train_data[1], indent=4))

full_val_image_list = []
full_train_image_list = []

for data in vcloze_train_data:
	full_train_image_list.extend(data['choice_list'])
	full_train_image_list.extend(data['question'])

for data in vcloze_val_data:
	full_val_image_list.extend(data['choice_list'])
	full_val_image_list.extend(data['question'])

def set_remove(x, y):
	z = set(x)
	z.remove(y)
	return z

full_train_image_set = set_remove(full_train_image_list, "@placeholder")
full_val_image_set = set_remove(full_val_image_list, "@placeholder")

full_val_image_list = list(full_val_image_set)
full_train_image_list = list(full_train_image_set)

reverse_index_mapping = {True:{}, False:{}}

for i, img in enumerate(full_train_image_list):
	reverse_index_mapping[True][img] = i

for i, img in enumerate(full_val_image_list):
	reverse_index_mapping[False][img] = i

# print(len(full_image_list), len(vcloze_data))

def extend_choices_and_shuffle(choice_list, question, answer, train):
	image_list = full_val_image_list
	if train:
		image_list = full_train_image_list

	max_img_ct = len(image_list) - len(question)
	prob_dist = [1.0/max_img_ct]*len(image_list)

	# print(len(choice_list))

	for c in choice_list:
		prob_dist[reverse_index_mapping[train][c]] = 0

	for q in question:
		prob_dist[reverse_index_mapping[train][q]] = 0

	prob_dist[reverse_index_mapping[train][answer]] = 0

	choice_list.append(answer)

	random_images = np.random.choice(image_list, MAX_CHOICES - len(choice_list), prob_dist)

	choice_list.extend(random_images)

	np.random.shuffle(choice_list)

	
def modify_dataset(train):

	vcloze_data = vcloze_val_data
	if train:
		vcloze_data = vcloze_train_data

	for data in vcloze_data:
		placeholder = data["question"].index("@placeholder")

		# replace placeholder with correct answer and select new answer
		data["question"][placeholder] = data["choice_list"][data["answer"]]
		new_answer = data["question"][-1]
		data["question"] = data["question"][:-1]

		# remove correct answer and fill with random images till MAX_CHOICES 
		new_choice_list = data["choice_list"][ : data["answer"]] + (data["choice_list"][data["answer"] + 1 : ])
		extend_choices_and_shuffle(new_choice_list, data["question"], new_answer, train)

		answer_index = new_choice_list.index(new_answer)
		data["answer"] = answer_index
		data["choice_list"] = new_choice_list

modify_dataset(train=False)
modify_dataset(train=True)

ujson.dump(vcloze_train_data, open("recipeqa/new_train.json", "w"), indent=4)
ujson.dump(vcloze_val_data, open("recipeqa/new_val.json", "w"), indent=4)

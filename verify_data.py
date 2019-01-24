import ujson

MAX_CHOICES = 20

train_data = ujson.load(open("recipeqa/train.json", "rb"))
val_data = ujson.load(open("recipeqa/val.json", "rb"))

old_train_data = [x for x in train_data['data'] if x['task']=="visual_cloze"]
old_val_data = [x for x in val_data['data'] if x['task']=="visual_cloze"]

new_train_data = ujson.load(open("recipeqa/new_train.json", "rb"))
new_val_data = ujson.load(open("recipeqa/new_val.json", "rb"))

def verify_data(train):
	
	new_data = new_val_data
	old_data = old_val_data

	if train:
		new_data = new_train_data
		old_data = old_train_data

	for i, new_d in enumerate(new_data):
		old_d = old_data[i]
		try:
			old_d['question'].index("@placeholder")
		except ValueError:
			pass

		exp_new_answer = old_d["question"][-1] if old_d["question"][-1]!="@placeholder" else old_d["choice_list"][old_d["answer"]]

		assert(exp_new_answer == new_d["choice_list"][new_d["answer"]])

		assert(len(new_d["choice_list"]) == MAX_CHOICES)

		assert(len(new_d["question"]) == len(old_d["question"])-1)
		# print(i, train)


verify_data(train=False)
verify_data(train=True)



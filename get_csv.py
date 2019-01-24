import ujson

train_data = ujson.load(open("recipeqa/new_train_cleaned.json", "r"))
# val_data = ujson.load(open("recipeqa/new_val_cleaned.json", "r"))

with open("sentences.csv", "w") as f:
	f.write("text\n")

	for data in train_data:
		for c in data["context"]:
			f.write("\"" + c["cleaned_body"] + "\"" + "\n")

import ujson

train_data = ujson.load(open("recipeqa/new_train_cleaned.json", "r"))
val_data = ujson.load(open("recipeqa/new_val_cleaned.json", "r"))

def create_csv(dataset, name):
	with open("sentences_{}.csv".format(name), "w") as f:
		f.write("text\n")
		ct = 0
		for data in dataset:
			data["context_base_id"] = ct
			for c in data["context"]:
				f.write("\"" + c["cleaned_body"] + "\"" + "\n")
				ct += 1


create_csv(train_data, "train")
create_csv(val_data, "val")

ujson.dump(train_data, open("recipeqa/new_train_cleaned.json", "w"), indent=4)
ujson.dump(val_data, open("recipeqa/new_val_cleaned.json", "w"), indent=4)

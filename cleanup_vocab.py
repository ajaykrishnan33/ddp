import ujson

word_counts = ujson.load(open("recipeqa/word_counts.json", "r"))

MIN_FREQ = 10

ct = 0

UNKNOWN = "<UNKNOWN>"

with open("recipeqa/vocab_clean.txt", "w") as f:
	f.write(UNKNOWN+"\n")
	for w in word_counts:
		if word_counts[w] >= MIN_FREQ:
			f.write(w + "\n")
			ct += 1

train_data = ujson.load(open("recipeqa/new_train_cleaned.json", "r"))
val_data = ujson.load(open("recipeqa/new_val_cleaned.json", "r"))

unk_cts = {}

def process_dataset(dataset):
	for item in dataset:
		unk_cts[item["qid"]] = 0
		for c in item["context"]:
			words = []
			for w in c["cleaned_body"].split():
				if word_counts[w] < MIN_FREQ:
					words.append(UNKNOWN)
					unk_cts[item["qid"]] += 0
				else:
					words.append(w)
			c["cleaned_body"] = " ".join(words)


process_dataset(train_data)
process_dataset(val_data)

print("Vocab size: " + str(ct))
ujson.dump(train_data, open("recipeqa/new_train_cleaned.json", "w"), indent=4)
ujson.dump(val_data, open("recipeqa/new_val_cleaned.json", "w"), indent=4)
ujson.dump(unk_cts, open("recipeqa/unk_cts.json", "w"), indent=4)


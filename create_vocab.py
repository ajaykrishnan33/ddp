import re
import ujson

train_data = ujson.load(open("recipeqa/new_train.json", "r"))
val_data = ujson.load(open("recipeqa/new_val.json", "r"))

regex0 = re.compile(r'i\.e\.')
regex1 = re.compile(r'([0-9])\/([0-9])')
regex2 = re.compile(r'\/')
regex3 = re.compile(r'[^\w\s]')

word_counts = {}

def process_string_into_words(sentence):
	# first make everything lowercase
	sentence = sentence.lower()
	# replacing i.e. with ie
	sentence = regex0.sub('ie', sentence)
	# replacing 3/4 with 3by4
	sentence = regex1.sub(r'\1by\2', sentence)
	# replacing cat/dog with cat or dog
	sentence = regex2.sub(r' or ', sentence)
	# replacing all other punctuations with spaces
	sentence = regex3.sub(' ', sentence)
	# splitting the sentence by whitespaces
	ws = sentence.split()

	for w in ws:
		if w in word_counts:
			word_counts[w] += 1
		else:
			word_counts[w] = 1

	return ws


words = []

def process_dataset(dataset):
	for item in dataset:
		for c in item["context"]:
			ws = process_string_into_words(c["body"])
			c["cleaned_body"] = " ".join(ws)
			words.extend(ws)


process_dataset(train_data)
process_dataset(val_data)

ujson.dump(train_data, open("recipeqa/new_train_cleaned.json", "w"), indent=4)
ujson.dump(val_data, open("recipeqa/new_val_cleaned.json", "w"), indent=4)

words_set = set(words)

print("Vocabulary size: " + str(len(words_set)))

with open("recipeqa/vocab.txt", "w") as f:
	for w in words_set:
		f.write(w+"\n")

ujson.dump(word_counts, open("recipeqa/word_counts.json", "w"), indent=4)

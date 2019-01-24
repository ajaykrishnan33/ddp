import ujson

word_counts = ujson.load(open("recipeqa/word_counts.json", "r"))

words_with_count = [[] for i in range(10)]

for w in word_counts:
	if word_counts[w]<=10:
		words_with_count[word_counts[w]-1].append(w)



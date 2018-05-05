import json
import sys

glossing_dict = {}
threshold = 0.3

phrase_table = sys.argv[1]
output_glossing = sys.argv[2]

with open(phrase_table) as f:
	for line in f:
		fields = line.split('|||')
		grouped_phonemes = fields[0].split()
		# only look at single word phonemes
		if len(grouped_phonemes) > 1:
			continue
		target_lan_words = fields[1].split()
		prob = float(fields[2].split()[2])
		# store to glossing dictionary if greater than threshold
		if prob > threshold:
			glossings = fields[3].split()
			for glossing in glossings:
				phoneme_index, target_index = map(int, glossing.split('-'))
				phoneme = grouped_phonemes[phoneme_index]
				target_word = target_lan_words[target_index]
				if phoneme not in glossing_dict:
					glossing_dict[phoneme] = [(target_word, prob)]
				else:
					glossing_dict[phoneme].append((target_word, prob))
# sort the glossings by probability
for key, value in glossing_dict.items():
	my_set = {x[0] for x in value}
	my_sums = [(i,sum(x[1] for x in value if x[0] == i)) for i in my_set]
	glossing_dict[key] = sorted(my_sums, key=lambda x: x[1], reverse=True)


with open(output_glossing, 'w') as f:
	f.write(json.dumps(glossing_dict))









import json
import sys

glossing_dict = {}
threshold = 0.8

phrase_table = sys.argv[1]
output_glossing = sys.argv[2]

with open(phrase_table) as f:
	for line in f:
		fields = line.split('|||')
		grouped_phonemes = fields[0].split()
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
					glossing_dict[phoneme] = set([target_word])
				else:
					glossing_dict[phoneme].add(target_word)
for key, value in glossing_dict.items():
	glossing_dict[key] = list(value)

with open(output_glossing, 'w') as f:
	f.write(json.dumps(glossing_dict))









import re
import string
import sys
def main():
	filename = sys.argv[1]
	delimiter = sys.argv[2]
	strip_punctuation = sys.argv[3]
	
	word_count = 0
	word_set = set()
	regex = re.compile('[%s]' % re.escape(string.punctuation+strip_punctuation))
	

	with open(filename, 'r') as file:	
		for line in file:
			line = regex.sub('', line.lower())
			words = line.strip().split(delimiter)
			for w in words:
				if w != "":
					word_set.add(w)
					word_count +=1
	file.closed
	distinct_words = len(word_set)
	ttr = distinct_words/ word_count
	print ("distinct_words",distinct_words)
	print ("word_count",word_count)
	print ("ttr",ttr)
	#print (word_set)
if __name__ == "__main__":
    main()

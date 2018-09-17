import re
import string
import sys


def main():
    filename, delimiter, strip_punctuation = sys.argv[1:]

    word_count = 0
    word_set = set()
    strip_punctuation = re.compile('[%s]' % re.escape(strip_punctuation +
                                                      string.punctuation +
                                                      '\n'))

    with open(filename, 'r') as file:
        for line in file:
            line = strip_punctuation.sub('', line.lower())
            words = line.split(delimiter)
            for w in words:
                if w != '':
                    word_set.add(w)
                    word_count += 1

    distinct_word_count = len(word_set)
    ttr = distinct_word_count / word_count

    print ('distinct_word_count =', distinct_word_count)
    print ('word_count =', word_count)
    print ('ttr =', ttr)

    #print ('word_set =', word_set)


if __name__ == '__main__':
    main()

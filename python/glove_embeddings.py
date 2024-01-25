import numpy as np
import scipy.spatial

def read_embeddings():
    FILE_NAME = '../data/glove/glove.6B.100d.txt'

    embeddings = {}

    file = open(FILE_NAME, 'r', encoding='utf-8')

    for line in file:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:],
                dtype='float32')

        embeddings[word] = vector

    file.close()

    print('Read %s embeddings.' % len(embeddings))

    return embeddings

def print_n_closest(embeddings, vec0, n):
    word_distances = {}

    for (word, vec1) in embeddings.items():
        distance = scipy.spatial.distance.cosine(vec1, vec0)

        word_distances[distance] = word

    for distance in sorted(word_distances.keys())[:n]:
        word = word_distances[distance]

        print(word + ': %6.3f' % distance)


embeddings = read_embeddings()

lookup_word = 'hello'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings, embeddings[lookup_word], 3)

lookup_word = 'precisely'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings, embeddings[lookup_word], 3)

lookup_word = 'dog'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings, embeddings[lookup_word], 3)

print('Now do some vector math')
lookup_word = 'king'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings, embeddings[lookup_word], 3)

lookup_word = 'king - man + woman'
print('\nWords closest to ' + lookup_word)
vec = embeddings['king'] - embeddings['man'] + embeddings['woman']
print_n_closest(embeddings, vec, 3)

print('\nNow some funny other math stuff')
print('\nThis shows that removing spain from Madrid, its capital, and adding sweden, gives the swedish capital')
lookup_word = 'sweden'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings, embeddings[lookup_word], 3)

lookup_word = 'madrid'
print('\nWords closest to ' + lookup_word)
print_n_closest(embeddings, embeddings[lookup_word], 3)


lookup_word = 'madrid- spain + sweden'
print('\nWords closest to ' + lookup_word)
vec = embeddings['madrid'] - embeddings['spain'] + embeddings['sweden']
print_n_closest(embeddings, vec, 3)



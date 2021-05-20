import numpy as np

words = []
vectors = []
with open("glove840b300d.txt", 'r') as f:
    for line in f:
        values = line.strip().split(' ')
        word = values[0]
        vector = np.asarray(values[1:], "float32")[:300]
        if len(vector.shape) == 1 and vector.shape[0] == 300:
                words.append(word)
                vectors.append(vector)


vectors = np.stack(vectors).astype('float32')

 
with open('glove.word', 'w') as filehandle:
    filehandle.writelines("%s\n" % word for word in words)
 
np.save("word_embeddings.npy", vectors )

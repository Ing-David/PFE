import numpy as np

embeddings_dict = {}
with open("glove840b300d.txt", 'r') as f:
    for line in f:
        values = line.strip().split(' ')
        word = values[0]
        vector = np.asarray(values[1:], "float32")
        embeddings_dict[word] = vector 


words =  list(embeddings_dict.keys())
vectors = np.array([embeddings_dict[word] for word in words])

 
with open('glove.word', 'w') as filehandle:
    filehandle.writelines("%s\n" % word for word in words)
 
np.save("word_embeddings.npy", vectors )

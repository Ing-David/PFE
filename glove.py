import numpy as np
 
glove = np.loadtxt("glove840b300d.txt", dtype='str', comments=None)
words = glove[:, 0]
vectors = glove[:, 1:].astype('float')
 
words=words.tolist()
with open('glove.word', 'w') as filehandle:
    filehandle.writelines("%s\n" % word for word in words)
 
np.save("word_embeddings.npy", vectors )
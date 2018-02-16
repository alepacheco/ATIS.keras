from keras.models import load_model
import data.load
import numpy as np

model = load_model('best_model.h5')

train_set, valid_set, dicts = data.load.atisfull()
w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2ne = {ne2idx[k]:k for k in ne2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}

sentence = "flights from new york to chicago for next friday".split(' ')
encoded_sentence = []
for word in sentence:
    encoded_sentence.append(w2idx[word])

print(encoded_sentence)


pred = model.predict_on_batch(np.array(encoded_sentence))
pred = np.reshape(np.argmax(pred,-1), -1)

print(pred)
for id in pred:
    print(idx2la[id], end=' ')


for i in range(len(sentence)):
    print(sentence[i], idx2la[pred[i]])

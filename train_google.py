import gensim
word2vector = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
vector2word = lambda x: word2vector.most_similar(positive=[x], topn=1)[0][0]

train_set, valid_set, dicts = data.load.atisfull()
word2idx, labels2idx = dicts['words2idx'], dicts['labels2idx']
idx2word  = {w2idx[k]:k for k in w2idx}
idx2label = {labels2idx[k]:k for k in labels2idx}

n_classes = len(idx2la)
n_vocab = len(idx2w)


model = Sequential()
model.add(Embedding(n_vocab,100)) # try without this
model.add(Convolution1D(64,5,border_mode='same', activation='relu'))
model.add(Dropout(0.25))
model.add(GRU(100,return_sequences=True))
model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
model.compile('rmsprop', 'categorical_crossentropy')

train_x, train_ne, train_label = train_set
val_x, val_ne, val_label = valid_set
words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]


### Training
n_epochs = 10

train_f_scores = []
val_f_scores = []
best_val_f1 = 0

for i in range(n_epochs):
    print("Epoch {}".format(i))

    print("Training =>")
    train_pred_label = []
    avgLoss = 0

    bar = progressbar.ProgressBar(max_value=len(train_x))
    for n_batch, sent in bar(enumerate(train_x)):
        label = train_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]

        if sent.shape[1] > 1: #some bug in keras
            loss = model.train_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        train_pred_label.append(pred)

    avgLoss = avgLoss/n_batch

    predword_train = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict = conlleval(predword_train, groundtruth_train, words_train, 'r.txt')
    train_f_scores.append(con_dict['f1'])
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))


    print("Validating =>")

    val_pred_label = []
    avgLoss = 0

    bar = progressbar.ProgressBar(max_value=len(val_x))


    for n_batch, sent in bar(enumerate(val_x)):
        label = val_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]

        if sent.shape[1] > 1: #some bug in keras
            loss = model.test_on_batch(sent, label)
            avgLoss += loss

        print('sent')
        for w in sent[0]:
            # print('w')
            # print(w)
            # print('w')
            print(idx2w[w], end=' ')
        print('sent')
        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        print('pred')
        for w in pred:
            print(idx2la[w], end=' ')
        print('pred')
        val_pred_label.append(pred)

    avgLoss = avgLoss/n_batch

    predword_val = [ list(map(lambda x: idx2la[x], y)) for y in val_pred_label]
    con_dict = conlleval(predword_val, groundtruth_val, words_val, 'r.txt')
    val_f_scores.append(con_dict['f1'])

    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['r'], con_dict['p'], con_dict['f1']))

    if con_dict['f1'] > best_val_f1:
    	best_val_f1 = con_dict['f1']
    	open('model_architecture.json','w').write(model.to_json())
    	model.save('best_model.h5',overwrite=True)
    	print("Best validation F1 score = {}".format(best_val_f1))
    print()

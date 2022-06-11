# 检验数据分布是否一致

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input
from tensorflow.keras.models import Model
from utils import *
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import auc, roc_curve, roc_auc_score
import tensorflow as tf

vocab_size = 20000
maxlen = 100
wvdim = 64
hidden_size = 64

df1 = pd.read_csv('datasets/20NG/20ng-train-all-terms.csv')
df2 = pd.read_csv('datasets/20NG/20ng-test-all-terms.csv')
df1['split_label'] = 0
df2['split_label'] = 1
df = pd.concat([df1,df2])
df = df.dropna(axis=0, how='any')
datas = df['content'].tolist()
labels = df['split_label'].tolist()
print('data size: ', len(df))

label_num = sorted(list(set(labels)))
label2id = {name: id for (name, id) in zip(label_num, range(len(label_num)))}
label_ids = [label2id[name] for name in labels]
labels = np.array(label_ids)
labels = to_categorical(labels, num_classes=len(label_num))

corpus = []
X = []

for content in datas:
    content_words = content.split(' ')
    corpus += content_words
    X.append(content_words)
tokenizer, word_index, freq_word_index = fit_corpus(corpus,vocab_size=vocab_size)
datas = text2idx(tokenizer,X,maxlen=maxlen)

train_data, test_data, train_label, test_label = train_test_split(datas, labels, test_size=0.2, random_state=777, shuffle=True, stratify=labels)
print('train size: ', len(train_data))
print('test size: ', len(test_data))


class MLP:
    def __init__(self, vocab_size, maxlen, wvdim, hidden_size, num_classes):
        self.vocab_size = vocab_size
        self.maxlen = maxlen
        self.wvdim = wvdim
        self.hidden_size = hidden_size
        self.num_classes = num_classes

    def mlp_model(self):
        def auc_metrics(y_true, y_pred):
            Y_t, Y_p = [], []
            Y_t.extend(y_true)
            Y_p.extend(y_pred)
            return roc_auc_score(Y_t, Y_p, multi_class='ovo')
        input_vec = Input(shape=(self.maxlen, ), name='text_input')
        input_emb = Embedding(self.vocab_size+1, self.wvdim, input_length=self.maxlen, name='text_emb')(input_vec)
        hidden = Dense(hidden_size)(input_emb)
        output = Flatten()(hidden)
        output = Dense(self.num_classes, activation='softmax')(output)
        model = Model(input_vec, output)
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      # metrics=[auc_metrics])
                      metrics=[tf.keras.metrics.AUC()])
        model.summary()
        return model


filepath = 'analysis_data_model'
if not os.path.exists(filepath):
    os.mkdir(filepath)
checkpiont = tf.keras.callbacks.ModelCheckpoint(os.path.join(filepath, 'model.hdf5'), monitor='val_auc', save_best_only=True, mode='min')
model_class = MLP(vocab_size, maxlen, wvdim, hidden_size, len(label_num))
model = model_class.mlp_model()
model.fit(train_data, train_label, batch_size=64, epochs=10, validation_data=(test_data, test_label), verbose=2, callbacks=[checkpiont])

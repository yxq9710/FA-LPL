import datetime
import sys
import pandas as pd
import numpy as np
from utils import *
from sklearn.utils import shuffle
from models import lstm
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split


from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)

# ========== parameters & dataset: ==========
vocab_size = 20000
maxlen = 100
wvdim = 64
hidden_size = 64
alpha = 4 # new model的loss中的alpha
batch_size = 128
epochs = 40
emb_type = ''

# log_txt_name = '20Ng_log' + '_8NG_H'
# label_groups = {'comp.graphics':0, 'comp.os.ms-windows.misc':1, 'comp.sys.ibm.pc.hardware':2, 'comp.windows.x':3,
#                 'sci.crypt':4, 'sci.electronics':5, 'sci.med':6, 'sci.space':7}

# log_txt_name = '20Ng_log' + '_8NG_E'
# label_groups = {'comp.windows.x':0, 'rec.sport.baseball':1, 'alt.atheism':2, 'sci.med':3,
#                 'talk.politics.guns':4, 'misc.forsale':5, 'soc.religion.christian':6, 'talk.politics.misc':7}

# log_txt_name = '20Ng_log' + '_4NG_H'
# label_groups = {'rec.autos':0, 'rec.motorcycles':1, 'rec.sport.baseball':2, 'rec.sport.hockey':3}

log_txt_name = '20Ng_log' + '_4NG_E'
label_groups = {'comp.windows.x':0, 'rec.sport.baseball':1, 'alt.atheism':2, 'sci.med':3}
epochs = 20

df, num_classes = load_subset_of_20NG(label_groups)

print('raw data size:',len(df))
df = df.dropna(axis=0,how='any')

df = shuffle(df)[:50000]
print('data size:',len(df))

# ========== data pre-processing: ==========
labels = sorted(list(set(df.label)))

label2idx = {name:i for name,i in zip(labels,range(num_classes))}
num_classes = len(label2idx)

assert len(labels) == num_classes,'wrong num of classes'

corpus = []
X_words = []
Y = []
i = 0

m_len = 0
lens = []
contents = []
for content,y in zip(list(df.content),list(df.label)):
    i += 1
    if i%1000 == 0:
        print(i)
    # English:
    content_words = content.split(' ')
    # Chinese:
    # content_words = jieba.lcut(content)

    corpus += content_words
    X_words.append(content_words)
    Y.append(label2idx[y])


tokenizer, word_index, freq_word_index = fit_corpus(corpus,vocab_size=vocab_size)
X = text2idx(tokenizer,X_words,maxlen=maxlen)
y = np.array(Y)

# train_data, X, train_label, y = train_test_split(X, y, test_size=0.08, shuffle=True, stratify=y)


# ========== model traing: ==========
old_list = []
ls_list = []
lcm_list = []   # lcm
lcm_word_level_list = []
lcm_loss = []   # loss1
lcm_loss1 = []  # lcm + loss1
N = 10
for n in range(N):
    # randomly split train and test each time:
    np.random.seed(n) # 这样保证了每次试验的seed一致
    random_indexs = np.random.permutation(range(len(X)))
    train_size = int(len(X)*0.6)
    val_size = int(len(X)*0.15)
    X_train = X[random_indexs][:train_size]
    X_val = X[random_indexs][train_size:train_size+val_size]
    X_test = X[random_indexs][train_size+val_size:]
    y_train = y[random_indexs][:train_size]
    y_val = y[random_indexs][train_size:train_size+val_size]
    y_test = y[random_indexs][train_size+val_size:]
    data_package = [X_train,y_train,X_val,y_val,X_test,y_test]

    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(str(datetime.datetime.now()),file=f)
        print('--Round',n+1,file=f)
    print('--Round ', n+1)
    # print('====Original:============')
    # basic_model = lstm.LSTM_Basic(maxlen,vocab_size,wvdim,hidden_size,num_classes,None)
    # best_val_score,val_score_list,final_test_score,final_train_score = basic_model.train_val(data_package,batch_size,epochs)
    # old_list.append(final_test_score)
    # with open('output/%s.txt'%log_txt_name,'a') as f:
    #     print(n,'old:',final_train_score,best_val_score,final_test_score,file=f)
    #
    # print('====LS:============')
    # ls_model = lstm.LSTM_LS(maxlen,vocab_size,wvdim,hidden_size,num_classes,None)
    # best_val_score,val_score_list,final_test_score,final_train_score = ls_model.train_val(data_package,batch_size,epochs)
    # ls_list.append(final_test_score)
    # with open('output/%s.txt'%log_txt_name,'a') as f:
    #     print(n,'ls:',final_train_score,best_val_score,final_test_score,file=f)
    #
    # print('====LCM:============')
    # dy_lcm_model = lstm.LSTM_LCM_dynamic(maxlen,vocab_size,wvdim,hidden_size,num_classes,alpha,None,None)
    # best_val_score,val_score_list,final_test_score,final_train_score = dy_lcm_model.train_val(data_package,batch_size,epochs,lcm_stop=5)
    # lcm_list.append(final_test_score)
    # with open('output/%s.txt'%log_txt_name,'a') as f:
    #     print(n,'lcm:',final_train_score,best_val_score,final_test_score,file=f)

    for alpha in [3]:
        print("====loss1:==========")
        loss_lcm = lstm.LSTM_two_loss(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None)
        best_val_score, val_score_list, final_test_score, final_train_score = loss_lcm.train_val(data_package, batch_size, epochs, lcm_stop=4)
        lcm_loss.append(final_test_score)
        with open('output/%s.txt'%(log_txt_name), 'a') as f:
            print(n, 'loss1:', final_train_score, best_val_score, final_test_score, file=f)
            print('val acc list:\n', str(val_score_list), '\n', file=f)

print('old_list mean : {:.4f}, {:.4f}'.format(np.mean(old_list), np.std(old_list)))
print('ls_list mean : {:.4f}, {:.4f}'.format(np.mean(ls_list), np.std(ls_list)))
print('lcm_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_list), np.std(lcm_list)))
print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)))
with open('output/%s.txt'%log_txt_name, 'a') as f:
    print('old_list mean: {:.4f}, {:.4f}'.format(np.mean(old_list), np.std(old_list)), file=f)
    print('ls_list mean : {:.4f}, {:.4f}'.format(np.mean(ls_list), np.std(ls_list)), file=f)
    print('lcm_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_list), np.std(lcm_list)), file=f)
    print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)), file=f)

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
# num_filters = 100
# filter_sizes = [3,10,25] # 不能超过maxlen
alpha = 4 # new model的loss中的alpha
# alpha = 0.5 # new model的loss中的alpha
# batch_size = 512
batch_size = 128
# epochs = 80
epochs = 40
# epochs = 10   # lcm及之后的模型10个epoch就够了
emb_type = ''

log_txt_name = '20NG_log'   # lcm_stop: 两个都是10
num_classes = 20
df1 = pd.read_csv('datasets/20NG/20ng-train-all-terms.csv')
df2 = pd.read_csv('datasets/20NG/20ng-test-all-terms.csv')
df = pd.concat([df1,df2])
# epochs = 60
# epochs = 20

# log_txt_name = 'AG'  # AdamLR(0.01, lr_schedule={500: 1,3000: 0.1})  # 使用return_sequence=True, 然后全局池化 lr: 1e-4 lcm_stop: 10
# num_classes = 4
# df1 = pd.read_csv('datasets/AG_news/train.csv', header=None, index_col=None)
# df2 = pd.read_csv('datasets/AG_news/test.csv', header=None, index_col=None)
# df1.columns = ['label', 'title', 'content']
# df2.columns = ['label', 'title', 'content']
# df = pd.concat([df1, df2])
# epochs = 30

# log_txt_name = 'DBPedia' # AG_log     # 使用return_sequence=True, 然后全局池化  lcm_stop: 10
# num_classes = 14
# df1 = pd.read_csv('datasets/DBPedia/train.csv',header=None,index_col=None)
# df2 = pd.read_csv('datasets/DBPedia/test.csv',header=None,index_col=None)
# df1.columns = ['label','title','content']
# df2.columns = ['label','title','content']
# df = pd.concat([df1,df2])

# log_txt_name = 'Fudan'
# num_classes = 20
# df = pd.read_csv('datasets/fudan_news.csv')

# log_txt_name = 'THU_log'   # epoch = 40, two_loss: 30
# df = pd.read_csv('datasets/thucnews_subset.csv')
# num_classes = 13
# epoch = 40

print('raw data size:',len(df))
df = df.dropna(axis=0,how='any')

# df1, df2 = train_test_split(df, test_size=0.4, stratify=df.label)  # DBP: 0.08  # AG: 0.4
# df = df2
# vocab_size = 50000

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
    print('====Original:============')
    basic_model = lstm.LSTM_Basic(maxlen,vocab_size,wvdim,hidden_size,num_classes,None)
    best_val_score,val_score_list,final_test_score,final_train_score = basic_model.train_val(data_package,batch_size,epochs)
    old_list.append(final_test_score)
    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(n,'old:',final_train_score,best_val_score,final_test_score,file=f)

    print('====LS:============')
    ls_model = lstm.LSTM_LS(maxlen,vocab_size,wvdim,hidden_size,num_classes,None)
    best_val_score,val_score_list,final_test_score,final_train_score = ls_model.train_val(data_package,batch_size,epochs)
    ls_list.append(final_test_score)
    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(n,'ls:',final_train_score,best_val_score,final_test_score,file=f)

    print('====LCM:============')
    dy_lcm_model = lstm.LSTM_LCM_dynamic(maxlen,vocab_size,wvdim,hidden_size,num_classes,alpha,None,None)
    best_val_score,val_score_list,final_test_score,final_train_score = dy_lcm_model.train_val(data_package,batch_size,epochs,lcm_stop=10)
    lcm_list.append(final_test_score)
    with open('output/%s.txt'%log_txt_name,'a') as f:
        print(n,'lcm:',final_train_score,best_val_score,final_test_score,file=f)

    for alpha in [3]:
        print("====loss1:==========")
        loss_lcm = lstm.LSTM_two_loss(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None)
        best_val_score, val_score_list, final_test_score, final_train_score = loss_lcm.train_val(data_package, batch_size, epochs, lcm_stop=10)
        lcm_loss.append(final_test_score)
        with open('output/%s.txt'%(log_txt_name), 'a') as f:
            print(n, 'loss1:', final_train_score, best_val_score, final_test_score, file=f)
            print('val acc list:\n', str(val_score_list), '\n', file=f)

    # print("====lcm loss1:==========")
    # loss_lcm = lstm.LSTM_LCM_loss1(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None)
    # best_val_score, val_score_list, final_test_score, final_train_score = loss_lcm.train_val(data_package, batch_size,
    #                                                                                          epochs, lcm_stop=20)
    # lcm_loss1.append(final_test_score)
    # with open('output/%s.txt' % log_txt_name, 'a') as f:
    #     print(n, 'lcm loss1:', final_train_score, best_val_score, final_test_score, file=f)

print('old_list mean : {:.4f}, {:.4f}'.format(np.mean(old_list), np.std(old_list)))
print('ls_list mean : {:.4f}, {:.4f}'.format(np.mean(ls_list), np.std(ls_list)))
print('lcm_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_list), np.std(lcm_list)))
print('lcm_word_level_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_word_level_list), np.std(lcm_word_level_list)))
print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)))
print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss1), np.std(lcm_loss1)))
with open('output/%s.txt'%log_txt_name, 'a') as f:
    print('old_list mean: {:.4f}, {:.4f}'.format(np.mean(old_list), np.std(old_list)), file=f)
    print('ls_list mean : {:.4f}, {:.4f}'.format(np.mean(ls_list), np.std(ls_list)), file=f)
    print('lcm_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_list), np.std(lcm_list)), file=f)
    print('lcm_word_level_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_word_level_list), np.std(lcm_word_level_list)), file=f)
    print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)), file=f)
    print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss1), np.std(lcm_loss1)), file=f)

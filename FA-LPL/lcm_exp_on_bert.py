import datetime
import numpy as np
from utils import load_dataset, create_asy_noise_labels
from sklearn.utils import shuffle
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding
from models import bert
import matplotlib.pyplot as plt
import time

#%%
# ========== parameters: ==========
maxlen = 128
hidden_size = 64
batch_size = 128

# epochs = 150
epochs = 100
# epochs = 50

# ========== bert config: ==========
# # for English, use bert_tiny:
bert_type = 'bert'
config_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_config.json'
checkpoint_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_model.ckpt'
vocab_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/vocab.txt'

# for Chinese, use albert_tiny:
# bert_type = 'albert'
# config_path = 'bert_weights/albert_tiny_google_zh_489k/albert_config.json'
# checkpoint_path = 'bert_weights/albert_tiny_google_zh_489k/albert_model.ckpt'
# vocab_path = 'bert_weights/albert_tiny_google_zh_489k/vocab.txt'

tokenizer = Tokenizer(vocab_path, do_lower_case=True)

# ========== dataset: ==========
# dataset_name = '20NG'  # epoch = 100,  lcm: 50, lcm_two_loss: 40
# epochs = 50
# dataset_name = 'AG'      # epoch = 100,  lcm: 40, lcm_two_loss: 40
dataset_name = 'DBP'      # epoch = 40,  lcm: 10, lcm_two_loss: 10

# dataset_name = 'FDU'       # epoch = 150, lcm: 50, lcm_two_loss: 50
# dataset_name = 'THU'         # epoch = 100, lcm: 50, lcm_two_loss: 40  # lr: 5e-5
# group_noise_rate = 0.3
group_noise_rate = 0
df,num_classes,label_groups = load_dataset(dataset_name)
# define log file name:
log_txt_name = '%s_BERT_log(group_noise=%s,comp+rec+talk)' % (dataset_name,group_noise_rate)


df = df.dropna(axis=0,how='any')
df = shuffle(df)[:50000]

print('data size:',len(df))

#%%
# ========== data preparation: ==========
labels = sorted(list(set(df.label)))
assert len(labels) == num_classes,'wrong num of classes!'
label2idx = {name:i for name,i in zip(labels,range(num_classes))}
#%%
print('start tokenizing...')
t = time.time()
X_token = []
X_seg = []
y = []
i = 0
lens = []

for content,label in zip(list(df.content),list(df.label)):
    i += 1
    if i%1000 == 0:
        print(i)
    token_ids, seg_ids = tokenizer.encode(content, maxlen=maxlen)
    X_token.append(token_ids)
    X_seg.append(seg_ids)
    y.append(label2idx[label])

# the sequences we obtained from above may have different length, so use Padding：
X_token = sequence_padding(X_token)
X_seg = sequence_padding(X_seg)
y = np.array(y)
print('tokenizing time cost:',time.time()-t,'s.')

#%%
# ========== model traing: ==========
old_list = []
ls_list = []
lcm_list = []
lcm_loss = []   # loss1
lcm_loss1 = []  # lcm + loss1
N = 5
# N = 3
for n in range(N):
    # randomly split train and test each time:
    np.random.seed(n) # 这样保证了每次试验的seed一致
    random_indexs = np.random.permutation(range(len(X_token)))
    train_size = int(len(X_token)*0.6)
    val_size = int(len(X_token)*0.15)
    X_token_train = X_token[random_indexs][:train_size]
    X_token_val = X_token[random_indexs][train_size:train_size+val_size]
    X_token_test = X_token[random_indexs][train_size+val_size:]
    X_seg_train = X_seg[random_indexs][:train_size]
    X_seg_val = X_seg[random_indexs][train_size:train_size + val_size]
    X_seg_test = X_seg[random_indexs][train_size + val_size:]
    y_train = y[random_indexs][:train_size]
    y_val = y[random_indexs][train_size:train_size+val_size]
    y_test = y[random_indexs][train_size+val_size:]
    data_package = [X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test]

    # apply noise only on train set:  # 只对bert下的20NG有影响，对其它的均没有影响
    # if group_noise_rate>0:
    #     _, overall_noise_rate, y_train = create_asy_noise_labels(y_train,label_groups,label2idx,group_noise_rate)
    #     data_package = [X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test,
    #                     y_test]
    #     with open('output/%s.txt' % log_txt_name, 'a') as f:
    #         print('-'*30,'\nNOITCE: overall_noise_rate=%s'%round(overall_noise_rate,2), file=f)

    with open('output/%s.txt'%log_txt_name,'a') as f:
        print('\n',str(datetime.datetime.now()),file=f)
        print('\n ROUND & SEED = ',n,'-'*20,file=f)

    model_to_run = [""]
    # print('====Original:============')
    # model = bert.BERT_Basic(config_path,checkpoint_path,hidden_size,num_classes,bert_type)
    # train_score_list, val_socre_list, best_val_score, test_score = model.train_val(data_package,batch_size,epochs, False)
    # # plt.plot(train_score_list, label='train')
    # # plt.plot(val_socre_list, label='val')
    # # plt.title('BERT')
    # # plt.legend()
    # # plt.show()
    # old_list.append(test_score)
    # with open('output/%s.txt'%log_txt_name,'a') as f:
    #     print('\n*** Orig BERT ***:',file=f)
    #     print('test acc:', str(test_score), file=f)
    #     print('best val acc:',str(best_val_score),file=f)
    #     # print('train acc list:\n',str(train_score_list),file=f)
    #     # print('val acc list:\n',str(val_socre_list),'\n',file=f)
    #
    # print('====Label Smooth:============')
    # ls_e = 0.1
    # model = bert.BERT_LS(config_path,checkpoint_path,hidden_size,num_classes,ls_e,bert_type)
    # train_score_list, val_socre_list, best_val_score, test_score = model.train_val(data_package,batch_size,epochs, False)
    # # plt.plot(train_score_list, label='train')
    # # plt.plot(val_socre_list, label='val')
    # # plt.title('BERT with LS')
    # # plt.legend()
    # # plt.show()
    # ls_list.append(test_score)
    # with open('output/%s.txt'%log_txt_name,'a') as f:
    #     print('\n*** Orig BERT with LS (e=%s) ***:'%ls_e, file=f)
    #     print('test acc:', str(test_score), file=f)
    #     print('best val acc:', str(best_val_score), file=f)
    #     # print('train acc list:\n', str(train_score_list), file=f)
    #     # print('val acc list:\n', str(val_socre_list), '\n', file=f)
    #
    # print('====LCM:============')
    # # alpha = 3
    # # for alpha in [3,4,5]:
    # for alpha in [4]:
    #     wvdim = 128
    #     lcm_stop = 50
    #     params_str = 'a=%s, wvdim=%s, lcm_stop=%s'%(alpha,wvdim,lcm_stop)
    #     model = bert.BERT_LCM(config_path,checkpoint_path,hidden_size,num_classes,alpha,wvdim,bert_type)
    #     train_score_list, val_socre_list, best_val_score, test_score = model.train_val(data_package, batch_size,epochs,lcm_stop, save_best=False)
    #     # plt.plot(train_score_list, label='train')
    #     # plt.plot(val_socre_list, label='val')
    #     # plt.title('BERT with LCM')
    #     # plt.legend()
    #     # plt.show()
    #     lcm_list.append(test_score)
    #     with open('output/%s.txt'%log_txt_name,'a') as f:
    #         print('\n*** Orig BERT with LCM (%s) ***:'%params_str,file=f)
    #         print('test acc:', str(test_score), file=f)
    #         print('best val acc:', str(best_val_score), file=f)
    #         # print('train acc list:\n', str(train_score_list), file=f)
    #         # print('val acc list:\n', str(val_socre_list), '\n', file=f)

    print('====loss1:============')
    # alpha = 3
    # for alpha in [3,4,5]:
    for alpha in [3]:
        wvdim = 128
        lcm_stop = 50
        # lcm_stop = 40
        params_str = 'a=%s, wvdim=%s, lcm_stop=%s'%(alpha,wvdim,lcm_stop)
        model = bert.BERT_two_loss(maxlen, config_path,checkpoint_path,hidden_size,num_classes,alpha,wvdim,bert_type)
        train_score_list, val_socre_list, best_val_score, test_score = model.train_val(data_package, batch_size,epochs=40,lcm_stop=10, save_best=False)
        lcm_loss.append(test_score)
        with open('output/%s.txt'%log_txt_name,'a') as f:
            print('\n*** Orig BERT with loss1 (%s) ***:'%params_str,file=f)
            print('test acc:', str(test_score), file=f)
            print('best val acc:', str(best_val_score), file=f)
            # print('train acc list:\n', str(train_score_list), file=f)
            # print('val acc list:\n', str(val_socre_list), '\n', file=f)

    # print('====lcm loss1:============')
    # for alpha in [4]:
    #     wvdim = 128
    #     lcm_stop = 50
    #     params_str = 'a=%s, wvdim=%s, lcm_stop=%s' % (alpha, wvdim, lcm_stop)
    #     model = bert.BERT_LCM_loss1(maxlen, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim, bert_type)
    #     train_score_list, val_socre_list, best_val_score, test_score = model.train_val(data_package, batch_size, epochs,
    #                                                                                    lcm_stop, True)
    #     lcm_loss1.append(test_score)
    #     with open('output/%s.txt' % log_txt_name, 'a') as f:
    #         print('\n*** Orig BERT with lcm loss1 (%s) ***:' % params_str, file=f)
    #         print('test acc:', str(test_score), file=f)
    #         print('best val acc:', str(best_val_score), file=f)
    #         # print('train acc list:\n', str(train_score_list), file=f)
    #         # print('val acc list:\n', str(val_socre_list), '\n', file=f)
print('old_list mean : {:.4f}, {:.4f}'.format(np.mean(old_list), np.std(old_list)))
print('ls_list mean : {:.4f}, {:.4f}'.format(np.mean(ls_list), np.std(ls_list)))
print('lcm_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_list), np.std(lcm_list)))
print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)))
# print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss1), np.std(lcm_loss1)))
with open('output/%s.txt'%log_txt_name, 'a') as f:
    print('old_list mean: {:.4f}, {:.4f}'.format(np.mean(old_list), np.std(old_list)), file=f)
    print('ls_list mean : {:.4f}, {:.4f}'.format(np.mean(ls_list), np.std(ls_list)), file=f)
    print('lcm_list mean : {:.4f}, {:.4f}'.format(np.mean(lcm_list), np.std(lcm_list)), file=f)
    print('lcm_loss mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss), np.std(lcm_loss)), file=f)
    # print('lcm_loss1 mean : {:.4f}, {:.4f}'.format(np.mean(lcm_loss1), np.std(lcm_loss1)), file=f)

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
# maxlen = 512
hidden_size = 64
# batch_size = 512
batch_size = 128
# epochs = 150
# epochs = 100
epochs = 50

# ========== bert config: ==========
# # for English, use bert_tiny:
# bert_type = 'bert'
# config_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_config.json'
# checkpoint_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_model.ckpt'
# vocab_path = 'bert_weights/bert_tiny_uncased_L-2_H-128_A-2/vocab.txt'

# for Chinese, use albert_tiny:
bert_type = 'albert'
config_path = '../bert_weights/albert_tiny_google_zh_489k/albert_config.json'
checkpoint_path = '../bert_weights/albert_tiny_google_zh_489k/albert_model.ckpt'
vocab_path = '../bert_weights/albert_tiny_google_zh_489k/vocab.txt'

tokenizer = Tokenizer(vocab_path, do_lower_case=True)

# ========== dataset: ==========
dataset_name = '20NG'
# dataset_name = 'THU'
# group_noise_rate = 0.3
group_noise_rate = 0
df,num_classes,label_groups = load_dataset(dataset_name)

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

# token_list = [i+1 for i in range(num_classes)]
# token_list += [102]
# seg_list = [0] * len(token_list)

for content,label in zip(list(df.content),list(df.label)):
    i += 1
    if i%1000 == 0:
        print(i)
    token_ids, seg_ids = tokenizer.encode(content, maxlen=maxlen)
    # token_id = [token_ids[0]] + token_list + token_ids[1:]
    # seg_id = [seg_ids[0]] + seg_list + seg_ids[1:]
    X_token.append(token_ids)
    X_seg.append(seg_ids)
    y.append(label2idx[label])

# the sequences we obtained from above may have different length, so use Padding：
X_token = sequence_padding(X_token)
X_seg = sequence_padding(X_seg)
y = np.array(y)
print('tokenizing time cost:',time.time()-t,'s.')
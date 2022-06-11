# checked at 2020.9.14
import numpy as np
import time
import keras
import tensorflow as tf
from keras.models import Sequential,Model
from keras.layers import Input,Dense,LSTM,Embedding,Conv1D,MaxPooling1D, Softmax, GlobalAveragePooling1D
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
import keras.backend as K
from keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr

AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

class LSTM_Basic:
    """
    input->embedding->lstm->softmax_dense
    """
    def __init__(self,maxlen,vocab_size,wvdim,hidden_size,num_classes,embedding_matrix=None):
        text_input =  Input(shape=(maxlen,),name='text_input')
        if embedding_matrix is None: # 不使用pretrained embedding
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,name='text_emb')(text_input) #(V,wvdim)
        else:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
        input_vec = LSTM(hidden_size)(input_emb)
        # input_vec = LSTM(hidden_size, return_sequences=True)(input_emb)
        # input_vec = GlobalAveragePooling1D()(input_vec)
        # input_vec = Dropout(0.4)(input_vec)

        pred_probs = Dense(num_classes,activation='softmax',name='pred_probs')(input_vec)
        self.model = Model(inputs=text_input,outputs=pred_probs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_val(self,data_package,batch_size,epochs,save_best=False):
        X_train,y_train,X_val,y_val,X_test,y_test = data_package
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """
        for i in range(epochs):
            t1 = time.time()
            self.model.fit(X_train,to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
            pred_probs = self.model.predict(X_val)
            predictions = np.argmax(pred_probs,axis=1)
            val_score = round(accuracy_score(y_val,predictions),5)
            t2 = time.time()
            print('(Orig)Epoch',i+1,'| time: %.3f s'%(t2-t1),'| current val accuracy:',val_score)
            if val_score>best_val_score-0.001:
                best_val_score = val_score
                # 使用当前val上最好模型进行test:
                pred_probs = self.model.predict(X_test)
                predictions = np.argmax(pred_probs,axis=1)
                current_test_score = round(accuracy_score(y_test,predictions),5)
                if current_test_score > final_test_score:
                    final_test_score = current_test_score
                    print('  Current Best model! Test score:',final_test_score)
                    # 同时记录一下train上的score：
                    pred_probs = self.model.predict(X_train)
                    predictions = np.argmax(pred_probs, axis=1)
                    final_train_score = round(accuracy_score(y_train, predictions),5)
                    print('  Current Best model! Train score:', final_train_score)
                    if save_best:
                        self.model.save('best_model_lstm.h5')
                        print('  best model saved!')
            val_socre_list.append(val_score)
        return best_val_score,val_socre_list,final_test_score,final_train_score


class LSTM_LS:
    """
    input->embedding->lstm->softmax_dense
    """
    def __init__(self,maxlen,vocab_size,wvdim,hidden_size,num_classes,embedding_matrix=None):
        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
            return (1-e)*loss1 + e*loss2
        
        text_input =  Input(shape=(maxlen,),name='text_input')
        if embedding_matrix is None: # 不使用pretrained embedding
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,name='text_emb')(text_input) #(V,wvdim)
        else:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
        input_vec = LSTM(hidden_size)(input_emb)
        # input_vec = LSTM(hidden_size, return_sequences=True)(input_emb)
        # input_vec = GlobalAveragePooling1D()(input_vec)
        # input_vec = Dropout(0.4)(input_vec)

        pred_probs = Dense(num_classes,activation='softmax',name='pred_probs')(input_vec)
        self.model = Model(inputs=text_input,outputs=pred_probs) 
        self.model.compile(loss=ls_loss, optimizer='adam', metrics=['accuracy']) # 'adam'
    

    def train_val(self,data_package,batch_size,epochs,save_best=False):
        X_train,y_train,X_val,y_val,X_test,y_test = data_package
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            t1 = time.time()
            self.model.fit(X_train,to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
            # val:
            pred_probs = self.model.predict(X_val)
            predictions = np.argmax(pred_probs,axis=1)
            val_score = round(accuracy_score(y_val,predictions),5)
            t2 = time.time()
            print('(LS)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
            if val_score>best_val_score-0.001:
                best_val_score = val_score
                # 使用当前val上最好模型进行test:
                pred_probs = self.model.predict(X_test)
                predictions = np.argmax(pred_probs, axis=1)
                current_test_score = round(accuracy_score(y_test, predictions),5)
                if current_test_score > final_test_score:
                    final_test_score = current_test_score
                    print('  Current Best model! Test score:', final_test_score)
                    # 同时记录一下train上的score：
                    pred_probs = self.model.predict(X_train)
                    predictions = np.argmax(pred_probs, axis=1)
                    final_train_score = round(accuracy_score(y_train, predictions),5)
                    print('  Current Best model! Train score:', final_train_score)
                    if save_best:
                        self.model.save('best_model_ls.h5')
                        print('  best model saved!')
            val_score_list.append(val_score)
        return best_val_score,val_score_list,final_test_score,final_train_score


class LSTM_LCM_dynamic:
    """
    LCM dynamic,跟LCM的主要差别在于：
    1.可以设置early stop，即设置在某一个epoch就停止LCM的作用；
    2.在停止使用LCM之后，可以选择是否使用label smoothing来计算loss。
    """
    def __init__(self,maxlen,vocab_size,wvdim,hidden_size,num_classes,alpha,default_loss='ls',text_embedding_matrix=None,label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true,y_pred,alpha=alpha):
            pred_porbs = y_pred[:,:num_classes]
            label_sim_dist = y_pred[:,num_classes:]
            simulated_y_true = K.softmax(label_sim_dist+4*y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true,simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true,pred_probs)
            return loss1+loss2
        
        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
            return (1-e)*loss1 + e*loss2
        
        # basic_predictor:
        text_input =  Input(shape=(maxlen,),name='text_input')
        if text_embedding_matrix is None:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,name='text_emb')(text_input) #(V,wvdim)
        else:
            input_emb = Embedding(vocab_size+1,wvdim,input_length=maxlen,weights=[text_embedding_matrix],name='text_emb')(text_input) #(V,wvdim)
        input_vec = LSTM(hidden_size)(input_emb)
        # input_vec = LSTM(hidden_size, return_sequences=True)(input_emb)
        # input_vec = GlobalAveragePooling1D()(input_vec)
        # input_vec = Dropout(0.4)(input_vec)

        pred_probs = Dense(num_classes,activation='softmax',name='pred_probs')(input_vec)
        # input_vec = Dense(num_classes, name='pred_probs')(input_vec)
        # pred_probs = Softmax()(input_vec)
        self.basic_predictor = Model(inputs=text_input,outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else:
            self.basic_predictor.compile(loss='categorical_crossentropy', optimizer='adam')

        # LCM:
        label_input = Input(shape=(num_classes,),name='label_input')
        label_emb = Embedding(num_classes,wvdim,input_length=num_classes,name='label_emb1')(label_input) # (n,wvdim)
        label_emb = Dense(hidden_size,activation='tanh',name='label_emb2')(label_emb)
        # label_emb = Dense(num_classes,activation='tanh',name='label_emb2')(label_emb)
        # similarity part:
        doc_product = Dot(axes=(2,1))([label_emb,input_vec]) # (n,d) dot (d,1) --> (n,1)
        label_sim_dict = Dense(num_classes,activation='softmax',name='label_sim_dict')(doc_product)  # 没有加one hot？
        # concat output:
        concat_output = Concatenate()([pred_probs,label_sim_dict])
        # compile；
        self.model = Model(inputs=[text_input,label_input],outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')

    def my_evaluator(self,model,inputs,label_list):
        outputs = model.predict(inputs)
        pred_probs = outputs[:,:self.num_classes]
        predictions = np.argmax(pred_probs,axis=1)
        acc = round(accuracy_score(label_list,predictions),5)
        # recall = recall_score(label_list,predictions,average='weighted')
        # f1 = f1_score(label_list,predictions,average='weighted')
        return acc

    def train_val(self,data_package,batch_size,epochs,lcm_stop=50,save_best=False):
        X_train,y_train,X_val,y_val,X_test,y_test = data_package
        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_test))])
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            if i < lcm_stop:
                t1 = time.time()
                self.model.fit([X_train,L_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                val_score = self.my_evaluator(self.model,[X_val,L_val],y_val)
                t2 = time.time()
                print('(LCM)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score>best_val_score-0.001:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model,[X_test,L_test],y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:',final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model.h5')
                            print('best model saved!')
                val_score_list.append(val_score)
            else: # 停止LCM的作用
                t1 = time.time()
                self.basic_predictor.fit(X_train,to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                pred_probs = self.basic_predictor.predict(X_val)
                predictions = np.argmax(pred_probs,axis=1)
                val_score = round(accuracy_score(y_val,predictions),5)
                t2 = time.time()
                print('(LCM-stop)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score>best_val_score-0.001:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model_lcm.h5')
                            print('  best model saved!')
                val_score_list.append(val_score)
        return best_val_score,val_score_list,final_test_score,final_train_score


class LSTM_LCM_word_level:
    """
    LCM dynamic,跟LCM的主要差别在于：
    1.可以设置early stop，即设置在某一个epoch就停止LCM的作用；
    2.在停止使用LCM之后，可以选择是否使用label smoothing来计算loss。
    """

    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, default_loss='ls',
                 text_embedding_matrix=None, label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=alpha):
            pred_porbs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        def loss2(y_true, y_pred):   # lcm_loss = loss2 + one_hot
            pred_prob = y_pred[:, :num_classes]
            simulated_y_true = y_pred[:, num_classes:]
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
            return loss1 + loss2

        # basic_predictor:
        text_input = Input(shape=(maxlen,), name='text_input')
        if text_embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)  # (V,wvdim)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[text_embedding_matrix],
                                  name='text_emb')(text_input)  # (V,wvdim)
        # LCM label embedding
        label_input = Input(shape=(num_classes,), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(label_input)  # (n,wvdim)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)

        # input_vec = LSTM(hidden_size)(input_emb)
        input_vec = LSTM(hidden_size, return_sequences=True)(input_emb)
        doc_product = Dot(axes=(2))([input_vec, label_emb])
        doc_product_softmax = Lambda(lambda x: K.tanh(x))(doc_product)
        weight = Lambda(lambda x: K.expand_dims(x, axis=2))(doc_product_softmax)
        input_vec = Lambda(lambda x: K.expand_dims(x, axis=-1))(input_vec)
        input_vec = Lambda(lambda x: K.sum(tf.matmul(x[0], x[1]), axis=-1))([input_vec, weight])
        input_vec = GlobalAveragePooling1D()(input_vec)

        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec)
        self.basic_predictor = Model(inputs=[text_input, label_input], outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else:
            self.basic_predictor.compile(loss='categorical_crossentropy', optimizer='adam')
        self.basic_predictor.summary()

        doc_product = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(doc_product)
        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.reshape(x, shape=(-1, num_classes)))(label_sim_dict)
        label_sim_dict = Softmax(name='label_sim_dict')(label_sim_dict)
        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')
        self.model.summary()

    def my_evaluator(self, model, inputs, label_list):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(label_list, predictions), 5)
        # recall = recall_score(label_list,predictions,average='weighted')
        # f1 = f1_score(label_list,predictions,average='weighted')
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_train, y_train, X_val, y_val, X_test, y_test = data_package
        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_test))])
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            if i < lcm_stop:
                t1 = time.time()
                self.model.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                val_score = self.my_evaluator(self.model, [X_val, L_val], y_val)
                t2 = time.time()
                print('(LCM word level loss)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model.h5')
                            print('best model saved!')
                val_score_list.append(val_score)
            else:  # 停止LCM的作用
                t1 = time.time()
                self.basic_predictor.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                pred_probs = self.basic_predictor.predict([X_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                t2 = time.time()
                print('(LCM-word-level-stop)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model_lcm.h5')
                            print('  best model saved!')
                val_score_list.append(val_score)
        return best_val_score, val_score_list, final_test_score, final_train_score


class LSTM_two_loss:

    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, default_loss='ls',
                 text_embedding_matrix=None, label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]

            # simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            simulated_y_true = K.softmax((label_sim_dist + 4 * y_true) / alpha)
            pred_probs = K.softmax(pred_probs / alpha)

            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            y_pred = K.softmax(y_pred)
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        def two_loss(y_true, y_pred):
            pred_prob = y_pred[:, :num_classes]
            simulated_y_true = y_pred[:, num_classes:]
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
            return loss1 + loss2 + K.categorical_crossentropy(y_true, simulated_y_true)

        def loss1(y_true, y_pred):
            pred_prob = y_pred[:, :num_classes]
            return K.categorical_crossentropy(y_true, pred_prob)

        def loss2(y_true, y_pred):
            pred_prob = y_pred[:, :num_classes]
            simulated_y_true = y_pred[:, num_classes:]
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
            return loss1 + loss2

        # basic_predictor:
        text_input = Input(shape=(maxlen,), name='text_input')
        if text_embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)  # (V,wvdim)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[text_embedding_matrix],
                                  name='text_emb')(text_input)  # (V,wvdim)
        # LCM label embedding
        label_input = Input(shape=(num_classes,), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(label_input)  # (n,wvdim)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)

        input_vec = LSTM(hidden_size, return_sequences=True)(input_emb)
        input_vec = Dropout(0.4)(input_vec)

        text_emb_d = Lambda(lambda x: K.expand_dims(x, axis=1))(input_vec)
        label_emb_d = Lambda(lambda x: K.expand_dims(x, axis=2))(label_emb)
        output_1 = Dense(hidden_size * 4)(text_emb_d)
        output_2 = Dense(hidden_size * 4)(label_emb_d)
        output1 = Lambda(lambda x: x[0] + x[1])([output_1, output_2])
        output1 = Lambda(lambda x: K.tanh(x))(output1)

        # # 引入dimensional 维度的attention
        weight_d = Softmax()(output1)
        weight = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([weight_d, output1])

        doc_product = Lambda(lambda x: K.squeeze(x, axis=-1))(weight)
        weight = Lambda(lambda x: K.softmax(x))(weight)   # 对注意力使用softmax
        input_vec = Lambda(lambda x: K.sum(x[0] * x[1], axis=1))([weight, text_emb_d])

        input_vec_1 = GlobalAveragePooling1D()(input_vec)
        pred_probs = Dense(num_classes, name='pred_probs')(input_vec_1)

        # pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec_1)
        self.basic_predictor = Model(inputs=[text_input, label_input], outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else: # tf.keras.losses.CategoricalCrossentropy(from_logits=True)   'categorical_crossentropy'
            self.basic_predictor.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam')

        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.squeeze(x, axis=-1))(label_sim_dict)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(label_sim_dict)
        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')

    def my_evaluator(self, model, inputs, label_list):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(label_list, predictions), 5)
        # recall = recall_score(label_list,predictions,average='weighted')
        # f1 = f1_score(label_list,predictions,average='weighted')
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_train, y_train, X_val, y_val, X_test, y_test = data_package
        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_test))])
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            if i < lcm_stop:
                t1 = time.time()
                self.model.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                val_score = self.my_evaluator(self.model, [X_val, L_val], y_val)
                t2 = time.time()
                print('(LCM two loss)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model.h5')
                            print('best model saved!')
                val_score_list.append(val_score)
            else:  # 停止LCM的作用
                t1 = time.time()
                self.basic_predictor.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                pred_probs = self.basic_predictor.predict([X_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                t2 = time.time()
                print('(LCM-two-loss-stop)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model_lcm.h5')
                            print('  best model saved!')
                val_score_list.append(val_score)
        return best_val_score, val_score_list, final_test_score, final_train_score


class LSTM_two_loss_no_attention:
    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, default_loss='ls',
                 text_embedding_matrix=None, label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            simulated_y_true = K.softmax((label_sim_dist + 4 * y_true) / alpha)
            pred_probs = K.softmax(pred_probs / alpha)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            y_pred = K.softmax(y_pred)
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        # basic_predictor:
        text_input = Input(shape=(maxlen,), name='text_input')
        if text_embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)  # (V,wvdim)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[text_embedding_matrix],
                                  name='text_emb')(text_input)  # (V,wvdim)
        # LCM label embedding
        label_input = Input(shape=(num_classes,), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(label_input)  # (n,wvdim)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)
        input_vec = LSTM(hidden_size, return_sequences=True)(input_emb)
        input_vec = Dropout(0.4)(input_vec)

        text_emb_d = Lambda(lambda x: K.expand_dims(x, axis=1))(input_vec)
        label_emb_d = Lambda(lambda x: K.expand_dims(x, axis=2))(label_emb)
        output_1 = Dense(hidden_size * 4)(text_emb_d)
        output_2 = Dense(hidden_size * 4)(label_emb_d)
        output1 = Lambda(lambda x: x[0] + x[1])([output_1, output_2])
        output1 = Lambda(lambda x: K.tanh(x))(output1)

        weight_d = Dense(hidden_size * 4)(label_emb_d)
        weight_d = Softmax()(weight_d)
        # # 引入dimensional 维度的attention
        weight = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([weight_d, output1])
        doc_product = Lambda(lambda x: K.squeeze(x, axis=-1))(weight)

        input_vec_1 = GlobalAveragePooling1D()(input_vec)
        pred_probs = Dense(num_classes, name='pred_probs')(input_vec_1)

        self.basic_predictor = Model(inputs=[text_input, label_input], outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else: # tf.keras.losses.CategoricalCrossentropy(from_logits=True)   'categorical_crossentropy'
            self.basic_predictor.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam')

        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.squeeze(x, axis=-1))(label_sim_dict)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(label_sim_dict)
        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')

    def my_evaluator(self, model, inputs, label_list):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(label_list, predictions), 5)
        # recall = recall_score(label_list,predictions,average='weighted')
        # f1 = f1_score(label_list,predictions,average='weighted')
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_train, y_train, X_val, y_val, X_test, y_test = data_package
        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_test))])
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            if i < lcm_stop:
                t1 = time.time()
                self.model.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                val_score = self.my_evaluator(self.model, [X_val, L_val], y_val)
                t2 = time.time()
                print('(LCM two loss no attention)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model.h5')
                            print('best model saved!')
                val_score_list.append(val_score)
            else:  # 停止LCM的作用
                t1 = time.time()
                self.basic_predictor.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                pred_probs = self.basic_predictor.predict([X_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                t2 = time.time()
                print('(LCM-two-loss-stop-no-attention)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model_lcm.h5')
                            print('  best model saved!')
                val_score_list.append(val_score)
        return best_val_score, val_score_list, final_test_score, final_train_score



class LSTM_LCM_loss1:

    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, default_loss='ls',
                 text_embedding_matrix=None, label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]

            # simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            simulated_y_true = K.softmax((label_sim_dist + alpha * y_true) / 3)
            pred_probs = K.softmax(pred_probs / 3)

            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def simulated_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            simulated_y_true = label_sim_dist
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        def second_loss(y_true, y_pred):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            loss1 = K.categorical_crossentropy(y_true, pred_probs)
            loss2 = K.categorical_crossentropy(y_true, label_sim_dist)
            return loss2

        # basic_predictor:
        text_input = Input(shape=(maxlen,), name='text_input')
        if text_embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(
                text_input)  # (V,wvdim)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[text_embedding_matrix],
                                      name='text_emb')(text_input)  # (V,wvdim)
        # LCM label embedding
        label_input = Input(shape=(num_classes,), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(label_input)  # (n,wvdim)
        label_emb = Dense(num_classes, activation='tanh', name='label_emb2')(label_emb)

        # input_vec = LSTM(hidden_size)(input_emb)
        input_vec = LSTM(num_classes, return_sequences=True)(input_emb)
        # input_vec = Dropout(0.4)(input_vec)

        text_emb_d = Lambda(lambda x: K.expand_dims(x, axis=1))(input_vec)
        label_emb_d = Lambda(lambda x: K.expand_dims(x, axis=2))(label_emb)
        output_1 = Dense(hidden_size * 4)(text_emb_d)
        output_2 = Dense(hidden_size * 4)(label_emb_d)
        output1 = Lambda(lambda x: x[0] + x[1])([output_1, output_2])
        output1 = Lambda(lambda x: K.tanh(x))(output1)

        # 引入dimensional 维度的attention
        weight_d = Softmax()(output1)
        weight = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([weight_d, output1])

        # doc_product_d = weight
        doc_product = Lambda(lambda x: K.squeeze(x, axis=-1))(weight)
        input_vec = Lambda(lambda x: K.sum(x[0] * x[1], axis=1))([weight, text_emb_d])

        input_vec_1 = GlobalAveragePooling1D()(input_vec)
        pred_probs = Dense(num_classes, name='pred_probs')(input_vec_1)

        # pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec_1)
        self.basic_predictor = Model(inputs=[text_input, label_input], outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else:  # tf.keras.losses.CategoricalCrossentropy(from_logits=True)   'categorical_crossentropy'
            self.basic_predictor.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                                             optimizer='adam')

        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.squeeze(x, axis=-1))(label_sim_dict)
        # label_sim_dict = Softmax(name='label_sim_dict')(label_sim_dict)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(label_sim_dict)
        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        # self.model.compile(loss=simulated_loss, optimizer='adam')
        self.model.compile(loss=lcm_loss, optimizer='adam')

        self.second_model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.second_model.compile(loss=second_loss, optimizer='adam')

    def my_evaluator(self, model, inputs, label_list):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(label_list, predictions), 5)
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_train, y_train, X_val, y_val, X_test, y_test = data_package
        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_test))])
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_score_list = []
        for i in range(epochs):
            if i < lcm_stop:
                t1 = time.time()
                self.model.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                val_score = self.my_evaluator(self.model, [X_val, L_val], y_val)
                t2 = time.time()
                print('(LCM loss1)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model.h5')
                            print('best model saved!')
                val_score_list.append(val_score)
            else:  # 停止LCM的作用
                t1 = time.time()
                self.basic_predictor.fit([X_train, L_train], to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
                pred_probs = self.basic_predictor.predict([X_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                t2 = time.time()
                print('(LCM-loss1-stop)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
                if val_score > best_val_score:
                    best_val_score = val_score
                    # test:
                    current_test_score = self.my_evaluator(self.model, [X_test, L_test], y_test)
                    if current_test_score > final_test_score:
                        final_test_score = current_test_score
                        print('  Current Best model! Test score:', final_test_score)
                        # train:
                        final_train_score = self.my_evaluator(self.model, [X_train, L_train], y_train)
                        print('  Current Best model! Train score:', final_train_score)
                        if save_best:
                            self.model.save('best_model_lcm.h5')
                            print('  best model saved!')
                val_score_list.append(val_score)
        return best_val_score, val_score_list, final_test_score, final_train_score


if __name__ == '__main__':
    vocab_size = 20000
    maxlen = 100
    wvdim = 64
    hidden_size = 64
    alpha = 0.5
    num_classes = 20
    # model = LSTM_two_loss(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None)
    model = LSTM_LCM_loss1(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None)

import numpy as np
import keras
import tensorflow as tf
import time
from keras.models import Sequential,Model
from keras.layers import Input, Dense, LSTM, Embedding, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Softmax
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional, Flatten
import keras.backend as K
from keras.utils import to_categorical
# from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr

class BERT_Basic:

    def __init__(self,config_path,checkpoint_path,hidden_size,num_classes,model_type='bert'):
        self.num_classes = num_classes
        bert = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path,
                                       model=model_type,return_keras_model=False, dropout_rate=0.3)
        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        # text_emb = Lambda(lambda x: x[:, 1: -1], name='CLS-token')(bert.model.output)
        # text_emb = GlobalAveragePooling1D()(text_emb)

        text_emb = Dense(hidden_size,activation='tanh')(text_emb)
        output = Dense(num_classes,activation='softmax')(text_emb)
        self.model = Model(bert.model.input,output)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model.compile(loss='categorical_crossentropy',optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1,2000: 0.1}))
        # self.model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=5e-5))

    def train_val(self,data_package,batch_size,epochs,save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """
        learning_curve = []
        for i in range(epochs):
            t1 = time.time()
            self.model.fit([X_token_train,X_seg_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
            # record train set result:
            pred_probs = self.model.predict([X_token_train, X_seg_train])
            predictions = np.argmax(pred_probs, axis=1)
            train_score = round(accuracy_score(y_train, predictions), 5)
            train_score_list.append(train_score)
            # validation:
            pred_probs = self.model.predict([X_token_val,X_seg_val])
            predictions = np.argmax(pred_probs,axis=1)
            val_score = round(accuracy_score(y_val, predictions), 5)
            val_socre_list.append(val_score)
            t2 = time.time()
            print('(Orig)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:', val_score)
            # save best model according to validation & test result:
            if val_score>best_val_score:
                best_val_score = val_score
                print('Current Best model!','current epoch:',i+1)
                # test on best model:
                pred_probs = self.model.predict([X_token_test,X_seg_test])
                predictions = np.argmax(pred_probs, axis=1)
                current_test_score = round(accuracy_score(y_test, predictions), 5)
                if current_test_score > test_score:
                    test_score = current_test_score
                    print('  Current Best model! Test score:', test_score)
                    if save_best:
                        self.model.save('best_model_bert.h5')
                        print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_LS:
    def __init__(self, config_path, checkpoint_path, hidden_size, num_classes, ls_e=0.1, model_type='bert'):

        def ls_loss(y_true, y_pred, e=ls_e):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        self.num_classes = num_classes
        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                       model=model_type, return_keras_model=False, dropout_rate=0.3)
        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        # text_emb = Lambda(lambda x: x[:, 1: -1], name='CLS-token')(bert.model.output)
        # text_emb = GlobalAveragePooling1D()(text_emb)

        text_emb = Dense(hidden_size, activation='tanh')(text_emb)
        output = Dense(num_classes, activation='softmax')(text_emb)
        self.model = Model(bert.model.input, output)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model.compile(loss=ls_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=ls_loss, optimizer=Adam(learning_rate=5e-5))

    def train_val(self, data_package, batch_size, epochs, save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """
        for i in range(epochs):
            t1 = time.time()
            self.model.fit([X_token_train, X_seg_train], to_categorical(y_train), batch_size=batch_size, verbose=0,
                           epochs=1)
            # record train set result:
            pred_probs = self.model.predict([X_token_train, X_seg_train])
            predictions = np.argmax(pred_probs, axis=1)
            train_score = round(accuracy_score(y_train, predictions), 5)
            train_score_list.append(train_score)
            # validation:
            pred_probs = self.model.predict([X_token_val, X_seg_val])
            predictions = np.argmax(pred_probs, axis=1)
            val_score = round(accuracy_score(y_val, predictions), 5)
            val_socre_list.append(val_score)
            t2 = time.time()
            print('(LS)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                  val_score)
            # save best model according to validation & test result:
            if val_score > best_val_score:
                best_val_score = val_score
                print('Current Best model!', 'current epoch:', i + 1)
                # test on best model:
                pred_probs = self.model.predict([X_token_test, X_seg_test])
                predictions = np.argmax(pred_probs, axis=1)
                current_test_score = round(accuracy_score(y_test, predictions), 5)
                if current_test_score > test_score:
                    test_score = current_test_score
                    print('  Current Best model! Test score:', test_score)
                    if save_best:
                        self.model.save('best_model_bert_ls.h5')
                        print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_LCM:
    def __init__(self,config_path,checkpoint_path,hidden_size,num_classes,alpha,wvdim=768,model_type='bert',label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true,y_pred,alpha=alpha):
            pred_probs = y_pred[:,:num_classes]
            label_sim_dist = y_pred[:,num_classes:]
            simulated_y_true = K.softmax(label_sim_dist+alpha*y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true,simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true,pred_probs)
            return loss1+loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred)/num_classes, y_pred)
            return (1-e)*loss1 + e*loss2     

        # text_encoder:
        bert = build_transformer_model(config_path=config_path,checkpoint_path=checkpoint_path,
                                       model=model_type,return_keras_model=False, dropout_rate=0.3)
        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        # text_emb = Lambda(lambda x: x[:, 1: -1], name='CLS-token')(bert.model.output)
        # text_emb = GlobalAveragePooling1D()(text_emb)

        text_emb = Dense(hidden_size,activation='tanh')(text_emb)
        pred_probs = Dense(num_classes,activation='softmax')(text_emb)

        # text_emb = Dense(num_classes)(text_emb)
        # pred_probs = Softmax()(text_emb)

        self.basic_predictor = Model(bert.model.input,pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss='categorical_crossentropy',optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1,2000: 0.1}))
        # self.basic_predictor.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=5e-5))


        # label_encoder:
        label_input = Input(shape=(num_classes,),name='label_input')
        if label_embedding_matrix is None: # 不使用pretrained embedding
            label_emb = Embedding(num_classes,wvdim,input_length=num_classes,name='label_emb1')(label_input) # (n,wvdim)
        else:
            label_emb = Embedding(num_classes,wvdim,input_length=num_classes,weights=[label_embedding_matrix],name='label_emb1')(label_input)
#         label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
        label_emb = Dense(hidden_size,activation='tanh',name='label_emb2')(label_emb)

        # label_emb = Dense(num_classes,activation='tanh',name='label_emb2')(label_emb)

        # similarity part:
        doc_product = Dot(axes=(2,1))([label_emb,text_emb]) # (n,d) dot (d,1) --> (n,1)
        label_sim_dict = Dense(num_classes,activation='softmax',name='label_sim_dict')(doc_product)
        # concat output:
        concat_output = Concatenate()([pred_probs,label_sim_dict])
        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model = Model(bert.model.input+[label_input],concat_output)
        self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1,2000: 0.1}))
        # self.model.compile(loss=lcm_loss, optimizer=Adam(learning_rate=5e-5))


    def lcm_evaluate(self,model,inputs,y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:,:self.num_classes]
        predictions = np.argmax(pred_probs,axis=1)
        acc = round(accuracy_score(y_true,predictions),5)
        return acc

    def train_val(self, data_package, batch_size,epochs,lcm_stop=50,save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_token_train,X_seg_train,L_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model,[X_token_train,X_seg_train,L_train],y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model,[X_token_val,X_seg_val,L_val],y_val)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    current_test_score = self.lcm_evaluate(self.model,[X_token_test,X_seg_test,L_test],y_test)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm.h5')
                            print('  best model saved!')
            else:
                self.basic_predictor.fit([X_token_train,X_seg_train],to_categorical(y_train),batch_size=batch_size,verbose=0,epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions),5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions),5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test])
                    predictions = np.argmax(pred_probs, axis=1)
                    current_test_score = round(accuracy_score(y_test, predictions),5)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm.h5')
                            print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_two_loss_cls:
    def __init__(self, maxlen, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim=768, model_type='bert',
                 label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
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

        # text_encoder:

        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                       model=model_type, return_keras_model=False, sequence_length=maxlen, dropout_rate=0.3)

        # label_encoder:
        label_input = Input(shape=(num_classes,), name='label_input')
        if label_embedding_matrix is None:  # 不使用pretrained embedding
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(
                label_input)  # (n,wvdim)
        else:
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, weights=[label_embedding_matrix],
                                  name='label_emb1')(label_input)
        #         label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)

        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        # text_emb = bert.model.output
        text_emb = Dense(hidden_size, activation='tanh')(text_emb)
        doc_product = Dot(axes=(1, 2))([text_emb, label_emb])
        weight = Lambda(lambda x: K.expand_dims(x, axis=-2))(doc_product)
        input_vec = Lambda(lambda x: K.expand_dims(x, axis=-1))(text_emb)
        input_vec = Lambda(lambda x: K.sum(tf.matmul(x[0], x[1]), axis=-1))([input_vec, weight])
        # input_vec = GlobalAveragePooling1D()(input_vec)
        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec)
        self.basic_predictor = Model(bert.model.input + [label_input], pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss='categorical_crossentropy',
                                     optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))

        # similarity part:
        # doc_product = Dot(axes=(2, 1))([label_emb, text_emb])  # (n,d) dot (d,1) --> (n,1)
        # label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)

        doc_product = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(doc_product)
        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.reshape(x, shape=(-1, num_classes)))(label_sim_dict)
        label_sim_dict = Softmax(name='label_sim_dict')(label_sim_dict)
        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model = Model(bert.model.input + [label_input], concat_output)
        # self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=two_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        self.model.compile(loss=loss1, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=loss2, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        self.model.summary()

    def lcm_evaluate(self, model, inputs, y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(y_true, predictions), 5)
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                               verbose=0, epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model, [X_token_train, X_seg_train, L_train], y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model, [X_token_val, X_seg_val, L_val], y_val)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM two loss)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    current_test_score = self.lcm_evaluate(self.model, [X_token_test, X_seg_test, L_test], y_test)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm.h5')
                            print('  best model saved!')
            else:
                self.basic_predictor.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                                         verbose=0, epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train, L_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions), 5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_two_loss_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score,
                      ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test, L_test])
                    predictions = np.argmax(pred_probs, axis=1)
                    current_test_score = round(accuracy_score(y_test, predictions), 5)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm.h5')
                            print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


# class BERT_two_loss:
#     def __init__(self, maxlen, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim=768, model_type='bert',
#                  label_embedding_matrix=None):
#         self.num_classes = num_classes
#         self.dense = Dense(num_classes, activation='softmax')
#         self.dense1 = Dense(num_classes, activation='softmax')
#
#         def lcm_loss(y_true, y_pred, alpha=alpha):
#             pred_probs = y_pred[:, :num_classes]
#             label_sim_dist = y_pred[:,  num_classes: num_classes * 2]
#             label_input_dense = y_pred[0, num_classes * 2:]
#
#             # pred_probs = K.mean(pred_probs, axis=1)
#             # label_sim_dist = K.mean(label_sim_dist, axis=1)
#
#             simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
#             loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
#             loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
#
#             # n = K.argmax(y_true)
#             one_hot = K.eye(num_classes)
#             one_hot = K.reshape(one_hot, shape=(1, -1))
#             # return loss1 + loss2 + (K.categorical_crossentropy(y_true, pred_probs)) * 1 + K.categorical_crossentropy(one_hot, label_input_dense)
#             return loss1 + loss2 + K.categorical_crossentropy(one_hot, label_input_dense)
#
#         def ls_loss(y_true, y_pred, e=0.1):
#             loss1 = K.categorical_crossentropy(y_true, y_pred)
#             loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
#             return (1 - e) * loss1 + e * loss2
#
#         def two_loss(y_true, y_pred):
#             pred_prob = y_pred[:, :num_classes]
#             simulated_y_true = y_pred[:, num_classes:]
#             loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
#             loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
#             return loss1 + loss2 + (K.categorical_crossentropy(y_true, simulated_y_true)) * 1
#
#         def loss1(y_true, y_pred):
#             pred_prob = y_pred[:, :num_classes]
#             return K.categorical_crossentropy(y_true, pred_prob)
#
#         def loss2(y_true, y_pred):
#             pred_prob = y_pred[:, :num_classes]
#             simulated_y_true = y_pred[:, num_classes:]
#             loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
#             loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
#             return loss1 + loss2
#
#         def loss_y_true(y_true, y_pred):
#             pred_prob = y_pred[:, :num_classes]
#             simulated_y_true = y_pred[:, num_classes:]
#             loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
#             loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
#             return loss1 + loss2 + (K.categorical_crossentropy(y_true, pred_prob)) * 1
#
#         # text_encoder:
#
#         bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
#                                        model=model_type, return_keras_model=False, sequence_length=maxlen, dropout_rate=0.3)
#
#         # label_encoder:
#         label_input = Input(shape=(num_classes,), name='label_input')
#         if label_embedding_matrix is None:  # 不使用pretrained embedding
#             label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(
#                 label_input)  # (n,wvdim)
#         else:
#             label_emb = Embedding(num_classes, wvdim, input_length=num_classes, weights=[label_embedding_matrix],
#                                   name='label_emb1')(label_input)
#         #         label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
#         label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)
#
#         text_emb = Lambda(lambda x: x[:, 1: -1], name='CLS-token')(bert.model.output)
#         # text_emb = bert.model.output
#         text_emb = Dense(hidden_size, activation='tanh')(text_emb)
#         doc_product = Dot(axes=(2))([text_emb, label_emb])
#
#         # softmax_doc_product = Softmax()(doc_product)
#
#         weight = Lambda(lambda x: K.expand_dims(x, axis=-2))(doc_product)
#         input_vec = Lambda(lambda x: K.expand_dims(x, axis=-1))(text_emb)
#         # input_vec = Lambda(lambda x: K.sum(tf.matmul(x[0], x[1]), axis=-1))([input_vec, weight])
#         input_vec = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1))([input_vec, weight])
#         input_vec = GlobalAveragePooling1D()(input_vec)
#         pred_probs = self.dense(input_vec)
#         self.basic_predictor = Model(bert.model.input + [label_input], pred_probs)
#         AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
#         self.basic_predictor.compile(loss='categorical_crossentropy',
#                                      optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
#
#         # similarity part:
#         # doc_product = Dot(axes=(2, 1))([label_emb, text_emb])  # (n,d) dot (d,1) -  -> (n,1)
#         # label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)
#
#         # 对注意力矩阵加入和label_emb的交互机制
#         doc_product_emb = Lambda(lambda x: K.expand_dims(x, axis=-1))(doc_product)
#         label_input_emb = Lambda(lambda x: K.expand_dims(x, axis=1))(label_emb)
#         doc_product = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1))([doc_product_emb, label_input_emb])
#         # 结束
#
#         # doc_product = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(doc_product)
#         # label_sim_dict = Dense(1)(doc_product)
#         # label_sim_dict = Lambda(lambda x: K.reshape(x, shape=(-1, num_classes)))(label_sim_dict)
#         label_sim_dict = GlobalAveragePooling1D()(doc_product)
#         # label_sim_dict = doc_product
#         label_sim_dict = Softmax(name='label_sim_dict')(label_sim_dict)
#
#         # label_emb = GlobalAveragePooling1D()(label_emb)
#         label_input_dense = self.dense(label_emb)
#         label_input_dense = Flatten()(label_input_dense)
#         # label_sim_dict = K.tile(K.expand_dims(label_sim_dict, axis=1), n=[1, num_classes, 1])
#         # pred_probs = K.tile(K.expand_dims(pred_probs, axis=1), n=[1, num_classes, 1])
#
#         # concat output:
#         concat_output = Concatenate()([pred_probs, label_sim_dict, label_input_dense])
#         # compile；
#         AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
#         self.model = Model(bert.model.input + [label_input], concat_output)
#         self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
#         # self.model.compile(loss=two_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
#         # self.model.compile(loss=loss_y_true, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
#         # self.model.compile(loss=loss1, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
#         # self.model.compile(loss=loss2, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
#         # self.model.summary()
#
#     def lcm_evaluate(self, model, inputs, y_true):
#         outputs = model.predict(inputs)
#         pred_probs = outputs[:, :self.num_classes]
#         predictions = np.argmax(pred_probs, axis=1)
#         acc = round(accuracy_score(y_true, predictions), 5)
#         return acc
#
#     def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
#         X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
#         best_val_score = 0
#         test_score = 0
#         train_score_list = []
#         val_socre_list = []
#         """实验说明：
#         每一轮train完，在val上测试，记录其accuracy，
#         每当val-acc达到新高，就立马在test上测试，得到test-acc，
#         这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
#         """
#
#         L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
#         L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
#         L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])
#
#         for i in range(epochs):
#             t1 = time.time()
#             if i < lcm_stop:
#                 self.model.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
#                                verbose=0, epochs=1)
#                 # record train set result:
#                 train_score = self.lcm_evaluate(self.model, [X_token_train, X_seg_train, L_train], y_train)
#                 train_score_list.append(train_score)
#                 # validation:
#                 val_score = self.lcm_evaluate(self.model, [X_token_val, X_seg_val, L_val], y_val)
#                 val_socre_list.append(val_score)
#                 t2 = time.time()
#                 print('(LCM two loss)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
#                       val_score)
#                 # save best model according to validation & test result:
#                 if val_score > best_val_score:
#                     best_val_score = val_score
#                     print('Current Best model!', 'current epoch:', i + 1)
#                     # test on best model:
#                     current_test_score = self.lcm_evaluate(self.model, [X_token_test, X_seg_test, L_test], y_test)
#                     if current_test_score > test_score:
#                         test_score = current_test_score
#                         print('  Current Best model! Test score:', test_score)
#                         if save_best:
#                             self.model.save('best_model_bert_two_loss.h5')
#                             print('  best model saved!')
#             else:
#                 self.basic_predictor.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
#                                          verbose=0, epochs=1)
#                 # record train set result:
#                 pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train, L_train])
#                 predictions = np.argmax(pred_probs, axis=1)
#                 train_score = round(accuracy_score(y_train, predictions), 5)
#                 train_score_list.append(train_score)
#                 # validation:
#                 pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val, L_val])
#                 predictions = np.argmax(pred_probs, axis=1)
#                 val_score = round(accuracy_score(y_val, predictions), 5)
#                 val_socre_list.append(val_score)
#                 t2 = time.time()
#                 print('(LCM_two_loss_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score,
#                       ' | val acc:',
#                       val_score)
#                 # save best model according to validation & test result:
#                 if val_score > best_val_score:
#                     best_val_score = val_score
#                     print('Current Best model!', 'current epoch:', i + 1)
#                     # test on best model:
#                     pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test, L_test])
#                     predictions = np.argmax(pred_probs, axis=1)
#                     current_test_score = round(accuracy_score(y_test, predictions), 5)
#                     if current_test_score > test_score:
#                         test_score = current_test_score
#                         print('  Current Best model! Test score:', test_score)
#                         if save_best:
#                             self.model.save('best_model_bert_two_loss.h5')
#                             print('  best model saved!')
#         return train_score_list, val_socre_list, best_val_score, test_score


class BERT_LCM_loss1:
    def __init__(self, maxlen, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim=768, model_type='bert',
                 label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        def two_loss(y_true, y_pred):
            pred_prob = y_pred[:, :num_classes]
            simulated_y_true = y_pred[:, num_classes:]
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
            return loss1 + loss2 + K.categorical_crossentropy(y_true, simulated_y_true) + K.categorical_crossentropy(y_true, pred_prob)

        def loss1(y_true, y_pred):
            pred_prob = y_pred[:, :num_classes]
            return K.categorical_crossentropy(y_true, pred_prob)

        def loss2(y_true, y_pred):
            pred_prob = y_pred[:, :num_classes]
            simulated_y_true = y_pred[:, num_classes:]
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
            return loss1 + loss2

        def loss1_lcm(y_true, y_pred):
            pred_prob = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
            return loss1 + loss2 + K.categorical_crossentropy(y_true, pred_prob)


        # text_encoder:

        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                       model=model_type, return_keras_model=False, sequence_length=maxlen, dropout_rate=0.3)

        # label_encoder:
        label_input = Input(shape=(num_classes,), name='label_input')
        if label_embedding_matrix is None:  # 不使用pretrained embedding
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(
                label_input)  # (n,wvdim)
        else:
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, weights=[label_embedding_matrix],
                                  name='label_emb1')(label_input)
        #         label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)

        text_emb = Lambda(lambda x: x[:, 1: -1], name='CLS-token')(bert.model.output)
        # text_emb = bert.model.output
        text_emb = Dense(hidden_size, activation='tanh')(text_emb)
        doc_product = Dot(axes=(2))([text_emb, label_emb])

        # softmax_doc_product = Softmax()(doc_product)

        weight = Lambda(lambda x: K.expand_dims(x, axis=2))(doc_product)
        input_vec = Lambda(lambda x: K.expand_dims(x, axis=-1))(text_emb)
        input_vec = Lambda(lambda x: K.sum(tf.matmul(x[0], x[1]), axis=-1))([input_vec, weight])
        input_vec = GlobalAveragePooling1D()(input_vec)
        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec)
        self.basic_predictor = Model(bert.model.input + [label_input], pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss='categorical_crossentropy',
                                     optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))

        # similarity part:
        # doc_product = Dot(axes=(2, 1))([label_emb, text_emb])  # (n,d) dot (d,1) --> (n,1)
        # label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)
        doc_product = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(doc_product)
        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.reshape(x, shape=(-1, num_classes)))(label_sim_dict)
        label_sim_dict = Softmax(name='label_sim_dict')(label_sim_dict)
        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model = Model(bert.model.input + [label_input], concat_output)
        # self.model.compile(loss=loss1_lcm, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        self.model.compile(loss=two_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        self.model.summary()

    def lcm_evaluate(self, model, inputs, y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(y_true, predictions), 5)
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                               verbose=0, epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model, [X_token_train, X_seg_train, L_train], y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model, [X_token_val, X_seg_val, L_val], y_val)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM loss1)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    current_test_score = self.lcm_evaluate(self.model, [X_token_test, X_seg_test, L_test], y_test)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm_loss1.h5')
                            print('  best model saved!')
            else:
                self.basic_predictor.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train),
                                         batch_size=batch_size,
                                         verbose=0, epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train, L_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions), 5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_loss1_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score,
                      ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test, L_test])
                    predictions = np.argmax(pred_probs, axis=1)
                    current_test_score = round(accuracy_score(y_test, predictions), 5)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm_loss1.h5')
                            print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_LCM_Att:

    """
    使用self-attention构建的LCM
    """
    def __init__(self, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim=768, model_type='bert',
                 label_embedding_matrix=None):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:, num_classes:]
            simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)
            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        # text_encoder:
        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                       model=model_type, return_keras_model=False)
        text_emb = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
        text_emb = Dense(hidden_size, activation='tanh')(text_emb)
        pred_probs = Dense(num_classes, activation='softmax')(text_emb)
        self.basic_predictor = Model(bert.model.input, pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss='categorical_crossentropy',
                                     optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))

        # label_encoder by attention:
        label_input = Input(shape=(num_classes,), name='label_input')
        if label_embedding_matrix is None:  # 不使用pretrained embedding
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(
                label_input)  # (n,wvdim)
        else:
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, weights=[label_embedding_matrix],
                                  name='label_emb1')(label_input)
        # self-attention part:
        doc_attention = Lambda(lambda pair: tf.matmul(pair[0],pair[1],transpose_b=True))([label_emb,label_emb]) #  (n,d) * (d,n) -> (n,n)
        attention_scores = Lambda(lambda x:tf.nn.softmax(x))(doc_attention) # (n,n)
        self_attention_seq = Lambda(lambda pair: tf.matmul(pair[0],pair[1]))([attention_scores, label_emb])  # (n,n)*(n,d)->(n,d)

        # similarity part:
        doc_product = Dot(axes=(2, 1))([self_attention_seq, text_emb])  # (n,d) dot (d,1) --> (n,1)
        label_sim_dist = Lambda(lambda x: tf.nn.softmax(x), name='label_sim_dict')(doc_product)
        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dist])
        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.model = Model(bert.model.input + [label_input], concat_output)
        self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))

    def lcm_evaluate(self, model, inputs, y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(y_true, predictions), 5)
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                               verbose=0, epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model, [X_token_train, X_seg_train, L_train], y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model, [X_token_val, X_seg_val, L_val], y_val)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    current_test_score = self.lcm_evaluate(self.model, [X_token_test, X_seg_test, L_test], y_test)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm.h5')
                            print('  best model saved!')
            else:
                self.basic_predictor.fit([X_token_train, X_seg_train], to_categorical(y_train), batch_size=batch_size,
                                         verbose=0, epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions), 5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score,
                      ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test])
                    predictions = np.argmax(pred_probs, axis=1)
                    current_test_score = round(accuracy_score(y_test, predictions), 5)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_lcm.h5')
                            print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_two_loss:
    def __init__(self, maxlen, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim=768, model_type='bert',
                 label_embedding_matrix=None):
        self.num_classes = num_classes

        def  lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:,  num_classes: num_classes * 2]
            # label_input_dense = y_pred[0, num_classes * 2:]

            # pred_probs = K.mean(pred_probs, axis=1)
            # label_sim_dist = K.mean(label_sim_dist, axis=1)

            # simulated_y_true = K.softmax(label_sim_dist + alpha * y_true)
            simulated_y_true = K.softmax((label_sim_dist + 4 * y_true) / alpha)
            # def parameter(x):
            #     return 1 / (1 + K.exp(-1 * x + 0.5)) - 0.5
            # sigma = K.max(simulated_y_true, axis=-1, keepdims=True)
            # sigma = parameter(sigma)
            # simulated_y_true = simulated_y_true * sigma

            pred_probs = K.softmax(pred_probs / alpha)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)

            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        def loss_y_true(y_true, y_pred):

            def parameter(x):
                return 1 / (1 + K.exp(-1 * x + 0.5)) - 0.5
            pred_prob = y_pred[:, :num_classes]
            simulated_y_true = y_pred[:, num_classes: num_classes * 2]
            # for i, (x, y) in enumerate(zip(y_true, simulated_y_true)):
            #     if K.argmax(y_true[ i]) != K.argmax(simulated_y_true[i]):
            #         simulated_y_true[i] = y_true[i]
            # simulated_y_true = [x if K.argmax(x) != K.argmax(y) else y for x, y in zip(y_true, simulated_y_true)]

            simulated_y_true = K.softmax(simulated_y_true + alpha * y_true)
            sigma = K.max(simulated_y_true, axis=-1, keepdims=True)
            sigma = parameter(sigma) * 4
            theta = K.relu(sigma)
            # beta = K.cast(K.argmax(simulated_y_true, axis=-1) == K.argmax(y_true, axis=-1), K.floatx())

            # simulated_y_true = beta * (simulated_y_true * parameter(sigma)) + (1 - beta) * y_true
            simulated_y_true = simulated_y_true * theta + (1 - theta) * y_true
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_prob)
            return loss1 + loss2 + K.categorical_crossentropy(y_true, simulated_y_true)

        # text_encoder:

        self.loss_y_true = loss_y_true
        self.lcm_loss = lcm_loss

        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                       model=model_type, return_keras_model=False, sequence_length=maxlen, dropout_rate=0.3)

        # label_encoder:
        label_input = Input(shape=(num_classes,), name='label_input_1')
        if label_embedding_matrix is None:  # 不使用pretrained embedding
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(
                label_input)  # (n,wvdim)
        else:
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, weights=[label_embedding_matrix],
                                  name='label_emb1')(label_input)
        #         label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)
        # label_emb = Dense(num_classes, activation='tanh', name='label_emb2')(label_emb)

        text_emb = Lambda(lambda x: x[:, 1: -1], name='CLS-token')(bert.model.output)
        # text_emb = bert.model.output
        text_emb = Dense(hidden_size, activation='tanh')(text_emb)
        # text_emb = Dense(num_classes, activation='tanh')(text_emb)
        text_emb = Dropout(0.4)(text_emb)

        text_emb_d = Lambda(lambda x: K.expand_dims(x, axis=1))(text_emb)
        label_emb_d = Lambda(lambda x: K.expand_dims(x, axis=2))(label_emb)
        output_1 = Dense(hidden_size * 4)(text_emb_d)
        output_2 = Dense(hidden_size * 4)(label_emb_d)
        output1 = Lambda(lambda x: x[0] + x[1])([output_1, output_2])
        output1 = Lambda(lambda x: K.tanh(x))(output1)

        # weight = Dense(1)(output1)
        weight_d = Softmax()(output1)
        weight = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([weight_d, output1])

        doc_product_d = Lambda(lambda x: K.softmax(x, axis=1))(weight)
        doc_product = Lambda(lambda x: K.squeeze(x, axis=-1))(weight)
        input_vec = Lambda(lambda x: K.sum(x[0] * x[1], axis=1))([doc_product_d, text_emb_d])

        # input_vec = text_emb
        input_vec_1 = GlobalAveragePooling1D()(input_vec)
        # pred_probs = input_vec_1
        # pred_probs = Softmax()(input_vec_1)
        pred_probs = Dense(num_classes, name='pred_probs')(input_vec_1)

        self.basic_predictor = Model(bert.model.input + [label_input], pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),  # categorical_crossentropy  # keras.losses.CategoricalCrossentropy(from_logits=True)
                                     optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.basic_predictor.compile(loss='categorical_crossentropy',
        #                              optimizer=Adam(learning_rate=5e-5))

        # similarity part:
        # doc_product = Dot(axes=(2, 1))([label_emb, text_emb])  # (n,d) dot (d,1) -  -> (n,1)
        # label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)

        # 对注意力矩阵加入和label_emb的交互机制
        # doc_product_emb = Lambda(lambda x: K.expand_dims(x, axis=-1))(doc_product)
        # label_input_emb = Lambda(lambda x: K.expand_dims(x, axis=1))(label_emb)
        # doc_product = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1))([doc_product_emb, label_input_emb])

        # doc_product_l = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1, 3]))(weight)
        # label_emb_l = Lambda(lambda x: K.expand_dims(x, axis=1))(label_emb)
        # label_vec = Lambda(lambda x: K.sum(x[0] * x[1], axis=2))([doc_product_l, label_emb_l])

        # 结束

        # label_sim_dict = Lambda(lambda x: tf.matmul(x[0], x[1]))([doc_product, input_vec])
        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.squeeze(x, axis=-1))(label_sim_dict)
        # label_sim_dict = Softmax()(label_sim_dict)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(label_sim_dict)

        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        # self.model = Model(bert.model.input + [label_input], concat_output)
        self.model = Model(bert.model.input + [label_input], concat_output)
        self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=lcm_loss, optimizer=Adam(learning_rate=5e-5))
        # self.model.compile(loss=first_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=lcm_loss, optimizer=Adam(learning_rate=1e-5))
        # self.model.compile(loss=two_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=loss_y_true, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=loss1, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.compile(loss=loss2, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.model.summary()

    def lcm_evaluate(self, model, inputs, y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(y_true, predictions), 5)
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                               verbose=0, epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model, [X_token_train, X_seg_train,L_train], y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model, [X_token_val, X_seg_val, L_val], y_val)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM two loss)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score - 0.001:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    current_test_score = self.lcm_evaluate(self.model, [X_token_test, X_seg_test, L_test], y_test)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_two_loss.h5')
                            print('  best model saved!')
            else:
                self.basic_predictor.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                                         verbose=0, epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train, L_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions), 5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_two_loss_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score,
                      ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test, L_test])
                    predictions = np.argmax(pred_probs, axis=1)
                    current_test_score = round(accuracy_score(y_test, predictions), 5)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_two_loss.h5')
                            print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


class BERT_two_loss_no_attention:
    def __init__(self, maxlen, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim=768, model_type='bert',
                 label_embedding_matrix=None):
        self.num_classes = num_classes

        def  lcm_loss(y_true, y_pred, alpha=alpha):
            pred_probs = y_pred[:, :num_classes]
            label_sim_dist = y_pred[:,  num_classes: num_classes * 2]
            simulated_y_true = K.softmax((label_sim_dist + 4 * y_true) / alpha)
            pred_probs = K.softmax(pred_probs / alpha)
            loss1 = -K.categorical_crossentropy(simulated_y_true, simulated_y_true)
            loss2 = K.categorical_crossentropy(simulated_y_true, pred_probs)

            return loss1 + loss2

        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        # text_encoder:
        bert = build_transformer_model(config_path=config_path, checkpoint_path=checkpoint_path,
                                       model=model_type, return_keras_model=False, sequence_length=maxlen, dropout_rate=0.3)

        # label_encoder:
        label_input = Input(shape=(num_classes,), name='label_input_1')
        if label_embedding_matrix is None:  # 不使用pretrained embedding
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(
                label_input)  # (n,wvdim)
        else:
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, weights=[label_embedding_matrix],
                                  name='label_emb1')(label_input)
        #         label_emb = Bidirectional(LSTM(hidden_size,return_sequences=True),merge_mode='ave')(label_emb) # (n,d)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)

        text_emb = Lambda(lambda x: x[:, 1: -1], name='CLS-token')(bert.model.output)
        text_emb = Dense(hidden_size, activation='tanh')(text_emb)
        text_emb = Dropout(0.4)(text_emb)

        text_emb_d = Lambda(lambda x: K.expand_dims(x, axis=1))(text_emb)
        label_emb_d = Lambda(lambda x: K.expand_dims(x, axis=2))(label_emb)
        output_1 = Dense(hidden_size * 4)(text_emb_d)
        output_2 = Dense(hidden_size * 4)(label_emb_d)
        output1 = Lambda(lambda x: x[0] + x[1])([output_1, output_2])
        output1 = Lambda(lambda x: K.tanh(x))(output1)

        # weight = Dense(1)(output1)
        weight_d = Softmax()(output1)
        weight = Lambda(lambda x: K.sum(x[0] * x[1], axis=-1, keepdims=True))([weight_d, output1])

        doc_product = Lambda(lambda x: K.squeeze(x, axis=-1))(weight)
        input_vec = text_emb

        # input_vec = text_emb
        input_vec_1 = GlobalAveragePooling1D()(input_vec)
        # pred_probs = input_vec_1
        # pred_probs = Softmax()(input_vec_1)
        pred_probs = Dense(num_classes, name='pred_probs')(input_vec_1)

        self.basic_predictor = Model(bert.model.input + [label_input], pred_probs)
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        self.basic_predictor.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),  # categorical_crossentropy  # keras.losses.CategoricalCrossentropy(from_logits=True)
                                     optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))
        # self.basic_predictor.compile(loss='categorical_crossentropy',
        #                              optimizer=Adam(learning_rate=5e-5))

        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.squeeze(x, axis=-1))(label_sim_dict)
        # label_sim_dict = Softmax()(label_sim_dict)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(label_sim_dict)

        # concat output:
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        # compile；
        AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
        # self.model = Model(bert.model.input + [label_input], concat_output)
        self.model = Model(bert.model.input + [label_input], concat_output)
        self.model.compile(loss=lcm_loss, optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}))

    def lcm_evaluate(self, model, inputs, y_true):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(y_true, predictions), 5)
        return acc

    def train_val(self, data_package, batch_size, epochs, lcm_stop=50, save_best=False):
        X_token_train, X_seg_train, y_train, X_token_val, X_seg_val, y_val, X_token_test, X_seg_test, y_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，就立马在test上测试，得到test-acc，
        这样最终保留下来的test-acc就是在val上表现最好的模型在test上的accuracy
        """

        L_train = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_train))])
        L_val = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_val))])
        L_test = np.array([np.array(range(self.num_classes)) for i in range(len(X_token_test))])

        for i in range(epochs):
            t1 = time.time()
            if i < lcm_stop:
                self.model.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                               verbose=0, epochs=1)
                # record train set result:
                train_score = self.lcm_evaluate(self.model, [X_token_train, X_seg_train,L_train], y_train)
                train_score_list.append(train_score)
                # validation:
                val_score = self.lcm_evaluate(self.model, [X_token_val, X_seg_val, L_val], y_val)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM two loss no attention)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score, ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score - 0.001:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    current_test_score = self.lcm_evaluate(self.model, [X_token_test, X_seg_test, L_test], y_test)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_two_loss.h5')
                            print('  best model saved!')
            else:
                self.basic_predictor.fit([X_token_train, X_seg_train, L_train], to_categorical(y_train), batch_size=batch_size,
                                         verbose=0, epochs=1)
                # record train set result:
                pred_probs = self.basic_predictor.predict([X_token_train, X_seg_train, L_train])
                predictions = np.argmax(pred_probs, axis=1)
                train_score = round(accuracy_score(y_train, predictions), 5)
                train_score_list.append(train_score)
                # validation:
                pred_probs = self.basic_predictor.predict([X_token_val, X_seg_val, L_val])
                predictions = np.argmax(pred_probs, axis=1)
                val_score = round(accuracy_score(y_val, predictions), 5)
                val_socre_list.append(val_score)
                t2 = time.time()
                print('(LCM_two_loss_stopped_no_attention)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), ' | train acc:', train_score,
                      ' | val acc:',
                      val_score)
                # save best model according to validation & test result:
                if val_score > best_val_score:
                    best_val_score = val_score
                    print('Current Best model!', 'current epoch:', i + 1)
                    # test on best model:
                    pred_probs = self.basic_predictor.predict([X_token_test, X_seg_test, L_test])
                    predictions = np.argmax(pred_probs, axis=1)
                    current_test_score = round(accuracy_score(y_test, predictions), 5)
                    if current_test_score > test_score:
                        test_score = current_test_score
                        print('  Current Best model! Test score:', test_score)
                        if save_best:
                            self.model.save('best_model_bert_two_loss.h5')
                            print('  best model saved!')
        return train_score_list, val_socre_list, best_val_score, test_score


if __name__ == '__main__':
    bert_type = 'bert'
    config_path = '../bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_config.json'
    checkpoint_path = '../bert_weights/bert_tiny_uncased_L-2_H-128_A-2/bert_model.ckpt'
    vocab_path = '../bert_weights/bert_tiny_uncased_L-2_H-128_A-2/vocab.txt'
    hidden_size = 64
    num_classes = 20
    alpha = 4
    wvdim = 256
    maxlen = 128
    model = BERT_two_loss(maxlen, config_path, checkpoint_path, hidden_size, num_classes, alpha, wvdim, bert_type)

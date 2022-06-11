# checked at 2022.2.27
import numpy as np
import time
# import tensorflow.keras
import tensorflow as tf
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Input,Dense,LSTM,Embedding,Conv1D,MaxPooling1D, Softmax, GlobalAveragePooling1D
from tensorflow.keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from gensim.models import Word2Vec
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score


class CNN_Basic:
    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, embedding_matrix=None, filter_sizes=None, num_filters=100):
        text_input = Input(shape=(maxlen, ), name='text_input')
        if embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[embedding_matrix], name='text_emb')(text_input)
        input_emb = Dropout(0.4)(input_emb)

        kernel_sizes = filter_sizes
        out_pool = []
        for kernel_size in kernel_sizes:
            conv1 = Conv1D(num_filters, kernel_size, padding='same', activation='relu')(input_emb)
            pool1 = GlobalAveragePooling1D()(conv1)
            # pool1 = MaxPooling1D(maxlen)(conv1)
            out_pool.append(pool1)
        pooled = Concatenate(axis=-1)(out_pool)
        pooled = Dense(hidden_size)(pooled)
        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(pooled)
        self.model = Model(inputs=text_input, outputs=pred_probs)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_val(self, data_package, batch_size, epochs, save_best=False):
        X_train, y_train, X_val, y_val, X_test, y_test = data_package
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_socre_list = []
        for i in range(epochs):
            t1 = time.time()
            self.model.fit(X_train, to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
            pred_probs = self.model.predict(X_val)
            predictions = np.argmax(pred_probs, axis=1)
            val_score = round(accuracy_score(y_val, predictions), 5)
            t2 = time.time()
            print('(Orig)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
            if val_score > best_val_score - 0.001:
                best_val_score = val_score
                # 使用当前val上最好模型进行test:
                pred_probs = self.model.predict(X_test)
                predictions = np.argmax(pred_probs, axis=1)
                current_test_score = round(accuracy_score(y_test, predictions), 5)
                if current_test_score > final_test_score:
                    final_test_score = current_test_score
                    print('  Current Best model! Test score:', final_test_score)
                    # 同时记录一下train上的score：
                    pred_probs = self.model.predict(X_train)
                    predictions = np.argmax(pred_probs, axis=1)
                    final_train_score = round(accuracy_score(y_train, predictions), 5)
                    print('  Current Best model! Train score:', final_train_score)
                    if save_best:
                        self.model.save('best_model_lstm.h5')
                        print('  best model saved!')
            val_socre_list.append(val_score)
        return best_val_score, val_socre_list, final_test_score, final_train_score


class CNN_LS:
    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, embedding_matrix=None, filter_sizes=None, num_filters=100):
        def ls_loss(y_true, y_pred, e=0.1):
            loss1 = K.categorical_crossentropy(y_true, y_pred)
            loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / num_classes, y_pred)
            return (1 - e) * loss1 + e * loss2

        text_input = Input(shape=(maxlen,), name='text_input')
        if embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[embedding_matrix],
                                  name='text_emb')(text_input)
        input_emb = Dropout(0.4)(input_emb)

        kernel_sizes = filter_sizes
        out_pool = []
        for kernel_size in kernel_sizes:
            conv1 = Conv1D(num_filters, kernel_size, padding='same', activation='relu')(input_emb)
            pool1 = GlobalAveragePooling1D()(conv1)
            out_pool.append(pool1)
        pooled = Concatenate(axis=-1)(out_pool)
        pooled = Dense(hidden_size)(pooled)
        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(pooled)
        self.model = Model(inputs=text_input, outputs=pred_probs)
        self.model.compile(loss=ls_loss, optimizer='adam', metrics=['accuracy'])

    def train_val(self, data_package, batch_size, epochs, save_best=False):
        X_train, y_train, X_val, y_val, X_test, y_test = data_package
        best_val_score = 0
        final_test_score = 0
        final_train_score = 0
        val_socre_list = []
        for i in range(epochs):
            t1 = time.time()
            self.model.fit(X_train, to_categorical(y_train), batch_size=batch_size, verbose=0, epochs=1)
            pred_probs = self.model.predict(X_val)
            predictions = np.argmax(pred_probs, axis=1)
            val_score = round(accuracy_score(y_val, predictions), 5)
            t2 = time.time()
            print('(LS)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
            if val_score > best_val_score - 0.001:
                best_val_score = val_score
                # 使用当前val上最好模型进行test:
                pred_probs = self.model.predict(X_test)
                predictions = np.argmax(pred_probs, axis=1)
                current_test_score = round(accuracy_score(y_test, predictions), 5)
                if current_test_score > final_test_score:
                    final_test_score = current_test_score
                    print('  Current Best model! Test score:', final_test_score)
                    # 同时记录一下train上的score：
                    pred_probs = self.model.predict(X_train)
                    predictions = np.argmax(pred_probs, axis=1)
                    final_train_score = round(accuracy_score(y_train, predictions), 5)
                    print('  Current Best model! Train score:', final_train_score)
                    if save_best:
                        self.model.save('best_model_lstm.h5')
                        print('  best model saved!')
            val_socre_list.append(val_score)
        return best_val_score, val_socre_list, final_test_score, final_train_score


class CNN_LCM:
    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha,default_loss='ls', embedding_matrix=None, filter_sizes=None, num_filters=100):
        self.num_classes = num_classes

        def lcm_loss(y_true, y_pred, alpha=4):
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

        text_input = Input(shape=(maxlen,), name='text_input')
        if embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[embedding_matrix],
                                  name='text_emb')(text_input)
        input_emb = Dropout(0.4)(input_emb)

        kernel_sizes = filter_sizes
        out_pool = []
        for kernel_size in kernel_sizes:
            conv1 = Conv1D(num_filters, kernel_size, padding='same', activation='relu')(input_emb)
            pool1 = GlobalAveragePooling1D()(conv1)
            out_pool.append(pool1)
        pooled = Concatenate(axis=-1)(out_pool)
        input_vec = Dense(hidden_size)(pooled)
        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec)
        self.basic_predictor = Model(inputs=text_input, outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else:
            self.basic_predictor.compile(loss='categorical_crossentropy', optimizer='adam')

        label_input = Input(shape=(num_classes, ), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb')(label_input)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)
        doc_product = Dot(axes=(2, 1))([label_emb, input_vec])
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')

    def my_evaluator(self, model, inputs, label_list):
        outputs = model.predict(inputs)
        pred_probs = outputs[:, :self.num_classes]
        predictions = np.argmax(pred_probs, axis=1)
        acc = round(accuracy_score(label_list, predictions), 5)
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


class CNN_two_loss:
    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, default_loss='ls',
               embedding_matrix=None, filter_sizes=None, num_filters=100):
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

        text_input = Input(shape=(maxlen,), name='text_input')
        if embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[embedding_matrix],
                                  name='text_emb')(text_input)

        label_input = Input(shape=(num_classes,), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb')(label_input)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)
        input_emb = Dropout(0.4)(input_emb)

        kernel_sizes = filter_sizes
        out_pool = []
        for kernel_size in kernel_sizes:
            conv1 = Conv1D(num_filters, kernel_size, padding='same', activation='relu')(input_emb)
            # pool1 = GlobalAveragePooling1D()(conv1)
            pool1 = conv1
            out_pool.append(pool1)
        pooled = Concatenate(axis=-1)(out_pool)
        input_vec = Dense(hidden_size)(pooled)

        text_emb_d = Lambda(lambda x: K.expand_dims(x, axis=1))(input_vec)
        label_emb_d = Lambda(lambda x: K.expand_dims(x, axis=2))(label_emb)
        output_1 = Dense(hidden_size * 4)(text_emb_d)
        output_2 = Dense(hidden_size * 4)(label_emb_d)
        output1 = Lambda(lambda x: x[0] + x[1])([output_1, output_2])
        output1 = Lambda(lambda x: K.tanh(x))(output1)

        weight_1 = Dense(hidden_size * 4)(label_emb_d)
        weight_2 = Dense(hidden_size * 4)(output1)
        weight_d = Lambda(lambda x: K.tanh(x[0] + x[1]))([weight_1, weight_2])

        weight = Lambda(lambda x:K.sum(x[0] * x[1], axis=-1, keepdims=True))([weight_d, output1])
        doc_product = Lambda(lambda x: K.squeeze(x, axis=-1))(weight)
        weight = Lambda(lambda x: K.softmax(x))(weight)
        input_vec = Lambda(lambda x: K.sum(x[0] * x[1], axis=1))([weight, text_emb_d])
        input_vec_1 = GlobalAveragePooling1D()(input_vec)
        pred_probs = Dense(num_classes, name='pred_probs')(input_vec_1)
        # pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec)
        self.basic_predictor = Model(inputs=[text_input, label_input], outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else: # 'categorical_crossentropy'  # 'adam'
            self.basic_predictor.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam')

        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.squeeze(x, axis=-1))(label_sim_dict)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(label_sim_dict)
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')

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
                print('(LCM_two_loss_stopped)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
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


class CNN_two_loss_no_attention:
    def __init__(self, maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, default_loss='ls',
               embedding_matrix=None, filter_sizes=None, num_filters=100):
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

        text_input = Input(shape=(maxlen,), name='text_input')
        if embedding_matrix is None:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, name='text_emb')(text_input)
        else:
            input_emb = Embedding(vocab_size + 1, wvdim, input_length=maxlen, weights=[embedding_matrix],
                                  name='text_emb')(text_input)

        label_input = Input(shape=(num_classes,), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb')(label_input)
        label_emb = Dense(hidden_size, activation='tanh', name='label_emb2')(label_emb)
        input_emb = Dropout(0.4)(input_emb)

        kernel_sizes = filter_sizes
        out_pool = []
        for kernel_size in kernel_sizes:
            conv1 = Conv1D(num_filters, kernel_size, padding='same', activation='relu')(input_emb)
            # pool1 = GlobalAveragePooling1D()(conv1)
            pool1 = conv1
            out_pool.append(pool1)
        pooled = Concatenate(axis=-1)(out_pool)
        input_vec = Dense(hidden_size)(pooled)

        text_emb_d = Lambda(lambda x: K.expand_dims(x, axis=1))(input_vec)
        label_emb_d = Lambda(lambda x: K.expand_dims(x, axis=2))(label_emb)
        output_1 = Dense(hidden_size * 4)(text_emb_d)
        output_2 = Dense(hidden_size * 4)(label_emb_d)
        output1 = Lambda(lambda x: x[0] + x[1])([output_1, output_2])
        output1 = Lambda(lambda x: K.tanh(x))(output1)
        # weight = Dense(1, activation='softmax')(output1)
        weight_d = Softmax()(output1)
        weight = Lambda(lambda x:K.sum(x[0] * x[1], axis=-1, keepdims=True))([weight_d, output1])
        doc_product = Lambda(lambda x: K.squeeze(x, axis=-1))(weight)

        input_vec_1 = GlobalAveragePooling1D()(input_vec)
        pred_probs = Dense(num_classes, name='pred_probs')(input_vec_1)
        # pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(input_vec)
        self.basic_predictor = Model(inputs=[text_input, label_input], outputs=pred_probs)
        if default_loss == 'ls':
            self.basic_predictor.compile(loss=ls_loss, optimizer='adam')
        else: # 'categorical_crossentropy'  # 'adam'
            self.basic_predictor.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), optimizer='adam')
        label_sim_dict = Dense(1)(doc_product)
        label_sim_dict = Lambda(lambda x: K.squeeze(x, axis=-1))(label_sim_dict)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(label_sim_dict)
        concat_output = Concatenate()([pred_probs, label_sim_dict])
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=lcm_loss, optimizer='adam')

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
                print('(LCM_two_loss_stopped_no_attention)Epoch', i + 1, '| time: %.3f s' % (t2 - t1), '| current val accuracy:', val_score)
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
    num_filters = 100
    filter_sizes = [3, 10, 25]  # 不能超过maxlen
    model = CNN_Basic(maxlen, vocab_size, wvdim, hidden_size, num_classes, None, filter_sizes=filter_sizes, num_filters=num_filters)
    # model = CNN_LS(maxlen,vocab_size,wvdim,hidden_size,num_classes,None, filter_sizes=filter_sizes, num_filters=num_filters)
    # model = CNN_LCM(maxlen,vocab_size,wvdim,hidden_size,num_classes,alpha,None,None, filter_sizes=filter_sizes, num_filters=num_filters)
    # model = CNN_two_loss(maxlen, vocab_size, wvdim, hidden_size, num_classes, alpha, None, None, filter_sizes=filter_sizes, num_filters=num_filters)
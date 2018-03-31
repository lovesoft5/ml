
import os
import datetime
import random
import tensorflow as tf
import numpy as np

class PoetryModel:

    def __init__(self):
        # 诗歌生成
        self.poetry = Poetry()
        # 单个cell训练序列个数
        self.batch_size = self.poetry.batch_size
        # 所有出现字符的数量
        self.word_len = len(self.poetry.word_to_int)
        # 隐层的数量
        self.rnn_size = 128

    @staticmethod
    def embedding_variable(inputs, rnn_size, word_len):
        with tf.variable_scope('embedding'):
            # 这里选择使用cpu进行embedding
            with tf.device("/cpu:0"):
                # 默认使用'glorot_uniform_initializer'初始化，来自源码说明:
                # If initializer is `None` (the default), the default initializer passed in
                # the variable scope will be used. If that one is `None` too, a
                # `glorot_uniform_initializer` will be used.
                # 这里实际上是根据字符数量分别生成state_size长度的向量
                embedding = tf.get_variable('embedding', [word_len, rnn_size])
                # 根据inputs序列中每一个字符对应索引 在embedding中寻找对应向量,即字符转为连续向量:[字]==>[1]==>[0,1,0]
                lstm_inputs = tf.nn.embedding_lookup(embedding, inputs)
        return lstm_inputs

    @staticmethod
    def soft_max_variable(rnn_size, word_len):
        # 共享变量
        with tf.variable_scope('soft_max'):
            w = tf.get_variable("w", [rnn_size, word_len])
            b = tf.get_variable("b", [word_len])
        return w, b

    def rnn_graph(self, batch_size, rnn_size, word_len, lstm_inputs, keep_prob):
        # cell.state_size ==> 128
        # 基础cell 也可以选择其他基本cell类型
        lstm = tf.nn.rnn_cell.BasicLSTMCell(num_units=rnn_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        # 多层cell 前一层cell作为后一层cell的输入
        cell = tf.nn.rnn_cell.MultiRNNCell([drop] * 2)
        # 初始状态生成(h0) 默认为0
        # initial_state.shape ==> (64, 128)
        initial_state = cell.zero_state(batch_size, tf.float32)
        # 使用dynamic_rnn自动进行时间维度推进 且 可以使用不同长度的时间维度
        # 因为我们使用的句子长度不一致
        lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, lstm_inputs, initial_state=initial_state)
        seq_output = tf.concat(lstm_outputs, 1)
        print("seq_output-->",seq_output)
        x = tf.reshape(seq_output, [-1, rnn_size])
        print("XXXXXXXXXXXXX",x)
        # softmax计算概率
        w, b = self.soft_max_variable(rnn_size, word_len)
        logits = tf.matmul(x, w) + b
        prediction = tf.nn.softmax(logits, name='predictions')
        print("logits---------", logits)
        print("prediction---------",prediction)
        return logits, prediction, initial_state, final_state

    @staticmethod
    def loss_graph(word_len, targets, logits):
        # 将y序列按序列值转为one_hot向量
        y_one_hot = tf.one_hot(targets, word_len)
        y_reshaped = tf.reshape(y_one_hot, [-1, word_len])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        return loss

    @staticmethod
    def optimizer_graph(loss, learning_rate):
        grad_clip = 5
        # 使用clipping gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
        return optimizer

    def train(self, epoch):
        # 输入句子长短不一致 用None自适应
        inputs = tf.placeholder(tf.int32, shape=(self.batch_size, None), name='inputs')
        # 输出为预测某个字后续字符 故输出也不一致
        targets = tf.placeholder(tf.int32, shape=(self.batch_size, None), name='targets')
        # 防止过拟合
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        # 将输入字符对应索引转化为变量
        print("self.rnn_size",self.rnn_size)
        print("self.word_len",self.word_len)
        lstm_inputs = self.embedding_variable(inputs, self.rnn_size, self.word_len)
        print("lstm_inputs", lstm_inputs)
        # rnn模型
        logits, _, initial_state, final_state = self.rnn_graph(self.batch_size, self.rnn_size, self.word_len, lstm_inputs, keep_prob)
        print("logits",logits)
        print("initial_state",initial_state)
        print("final_state",final_state)
        # 损失
        loss = self.loss_graph(self.word_len, targets, logits)
        # 优化
        learning_rate = tf.Variable(0.0, trainable=False)
        optimizer = self.optimizer_graph(loss, learning_rate)

        # 开始训练
        saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step = 0
        new_state = sess.run(initial_state)
        for i in range(epoch):
            # 训练数据生成器
            batches = self.poetry.batch()
            # 随模型进行训练 降低学习率
            sess.run(tf.assign(learning_rate, 0.001 * (0.97 ** i)))
            for batch_x, batch_y in batches:
                print("batch_xiiiiiiiiiiiii",batch_x)
                print("batch_yiiiiiiiiiiiii",batch_y)
                print("batch_x++++++++++++", batch_x.shape)
                print("batch_y------------", batch_y.shape)
                feed = {inputs: batch_x, targets: batch_y, initial_state: new_state, keep_prob: 0.5}
                batch_loss, _, new_state = sess.run([loss, optimizer, final_state], feed_dict=feed)
                print(datetime.datetime.now().strftime('%c'), ' i:', i, 'step:', step, ' batch_loss:', batch_loss)
                step += 1
        model_path = os.getcwd() + os.sep + "poetry.model"
        saver.save(sess, model_path, global_step=step)
        sess.close()

    def gen(self, poem_len):
        def to_word(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            sample = int(np.searchsorted(t, np.random.rand(1) * s))
            return self.poetry.int_to_word[sample]

        # 输入
        # 句子长短不一致 用None自适应
        self.batch_size = 1
        inputs = tf.placeholder(tf.int32, shape=(self.batch_size, 1), name='inputs')
        # 防止过拟合
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        lstm_inputs = self.embedding_variable(inputs, self.rnn_size, self.word_len)
        # rnn模型
        _, prediction, initial_state, final_state = self.rnn_graph(self.batch_size, self.rnn_size, self.word_len, lstm_inputs, keep_prob)

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, tf.train.latest_checkpoint('.'))
            new_state = sess.run(initial_state)

            # 在所有字中随机选择一个作为开始
            x = np.zeros((1, 1))
            x[0, 0] = self.poetry.word_to_int[self.poetry.int_to_word[random.randint(1, self.word_len-1)]]
            feed = {inputs: x, initial_state: new_state, keep_prob: 1}

            predict, new_state = sess.run([prediction, final_state], feed_dict=feed)
            word = to_word(predict)
            poem = ''
            while len(poem) < poem_len:
                poem += word
                x = np.zeros((1, 1))
                x[0, 0] = self.poetry.word_to_int[word]
                feed = {inputs: x, initial_state: new_state, keep_prob: 1}
                predict, new_state = sess.run([prediction, final_state], feed_dict=feed)
                word = to_word(predict)
            return poem



class Poetry:

    def __init__(self):
        self.poetry_file = 'poetry.txt'
        self.poetry_list = self._get_poetry()
        self.poetry_vectors, self.word_to_int, self.int_to_word = self._gen_poetry_vectors()
        self.batch_size = 64
        self.chunk_size = len(self.poetry_vectors) // self.batch_size

    def _get_poetry(self):
        with open(self.poetry_file, "r", encoding='utf-8') as f:
            poetry_list = [line for line in f]
        print("poetry_list",poetry_list[0])
        print("poetry_list.shaope------", poetry_list[1])
        return poetry_list[:300]

    def _gen_poetry_vectors(self):
        words = sorted(set(''.join(self.poetry_list)+' '))
        for i in range(10):
            print("words",words[i])
        # 每一个字符分配一个索引 为后续诗词向量化做准备
        int_to_word = {i: word for i, word in enumerate(words)}
        word_to_int = {v: k for k, v in int_to_word.items()}
        to_int = lambda word: word_to_int.get(word)
        print("int_to_word", int_to_word.get(0))
        print("to_int",map(to_int, self.poetry_list))
        poetry_vectors = [list(map(to_int, poetry)) for poetry in self.poetry_list]
        print("poetry_vectors[0]", poetry_vectors[0])
        print("poetry_vectors[1]", poetry_vectors[1])
        for i in range(1):
            print("word_to_int", word_to_int.items())
        return poetry_vectors, word_to_int, int_to_word

    def batch(self):
        # 生成器
        start = 0
        end = self.batch_size
        print("self.chunk_size",self.chunk_size)
        for _ in range(self.chunk_size):
            batches = self.poetry_vectors[start:end]
            print("batches",batches[0])
            print("batches============", len(batches))
            # 输入数据 按每块数据中诗句最大长度初始化数组，缺失数据补全
            x_batch = np.full((self.batch_size, max(map(len, batches))), self.word_to_int[' '], np.int32)
            print("x_batch1",x_batch.shape)
            for row in range(self.batch_size): x_batch[row, :len(batches[row])] = batches[row]
            print("batches[row]", batches[row])
            # 标签数据 根据上一个字符预测下一个字符 所以这里y_batch数据应为x_batch数据向后移一位
            y_batch = np.copy(x_batch)
            y_batch[:, :-1], y_batch[:, -1] = x_batch[:, 1:], x_batch[:, 0]
            print("x_batch2",x_batch)
            print("y_batch",y_batch)
            yield x_batch, y_batch
            start += self.batch_size
            end += self.batch_size
            print("end",end)
'''
if __name__ == '__main__':
    data = Poetry().batch()
    for x, y in data:
        print(x)
'''





if __name__ == '__main__':
    poetry = PoetryModel()
    poetry.train(epoch=1)

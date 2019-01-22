import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
strokes = np.load('../data/strokes.npy', encoding='latin1')
stroke = strokes[0]

lstm_size = 100
lstm_sizes = [lstm_size]
nb_mixture = 12
batch_size = 1
T = 300 # = corresponding to the number of point in a stroke
learning_rate = 0.05
epoch = 10
nb_batch = 10
def create_and_train():

    def model_inputs():
        input_data = tf.placeholder(dtype=tf.float32,
                                    shape=[None, None, 3], name="input_data")
        target_data = tf.placeholder(dtype=tf.float32,
                                     shape=[None, T, 3], name='target_data')
        return input_data, target_data

    def build_lstm_layers(lstm_sizes, X, batch_size):
        lstms = [tf.contrib.rnn.BasicLSTMCell(size) for size in lstm_sizes]
        # drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1) for lstm in lstms]
        cell = tf.contrib.rnn.MultiRNNCell(lstms)
        initial_state = cell.zero_state(batch_size, tf.float32)
        state_in = tf.identity(initial_state, name='state_in')
        lstm_output, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state)
        return state_in, lstm_output, final_state, cell

    def build_output(lstm_output):
        output_w = tf.Variable(tf.random_normal([lstm_size, 6 * nb_mixture + 1]))
        output_b = tf.Variable(tf.zeros([6 * nb_mixture + 1]))
        output = tf.nn.xw_plus_b(lstm_output, output_w, output_b)
        return output

    def get_mixture_coef(output):
        eos = output[:, 0:1]
        pi, mu1, mu2, sigma1, sigma2, corr = tf.split(axis=1, num_or_size_splits=6, value=output[:, 1:])
        eos = tf.sigmoid(eos)
        pi = tf.nn.softmax(pi)
        #max_pi = tf.reduce_max(pi, 1, keepdims=True)
        #pi = tf.subtract(pi, max_pi)
        #pi = tf.exp(pi)
        #normalize_pi = tf.reciprocal(tf.reduce_sum(pi, 1, keepdims=True))
        #pi = tf.multiply(normalize_pi, pi)
        sigma1 = tf.exp(sigma1)
        sigma2 = tf.exp(sigma2)
        corr = tf.tanh(corr)
        corr = tf.clip_by_value(corr, -0.99999, 0.99999)  # Make sure denom > O
        return pi, mu1, mu2, sigma1, sigma2, corr, eos

    def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
        #eq 24 & 25
        norm1 = tf.subtract(x1, mu1)
        norm2 = tf.subtract(x2, mu2)
        s1s2 = tf.multiply(s1, s2)
        z = tf.square(tf.div(norm1, s1)) + tf.square(tf.div(norm2, s2)) - \
            2 * tf.div(tf.multiply(rho, tf.multiply(norm1, norm2)), s1s2)
        rhoterm = 1 - tf.square(rho)
        result = tf.exp(tf.div(-z, 2 * rhoterm))
        denom = 2 * np.pi * tf.multiply(s1s2, tf.sqrt(rhoterm))
        result = tf.div(result, denom)
        return result

    def get_lossfunc(pi, mu1, mu2, sigma1,sigma2,
                    corr, eos, x1_target, x2_target, eos_target):
        #eq 26
        epsilon = 0.000001
        res = tf.multiply(tf_2d_normal(x1_target, x2_target, mu1, mu2, sigma1, sigma2, corr), pi)
        res = tf.reduce_sum(res, 1, keep_dims=True)
        res = -tf.log(res + epsilon) #make sure log(x) with x>0
        res2 = tf.multiply(eos, eos_target) + tf.multiply(1 - eos, 1 - eos_target)
        res2 = -tf.log(res2 + epsilon)
        return tf.reduce_mean(res + res2)

    #build the model
    input_data, target_data = model_inputs()

    state_in, lstm_outputs, final_state, cell = build_lstm_layers(lstm_sizes, input_data, batch_size)

    lstm_outputs = tf.reshape(tf.concat(axis=1, values=lstm_outputs), [-1, lstm_size])

    output = build_output(lstm_outputs)

    [pi, mu1, mu2, sigma1, sigma2, corr, eos] = get_mixture_coef(output)

    target_data_new = tf.reshape(target_data, [-1, 3])

    [eos_data, x1_data, x2_data] = tf.split(axis=1, num_or_size_splits=3, value=target_data_new)

    cost = get_lossfunc(pi, mu1, mu2, sigma1, sigma2, corr, eos, x1_data, x2_data, eos_data)
    train = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

    def launch_train(epoch, nb_batch):
        save_cost = np.zeros(epoch)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        for e in range(epoch):
            state = state_in.eval(session=sess)

            for b in range(nb_batch):
                x_data = np.zeros((batch_size, 300, 3))
                y_data = np.zeros((batch_size, 300, 3))
                for i in range(batch_size):
                    x_data[i] = strokes[b + i][0: 300]
                    y_data[i] = strokes[b + i][1: 301]

                cost_, train_, state = sess.run([cost, train, final_state],
                                           feed_dict={
                                               input_data: x_data,
                                               target_data: y_data,
                                               state_in: state
                                           })
                print(' epoch n°' + str(e) + ' batch n°' + str(b) + ' cost = ' + str(cost_))

            save_cost[e] = cost_

        plt.plot(save_cost)
        return sess
    sess = launch_train(epoch,nb_batch)
    return sess, cell, mu1, mu2, sigma1, sigma2, corr, eos, final_state, input_data, state_in

sess, cell, mu1, mu2, sigma1, sigma2, corr, eos, final_state, input_data, state_in = create_and_train()

def generate_unconditionally(random_seed=1):
    def generate_stroke(sess, T):
        prev_state = sess.run(cell.zero_state(1, tf.float32))

        def random_gaussian_2d(mu1, mu2, sigma1, sigma2, corr):
            mean = [mu1, mu2]
            cov = [[sigma1 * sigma1, corr * sigma1 * sigma2], [corr * sigma1 * sigma2, sigma2 * sigma2]]
            x = np.random.multivariate_normal(mean, cov, 1)
            return x[0][0], x[0][1]

        prev_x = np.zeros((1, 1, 3), dtype=np.float32)
        prev_x[0, 0, 0] = 1
        strokes = np.zeros((T, 3), dtype=np.float32)
        for i in range(T):
            [mu1_, mu2_, sigma1_, sigma2_,
             corr_, eos_, next_state] = sess.run([mu1, mu2, sigma1, sigma2, corr, eos, final_state],
                                    feed_dict={input_data: prev_x, state_in: prev_state}
                                    )

            index = random.randint(0, nb_mixture - 1)
            if eos_[0][0] < 0.5:
                new_eos = 0
            else:
                new_eos = 1
            next_x1, next_x2 = random_gaussian_2d(
                mu1_[0][index], mu2_[0][index], sigma1_[0][index], sigma2_[0][index], corr_[0][index])

            strokes[i, :] = [new_eos, next_x1, next_x2]

            prev_x = np.zeros((1, 1, 3), dtype=np.float32)
            prev_x[0][0] = np.array([new_eos, next_x1, next_x2], dtype=np.float32)
            prev_state = next_state
        return strokes

    stroke = generate_stroke(sess, T)
    return stroke


def generate_conditionally(text='welcome to lyrebird', random_seed=1):
    # Input:
    #   text - str
    #   random_seed - integer

    # Output:
    #   stroke - numpy 2D-array (T x 3)
    return stroke


def recognize_stroke(stroke):
    # Input:
    #   stroke - numpy 2D-array (T x 3)

    # Output:
    #   text - str
    return 'welcome to lyrebird'
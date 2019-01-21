import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import random
sys.path.insert(0, '..')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


strokes = np.load('../data/strokes.npy', encoding='latin1')


lstm_size = 100
lstm_sizes = [lstm_size]
nb_mixture = 12
batch_size = 10
len_sequence = 300


def model_inputs():
    input_data = tf.placeholder(dtype=tf.float32,
                                shape=[None, None, 3], name="input_data")
    target_data = tf.placeholder(dtype=tf.float32,
                                 shape=[None, len_sequence, 3], name='target_data')
    return input_data, target_data


def build_lstm_layers(lstm_sizes, X, batch_size):
    lstms = [tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True) for size in lstm_sizes]
    drops = [tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=1) for lstm in lstms]
    cell = tf.contrib.rnn.MultiRNNCell(drops)
    initial_state = cell.zero_state(batch_size, tf.float32)
    state_in = tf.identity(initial_state, name='state_in')
    lstm_output, final_state = tf.nn.dynamic_rnn(cell, X, initial_state=initial_state)
    return state_in, lstm_output, final_state, cell

def build_output(lstm_output):
    output_w = tf.Variable(tf.random_normal([lstm_size, 6*nb_mixture+1]))
    output_b = tf.Variable(tf.zeros([6*nb_mixture+1]))
    output = tf.nn.xw_plus_b(lstm_output, output_w, output_b)
    return output


def get_mixture_coef(output):
    eos = output[:, 0:1]
    pi, mu1, mu2, sigma1, sigma2, corr = tf.split(axis=1, num_or_size_splits=6, value=output[:, 1:])
    eos = tf.sigmoid(eos)
    max_pi = tf.reduce_max(pi, 1, keepdims=True)
    pi = tf.subtract(pi, max_pi)
    pi = tf.exp(pi)
    normalize_pi = tf.reciprocal(
        tf.reduce_sum(pi, 1, keepdims=True))
    pi = tf.multiply(normalize_pi, pi)

    sigma1 = tf.exp(sigma1)
    sigma2 = tf.exp(sigma2)
    corr = tf.tanh(corr)
    corr= tf.clip_by_value(corr, -0.99999, 0.99999) #Mke sure denom >O
    return pi, mu1, mu2, sigma1, sigma2, corr, eos


def tf_2d_normal(x1, x2, mu1, mu2, s1, s2, rho):
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

def get_lossfunc(
        pi,
        mu1,
        mu2,
        sigma1,
        sigma2,
        corr,
        eos,
        x1_data,
        x2_data,
        eos_data):
    result0 = tf_2d_normal(x1_data, x2_data,mu1,mu2,sigma1,sigma2,corr)
    epsilon = 0.000001
    result1 = tf.multiply(result0, pi)
    result1 = tf.reduce_sum(result1, 1, keep_dims=True)
    result1 = -tf.log(tf.maximum(result1, epsilon))
    result2 = tf.multiply(eos, eos_data) + tf.multiply(1 - eos, 1 - eos_data)
    result2 = -tf.log(tf.maximum(result2, epsilon))
    result = result1 + result2
    return tf.reduce_mean(result)

input_data, target_data = model_inputs()
state_in, lstm_outputs, final_state, cell = build_lstm_layers(lstm_sizes, input_data, batch_size)

lstm_outputs = tf.reshape(
            tf.concat(axis=1, values=lstm_outputs), [-1, lstm_size])

output = build_output(lstm_outputs)

[pi, mu1, mu2, sigma1, sigma2, corr, eos] = get_mixture_coef(output)

target_data_new = tf.reshape(target_data, [-1, 3])

[eos_data, x1_data, x2_data] = tf.split(
           axis=1, num_or_size_splits=3, value=target_data_new)

cost = get_lossfunc(pi, mu1, mu2, sigma1, sigma2, corr, eos, x1_data, x2_data, eos_data)
train = tf.train.AdadeltaOptimizer(1).minimize(cost)

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
                x_data[i] = strokes[b+i][0: 300]
                y_data[i] = strokes[b + i][1: 301]

            cost_, _, state = sess.run([cost, train, final_state],
                        feed_dict={
                          input_data: x_data,
                          target_data: y_data,
                          state_in: state
                         })
            print(' epoch n°' + str(e)+ ' batch n°' + str(b) + ' cost = '+ str(cost_))

        save_cost[e] = cost_  # *(batch_size*len_sequence)

    plt.plot(save_cost)
    return sess

def sample(sess, len_sequence):
    prev_state = sess.run(cell.zero_state(1, tf.float32))
    def random_gaussian_2d(mu1, mu2, s1, s2, rho):
        mean = [mu1, mu2]
        cov = [[s1 * s1, rho * s1 * s2], [rho * s1 * s2, s2 * s2]]
        x = np.random.multivariate_normal(mean, cov, 1)
        return x[0][0], x[0][1]

    prev_x = np.zeros((1, 1, 3), dtype=np.float32)
    prev_x[0, 0, 0] = 1
    strokes = np.zeros((len_sequence, 3), dtype=np.float32)
    for i in range(len_sequence):
        [pi_,
         mu1_,
         mu2_,
         sigma1_,
         sigma2_,
         corr_,
         eos_,
         next_state] = sess.run([pi, mu1, mu2, sigma1, sigma2, corr, eos, final_state],
                                feed_dict={input_data: prev_x, state_in: prev_state}
                                )

        index = random.randint(0, nb_mixture-1)
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


def plot_stroke(stroke, save_name=None):
    f, ax = plt.subplots()

    x = np.cumsum(stroke[:, 1])
    y = np.cumsum(stroke[:, 2])

    size_x = x.max() - x.min() + 1.
    size_y = y.max() - y.min() + 1.

    f.set_size_inches(5. * size_x / size_y, 5.)

    cuts = np.where(stroke[:, 0] == 1)[0]
    start = 0

    for cut_value in cuts:
        ax.plot(x[start:cut_value], y[start:cut_value],
                'k-', linewidth=3)
        start = cut_value + 1
    ax.axis('equal')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if save_name is None:
        plt.show()
    else:
        try:
            plt.savefig(
                save_name,
                bbox_inches='tight',
                pad_inches=0.5)
        except Exception:
            print ("Error building image!: " + save_name)

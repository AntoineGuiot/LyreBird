import numpy as np
import matplotlib.pyplot as plt
import sys
import tensorflow as tf
import random
sys.path.insert(0, '..')
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


K = 15
size = 100
batch_size = 1
len_text = 26 # set as 26 for a test of the code
nb_mixtures = 10
T = 55
epoch = 50
nb_batch = 1
alphabeat ='ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz.,/?"'
char_vec_len = len(alphabeat)
learning_rate = 0.01

strokes = np.load('../data/strokes.npy', encoding='latin1')
with open('../data/sentences.txt') as f:
    texts = f.readlines()

def get_one_hot(text, char_list):
    one_hots = []
    char_vec_len = len(char_list)
    for char in text:
        one_hot_char = np.zeros(char_vec_len)
        one_hot_char[char_list.index(char)] = 1
        one_hots.append(one_hot_char)
    return np.array(one_hots)

def model_inputs():
    input_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name="input_data")
    target_data = tf.placeholder(dtype=tf.float32, shape=[None, None, 3], name='target_data')
    char_seq = tf.placeholder(dtype=tf.float32, shape=[None, len_text, char_vec_len]) # matrix representing the sentence
    return input_data, target_data, char_seq


def build_lstm1_layers(X, batch_size):
    X = tf.split(axis=1, num_or_size_splits=T, value=X)
    X = [tf.squeeze(x_i, [1]) for x_i in X]
    lstm1 = tf.contrib.rnn.BasicLSTMCell(size)
    initial_state1 = lstm1.zero_state(batch_size, tf.float32)
    lstm1_output, final_state1 = tf.contrib.legacy_seq2seq.rnn_decoder(X, initial_state1, lstm1, loop_function=None, scope='lstm1')
    return initial_state1, lstm1_output, final_state1, lstm1, X

def get_K_mixtures_input(lstm1_outputs, reuse=True):
    with tf.variable_scope('window', reuse=reuse):
        output_w = tf.Variable(tf.random_normal([size, 3*K]))
        output_b = tf.Variable(tf.zeros([3*K]))
    output = tf.nn.xw_plus_b(lstm1_outputs, output_w, output_b)
    return output

def get_K_gaussian_Coef(output, kappa_prev):
    output = tf.exp(tf.reshape(output, [-1, 3 * K, 1]))
    alpha_hat, beta_hat, kappa_hat = tf.split(axis=1, num_or_size_splits=3, value=output[:, :])
    alpha = tf.exp(alpha_hat)
    alpha = tf.clip_by_value(alpha, 0, 1e20)# make sure alpha, beta, kappa != inf
    beta = tf.exp(beta_hat)
    beta = tf.clip_by_value(beta, 0, 1e20)
    kappa = kappa_prev + tf.exp(kappa_hat)
    kappa = tf.clip_by_value(kappa, 0, 1e20)

    return alpha, beta, kappa

def get_phi(alpha, beta, kappa,len_text):
    u = np.linspace(0, len_text - 1, len_text)
    kappa = tf.square(tf.subtract(kappa, u))
    exp = tf.multiply(-beta, kappa)
    phi = tf.multiply(alpha, tf.exp(exp))
    phi = tf.reduce_sum(phi, 1, keep_dims=True)
    return phi

def get_window(alpha, beta, kappa, char_seq):
    phi = get_phi(alpha, beta, kappa, len_text)
    window = tf.matmul(phi, char_seq)
    window = tf.squeeze(window, [1])
    return window, phi


def build_lstm2_layers(lstm1_output, batch_size):
    lstm2 = tf.contrib.rnn.BasicLSTMCell(size)
    initial_state2 = lstm2.zero_state(batch_size, tf.float32)
    lstm2_output, final_state2 = tf.contrib.legacy_seq2seq.rnn_decoder(lstm1_output, initial_state2, lstm2, loop_function=None)
    return initial_state2, lstm2_output, final_state2, lstm2


def get_mdn_mixtures_input(lstm2_outputs):
    with tf.variable_scope('mdn'):
        output_w2 = tf.Variable(tf.random_normal([size, 6 * nb_mixtures + 1]))
        output_b2 = tf.Variable(tf.zeros([6 * nb_mixtures + 1]))
    output2 = tf.nn.xw_plus_b(tf.reshape(tf.concat(axis=1, values=lstm2_outputs), [-1, size]), output_w2, output_b2)
    return output2


def get_mixture_coef(output):
    eos = output[:, 0:1]
    pi, mu1, mu2, sigma1, sigma2, corr = tf.split(axis=1, num_or_size_splits=6, value=output[:, 1:])
    eos = tf.sigmoid(eos)
    pi = tf.nn.softmax(pi)
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
    res = -tf.log(res + epsilon) #make sure log(x) with x > 0
    res2 = tf.multiply(eos, eos_target) + tf.multiply(1 - eos, 1 - eos_target)
    res2 = -tf.log(res2 + epsilon)
    return tf.reduce_mean(res + res2)

# build the model
input_data, target_data, char_seq = model_inputs()

init_kappa = tf.placeholder(dtype=tf.float32, shape=[None, K, 1])
kappa_prev = init_kappa
w_prev = char_seq[:, 0, :]
reuse = False
initial_state1, lstm1_output, final_state1, lstm1, input = build_lstm1_layers(input_data, batch_size)
for i in range(len(lstm1_output)): # concat input, window and lstm1_output before the second lstm layer
    k_mixtures_input = get_K_mixtures_input(lstm1_output[i], reuse)
    [alpha, beta, new_kappa] = get_K_gaussian_Coef(k_mixtures_input, kappa_prev)
    window, phi = get_window(alpha, beta, new_kappa, char_seq)
    lstm1_output[i] = tf.concat((lstm1_output[i], window), 1)
    lstm1_output[i] = tf.concat((lstm1_output[i], input[i]), 1)
    prev_kappa = new_kappa
    prev_window = window
    reuse = True

initial_state2, lstm2_output, final_state2, lstm2 = build_lstm2_layers(lstm1_output, batch_size)

lstm2_output = tf.reshape(tf.concat(axis=1, values=lstm2_output), [-1, size])

mdn_mixtures_input = get_mdn_mixtures_input(lstm2_output)

[z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos] = get_mixture_coef(mdn_mixtures_input)

target_data_new = tf.reshape(target_data, [-1, 3])

[eos_data, x1_data, x2_data] = tf.split(axis=1, num_or_size_splits=3, value=target_data_new)

cost = get_lossfunc(z_pi, z_mu1, z_mu2, z_sigma1, z_sigma2, z_corr, z_eos, x1_data, x2_data, eos_data)

train = tf.train.AdadeltaOptimizer(learning_rate).minimize(cost)

# here we just try to train the model with one example (with nb_batch = 1)


save_cost = np.zeros(epoch)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(epoch):
    state1 = sess.run(initial_state1)
    state2 = sess.run(initial_state2)
    kappa = np.zeros((batch_size, K, 1))
    for b in range(nb_batch):
        x_data = np.zeros((batch_size, T, 3))
        y_data = np.zeros((batch_size, T, 3))
        text_data = np.zeros((batch_size, len_text, char_vec_len)) # for the test len_text is 26
        for i in range(batch_size):
            x_data[i] = strokes[b + i][0: T]
            y_data[i] = strokes[b + i][1: T+1]
            text_data[i] = get_one_hot(texts[i][0:len(texts[0])-1], alphabeat)

        cost_, _, state1, state2 = sess.run([cost, train, final_state1, final_state2],
                    feed_dict={
                      input_data: x_data,
                      target_data: y_data,
                      char_seq: text_data,
                      initial_state2: state2,
                      initial_state1: state1,
                      init_kappa: kappa
                    })

        print(' epoch n°' + str(e)+ ' batch n°' + str(b))
        print("Train Loss= " + str(cost_))
    save_cost[e] = cost_  # *(batch_size*len_sequence)

plt.plot(save_cost)

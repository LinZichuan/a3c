import tensorflow as tf
import threading
import gym
from gymenv import GymEnv
import time
from config import Config


T_MAX = 80000000
t_max = 32

class Agent:

    def __init__(self, sess, config):
        self.num_actions = 6
        self.width, self.height = 84,84
        self.hist_len = 4
        self.learning_rate = 0.00025
        self.graph = self.build_graph()
        self.saver = tf.train.Saver()
        self.sess = sess
        self.thread_num = 2
        self.env_name = 'Breakout-v0'
        self.config = config
        self.gamma = 0.99
        self.is_train = True
        self.env = GymEnv(self.config)
        self.num_actions = self.env.action_size

    def build_graph(self):
        # create shared policy and value networks
        self.build_policy_and_value_network()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        R_t = tf.placeholder('float', [None])
        a_t = tf.placeholder('float', [None, self.num_actions])
        log_prob = tf.log(tf.reduce_sum(tf.mul(self.action_probs, a_t), reduction_indices=1))
        p_loss = -log_prob * (R_t - self.state_value)
        v_loss = tf.reduce_mean(tf.square(R_t - self.state_value))
        total_loss = p_loss + 0.5 * v_loss
        self.minimize = optimizer.minimize(total_loss)
        return self.state, a_t, R_t, self.minimize
    
    def build_policy_and_value_network(self):
        data_format = 'NCHW'
        init = tf.truncated_normal_initializer(0, 0.02)

        self.state = tf.placeholder('float', [None, self.hist_len, self.width, self.height], 'state')

        w1 = tf.get_variable('w1', [8,8,self.hist_len,16], initializer=init)
        b1 = tf.get_variable('b1', [16], initializer=init)
        conv1 = tf.nn.conv2d(self.state, w1, [1,1,4,4], 'VALID', data_format=data_format)
        bias1 = tf.nn.bias_add(conv1, b1, data_format)
        out1 = tf.nn.relu(bias1)

        w2 = tf.get_variable('w2', [4,4,16,32], initializer=init)
        b2 = tf.get_variable('b2', [32], initializer=init)
        conv2 = tf.nn.conv2d(out1, w2, [1,1,2,2], 'VALID', data_format=data_format)
        bias2 = tf.nn.bias_add(conv2, b2, data_format)
        out2 = tf.nn.relu(bias2)

        shape = out2.get_shape().as_list()
        out2_flat = tf.reshape(out2, [-1, reduce(lambda x,y: x*y, shape[1:])])
        shape = out2_flat.get_shape().as_list()

        w3 = tf.get_variable('w3', [shape[1], 256], initializer=init)
        b3 = tf.get_variable('b3', [256], initializer=init)
        bias3 = tf.nn.bias_add(tf.matmul(out2_flat, w3), b3, data_format)
        out3 = tf.nn.relu(bias3)

        w4 = tf.get_variable('w4', [256, self.num_actions], initializer=init)
        b4 = tf.get_variable('b4', [self.num_actions], initializer=init)
        logits = tf.nn.bias_add(tf.matmul(out3, w4), b4, data_format)
        self.action_probs = tf.nn.softmax(logits)

        w5 = tf.get_variable('w5', [256, 1], initializer=init)
        b5 = tf.get_variable('b5', [1], initializer=init)
        self.state_value = tf.nn.bias_add(tf.matmul(out3, w5), b5, data_format)


    def thread_learning(self, thread_id):
        #start_thread
        global T_MAX, T
        T = 0 #shared
        env = GymEnv(self.config)
        time.sleep(2*thread_id)
        #get initial state
        s_t, _, _, _ = env.newRandomGame() #TODO
        history = [s_t for i in range(4)]
        terminal = False
        while T <= T_MAX:
            #reset gradient
            #sync parameters
            t = 0
            t_start = t
            s_batch = []
            a_batch = []
            r_batch = []
            while not terminal and (t - t_start) != t_max:
                #use policy network to choose a_t
                probs = self.sess.run(self.action_probs, feed_dict={self.state: [history]}) ##TODO
                action_index = self.sample_policy_action(self.num_actions, probs)
                a_t = np.zeros([self.num_actions]) #TODO
                a_t[action_index] = 1
                s_batch.append(history)
                a_batch.append(a_t)
                #get reward and s_t_plus_1
                s_t_plus_1, r_t, terminal = env.step(action_index)
                history[:-1] = history[1:]
                history[-1] = s_t_plus_1
                r_t = np.clip(r_t, -1, 1)
                r_batch.append(r_t)
                t += 1
                T += 1
                s_t = s_t_plus_1
            R_t = 0 if terminal else self.sess.run(self.state_value, feed_dict={state: [history]}) #V(s_t_plus_1)
            R_batch = np.zeros(t)
            for i in xrange(t-1, t_start): #from back to front
                R_t = r_batch[i] + self.gamma * R_t
                R_batch[i] = R_t

            self.sess.run(self.minimize, feed_dict={
                R: R_batch, a: a_batch, s: s_batch
            })

            if terminal:
                #self.sess.run()
                s_t, _, _, _ = env.newRandomGame()
                terminal = False

            #async update d_w and d_v to w and v
        pass

    def train(self):
        self.threads = []
        for i in xrange(self.thread_num):
            thread = threading.Thread(target=self.thread_learning, args=(i,), name=('thread_%d'%i))
            print "create thread_%d" % i
            self.threads.append(thread)

        for thread in self.threads:
            thread.start()

        for thread in self.threads:
            thread.join()




if __name__ == '__main__':

    with tf.Session() as sess:
        config = Config()
        agent = Agent(sess, config)
        if agent.is_train:
            agent.train()
        











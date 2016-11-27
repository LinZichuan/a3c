import tensorflow as tf
import threading
import gym
from gymenv import GymEnv
import time
from config import Config
import numpy as np


T_MAX = 80000000
t_max = 32

class Agent:

    def __init__(self, sess, config):
        self.width, self.height = 84,84
        self.hist_len = 4
        self.learning_rate = 0.0001 #0.00025
        self.thread_num = 4
        self.env_name = 'Breakout-v0'
        self.config = config
        self.gamma = 0.99
        self.is_train = True
        self.env = GymEnv(self.config)
        self.num_actions = 3 #self.env.action_size
        self.graph = self.build_graph()
        self.saver = tf.train.Saver()
        self.sess = sess

    def build_graph(self):
        # create shared policy and value networks
        self.build_policy_and_value_network()
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.R_t_place = tf.placeholder('float', [None])
        self.a_t_place = tf.placeholder('float', [None, self.num_actions])

        log_pi = tf.log(self.action_probs + 1e-6)
        entropy = - tf.reduce_sum(self.action_probs * log_pi, reduction_indices=1)
        td = self.R_t_place - self.state_value
        policy_loss = - tf.reduce_sum(tf.reduce_sum(tf.mul(log_pi, self.a_t_place), reduction_indices=1) * td + entropy * 0.01)
        value_loss = 0.5 * tf.nn.l2_loss(self.R_t_place - self.state_value)
        self.total_loss = policy_loss + value_loss
        '''
        log_prob = tf.log(tf.reduce_sum(tf.mul(self.action_probs, self.a_t_place), reduction_indices=1))
        p_loss = -log_prob * (self.R_t_place - self.state_value)
        xentropy_loss = tf.reduce_sum(self.action_probs * log_prob, name='xentropy_loss')
        v_loss = tf.reduce_mean(tf.square(self.R_t_place - self.state_value))
        self.total_loss = p_loss + 0.01 * xentropy_loss + v_loss
        self.loss = tf.reduce_mean(self.total_loss)
        '''
        '''
        #log_probs = tf.log(self.action_probs  + 1e-6)
        #log_pi_a_given_s = tf.reduce_sum(log_probs * tf.one_hot(self.a_t_place, self.num_actions), 1)
        #p_loss = tf.reduce_sum(log_pi_a_given_s * (self.R_t_place - self.state_value))
        #v_loss = tf.nn.l2_loss(self.state_value - self.R_t_place)
        '''
        self.minimize = optimizer.minimize(self.total_loss)
        return self.state, self.a_t_place, self.R_t_place, self.minimize
    
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
        bias3 = tf.nn.bias_add(tf.matmul(out2_flat, w3), b3)
        out3 = tf.nn.relu(bias3)

        w4 = tf.get_variable('w4', [256, self.num_actions], initializer=init)
        b4 = tf.get_variable('b4', [self.num_actions], initializer=init)
        logits = tf.nn.bias_add(tf.matmul(out3, w4), b4)
        self.action_probs = tf.nn.softmax(logits)

        w5 = tf.get_variable('w5', [256, 1], initializer=init)
        b5 = tf.get_variable('b5', [1], initializer=init)
        self.state_value = tf.nn.bias_add(tf.matmul(out3, w5), b5)

    def sample_policy_action(self, num_actions, probs):
        """
        Sample an action from an action probability distribution output by
        the policy network.
        """
        # Subtract a tiny value from probabilities in order to avoid
        # "ValueError: sum(pvals[:-1]) > 1.0" in numpy.multinomial
        probs = probs - np.finfo(np.float32).epsneg

        histogram = np.random.multinomial(1, probs)
        action_index = int(np.nonzero(histogram)[0])
        return action_index 


    def thread_learning(self, thread_id):
        #start_thread
        global T_MAX, T
        T = 0 #shared
        env = GymEnv(self.config)
        time.sleep(5*thread_id)
        # set up per-episode counters
        ep_reward = 0
        ep_avg_v = 0
        v_steps = 0
        ep_t = 0
        #get initial state
        s_t, _, _, _ = env.newRandomGame() #TODO
        history = [s_t for i in range(4)]
        terminal = False
        loss_eval_step = 100
        loss_eval = 0
        loss_counter = 0
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
                probs = self.sess.run(self.action_probs, feed_dict={self.state: [history]})[0] ##TODO
                action_index = self.sample_policy_action(self.num_actions, probs)
                a_t = np.zeros([self.num_actions]) #TODO
                a_t[action_index] = 1
                s_batch.append(history)
                a_batch.append(a_t)
                #get reward and s_t_plus_1
                s_t_plus_1, r_t, terminal = env.act(action_index)
                ep_reward += r_t
                history[:-1] = history[1:]
                history[-1] = s_t_plus_1
                r_t = np.clip(r_t, -1, 1)
                r_batch.append(r_t)
                t += 1
                T += 1
                ep_t += 1
                s_t = s_t_plus_1
            R_t = 0 if terminal else self.sess.run(self.state_value, feed_dict={self.state: [history]})[0][0] #V(s_t_plus_1)
            R_batch = np.zeros(t)
            for i in xrange(t-1, t_start): #from back to front
                R_t = r_batch[i] + self.gamma * R_t
                R_batch[i] = R_t
                #R_batch.append(R_t)

            _, loss = self.sess.run([self.minimize, self.total_loss], feed_dict={
                self.R_t_place: R_batch, self.a_t_place: a_batch, self.state: s_batch
            })
            loss_eval += loss
            loss_counter += 1
            if loss_counter == loss_eval_step:
                avg_loss = loss_eval / loss_counter
                print "THREAD:", thread_id, "/LOSS: ", abs(avg_loss)
                loss_counter = 0
                loss_eval = 0

            if terminal:
                #self.sess.run()
                s_t, _, _, _ = env.newRandomGame()
                terminal = False
                print "THREAD:", thread_id, "/ TIME:", T, "/ REWARD:", ep_reward
                #reset
                ep_reward = 0
                ep_t = 0

            #async update d_w and d_v to w and v
        pass

    def train(self):
        self.sess.run(tf.initialize_all_variables())
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
        











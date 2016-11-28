import tensorflow as tf
import threading
import gym
from gymenv import GymEnv
import time
from config import Config
import numpy as np
from layers import Conv2D, MaxPooling, Linear


T_MAX = 80000000
t_max = 32

class Agent:

    def __init__(self, sess, config):
        self.width, self.height = 84,84
        self.hist_len = 4
        self.learning_rate = 0.0001 #0.00025
        self.thread_num = 16
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

        log_probs = tf.log(self.logits + 1e-6)
        log_pi_a_given_s = tf.reduce_sum(log_probs * self.a_t_place, 1)
        advantage = tf.sub(tf.stop_gradient(self.value), self.R_t_place, name='advantage')
        policy_loss = tf.reduce_sum(log_pi_a_given_s * advantage, name='policy_loss')
        xentropy_loss = tf.reduce_sum(
                                self.logits * log_probs, name='xentropy_loss')
        value_loss = tf.nn.l2_loss(self.value - self.R_t_place, name='value_loss')
        pred_reward = tf.reduce_mean(self.value, name='predict_reward')
        self.total_loss = tf.add_n([policy_loss, xentropy_loss * 0.01, value_loss])
        self.total_loss = tf.truediv(self.total_loss,
                tf.cast(tf.shape(self.R_t_place)[0], tf.float32),
                name='total_loss')

        self.minimize = optimizer.minimize(self.total_loss)
        return self.state, self.a_t_place, self.R_t_place, self.minimize
    
    def build_policy_and_value_network(self):
        data_format = 'NHWC'
        init = tf.truncated_normal_initializer(0, 0.02)

        self.state = tf.placeholder('float', [None, self.width, self.height, self.hist_len], 'state')
        print self.state.get_shape()
        l = Conv2D('conv0', self.state, out_channel=32, kernel_shape=5)
        l = MaxPooling('pool0', l, 2)
        l = Conv2D('conv1', l, out_channel=32, kernel_shape=5)
        l = MaxPooling('pool1', l, 2)
        l = Conv2D('conv2', l, out_channel=64, kernel_shape=4)
        l = MaxPooling('pool2', l, 2)
        l = Conv2D('conv3', l, out_channel=64, kernel_shape=3)

        l = Linear('linear1', l, out_dim=512, nl=tf.identity)
        #l = tf.nn.relu(l, name='relu')
        self.logits = tf.nn.softmax(Linear('fc-pi', l, out_dim=self.num_actions, nl=tf.identity))
        self.value = Linear('fc-v', l, 1, nl=tf.identity)


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
                probs = self.sess.run(self.logits, feed_dict={self.state: [np.transpose(history, (1,2,0))]})[0] ##TODO
                action_index = self.sample_policy_action(self.num_actions, probs)
                a_t = np.zeros([self.num_actions]) #TODO
                a_t[action_index] = 1
                s_batch.append(history)
                a_batch.append(a_t)
                #get reward and s_t_plus_1
                s_t_plus_1, r_t, terminal = env.act(action_index)
                #env.env.render()
                ep_reward += r_t
                history[:-1] = history[1:]
                history[-1] = s_t_plus_1
                r_t = np.clip(r_t, -1, 1)
                r_batch.append(r_t)
                t += 1
                T += 1
                ep_t += 1
                s_t = s_t_plus_1
            R_t = 0 if terminal else self.sess.run(self.value, feed_dict={self.state: [np.transpose(history, (1,2,0))]})[0][0] #V(s_t_plus_1)
            R_batch = np.zeros(t)
            for i in xrange(t-1, t_start): #from back to front
                R_t = r_batch[i] + self.gamma * R_t
                R_batch[i] = R_t
                #R_batch.append(R_t)

            s_batch = np.transpose(s_batch, (0,2,3,1))
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
        











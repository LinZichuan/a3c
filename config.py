class Config:
    env_name = 'Breakout-v0'
    env_type='detail'
    cnn_format = 'NCHW'
    is_train = True
    display = False
    action_repeat = 1
    use_gpu = True
    scale = 10000
    max_step = 5000*scale
    test_step = 5 * scale
    memory_size = 100 * scale
    learn_start = 5 * scale
    batch_size = 32
    discount = 0.99
    learning_rate = 0.00025
    learning_rate_minimum = 0.00025
    learning_rate_decay = 0.96
    learning_rate_decay_step = 5*scale
    ep_end = 0.1
    ep_start = 1.0
    ep_end_t = memory_size
    hist_len = 4
    screen_h = 84
    screen_w = 84
    train_frequency = 4
    target_q_update_step = 1 * scale
    step_input = 0
    max_delta = 1
    min_delta = -1
    max_reward = 1.
    min_reward = -1.
    
    checkpoint_dir = 'checkpoints/'
    double_q = False
    dueling = True
    random_start = 30

# cannot add variable casually, because it will affect the model_dir path!
    @property
    def model_dir(self):
        _model_dir = self.env_name
        for k, v in sorted(vars(Config).iteritems()):
            if not k.startswith('__') and not callable(k) and k not in ['model_dir', 'checkpoint_dir', 'env_name']:
                v = ','.join([str(i) for i in v]) if type(v) == list else v
                _model_dir += "/%s-%s" % (k, v)
        return _model_dir + '/'

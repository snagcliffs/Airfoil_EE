import numpy as np
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

class LSTM_ROM(tf.keras.Model):

    def __init__(self, data_params=None, net_params=None, learning_params=None, restart_file=None, restart_dict=None):

        if restart_dict is None and restart_file is not None:
            restart_dict = np.load(restart_file,allow_pickle=True).item()

        if restart_dict is not None:
            data_params = restart_dict['data_params']
            net_params = restart_dict['net_params']
            learning_params = restart_dict['learning_params']
        
        super(LSTM_ROM, self).__init__()
        
        r, m_hist,stride,n_steps = data_params
        self.m_hist = m_hist    # number of history points
        self.stride = stride    # number of history points
        self.n_steps = n_steps  # number of history points
        self.r = r             # dimension of space
        
        layer_sizes, activation = net_params
        self.layer_sizes = layer_sizes
        self.activation = activation

        l1_reg, l2_reg, lr, decay_steps, decay_rate = learning_params
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.reg = tf.keras.regularizers.L1L2(l1=l1_reg, l2=l2_reg)
        self.lr = lr
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate

        # Build networks
        self.build_network()

        # For now set deault optimizer to be Adam
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                            initial_learning_rate=lr,
                            decay_steps=decay_steps,
                            decay_rate=decay_rate)
        self.optimizer = tf.keras.optimizers.Adam(lr_schedule)
        self.train_loss = []
        self.val_loss = []
        self.test_loss = np.inf

        if restart_dict is not None:
            self.set_weights(restart_dict['weights'])
            self.train_loss = restart_dict['train_loss']
            self.val_loss = restart_dict['val_loss']
            self.test_loss = restart_dict['test_loss']

    def build_network(self):

        if self.m_hist == 1: layers = [tf.keras.layers.Input(shape=(self.r))]    
        else: layers = [tf.keras.layers.Input(shape=(self.m_hist,self.r))]
        
        # Pre-LSTM dense layers
        for l in self.layer_sizes[0]:
            layers.append(tf.keras.layers.Dense(l,activation=self.activation,
                                                kernel_regularizer=self.reg)(layers[-1]))
        
        # LSTM
        if self.m_hist > 1: 
            for j in range(len(self.layer_sizes[1])):

                l = self.layer_sizes[1][j]
                
                # Intermediate LSTM layers should return sequences
                if j == len(self.layer_sizes[1]) - 1:
                    rs = False
                else: rs =True 

                layers.append(tf.keras.layers.LSTM(l,return_sequences=rs)(layers[-1]))
        
        # Post-LSTM dense layers
        for l in self.layer_sizes[2]:
            layers.append(tf.keras.layers.Dense(l,activation=self.activation,
                                                kernel_regularizer=self.reg)(layers[-1]))
            
        layers.append(tf.keras.layers.Dense(self.r)(layers[-1]))
        self.network = tf.keras.Model(inputs=layers[0], outputs=layers[0][:,-1,:] + layers[-1])

    @tf.function
    def compute_loss(self, Xi_hist, Xi_forecast):

        Xi_est = []

        for step in range(self.n_steps):

            Xi_est.append(tf.expand_dims(self.network(Xi_hist), axis=1))
            Xi_hist = tf.concat([Xi_hist[:,1:,:], Xi_est[-1]], axis=1)

        # For now just use MSE
        reconstruction_loss = tf.reduce_mean((tf.concat(Xi_est,axis=1)-Xi_forecast)**2)

        return reconstruction_loss

    @tf.function
    def train_step(self, Xi_hist, Xi_forecast):
        """
        Single step of optimization algorithm
        """
        
        with tf.GradientTape() as tape:
            loss = self.compute_loss(Xi_hist, Xi_forecast)
            reg_loss = loss + tf.reduce_sum(self.network.losses)

        gradients = tape.gradient(reg_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        return loss

    def train_epoch(self, gen):
        
        train_batches, val_batches = gen.batches_per_epoch()[:2]
        
        train_loss = tf.keras.metrics.Mean()
        for j in range(train_batches):
            Xi_hist, Xi_forecast = gen.next_train()
            train_loss(self.train_step(Xi_hist, Xi_forecast))
            
                    
        val_loss = tf.keras.metrics.Mean()
        for j in range(val_batches):
            Xi_hist, Xi_forecast = gen.next_val()
            val_loss(self.compute_loss(Xi_hist, Xi_forecast))
        
        return train_loss.result(), val_loss.result()

    def compute_test_loss(self, gen):
        
        test_batches = gen.batches_per_epoch()[2]
                    
        test_loss = tf.keras.metrics.Mean()
        for j in range(test_batches):
            Xi_hist, Xi_forecast = gen.next_test()
            test_loss(self.compute_loss(Xi_hist, Xi_forecast))
        
        return test_loss.result()

    def train_model(self, training_params, gen, save_file):

        epochs, patience = training_params

        for epoch in range(epochs):

            losses = self.train_epoch(gen)
       
            self.train_loss.append(losses[0].numpy())
            self.val_loss.append(losses[1].numpy())
            
            print('Epoch: {}, Train Loss: {}, Val Loss: {}'.format(epoch+1, 
                                                                   np.round(self.train_loss[-1],6), 
                                                                   np.round(self.val_loss[-1],6)))

            # Save weights if val loss has improved
            if self.val_loss[-1] == np.min(self.val_loss):
                print('Val loss improved.  Saving NN weights.')
                self.save_weights(save_file)
            
            if epoch % 10 == 0:
                print('Test loss: {}'.format(self.compute_test_loss(gen)))

            if np.argmin(self.val_loss) <= epoch - patience: break

        self.load_weights(save_file)
        self.test_loss = self.compute_test_loss(gen)

    def save_weights(self, save_file):
        np.save(save_file, [w.numpy() for w in self.trainable_weights])

    def load_weights(self, save_file):
        self.set_weights(np.load(save_file+'.npy',allow_pickle=True))  

    def save_network(self, filename):
        np.save(filename, self.get_network_dict())

    def predict_full_data(self, gen):
        
        batch_size = gen.batch_size
        m = gen.m
        n_batches = int(np.ceil(m/batch_size))

        q = np.zeros((m,1))
        
        for j in range(n_batches):

            batch_inds = (gen.m_hist-1)*gen.stride + np.arange(j*batch_size, np.min([gen.m,(j+1)*batch_size]))
            
            P_hist_batch = gen.get_P_hist(gen.P, batch_inds, gen.m_hist, gen.stride)
            q[j*batch_size:(j+1)*batch_size,0] = self.network(P_hist_batch).numpy().flatten()

        return q

    def get_network_dict(self):

        data_params = [self.r,
                       self.m_hist,
                       self.stride,
                       self.n_steps]

        net_params = [self.layer_sizes,
                      self.activation]

        learning_params = [self.l1_reg,
                           self.l2_reg,
                           self.lr,
                           self.decay_steps, 
                           self.decay_rate]

        network_dict = {'data_params' : data_params,
                        'net_params' : net_params, 
                        'learning_params' : learning_params,
                        'weights' : [w.numpy() for w in self.trainable_weights],
                        'train_loss' : self.train_loss,
                        'val_loss' : self.val_loss,
                        'test_loss' : self.test_loss}

        return network_dict



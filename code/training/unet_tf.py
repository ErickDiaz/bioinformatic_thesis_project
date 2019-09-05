import tensorflow as tf    
import pandas as pd                 
import numpy as np                                       
import sklearn.model_selection     # For using KFold
import keras.preprocessing.image   # For using image generation
import datetime                    # To measure running time 
import skimage.transform           # For resizing images
import skimage.morphology          # For using image labeling
import cv2                         # To read and manipulate images
import os                          # For filepath, directory handling
import sys                         # System-specific parameters and functions
import tqdm                        # Use smart progress meter
import seaborn as sns              # For pairplots
import matplotlib.pyplot as plt    # Python 2D plotting library
import matplotlib.cm as cm         # Color map

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

LOGS_DIR_NAME = 'logs' 
SAVES_DIR_NAME = 'saves'

class NeuralNetwork():
    """ Implements a neural network.
        
        TensorFlow is used to implement the U-Net, which consists of convolutional
        and max pooling layers. Input and output shapes coincide. Methods are
        implemented to train the model, to save/load the complete session and to 
        attach summaries for visualization with TensorBoard. 
    """

    def __init__(self, nn_name='tmp', nn_type='UNet', log_step=0.2, keep_prob=0.33, 
                 mb_size=16, input_shape=[IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS], 
                 output_shape=[IMG_HEIGHT,IMG_WIDTH,1]):
        """Instance constructor."""
        
        # Tunable hyperparameters for training.
        self.mb_size = mb_size       # Mini batch size
        self.keep_prob = keep_prob   # Keeping probability with dropout regularization 
        self.learn_rate_step = 3     # Step size in terms of epochs
        self.learn_rate_alpha = 0.25 # Reduction of learn rate for each step 
        self.learn_rate_0 = 0.001    # Starting learning rate 
        self.dropout_proba = 0.1     # == 1-keep_probability
        
        # Set helper variables.
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nn_type = nn_type                # Type of neural network
        self.nn_name = nn_name                # Name of neural network
        self.params = {}                      # For storing parameters
        self.learn_rate_pos = 0                
        self.learn_rate = self.learn_rate_0
        self.index_in_epoch = 0 
        self.epoch = 0. 
        self.log_step = log_step              # Log results in terms of epochs
        self.n_log_step = 0                   # Count number of mini batches  
        self.use_tb_summary = False           # True = use TensorBoard summaries
        self.use_tf_saver = False             # True = save the session
        
        # Parameters that should be stored.
        self.params['train_loss']=[]
        self.params['valid_loss']=[]
        self.params['train_score']=[]
        self.params['valid_score']=[]
        
    def get_learn_rate(self):
        """Compute the current learning rate."""
        if False:
            # Fixed learnrate
            learn_rate = self.learn_rate_0
        else:
            # Decreasing learnrate each step by factor 1-alpha
            learn_rate = self.learn_rate_0*(1.-self.learn_rate_alpha)**self.learn_rate_pos
        return learn_rate

    def next_mini_batch(self):
        """Get the next mini batch."""
        start = self.index_in_epoch
        self.index_in_epoch += self.mb_size           
        self.epoch += self.mb_size/len(self.x_train)
        
        # At the start of the epoch.
        if start == 0:
            np.random.shuffle(self.perm_array) # Shuffle permutation array.
   
        # In case the current index is larger than one epoch.
        if self.index_in_epoch > len(self.x_train):
            self.index_in_epoch = 0
            self.epoch -= self.mb_size/len(self.x_train) 
            return self.next_mini_batch() # Recursive use of function.
        
        end = self.index_in_epoch
        
        # Original data.
        x_tr = self.x_train[self.perm_array[start:end]]
        y_tr = self.y_train[self.perm_array[start:end]]
        
        
        return x_tr, y_tr
 
    def weight_variable(self, shape, name=None):
        """ Weight initialization """
        #initializer = tf.truncated_normal(shape, stddev=0.1)
        initializer = tf.contrib.layers.xavier_initializer()
        #initializer = tf.contrib.layers.variance_scaling_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer)

    def bias_variable(self, shape, name=None):
        """Bias initialization."""
        #initializer = tf.constant(0.1, shape=shape)  
        initializer = tf.contrib.layers.xavier_initializer()
        #initializer = tf.contrib.layers.variance_scaling_initializer()
        return tf.get_variable(name, shape=shape, initializer=initializer)
     
    def conv2d(self, x, W, name=None):
        """ 2D convolution. """
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME', name=name)

    def max_pool_2x2(self, x, name=None):
        """ Max Pooling 2x2. """
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME',
                              name=name)
    
    def conv2d_transpose(self, x, filters, name=None):
        """ Transposed 2d convolution. """
        return tf.layers.conv2d_transpose(x, filters=filters, kernel_size=2, 
                                          strides=2, padding='SAME') 
    
    def leaky_relu(self, z, name=None):
        """Leaky ReLU."""
        return tf.maximum(0.01 * z, z, name=name)
    
    def activation(self, x, name=None):
        """ Activation function. """
        a = tf.nn.elu(x, name=name)
        #a = self.leaky_relu(x, name=name)
        #a = tf.nn.relu(x, name=name)
        return a 
    
    def loss_tensor(self):
        """Loss tensor."""
        if True:
            # Dice loss based on Jaccard dice score coefficent.
            print("loss Jaccard")
            axis=np.arange(1,len(self.output_shape)+1)
            offset = 1e-5
            corr = tf.reduce_sum(self.y_data_tf * self.y_pred_tf, axis=axis)
            l2_pred = tf.reduce_sum(tf.square(self.y_pred_tf), axis=axis)
            l2_true = tf.reduce_sum(tf.square(self.y_data_tf), axis=axis)
            dice_coeff = (2. * corr + 1e-5) / (l2_true + l2_pred + 1e-5)
            # Second version: 2-class variant of dice loss
            #corr_inv = tf.reduce_sum((1.-self.y_data_tf) * (1.-self.y_pred_tf), axis=axis)
            #l2_pred_inv = tf.reduce_sum(tf.square(1.-self.y_pred_tf), axis=axis)
            #l2_true_inv = tf.reduce_sum(tf.square(1.-self.y_data_tf), axis=axis)
            #dice_coeff = ((corr + offset) / (l2_true + l2_pred + offset) +
            #             (corr_inv + offset) / (l2_pred_inv + l2_true_inv + offset))
            loss = tf.subtract(1., tf.reduce_mean(dice_coeff))
        if False:
            # Sigmoid cross entropy. 
            print("loss cross entropy")
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.y_data_tf, logits=self.z_pred_tf))
        return loss 
    
    def optimizer_tensor(self):
        """Optimization tensor."""
        # Adam Optimizer (adaptive moment estimation). 
        optimizer = tf.train.AdamOptimizer(self.learn_rate_tf).minimize(
                    self.loss_tf, name='train_step_tf')
        return optimizer
   
    def batch_norm_layer(self, x, name=None):
        """Batch normalization layer."""
        if False:
            layer = tf.layers.batch_normalization(x, training=self.training_tf, 
                                                  momentum=0.9, name=name)
        else: 
            layer = x
        return layer
    
    def dropout_layer(self, x, name=None):
        """Dropout layer."""
        if False:
            layer = tf.layers.dropout(x, self.dropout_proba, training=self.training_tf,
                                     name=name)
        else:
            layer = x
        return layer

    def num_of_weights(self,tensors):
        """Compute the number of weights."""
        sum_=0
        for i in range(len(tensors)):
            m = 1
            for j in range(len(tensors[i].shape)):
              m *= int(tensors[i].shape[j])
            sum_+=m
        return sum_

    def build_UNet_graph(self):
        """ Create the UNet graph in TensorFlow. """
        # 1. unit 
        with tf.name_scope('1.unit'):
            W1_1 = self.weight_variable([3,3,self.input_shape[2],16], 'W1_1')
            b1_1 = self.bias_variable([16], 'b1_1')
            Z1 = self.conv2d(self.x_data_tf, W1_1, 'Z1') + b1_1
            A1 = self.activation(self.batch_norm_layer(Z1)) # (.,128,128,16)
            A1_drop = self.dropout_layer(A1)
            W1_2 = self.weight_variable([3,3,16,16], 'W1_2')
            b1_2 = self.bias_variable([16], 'b1_2')
            Z2 = self.conv2d(A1_drop, W1_2, 'Z2') + b1_2
            A2 = self.activation(self.batch_norm_layer(Z2)) # (.,128,128,16)
            P1 = self.max_pool_2x2(A2, 'P1') # (.,64,64,16)
        # 2. unit 
        with tf.name_scope('2.unit'):
            W2_1 = self.weight_variable([3,3,16,32], "W2_1")
            b2_1 = self.bias_variable([32], 'b2_1')
            Z3 = self.conv2d(P1, W2_1) + b2_1
            A3 = self.activation(self.batch_norm_layer(Z3)) # (.,64,64,32)
            A3_drop = self.dropout_layer(A3)
            W2_2 = self.weight_variable([3,3,32,32], "W2_2")
            b2_2 = self.bias_variable([32], 'b2_2')
            Z4 = self.conv2d(A3_drop, W2_2) + b2_2
            A4 = self.activation(self.batch_norm_layer(Z4)) # (.,64,64,32)
            P2 = self.max_pool_2x2(A4) # (.,32,32,32)
        # 3. unit
        with tf.name_scope('3.unit'):
            W3_1 = self.weight_variable([3,3,32,64], "W3_1")
            b3_1 = self.bias_variable([64], 'b3_1')
            Z5 = self.conv2d(P2, W3_1) + b3_1
            A5 = self.activation(self.batch_norm_layer(Z5)) # (.,32,32,64)
            A5_drop = self.dropout_layer(A5)
            W3_2 = self.weight_variable([3,3,64,64], "W3_2")
            b3_2 = self.bias_variable([64], 'b3_2')
            Z6 = self.conv2d(A5_drop, W3_2) + b3_2
            A6 = self.activation(self.batch_norm_layer(Z6)) # (.,32,32,64)
            P3 = self.max_pool_2x2(A6) # (.,16,16,64)
        # 4. unit
        with tf.name_scope('4.unit'):
            W4_1 = self.weight_variable([3,3,64,128], "W4_1")
            b4_1 = self.bias_variable([128], 'b4_1')
            Z7 = self.conv2d(P3, W4_1) + b4_1
            A7 = self.activation(self.batch_norm_layer(Z7)) # (.,16,16,128)
            A7_drop = self.dropout_layer(A7)
            W4_2 = self.weight_variable([3,3,128,128], "W4_2")
            b4_2 = self.bias_variable([128], 'b4_2')
            Z8 = self.conv2d(A7_drop, W4_2) + b4_2
            A8 = self.activation(self.batch_norm_layer(Z8)) # (.,16,16,128)
            P4 = self.max_pool_2x2(A8) # (.,8,8,128)
        # 5. unit 
        with tf.name_scope('5.unit'):
            W5_1 = self.weight_variable([3,3,128,256], "W5_1")
            b5_1 = self.bias_variable([256], 'b5_1')
            Z9 = self.conv2d(P4, W5_1) + b5_1
            A9 = self.activation(self.batch_norm_layer(Z9)) # (.,8,8,256)
            A9_drop = self.dropout_layer(A9)
            W5_2 = self.weight_variable([3,3,256,256], "W5_2")
            b5_2 = self.bias_variable([256], 'b5_2')
            Z10 = self.conv2d(A9_drop, W5_2) + b5_2
            A10 = self.activation(self.batch_norm_layer(Z10)) # (.,8,8,256)
        # 6. unit
        with tf.name_scope('6.unit'):
            W6_1 = self.weight_variable([3,3,256,128], "W6_1")
            b6_1 = self.bias_variable([128], 'b6_1')
            U1 = self.conv2d_transpose(A10, 128) # (.,16,16,128)
            U1 = tf.concat([U1, A8], 3) # (.,16,16,256)
            Z11 = self.conv2d(U1, W6_1) + b6_1
            A11 = self.activation(self.batch_norm_layer(Z11)) # (.,16,16,128)
            A11_drop = self.dropout_layer(A11)
            W6_2 = self.weight_variable([3,3,128,128], "W6_2")
            b6_2 = self.bias_variable([128], 'b6_2')
            Z12 = self.conv2d(A11_drop, W6_2) + b6_2
            A12 = self.activation(self.batch_norm_layer(Z12)) # (.,16,16,128)
        # 7. unit 
        with tf.name_scope('7.unit'):
            W7_1 = self.weight_variable([3,3,128,64], "W7_1")
            b7_1 = self.bias_variable([64], 'b7_1')
            U2 = self.conv2d_transpose(A12, 64) # (.,32,32,64)
            U2 = tf.concat([U2, A6],3) # (.,32,32,128)
            Z13 = self.conv2d(U2, W7_1) + b7_1
            A13 = self.activation(self.batch_norm_layer(Z13)) # (.,32,32,64)
            A13_drop = self.dropout_layer(A13)
            W7_2 = self.weight_variable([3,3,64,64], "W7_2")
            b7_2 = self.bias_variable([64], 'b7_2')
            Z14 = self.conv2d(A13_drop, W7_2) + b7_2
            A14 = self.activation(self.batch_norm_layer(Z14)) # (.,32,32,64)
        # 8. unit
        with tf.name_scope('8.unit'):
            W8_1 = self.weight_variable([3,3,64,32], "W8_1")
            b8_1 = self.bias_variable([32], 'b8_1')
            U3 = self.conv2d_transpose(A14, 32) # (.,64,64,32)
            U3 = tf.concat([U3, A4],3) # (.,64,64,64)
            Z15 = self.conv2d(U3, W8_1) + b8_1
            A15 = self.activation(self.batch_norm_layer(Z15)) # (.,64,64,32)
            A15_drop = self.dropout_layer(A15)
            W8_2 = self.weight_variable([3,3,32,32], "W8_2")
            b8_2 = self.bias_variable([32], 'b8_2')
            Z16 = self.conv2d(A15_drop, W8_2) + b8_2
            A16 = self.activation(self.batch_norm_layer(Z16)) # (.,64,64,32)
        # 9. unit 
        with tf.name_scope('9.unit'):
            W9_1 = self.weight_variable([3,3,32,16], "W9_1")
            b9_1 = self.bias_variable([16], 'b9_1')
            U4 = self.conv2d_transpose(A16, 16) # (.,128,128,16)
            U4 = tf.concat([U4, A2],3) # (.,128,128,32)
            Z17 = self.conv2d(U4, W9_1) + b9_1
            A17 = self.activation(self.batch_norm_layer(Z17)) # (.,128,128,16)
            A17_drop = self.dropout_layer(A17)
            W9_2 = self.weight_variable([3,3,16,16], "W9_2")
            b9_2 = self.bias_variable([16], 'b9_2')
            Z18 = self.conv2d(A17_drop, W9_2) + b9_2
            A18 = self.activation(self.batch_norm_layer(Z18)) # (.,128,128,16)
        # 10. unit: output layer
        with tf.name_scope('10.unit'):
            W10 = self.weight_variable([1,1,16,1], "W10")
            b10 = self.bias_variable([1], 'b10')
            Z19 = self.conv2d(A18, W10) + b10
            A19 = tf.nn.sigmoid(self.batch_norm_layer(Z19)) # (.,128,128,1)
        
        self.z_pred_tf = tf.identity(Z19, name='z_pred_tf') # (.,128,128,1)
        self.y_pred_tf = tf.identity(A19, name='y_pred_tf') # (.,128,128,1)
        
        print('Build UNet Graph: 10 layers, {} trainable weights'.format(
            self.num_of_weights([W1_1,b1_1,W1_2,b1_2,W2_1,b2_1,W2_2,b2_2,
                                 W3_1,b3_1,W3_2,b3_2,W4_1,b4_1,W4_2,b4_2,
                                 W5_1,b5_1,W5_2,b5_2,W6_1,b6_1,W6_2,b6_2,
                                 W7_1,b7_1,W7_2,b7_2,W8_1,b8_1,W8_2,b8_2,
                                 W9_1,b9_1,W9_2,b9_2,W10,b10])))
    
    def build_graph(self):
        """ Build the complete graph in TensorFlow. """
        tf.reset_default_graph()  
        self.graph = tf.Graph()

        with self.graph.as_default():
            
            # Input tensor.
            shape = [None]
            shape = shape.extend(self.input_shape)
            self.x_data_tf = tf.placeholder(dtype=tf.float32, shape=shape, 
                                            name='x_data_tf') # (.,128,128,3)
            
            # Generic tensors.
            self.keep_prob_tf = tf.placeholder_with_default(1.0, shape=(), 
                                                            name='keep_prob_tf') 
            self.learn_rate_tf = tf.placeholder(dtype=tf.float32,
                                                name="learn_rate_tf")
            self.training_tf = tf.placeholder_with_default(False, shape=(),
                                                           name='training_tf')
            # Build U-Net graph.
            self.build_UNet_graph()

            # Target tensor.
            shape = [None]
            shape = shape.extend(self.output_shape)
            self.y_data_tf = tf.placeholder(dtype=tf.float32, shape=shape, 
                                            name='y_data_tf') # (.,128,128,1)
            # Loss tensor
            self.loss_tf = tf.identity(self.loss_tensor(), name='loss_tf')

            # Optimisation tensor.
            self.train_step_tf = self.optimizer_tensor()
            
            # Extra operations required for batch normalization.
            self.extra_update_ops_tf = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            
    def train_graph(self, sess, x_train, y_train, x_valid, y_valid, n_epoch=1):
        """ Train the graph of the corresponding neural network. """
        # Set training and validation sets.
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        
        # Parameters.
        self.perm_array = np.arange(len(self.x_train))
        mb_per_epoch = self.x_train.shape[0]/self.mb_size
        train_loss, train_score, valid_loss, valid_score = [],[],[],[]
        
        # Start timer.
        start = datetime.datetime.now()
        print('Training the Neural Network')
        print('\tnn_name = {}, n_epoch = {}, mb_size = {}, learnrate = {:.7f}'.format(
               self.nn_name, n_epoch, self.mb_size, self.learn_rate))
        print('\tinput_shape = {}, output_shape = {}'.format(
            self.input_shape, self.output_shape))
        print('\tlearn_rate = {:.10f}, learn_rate_0 = {:.10f}, learn_rate_alpha = {}'.format(
            self.learn_rate, self.learn_rate_0, self.learn_rate_alpha))
        print('\tlearn_rate_step = {}, learn_rate_pos = {}, dropout_proba = {}'.format(
            self.learn_rate_step, self.learn_rate_pos, self.dropout_proba))
        print('\tx_train = {}, x_valid = {}'.format(x_train.shape, x_valid.shape))
        print('\ty_train = {}, y_valid = {}'.format(y_train.shape, y_valid.shape))
        print('Training started: {}'.format(datetime.datetime.now().strftime(
                                     '%d-%m-%Y %H:%M:%S')))
        
        ## Start training log file
        f = open(self.nn_name + ".log", "a")
        f.write('epoch,dataset,loss,score,IoU,precision,recall' + "\n")
        f.close()
        ##
        
        # Looping over mini batches.
        for i in range(int(n_epoch*mb_per_epoch)+1):

            # Adapt the learning rate.
            if not self.learn_rate_pos == int(self.epoch // self.learn_rate_step):
                self.learn_rate_pos = int(self.epoch // self.learn_rate_step)
                self.learn_rate = self.get_learn_rate()
                print('Update learning rate to {:.10f}. Running time: {}'.format(
                    self.learn_rate, datetime.datetime.now()-start))
            
            # Train the graph.
            x_batch, y_batch = self.next_mini_batch() # next mini batch
            sess.run([self.train_step_tf, self.extra_update_ops_tf], 
                     feed_dict={self.x_data_tf: x_batch, self.y_data_tf: y_batch, 
                                self.keep_prob_tf: self.keep_prob, 
                                self.learn_rate_tf: self.learn_rate,
                                self.training_tf: True})
            
            # Store losses and scores.
            if i%int(self.log_step*mb_per_epoch) == 0:
             
                self.n_log_step += 1 # Current number of log steps.
                
                # Train data used for evaluation.
                ids = np.arange(len(self.x_train))
                np.random.shuffle(ids)
                ids = ids[:len(x_valid)] # len(x_batch)
                x_trn = self.x_train[ids]
                y_trn = self.y_train[ids]
                
                # Valid data used for evaluation.
                ids = np.arange(len(self.x_valid))
                np.random.shuffle(ids)
                ids = ids[:len(x_valid)] # len(x_batch)
                x_vld = self.x_valid[ids]
                y_vld = self.y_valid[ids]
                
                feed_dict_train = {self.x_data_tf: x_trn, self.y_data_tf: y_trn, 
                                   self.keep_prob_tf: 1.0}
                feed_dict_valid = {self.x_data_tf: x_vld, self.y_data_tf: y_vld, 
                                   self.keep_prob_tf: 1.0}
                
                # Evaluate current loss and score
                train_loss, y_train_pred = sess.run([self.loss_tf, self.y_pred_tf], 
                                                   feed_dict = feed_dict_train)
                valid_loss, y_valid_pred = sess.run([self.loss_tf, self.y_pred_tf], 
                                                   feed_dict = feed_dict_valid)
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    ## train dataset scores
                    train_scores, train_metrics = get_score(y_trn, y_train_pred)
                    train_iou = np.nanmean(train_metrics[:,0])
                    train_precision = np.nanmean(train_metrics[:,1])
                    train_recall = np.nanmean(train_metrics[:,2])
                    train_score = np.mean(train_scores)
                    
                    ## Validation dataset scores
                    valid_scores, valid_metrics   = get_score(y_vld, y_valid_pred)
                    valid_iou = np.nanmean(valid_metrics[:,0])
                    valid_precision = np.nanmean(valid_metrics[:,1])
                    valid_recall = np.nanmean(valid_metrics[:,2])
                    valid_score = np.mean(valid_scores)
                    
                    train_log_line = ('{:.2f},train,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}' ).format(
                        self.epoch, train_loss, train_score, train_iou, train_precision, train_recall)
                    
                    val_log_line = ('{:.2f},validation,{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}' ).format(
                        self.epoch, valid_loss, valid_score, valid_iou, valid_precision,  valid_recall)
                    
                    f = open(self.nn_name + ".log", "a")
                    f.write(train_log_line + "\n")
                    f.write(val_log_line + "\n")
                    f.close()
                    
                    print(train_log_line)
                    print(val_log_line)

                # Store losses and scores.
                self.params['train_loss'].extend([train_loss])
                self.params['valid_loss'].extend([valid_loss])
                self.params['train_score'].extend([train_score])
                self.params['valid_score'].extend([valid_score])
        
                # Save summaries for TensorBoard.
                if self.use_tb_summary:
                    train_summary = sess.run(self.merged, feed_dict = feed_dict_train)
                    valid_summary = sess.run(self.merged, feed_dict = feed_dict_valid)
                    self.train_writer.add_summary(train_summary, self.n_log_step)
                    self.valid_writer.add_summary(valid_summary, self.n_log_step)
                
        # Store parameters.
        self.params['learn_rate'] = self.learn_rate
        self.params['learn_rate_step'] = self.learn_rate_step
        self.params['learn_rate_pos'] = self.learn_rate_pos
        self.params['learn_rate_alpha'] = self.learn_rate_alpha
        self.params['learn_rate_0'] = self.learn_rate_0
        self.params['keep_prob'] = self.keep_prob
        self.params['epoch'] = self.epoch
        self.params['n_log_step'] = self.n_log_step
        self.params['log_step'] = self.log_step
        self.params['input_shape'] = self.input_shape
        self.params['output_shape'] = self.output_shape
        self.params['mb_size'] = self.mb_size
        self.params['dropout_proba'] = self.dropout_proba
        
        print('Training ended. Running time: {}'.format(datetime.datetime.now()-start))
    
    def summary_variable(self, var, var_name):
        """ Attach summaries to a tensor for TensorBoard visualization. """
        with tf.name_scope(var_name):
            mean = tf.reduce_mean(var)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('mean', mean)
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

    def attach_summary(self, sess):
        """ Attach TensorBoard summaries to certain tensors. """
        self.use_tb_summary = True
        
        # Create summary tensors for TensorBoard.
        tf.summary.scalar('loss_tf', self.loss_tf)

        # Merge all summaries.
        self.merged = tf.summary.merge_all()

        # Initialize summary writer.
        timestamp = datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
        filepath = os.path.join(os.getcwd(), LOGS_DIR_NAME, (self.nn_name+'_'+timestamp))
        self.train_writer = tf.summary.FileWriter(os.path.join(filepath,'train'), sess.graph)
        self.valid_writer = tf.summary.FileWriter(os.path.join(filepath,'valid'), sess.graph)

    def attach_saver(self):
        """ Initialize TensorFlow saver. """
        with self.graph.as_default():
            self.use_tf_saver = True
            self.saver_tf = tf.train.Saver()

    def save_model(self, sess):
        """ Save parameters, tensors and summaries. """
        if not os.path.isdir(os.path.join(CW_DIR, SAVES_DIR_NAME)):
            os.mkdir(SAVES_DIR_NAME)
        filepath = os.path.join(os.getcwd(), SAVES_DIR_NAME , self.nn_name+'_params.npy')
        np.save(filepath, self.params) # save parameters of the network

        # TensorFlow saver
        if self.use_tf_saver:
            filepath = os.path.join(os.getcwd(),  self.nn_name)
            self.saver_tf.save(sess, filepath)

        # TensorBoard summaries
        if self.use_tb_summary:
            self.train_writer.close()
            self.valid_writer.close()
                  
        
    def load_session_from_file(self, filename, allow_growth=False, same_location=False):
        """ Load session from a file, restore the graph, and load the tensors. """
        tf.reset_default_graph()
        filepath = os.path.join(os.getcwd(), filename + '.meta')
        print(filepath)
        saver = tf.train.import_meta_graph(filepath)
        
        if allow_growth:
            gpu_options = tf.GPUOptions(allow_growth=True)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            sess = tf.Session() # default session
        
        saver.restore(sess, filename) # restore session
        print(filename)
        self.graph = tf.get_default_graph() # save default graph
        print(filename)
        self.load_parameters(filename, same_location) # load parameters
        self.load_tensors(self.graph) # define relevant tensors as variables 
        return sess
    
    def load_parameters(self, filename, same_location=False):
        '''Load helper and tunable parameters.'''
        if (same_location):
            filepath = os.path.join(os.getcwd(), filename+'_params.npy')
        else:
            filepath = os.path.join(os.getcwd(), SAVES_DIR_NAME, filename+'_params.npy')
            
        self.params = np.load(filepath, allow_pickle=True).item() # load parameters of network
        
        self.nn_name = filename
        self.learn_rate = self.params['learn_rate']
        self.learn_rate_0 = self.params['learn_rate_0']
        self.learn_rate_step = self.params['learn_rate_step']
        self.learn_rate_alpha = self.params['learn_rate_alpha']
        self.learn_rate_pos = self.params['learn_rate_pos']
        self.keep_prob = self.params['keep_prob']
        self.epoch = self.params['epoch'] 
        self.n_log_step = self.params['n_log_step']
        self.log_step = self.params['log_step']
        self.input_shape = self.params['input_shape']
        self.output_shape = self.params['output_shape'] 
        self.mb_size = self.params['mb_size']   
        self.dropout_proba = self.params['dropout_proba']
        
        print('Parameters of the loaded neural network')
        print('\tnn_name = {}, epoch = {:.2f}, mb_size = {}'.format(
            self.nn_name, self.epoch, self.mb_size))
        print('\tinput_shape = {}, output_shape = {}'.format(
            self.input_shape, self.output_shape))
        print('\tlearn_rate = {:.10f}, learn_rate_0 = {:.10f}, dropout_proba = {}'.format(
            self.learn_rate, self.learn_rate_0, self.dropout_proba))
        print('\tlearn_rate_step = {}, learn_rate_pos = {}, learn_rate_alpha = {}'.format(
            self.learn_rate_step, self.learn_rate_pos, self.learn_rate_alpha))

    def load_tensors(self, graph):
        """ Load tensors from a graph. """
        # Input tensors
        self.x_data_tf = graph.get_tensor_by_name("x_data_tf:0")
        self.y_data_tf = graph.get_tensor_by_name("y_data_tf:0")

        # Tensors for training and prediction.
        self.learn_rate_tf = graph.get_tensor_by_name("learn_rate_tf:0")
        self.keep_prob_tf = graph.get_tensor_by_name("keep_prob_tf:0")
        self.loss_tf = graph.get_tensor_by_name('loss_tf:0')
        self.train_step_tf = graph.get_operation_by_name('train_step_tf')
        self.z_pred_tf = graph.get_tensor_by_name('z_pred_tf:0')
        self.y_pred_tf = graph.get_tensor_by_name("y_pred_tf:0")
        self.training_tf = graph.get_tensor_by_name("training_tf:0")
        self.extra_update_ops_tf = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def get_prediction(self, sess, x_data, keep_prob=1.0):
        """ Prediction of the neural network graph. """
        return sess.run(self.y_pred_tf, feed_dict={self.x_data_tf: x_data,
                                                     self.keep_prob_tf: keep_prob})
       
    def get_loss(self, sess, x_data, y_data, keep_prob=1.0):
        """ Compute the loss. """
        return sess.run(self.loss_tf, feed_dict={self.x_data_tf: x_data, 
                                                 self.y_data_tf: y_data,
                                                 self.keep_prob_tf: keep_prob})
## here are two ranking and c'=c, as binary classification
import tensorflow as tf

class LSTM_RCE(object):

    def init_weights(self, input_dim, output_dim, name, std=0.1, reg=None):
        return tf.get_variable(name,shape=[input_dim, output_dim],initializer=tf.random_normal_initializer(0.0, std),regularizer = reg)
    def init_bias(self, output_dim, name):
        return tf.get_variable(name,shape=[output_dim],initializer=tf.constant_initializer(1.0))
    def no_init_weights(self, input_dim, output_dim, name):
        return tf.get_variable(name,shape=[input_dim, output_dim])
    def no_init_bias(self, output_dim, name):
        return tf.get_variable(name,shape=[output_dim])

    def __init__(self, input_dim, output_dim, hidden_dim, fc_dim, train):

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input = tf.placeholder('float', shape=[None, None, self.input_dim])#[batch size x seq length x input dim]
        self.labels = tf.placeholder('float', shape=[None,output_dim])
        self.rank= tf.placeholder('float', shape=[1])
        self.keep_prob = tf.placeholder(tf.float32)
        self.step = tf.placeholder(tf.int32)
        if train == 1:
            self.Wi = self.init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight',reg=None)
            self.Ui = self.init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight',reg=None)
            self.bi = self.init_bias(self.hidden_dim, name='Input_Hidden_bias')

            self.Wf = self.init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight',reg=None)
            self.Uf = self.init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight',reg=None)
            self.bf = self.init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            self.Wog = self.init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight',reg=None)
            self.Uog = self.init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight',reg=None)
            self.bog = self.init_bias(self.hidden_dim, name='Output_Hidden_bias')

            self.Wc = self.init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight',reg=None)
            self.Uc = self.init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight',reg=None)
            self.bc = self.init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            self.W_decomp = self.init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight',reg=None)
            self.b_decomp = self.init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            # fully connect output for one ranking net (sep)
            self.Wo = self.init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight',reg=None)#tf.contrib.layers.l2_regularizer(scale=0.001)
            self.bo = self.init_bias(fc_dim, name='Fc_Layer_bias')

            # fully connect output for the other ranking net (non-sep)
            self.Wo2 = self.init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight2',
                                        reg=None)  # tf.contrib.layers.l2_regularizer(scale=0.001)
            self.bo2 = self.init_bias(fc_dim, name='Fc_Layer_bias2')

            # score output for one ranking net (sep)
            self.W_score = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight',reg=None)
            self.b_score = self.init_bias(output_dim, name='Output_Layer_bias')

            # score output for the other ranking net (non-sep)
            self.W_score2 = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight2',reg=None)
            self.b_score2 = self.init_bias(output_dim, name='Output_Layer_bias2')

        else:
            self.Wi = self.no_init_weights(self.input_dim, self.hidden_dim, name='Input_Hidden_weight')
            self.Ui = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Input_State_weight')
            self.bi = self.no_init_bias(self.hidden_dim, name='Input_Hidden_bias')

            self.Wf = self.no_init_weights(self.input_dim, self.hidden_dim, name='Forget_Hidden_weight')
            self.Uf = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Forget_State_weight')
            self.bf = self.no_init_bias(self.hidden_dim, name='Forget_Hidden_bias')

            self.Wog = self.no_init_weights(self.input_dim, self.hidden_dim, name='Output_Hidden_weight')
            self.Uog = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Output_State_weight')
            self.bog = self.no_init_bias(self.hidden_dim, name='Output_Hidden_bias')

            self.Wc = self.no_init_weights(self.input_dim, self.hidden_dim, name='Cell_Hidden_weight')
            self.Uc = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Cell_State_weight')
            self.bc = self.no_init_bias(self.hidden_dim, name='Cell_Hidden_bias')

            self.W_decomp = self.no_init_weights(self.hidden_dim, self.hidden_dim, name='Decomposition_Hidden_weight')
            self.b_decomp = self.no_init_bias(self.hidden_dim, name='Decomposition_Hidden_bias_enc')

            self.Wo = self.no_init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight')
            self.bo = self.no_init_bias(fc_dim, name='Fc_Layer_bias')

            self.Wo2 = self.no_init_weights(self.hidden_dim, fc_dim, name='Fc_Layer_weight2')
            self.bo2 = self.no_init_bias(fc_dim, name='Fc_Layer_bias2')

            self.W_score = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight',
                                               reg=None)
            self.b_score = self.init_bias(output_dim, name='Output_Layer_bias')

            self.W_score2 = self.init_weights(fc_dim, output_dim, name='Output_Layer_weight2',
                                             reg=None)
            self.b_score2 = self.init_bias(output_dim, name='Output_Layer_bias2')

    def LSTM_Unit(self, prev_hidden_memory, input):

        prev_hidden_state, prev_cell = tf.unstack(prev_hidden_memory)
        x = input

        i = tf.sigmoid(tf.matmul(x, self.Wi) + tf.matmul(prev_hidden_state, self.Ui) + self.bi)
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(prev_hidden_state, self.Uf) + self.bf)
        o = tf.sigmoid(tf.matmul(x, self.Wog) + tf.matmul(prev_hidden_state, self.Uog) + self.bog)
        C = tf.nn.tanh(tf.matmul(x, self.Wc) + tf.matmul(prev_hidden_state, self.Uc) + self.bc)
        Ct = f * prev_cell + i * C
        current_hidden_state = o * tf.nn.tanh(Ct)

        return tf.stack([current_hidden_state, Ct])

    def get_states(self):

        batch_size = tf.shape(self.input)[0]
        scan_input_ = tf.transpose(self.input, perm=[2, 0, 1])
        scan_input = tf.transpose(scan_input_) # [seq_length x batch_size x input_dim]
        initial_hidden = tf.zeros([batch_size, self.hidden_dim], tf.float32)
        ini_state_cell = tf.stack([initial_hidden, initial_hidden])

        packed_hidden_states = tf.scan(self.LSTM_Unit, scan_input, initializer=ini_state_cell, name='states')
        all_states = packed_hidden_states[:, 0, :, :]

        return all_states

    def get_output(self, state):
        output = tf.nn.relu(tf.matmul(state, self.Wo) + self.bo)
        output = tf.nn.dropout(output, self.keep_prob)

        output = tf.matmul(output, self.W_score) + self.b_score
        return output

    def get_output2(self, state):
        output = tf.nn.relu(tf.matmul(state, self.Wo2) + self.bo2)
        output = tf.nn.dropout(output, self.keep_prob)

        output = tf.matmul(output, self.W_score2) + self.b_score2
        return output

    def get_outputs1(self): # Returns all the outputs of c0 rankings
        all_states = self.get_states()

        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :]

        return output

    def get_outputs2(self): # Returns all the outputs of c1 rankings
        all_states = self.get_states()

        all_outputs2 = tf.map_fn(self.get_output2, all_states)
        output2 = tf.reverse(all_outputs2, [0])[0, :, :]

        return output2

    def get_outputs_all(self): # Returns all the outputs of c rankings
        all_states = self.get_states()

        all_outputs = tf.map_fn(self.get_output, all_states)
        output = tf.reverse(all_outputs, [0])[0, :, :]

        all_outputs2 = tf.map_fn(self.get_output2, all_states)
        output2 = tf.reverse(all_outputs2, [0])[0, :, :]

        output= tf.concat(1,[output,output2])

        return output

    def get_cost_acc(self):

        # pairwise-ranking loss of one ranking
        output1 = self.get_outputs1()
        cross_entropy1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                                               logits=output1)) + tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rank, logits=output1[0] - output1[1])
        # pairwise-ranking loss of the other ranking
        output2 = self.get_outputs2()
        cross_entropy2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels,
                                                                               logits=output2)) + tf.nn.sigmoid_cross_entropy_with_logits(labels=self.rank, logits=output2[0] - output2[1])
        cross_entropy=cross_entropy1+cross_entropy2

        # ranking-based cross_entropy (RCE)
        logits = self.get_outputs_all()
        ranking_based_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits))  # ranking-based cross_entropy

        # total loss
        loss=cross_entropy+ranking_based_cross_entropy

        y_pred = tf.nn.softmax(logits)
        y = self.labels

        return loss, y_pred, y, logits, self.labels

    def get_cost_acc_for_test(self):

        logits = self.get_outputs_all() # ranking score
        ranking_based_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)) # ranking-based cross_entropy
        y_pred = tf.nn.softmax(logits)
        y = self.labels

        return ranking_based_cross_entropy, y_pred, y, logits, self.labels

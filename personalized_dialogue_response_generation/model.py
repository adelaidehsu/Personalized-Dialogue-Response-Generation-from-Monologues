import tensorflow as tf
import numpy as np
import copy

from units import *
import data_utils


class GAN:
    def __init__(
            self,
            mode,
            size,
            num_layers,
            vocab_size,
            buckets,
            feature_size = 6,
            baseline = 1.5,
            learning_rate=0.5,
            learning_rate_decay_factor=0.99,
            max_gradient_norm=5.0,
            learning_rate_star=1e-4,
            critic=None,
            critic_size=None,
            critic_num_layers=None,
            other_option=None,
            use_attn=False,
            output_sample=False,
            input_embed=True,
            batch_size=32,
            D_lambda=1, #the hyperparameter for Discriminator
            G_lambda=(1,1), #the hyperparameter for Generator
            D_lr=1e-4,
            D_lr_decay_factor=0.99,
            v_lr=1e-4,
            v_lr_decay_factor=0.99,
            dtype=tf.float32):

        # self-config
        self.size = size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        if vocab_size > 1000:
            num_sampled = 512
        elif vocab_size < 20:
            num_sampled = 5
        self.feature_size = feature_size
        self.buckets = buckets
        self.critic = None
        self.other_option = other_option
        self.use_attn = use_attn
        self.output_sample = output_sample
        self.input_embed = input_embed
        self.batch_size = batch_size

        # general vars
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, dtype=dtype)
        self.learning_rate_star = tf.Variable(float(learning_rate_star), trainable=False, dtype=dtype)
        self.D_lr = tf.Variable(float(D_lr), trainable=False, dtype=dtype)
        self.v_lr = tf.Variable(float(v_lr), trainable=False, dtype=dtype)
        self.op_lr_decay = self.learning_rate.assign(self.learning_rate * learning_rate_decay_factor)
        self.op_D_lr_decay = self.D_lr.assign(self.D_lr * D_lr_decay_factor)
        self.op_v_lr_decay = self.v_lr.assign(self.v_lr * v_lr_decay_factor)
        self.learning_rate_star_decay = self.learning_rate_star.assign(self.learning_rate_star * learning_rate_decay_factor)
        self.global_F_step = tf.Variable(0, trainable=False)
        self.global_B_step = tf.Variable(0, trainable=False)
        self.global_D_step = tf.Variable(0, trainable=False)
        self.global_preF_step = tf.Variable(0, trainable=False)
        self.global_preB_step = tf.Variable(0, trainable=False)
        self.D_lambda = D_lambda
        self.G_lambda = G_lambda

        
        
        ### ------------------------------------------------------------ ###
        ###                        Common Param                          ###
        ### ------------------------------------------------------------ ###
        # output projection 
        # also used for the backward model
        w = tf.get_variable('proj_w', [size, vocab_size])
        w_t = tf.transpose(w)
        b = tf.get_variable('proj_b', [vocab_size])
        self.output_projection = (w, b)

        feature_w = tf.get_variable('feature_proj_w', [feature_size, size/4])
        feature_w_t = tf.transpose(feature_w)
        
        #slef-config of glove
        self.gloveA = tf.placeholder(tf.float32, shape = [vocab_size, size], name = 'gloveA')
        self.gloveB = tf.placeholder(tf.float32, shape = [vocab_size, int(3*size/4)], name = 'gloveB')

        
        ### ------------------------------------------------------------ ###
        ###                        Forward Model                         ###
        ### ------------------------------------------------------------ ###
        ## ------------------------ ENCODER ---------------------------- ##
        with variable_scope.variable_scope('FORWARD') as scope:
            # core cells, encoder and decoder are separated
            self.forward_dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size, name='forward_dec_cell_{0}'.format(x)) for x in range(num_layers)])
            self.forward_enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size, name='forward_enc_cell_{0}'.format(y)) for y in range(num_layers)])
        
            # inputs should be a list of [batch_size]
            # encoder's placeholder (real_words)
            self.real_words = []
            for i in range(buckets[-1][0]):
                self.real_words.append(
                    tf.placeholder(tf.int32, shape = [None],
                                   name = 'real_word_{0}'.format(i)))

            self.forward_seq_len = tf.placeholder(
                tf.int32, shape = [None],
                name = 'forward_seq_len')

            ## ------------------------ DECODER ---------------------------- ##
            # target feature
            self.target_feature = []
            self.target_feature.append(tf.placeholder(tf.float32, \
                                                shape = [None, feature_size], name = 'target_feature'))

            # decoder's placeholder
            # decoder need one more input for token <EOS>
            self.forward_decoder_inputs = []
            for i in range(buckets[-1][1] + 1):
                self.forward_decoder_inputs.append(
                tf.placeholder(tf.int32, shape = [None],
                                name = 'forward_decoder_{0}'.format(i)))

            # forward target weight for pretraining
            self.forward_weights = []
            for i in range(buckets[-1][1] + 1):
                self.forward_weights.append(
                    tf.placeholder(tf.float32, shape = [None],
                                name = 'forward_weight_{0}'.format(i)))


        ### ------------------------------------------------------------ ###
        ###                        Backward Model                        ###
        ### ------------------------------------------------------------ ###
        with variable_scope.variable_scope('BACKWARD') as scope:
            ## ------------------------ ENCODER ---------------------------- ##
            # core cells, encoder and decoder are separated
            self.backward_dec_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size, name='backward_dec_cell_{0}'.format(x)) for x in range(num_layers)])
            self.backward_enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size, name='backward_enc_cell_{0}'.format(y)) for y in range(num_layers)])
        
            # inputs should be a list of [batch_size]
            # encoder's placeholder (backward_real_words)
            self.backward_real_words = []
            for i in range(buckets[-1][1]):
                self.backward_real_words.append(
                    tf.placeholder(tf.int32, shape = [None],
                               name = 'backward_real_word_{0}'.format(i)))

            self.backward_seq_len = tf.placeholder(
                tf.int32, shape = [None],
                name = 'backward_seq_len')

            ## ------------------------ DECODER ---------------------------- ##
    
            # original feature
            self.original_feature = []
            self.original_feature.append(tf.placeholder(tf.float32, \
                                                shape = [None, feature_size], name = 'original_feature'))

            # decoder's placeholder
            # decoder need one more input for token <EOS>
            self.backward_decoder_inputs = []
            for i in range(buckets[-1][0] + 1):
                self.backward_decoder_inputs.append(
                    tf.placeholder(tf.int32, shape = [None],
                                name = 'backward_decoder_{0}'.format(i)))

            # backward target weight
            self.backward_weights = []
            for i in range(buckets[-1][0] + 1):
                self.backward_weights.append(
                    tf.placeholder(tf.float32, shape = [None],
                                name = 'backward_weight_{0}'.format(i)))

        ### ------------------------------------------------------------ ###
        ###                    Discriminator Model                       ###
        ### ------------------------------------------------------------ ###
        with variable_scope.variable_scope('DISCRIMINATOR') as scope:
            ## ------------------------ ENCODER ---------------------------- ##
            # core cells, encoder cells
            self.D_enc_cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(size, name='D_enc_cell_{0}'.format(x)) for x in range(num_layers)])
        
            # output projection 
            # also used for the backward model
            wrf = tf.get_variable('disc_proj_wrf', [size, 1])
            wrf_t = tf.transpose(w)
            brf = tf.get_variable('disc_proj_brf', [1])
            self.Drf_output_projection = (wrf, brf)

            wc = tf.get_variable('disc_proj_wc', [size, feature_size])
            wc_t = tf.transpose(w)
            bc = tf.get_variable('disc_proj_bc', [feature_size])
            self.Dc_output_projection = (wc, bc)

            # inputs should be a list of [batch_size]
            # encoder's placeholder (real_data)
            self.real_data = []
            for i in range(buckets[-1][1]):
                self.real_data.append(
                    tf.placeholder(tf.int32, shape = [None],
                               name = 'real_data_{0}'.format(i)))

            self.real_seq_len = tf.placeholder(
                tf.int32, shape = [None],
                name = 'real_seq_len')

            # real feature
            self.real_feature = []
            self.real_feature.append(tf.placeholder(tf.float32, \
                                                shape = [None, feature_size], name = 'real_feature'))
        

        ### ------------------------------------------------------------ ###
        ###                          Training                            ###
        ### ------------------------------------------------------------ ###

        # seq2seq-specific functions
        def loop_function(prev):
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_symbol = math_ops.argmax(prev, axis=1)
            emb_prev = tf.cast(embedding_ops.embedding_lookup(self.gloveB, prev_symbol), tf.float32)
            return [emb_prev, prev_symbol]
        
        def sample_loop_function(prev):
            prev = nn_ops.xw_plus_b(prev, self.output_projection[0], self.output_projection[1])
            prev_index = tf.multinomial(tf.log(tf.nn.softmax(2*prev)), 1)
            prev_symbol = tf.reshape(prev_index, [-1])
            emb_prev = tf.cast(embedding_ops.embedding_lookup(self.gloveB, prev_symbol), tf.float32)
            return [emb_prev, prev_symbol]
        
        def softmax_loss_function(labels, inputs):
            labels = tf.reshape(labels, [-1, 1])
            local_w_t = tf.cast(w_t, tf.float32)
            local_b = tf.cast(b, tf.float32)
            local_inputs = tf.cast(inputs, tf.float32)
            return tf.cast(tf.nn.sampled_softmax_loss(
                weights = local_w_t,
                biases = local_b,
                inputs = local_inputs,
                labels = labels,
                num_sampled = num_sampled,
                num_classes = vocab_size),
                dtype = tf.float32)

        def seq_log_prob(logits, targets, rewards=None):
            if rewards is None:
                rewards = [tf.ones(tf.shape(target), tf.float32) for target in targets]
            with ops.name_scope("sequence_log_prob", logits + targets + rewards):
                log_perp_list = []
                tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
                tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
                weights = [math_ops.add_n(tmp[i:]) for i in range(len(tmp))]
                for logit, target, weight, reward in zip(logits, targets, weights, rewards):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight * reward)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                return log_perps

        def compute_loss(logits, targets, weights):
            with ops.name_scope("sequence_loss", logits + targets + weights):
                log_perp_list = []
                for logit, target, weight in zip(logits, targets, weights):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                cost = math_ops.reduce_sum(log_perps)
                batch_size = array_ops.shape(targets[0])[0]
                return cost / math_ops.cast(batch_size, cost.dtype)
        
        def recon_compute_loss(logits, targets, weights):
            with ops.name_scope("sequence_loss", logits + targets + weights):
                log_perp_list = []
                for logit, target, weight in zip(logits, targets, weights):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight)
                log_perps = math_ops.add_n(log_perp_list)
                total_size = math_ops.add_n(weights)
                total_size += 1e-12
                log_perps /= total_size
                cost = math_ops.reduce_sum(log_perps)
                batch_size = array_ops.shape(targets[0])[0]
                return log_perps / math_ops.cast(batch_size, cost.dtype)

        def uniform_weights(targets):
            tmp = [tf.cast(tf.equal(target, data_utils.EOS_ID), tf.float32) for target in targets]
            tmp[-1] = tf.cast(tf.equal(math_ops.add_n(tmp), 0.0), tf.float32)
            uniform_weights = [tf.cast(tf.equal(math_ops.add_n(tmp[i:]), math_ops.add_n(tmp)), tf.float32) \
                               for i in range(len(tmp))]
            return uniform_weights

        def each_perp(logits, targets, weights):
            with ops.name_scope("each_perp", logits + targets):
                log_perp_list = []
                for logit, target, weight in zip(logits, targets, weights):
                    crossent = softmax_loss_function(target, logit)
                    log_perp_list.append(crossent * weight)
                return log_perp_list 

        # for saver 
        def compute_rf_c(enc_outputs, uniW):
            rfs = []
            cs  = []
            for uni, enc_output in zip(uniW, enc_outputs):
                uni = tf.expand_dims(uni, 1)
                rf = nn_ops.xw_plus_b(enc_output, self.Drf_output_projection[0], self.Drf_output_projection[1])
                rf = tf.nn.sigmoid(rf)
                rfs.append(tf.multiply(rf,uni))
                c  = nn_ops.xw_plus_b(enc_output, self.Dc_output_projection[0], self.Dc_output_projection[1])
                cs.append(tf.multiply(c,uni))
                    
            seq_len = tf.expand_dims(math_ops.add_n(uniW),1)
            output_rf = tf.divide(math_ops.add_n(rfs), seq_len)
            output_c  = tf.divide(math_ops.add_n(cs), seq_len)          
            return output_rf, output_c
        
        if mode == 'GAN':
            ### ------------------------------------------------------------ ###
            ###                     GAN Training                         ###
            ### ------------------------------------------------------------ ###
            # generator model loss
            self.losses = []
            self.reconstructed_losses = []
            self.losses_updates = []

            self.recon_losses = []
            self.G_c_losses = []
            self.G_rf_losses = []
            self.rewards = []

            # discriminator loss
            self.D_losses = []

            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    f_enc_outputs, f_enc_state = \
                        encode(self.forward_enc_cell, self.gloveA, self.real_words[:bucket[0]], self.forward_seq_len)
                    samples_dists, samples, _ = \
                        decode(self.forward_dec_cell, f_enc_state, self.gloveB, \
                               self.forward_decoder_inputs,\
                               self.target_feature, feature_w, bucket[1]+1,\
                               feed_prev=True, loop_function=sample_loop_function)
                    
                self.fake_words = samples
                fake_uniW = uniform_weights(samples)
                seq_len_fakewords = math_ops.add_n(fake_uniW)
                

                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    b_enc_outputs, b_enc_state = \
                        encode(self.backward_enc_cell, self.gloveA, self.fake_words[:bucket[1]], seq_len_fakewords)
                    outputs, _, _ = \
                        decode(self.backward_dec_cell, b_enc_state, self.gloveB, \
                                self.backward_decoder_inputs[:bucket[0]],\
                                self.original_feature, feature_w, bucket[0]+1,\
                                feed_prev=False, loop_function=sample_loop_function)

                real_targets = [self.backward_decoder_inputs[i+1] for i in range(len(self.backward_decoder_inputs)-1)]
                reconstructed_loss = compute_loss(outputs, real_targets[:bucket[0]], self.backward_weights[:bucket[0]])
                self.reconstructed_losses.append(reconstructed_loss)
                    
                recon_loss = recon_compute_loss(outputs, real_targets[:bucket[0]], self.backward_weights[:bucket[0]])
                self.recon_losses.append(recon_loss)
                

                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    ## for the fake training 
                    fake_enc_outputs, fake_enc_state = \
                        encode(self.D_enc_cell, self.gloveA, self.fake_words[:bucket[1]], seq_len_fakewords)
                    fake_rf, fake_c = compute_rf_c(fake_enc_outputs, fake_uniW)
                    ## for the real training 
                    real_enc_outputs, real_enc_state = \
                        encode(self.D_enc_cell, self.gloveA, self.real_data[:bucket[1]], self.real_seq_len)
                    real_uniW = uniform_weights(self.real_data[:bucket[1]])
                    real_rf, real_c = compute_rf_c(real_enc_outputs, real_uniW)
                
                fake_D_rf_loss = -tf.log(1-fake_rf)
                real_D_rf_loss = -tf.log(real_rf)
                fake_rf_loss = -tf.log(fake_rf)

                fake_c_loss = tf.nn.softmax_cross_entropy_with_logits(logits=fake_c, labels=self.target_feature)
                real_c_loss = tf.nn.softmax_cross_entropy_with_logits(logits=real_c, labels=self.real_feature)

                D_rf_loss = tf.reduce_mean(real_D_rf_loss + fake_D_rf_loss)
                G_rf_loss = tf.reshape(fake_rf_loss, [-1,])

                D_c_loss  = tf.reduce_mean(real_c_loss)
                G_c_loss  = fake_c_loss

                self.D_losses.append(D_rf_loss + D_c_loss)
                self.G_rf_losses.append(G_rf_loss)
                self.G_c_losses.append(G_c_loss)
                    
                reward = 5*G_c_loss + G_rf_loss + recon_loss
                self.rewards.append(reward-tf.reduce_mean(reward))
                    
                loss = seq_log_prob(samples_dists, samples, fake_uniW)
                loss_update = tf.reduce_sum(loss)/batch_size
                    
                self.losses.append(loss)
                self.losses_updates.append(loss_update)
            
            self.forward_param = [w, b, feature_w] + self.forward_enc_cell.trainable_variables + \
                                                    self.forward_dec_cell.trainable_variables
            
            self.backward_param = [w, b, feature_w] + self.backward_enc_cell.trainable_variables + \
                                                    self.backward_dec_cell.trainable_variables
            
            self.discriminator_param = [wc, bc, wrf, brf] + self.D_enc_cell.trainable_variables

            self.all_param = self.forward_param + self.backward_param + self.discriminator_param
            
            # trainable variables
            params = tf.trainable_variables()

            # update parameters
            self.forward_solver = []
            self.backward_solver = []
            self.discriminator_solver = []

            # optimizer
            D_optimizer = tf.train.GradientDescentOptimizer(self.D_lr)
            optimizer = tf.train.GradientDescentOptimizer(self.learning_rate_star)

            for j, bucket in enumerate(buckets):
                # forward model
                for_grad = tf.gradients(self.losses[j], self.forward_param, self.rewards[j])
                clipped_F_grads, _ = tf.clip_by_global_norm(for_grad, max_gradient_norm)
                self.forward_solver.append(optimizer.apply_gradients(
                        zip(clipped_F_grads, self.forward_param),
                        global_step=self.global_F_step))
                # backward model
                back_grad = tf.gradients(self.reconstructed_losses[j], self.backward_param)
                clipped_B_grads, _ = tf.clip_by_global_norm(back_grad, max_gradient_norm)
                self.backward_solver.append(optimizer.apply_gradients(
                        zip(clipped_B_grads, self.backward_param),
                        global_step=self.global_B_step))
                # discriminator model
                disc_grad = tf.gradients(self.D_losses[j], self.discriminator_param)
                clipped_D_grads, _ = tf.clip_by_global_norm(disc_grad, max_gradient_norm)
                self.discriminator_solver.append(D_optimizer.apply_gradients(
                        zip(clipped_D_grads, self.discriminator_param),
                        global_step=self.global_D_step))
        
            ### ------------------------------------------------------------ ###
            ###                     Pretrain Training                        ###
            ### ------------------------------------------------------------ ###
            self.prefor_losses = []
            self.preback_losses = []
            
            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=tf.AUTO_REUSE):
                    prefor_enc_outputs, prefor_enc_state = \
                        encode(self.forward_enc_cell, self.gloveA, self.real_words[:bucket[0]], self.forward_seq_len)
                    # feed_prev false for the pretrain
                    prefor_output, _, _ = \
                        decode(self.forward_dec_cell, prefor_enc_state, self.gloveB, \
                               self.forward_decoder_inputs[:bucket[1]],\
                               self.target_feature, feature_w, bucket[1]+1,\
                               feed_prev=False, loop_function=sample_loop_function)
                    
                    targets_prefor = [self.forward_decoder_inputs[i+1] for i in range(len(self.forward_decoder_inputs)-1)]
                    prefor_loss = compute_loss(prefor_output, targets_prefor[:bucket[1]], self.forward_weights[:bucket[1]])
                    self.prefor_losses.append(prefor_loss)


                    with variable_scope.variable_scope(variable_scope.get_variable_scope(), reuse=tf.AUTO_REUSE):
                        preback_enc_outputs, preback_enc_state = \
                            encode(self.backward_enc_cell, self.gloveA, self.backward_real_words[:bucket[0]], self.backward_seq_len)
                        # feed_prev false for the pretrain
                        preback_output, _, _ = \
                            decode(self.backward_dec_cell, preback_enc_state, self.gloveB, \
                                   self.backward_decoder_inputs[:bucket[1]],\
                                   self.original_feature, feature_w, bucket[1]+1,\
                                   feed_prev=False, loop_function=sample_loop_function)
                    targets_preback = [self.backward_decoder_inputs[i+1] for i in range(len(self.backward_decoder_inputs)-1)]
                    preback_loss = compute_loss(preback_output, targets_preback[:bucket[1]], self.backward_weights[:bucket[1]])
                    self.preback_losses.append(preback_loss)


            # update parameter
            self.prefor_update = []
            self.preback_update = []

            prefor_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            preback_optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

            for j in range(len(self.buckets)):
                prefor_gradients = tf.gradients(self.prefor_losses[j], self.forward_param)
                preback_gradients = tf.gradients(self.preback_losses[j], self.backward_param)

                prefor_clipped_gradients, _ = tf.clip_by_global_norm(prefor_gradients, max_gradient_norm)
                preback_clipped_gradients, _ = tf.clip_by_global_norm(preback_gradients, max_gradient_norm)

                self.prefor_update.append(prefor_optimizer.apply_gradients(zip(prefor_clipped_gradients, self.forward_param),
                                                                    global_step=self.global_preF_step))
                self.preback_update.append(preback_optimizer.apply_gradients(zip(preback_clipped_gradients, self.backward_param),
                                                                    global_step=self.global_preB_step))
        
        elif mode == 'G_test':
            self.enc_state = []
            self.outputs = []
            
            enc_outputs, enc_state = \
                encode(self.forward_enc_cell, self.gloveA, self.real_words, self.forward_seq_len)
            
            self.A = enc_outputs 
            
            self.enc_state.append(enc_state)
            # for argmax test
            with variable_scope.variable_scope(
                    variable_scope.get_variable_scope(), reuse=None):
                outputs, _, _ = \
                    decode(self.forward_dec_cell, enc_state, self.gloveB, \
                           self.forward_decoder_inputs, self.target_feature, feature_w, buckets[-1][1], \
                           feed_prev=True, loop_function=loop_function)
            self.outputs.append(outputs)
            self.print_outputs = []
            self.tmp_outputs = []
            for j, outs in enumerate(self.outputs):
                self.print_outputs.append([])
                self.tmp_outputs.append([])
                for i in range(len(outs)):
                    self.print_outputs[j].append(nn_ops.xw_plus_b(outs[i], self.output_projection[0], self.output_projection[1]))
                    self.tmp_outputs[j].append(math_ops.argmax(self.print_outputs[j][i], axis=1))
            self.max_log_prob = - seq_log_prob(outputs, self.tmp_outputs[0])

        elif mode == 'D_test':
            enc_outputs, enc_state = \
                encode(self.D_enc_cell, self.gloveA, self.real_data, self.real_seq_len)
            real_uniW = uniform_weights(self.real_data)
            rf, c = compute_rf_c(enc_outputs, real_uniW)
            self.rf = rf
            self.c = tf.nn.softmax(c)
        self.saver = tf.train.Saver()

    def train_gan(
            self,
            sess,
            real_words, 
            forward_decoder_inputs,
            target_feature, 
            backward_decoder_inputs,
            backward_weights,
            original_feature,
            real_data,
            real_feature,
            bucket_id,
            gloveA,
            gloveB,
            forward=False,
            backward=False,
            disc=False,
            real_seq_len=None,
            forward_seq_len=None
    ):
        batch_size = real_words[0].shape[0]
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {}

        for l in range(encoder_size):
            input_feed[self.real_words[l].name] = real_words[l]
            input_feed[self.backward_decoder_inputs[l].name] = backward_decoder_inputs[l]
            input_feed[self.backward_weights[l].name] = backward_weights[l]

        input_feed[self.forward_seq_len] = forward_seq_len
        input_feed[self.real_seq_len] = real_seq_len
        input_feed[self.target_feature[0].name] = target_feature
        input_feed[self.gloveA.name] = gloveA
        input_feed[self.gloveB.name] = gloveB

        for l in range(decoder_size):
            input_feed[self.real_data[l].name] = real_data[l]
            input_feed[self.forward_decoder_inputs[l].name] = forward_decoder_inputs[l]
        input_feed[self.original_feature[0].name] = original_feature
        input_feed[self.real_feature[0].name] = real_feature
        
        last_target = self.backward_decoder_inputs[encoder_size].name
        input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)

        output_feed = []
        if forward:
            output_feed.append(self.forward_solver[bucket_id])
            output_feed.append(self.losses_updates[bucket_id])
        if backward:
            output_feed.append(self.backward_solver[bucket_id])
            output_feed.append(self.reconstructed_losses[bucket_id])
        if disc:
            output_feed.append(self.discriminator_solver[bucket_id])
            output_feed.append(self.D_losses[bucket_id])
        return sess.run(output_feed, input_feed)


    def train_previous(
            self,
            sess,
            real_words,
            target_feature,
            forward_decoder_inputs,
            forward_weights,
            backward_real_words,
            original_feature,
            backward_decoder_inputs,
            backward_weights,
            bucket_id,
            gloveA,
            gloveB,
            forward_seq_len=None,
            backward_seq_len=None
    ):
        batch_size = real_words[0].shape[0]
        encoder_size, decoder_size = self.buckets[bucket_id]
        input_feed = {}
        for l in range(encoder_size): # forward encoder && backward decoder
            input_feed[self.real_words[l].name] = real_words[l]
            input_feed[self.backward_decoder_inputs[l].name] = backward_decoder_inputs[l]
            input_feed[self.backward_weights[l].name] = backward_weights[l]
        input_feed[self.forward_seq_len] = forward_seq_len
        input_feed[self.target_feature[0].name] = target_feature
        input_feed[self.gloveA.name] = gloveA
        input_feed[self.gloveB.name] = gloveB

        for l in range(decoder_size): # forward decoder && backward encoder
            input_feed[self.forward_decoder_inputs[l].name] = forward_decoder_inputs[l]
            input_feed[self.forward_weights[l].name] = forward_weights[l]
            input_feed[self.backward_real_words[l].name] = backward_real_words[l]
        input_feed[self.backward_seq_len] = backward_seq_len
        input_feed[self.original_feature[0].name] = original_feature
        
        last_target = self.forward_decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([batch_size], dtype = np.int32)

        last_target_b = self.backward_decoder_inputs[encoder_size].name
        input_feed[last_target_b] = np.zeros([batch_size], dtype = np.int32)
        
        output_feed = [self.prefor_losses[bucket_id], self.preback_losses[bucket_id], \
                self.prefor_update[bucket_id], self.preback_update[bucket_id]]
        return sess.run(output_feed, input_feed)


    def dynamic_decode_G(
        self, 
        sess, 
        real_words, 
        forward_seq_len, 
        forward_decoder_inputs,
        target_feature,
        gloveA,
        gloveB
    ): 
        encoder_size = self.buckets[-1][0]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.real_words[l].name] = real_words[l]
        input_feed[self.forward_seq_len] = forward_seq_len
        input_feed[self.forward_decoder_inputs[0].name] = forward_decoder_inputs[0]
        input_feed[self.target_feature[0].name] = target_feature
        input_feed[self.gloveA.name] = gloveA
        input_feed[self.gloveB.name] = gloveB
        
        output_feed = [self.A, self.tmp_outputs[0], self.max_log_prob]
        return sess.run(output_feed, input_feed)


    def dynamic_decode_D(
        self,
        sess,
        real_data,
        real_seq_len,
        gloveA
    ): 
        print(real_data)
        print(real_seq_len)
        encoder_size = self.buckets[-1][1]
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.real_data[l].name] = real_data[l]
        input_feed[self.real_seq_len] = real_seq_len
        input_feed[self.gloveA.name] = gloveA

        output_feed = [self.rf, self.c]
        return sess.run(output_feed, input_feed)

import tensorflow as tf
import numpy as np
import pickle
import random
import re
import os
import sys
import time
import math

from model import *
from nltk.translate.bleu_score import sentence_bleu
import data_utils
import args


import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove


def parse_buckets(str_buck):
    _pair = re.compile(r"(\d+,\d+)") # find buckets like (*,*) --|
    _num = re.compile(r"\d+")        # parse out the number ------> (15,20) -> 15 20
    buck_list = _pair.findall(str_buck)
    if len(buck_list) < 1:
        raise ValueError("There should be at least 1 specific bucket.\nPlease set buckets in configuration.")
    buckets = []
    for buck in buck_list:
        tmp = _num.findall(buck)
        d_tmp = (int(tmp[0]), int(tmp[1]))
        buckets.append(d_tmp)
    return buckets


FLAGS = args.parse()
_buckets = parse_buckets(FLAGS.buckets)

glove_corpus_path = FLAGS.glove_model+".txt.voc%d" % FLAGS.vocab_size
sentences = list(itertools.islice(Text8Corpus(glove_corpus_path),None))
corpus = Corpus()
corpus.fit(sentences, window=30)
modelA = FLAGS.glove_model + "_%d.model" % FLAGS.size
modelB = FLAGS.glove_model + "_%d.model" % (FLAGS.size*3/4)

def train_GAN():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    set_session(tf.Session(config=config))

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    
    if not os.path.exists(FLAGS.pretrain_dir):
        os.makedirs(FLAGS.pretrain_dir)
    
    if not os.path.exists(FLAGS.gan_dir):
        os.makedirs(FLAGS.gan_dir)
    
    def build_summaries(): 
        train_loss = tf.Variable(0.)
        tf.summary.scalar("train_loss", train_loss)
        summary_vars = [train_loss]
        summary_ops = tf.summary.merge_all()
        return summary_ops, summary_vars
    
    
    feature, data, train, data_voc, train_voc = \
        data_utils.prepare_data(FLAGS.feature_path, FLAGS.feature_size, FLAGS.data_dir, \
                        FLAGS.data_path, FLAGS.train_path, FLAGS.vocab_size)
    
    vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
    data_utils.combine_corpus(data_voc, train_voc, vocab_path, glove_corpus_path , 28)

    if not os.path.exists(modelA) :
        gloveA = Glove(no_components=FLAGS.size, learning_rate=0.05)
        gloveA.fit(corpus.matrix, epochs=300, no_threads=4, verbose=True)
        gloveA.add_dictionary(corpus.dictionary)
        gloveA.save(modelA) # 512

    if not os.path.exists(modelB) :
        gloveB = Glove(no_components=int(FLAGS.size*3/4), learning_rate=0.05)
        gloveB.fit(corpus.matrix, epochs=300, no_threads=4, verbose=True)
        gloveB.add_dictionary(corpus.dictionary)
        gloveB.save(modelB) # 384
    
    gloveA = Glove.load(modelA)
    gloveA.add_dictionary(corpus.dictionary)
    gloveB = Glove.load(modelB)
    gloveB.add_dictionary(corpus.dictionary)
    
    vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

    with tf.Session() as sess:
        model = GAN(
            'GAN',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.feature_size, 
            FLAGS.baseline,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            batch_size=FLAGS.batch_size,
            D_lambda = FLAGS.lambda_dis,
            G_lambda = (FLAGS.lambda_one, FLAGS.lambda_two),
            dtype=tf.float32)
        # build summary and intialize
        summary_ops, summary_vars = build_summaries()
        sess.run(tf.variables_initializer(tf.global_variables()))
        log_dir = os.path.join(FLAGS.model_dir, 'log')
        writer = tf.summary.FileWriter(log_dir, sess.graph)
        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print ('read in model from {}'.format(ckpt.model_checkpoint_path))
            model.saver.restore(sess, ckpt.model_checkpoint_path)

        # load in train and dev(valid) data with buckets
        train_set = read_data_with_buckets(train, FLAGS.max_train_data_size)
        data_set = read_data_with_buckets(data, FLAGS.max_train_data_size)

        train_buckets_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_buckets_sizes))
        print ('each buckets has: {d}'.format(d=train_buckets_sizes))
        train_buckets_scale = [sum(train_buckets_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_buckets_sizes))]
        
        # main process
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        
        # glove embeddings
        gloveA_emb = gloveA.word_vectors[:FLAGS.vocab_size,:]
        gloveB_emb = gloveB.word_vectors[:FLAGS.vocab_size,:]
        ### ------------------------------------------------------------ ###
        ###                           Pretrain                           ###
        ### ------------------------------------------------------------ ###
        while True:
            # get batch from a random selected bucket
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01]) # random pick bucket

            # get batch for the pretraining data
            feature_inputs_f, encoder_inputs_f, decoder_inputs_f, weights_f, seq_lens_f, _,  \
            feature_inputs_b, encoder_inputs_b, decoder_inputs_b, weights_b, seq_lens_b, _,  = \
                get_batch_with_buckets(FLAGS.feature_size, data_set, FLAGS.batch_size, bucket_id)

            # pretrain start !
            start_time = time.time()
            forloss, _ , _, _ = model.train_previous(sess, encoder_inputs_f, feature_inputs_f, \
                                                decoder_inputs_f, weights_f, encoder_inputs_b, \
                                                feature_inputs_b, decoder_inputs_b, weights_b, \
                                                bucket_id, gloveA_emb, gloveB_emb, seq_lens_f, seq_lens_b)
            step_loss = forloss
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += np.mean(step_loss) / FLAGS.steps_per_checkpoint / (FLAGS.Gstep*2+FLAGS.Dstep+1)
            #print('pretrain : ',step_loss)
            ### ------------------------------------------------------------ ###
            ###                         Train GAN                            ###
            ### ------------------------------------------------------------ ###
            for _ in range(FLAGS.Dstep):
                # get batch from a random selected bucket
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                if train_buckets_scale[i] > random_number_01]) # random pick bucket

                # get batch for the pretraining data
                feature_inputs_f, encoder_inputs_f, decoder_inputs_f, seq_lens_f, \
                feature_inputs_b, decoder_inputs_b, weights_b, \
                real_inputs, real_feature , real_seq_lens= \
                    get_gan_data(feature, FLAGS.feature_size, train_set, FLAGS.batch_size, bucket_id)

                # D_step start !
                start_time = time.time()
                _, D_loss = model.train_gan(sess, encoder_inputs_f, decoder_inputs_f, feature_inputs_f, \
                                                decoder_inputs_b, weights_b, feature_inputs_b, \
                                                real_inputs, real_feature, bucket_id, gloveA_emb, gloveB_emb, \
                                                disc = True,real_seq_len=real_seq_lens, forward_seq_len=seq_lens_f)
                step_loss = D_loss
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += np.mean(step_loss) / FLAGS.steps_per_checkpoint / (FLAGS.Gstep*2+FLAGS.Dstep+1)
                #print('D_step : ', step_loss)
            for _ in range(FLAGS.Gstep):
                # get batch from a random selected bucket
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                if train_buckets_scale[i] > random_number_01]) # random pick bucket

                # get batch for the pretraining data
                feature_inputs_f, encoder_inputs_f, decoder_inputs_f, seq_lens_f, \
                feature_inputs_b, decoder_inputs_b, weights_b, \
                real_inputs, real_feature, real_seq_lens = \
                    get_gan_data(feature, FLAGS.feature_size, train_set, FLAGS.batch_size, bucket_id)

                # G_step start !
                start_time = time.time()
                _, for_reward = model.train_gan(sess, encoder_inputs_f, decoder_inputs_f, feature_inputs_f, \
                                                decoder_inputs_b, weights_b, feature_inputs_b, \
                                                real_inputs, real_feature, bucket_id, gloveA_emb, gloveB_emb, \
                                                forward = True,real_seq_len=real_seq_lens , forward_seq_len=seq_lens_f)
            
                step_loss = for_reward
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += np.mean(step_loss) / FLAGS.steps_per_checkpoint / (FLAGS.Gstep*2+FLAGS.Dstep+1)
                #print('for_loss :', step_loss)
                # get batch from a random selected bucket
                random_number_01 = np.random.random_sample()
                bucket_id = min([i for i in range(len(train_buckets_scale))
                                if train_buckets_scale[i] > random_number_01]) # random pick bucket

                # get batch for the pretraining data
                feature_inputs_f, encoder_inputs_f, decoder_inputs_f, seq_lens_f, \
                feature_inputs_b, decoder_inputs_b, weights_b, \
                real_inputs, real_feature, real_seq_lens = \
                    get_gan_data(feature, FLAGS.feature_size, train_set, FLAGS.batch_size, bucket_id)

                # G_step start !
                start_time = time.time()
                _, back_reward = model.train_gan(sess, encoder_inputs_f, decoder_inputs_f, feature_inputs_f, \
                                                decoder_inputs_b, weights_b, feature_inputs_b, \
                                                real_inputs, real_feature, bucket_id, gloveA_emb, gloveB_emb, \
                                                backward = True,real_seq_len=real_seq_lens , forward_seq_len=seq_lens_f)
            
                step_loss = back_reward
                step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
                loss += np.mean(step_loss) / FLAGS.steps_per_checkpoint / (FLAGS.Gstep*2+FLAGS.Dstep+1)
                #print('back_loss :', step_loss)
            current_step += 1
            # log, save and eval
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print ("Generator step %d; learning rate %.4f; learning_rate_star %.6f; D_lr %6f; step-time %.2f; perplexity "
                        "%.2f; loss %.2f"
                        % (model.global_F_step.eval(), model.learning_rate.eval(), model.learning_rate_star.eval(), model.D_lr.eval(),
                            step_time, perplexity, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.op_lr_decay)
                    sess.run(model.op_D_lr_decay)
                    sess.run(model.learning_rate_star_decay)
                previous_losses.append(loss)
                
                # write summary
                feed_dict = {}
                feed_dict[summary_vars[0]] = loss
                summary_str = sess.run(summary_ops,feed_dict=feed_dict)
                writer.add_summary(summary_str, model.global_F_step.eval())
                writer.flush()
                # Save checkpoint and zero timer and loss.
                ckpt_path = os.path.join(FLAGS.model_dir, "ckpt")
                model.saver.save(sess, ckpt_path, global_step=model.global_F_step)
                
                gan_path = os.path.join(FLAGS.gan_dir, "ckpt_prev")
                model.saver.save(sess, gan_path, global_step=model.global_F_step)
                step_time, loss = 0.0, 0.0

                sys.stdout.flush()


def test():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    gloveA = Glove.load(modelA)
    gloveA.add_dictionary(corpus.dictionary)
    gloveB = Glove.load(modelB)
    gloveB.add_dictionary(corpus.dictionary)
    
    # glove embeddings
    gloveA_emb = gloveA.word_vectors[:FLAGS.vocab_size,:]
    gloveB_emb = gloveB.word_vectors[:FLAGS.vocab_size,:]

    with tf.Session() as sess:
        model = GAN(
            'G_test',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.feature_size, 
            FLAGS.baseline,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            batch_size=FLAGS.batch_size,
            D_lambda = FLAGS.lambda_dis,
            G_lambda = (FLAGS.lambda_one, FLAGS.lambda_two),
            dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        sys.stdout.write('> ')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            if sentence.strip() == 'exit()':
                break
            number = 0
            for id , x in enumerate(sentence):
                if x == '|':
                    number = int(sentence[id+1:])
                    sentence = sentence[:id]
                    break
            feature = [[0 for _ in range(FLAGS.feature_size)]]
            if number < 6:
                feature[0][number] = 1
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
            token_ids.append(data_utils.EOS_ID)
            encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
            encoder_lens = [len(token_ids)]
            token_ids = list(token_ids) + encoder_pad
            encoder_inputs = []
            encoder_inputs.append([idx] for idx in token_ids)

            decoder_inputs = [[data_utils.GO_ID]]
                
                
            _, outputs, log_prob = model.dynamic_decode_G(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs, feature, gloveA_emb, gloveB_emb)
            outputs = [output_ids[0] for output_ids in outputs]
            if data_utils.EOS_ID in outputs:
                outputs = outputs[:outputs.index(data_utils.EOS_ID)]
            print(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                
            sys.stdout.write('> ')
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def seriestest():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    gloveA = Glove.load(modelA)
    gloveA.add_dictionary(corpus.dictionary)
    gloveB = Glove.load(modelB)
    gloveB.add_dictionary(corpus.dictionary)
    
    # glove embeddings
    gloveA_emb = gloveA.word_vectors[:FLAGS.vocab_size,:]
    gloveB_emb = gloveB.word_vectors[:FLAGS.vocab_size,:]
    
    with tf.Session() as sess:
        model = GAN(
            'G_test',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.feature_size, 
            FLAGS.baseline,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            batch_size=FLAGS.batch_size,
            D_lambda = FLAGS.lambda_dis,
            G_lambda = (FLAGS.lambda_one, FLAGS.lambda_two),
            dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        sys.stdout.write('> ')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            if sentence.strip() == 'exit()':
                break
            feature = []
            for f in range(FLAGS.feature_size):
                feature.append([[3 if x == f else 0 for x in range(FLAGS.feature_size)]])
                        
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
            token_ids.append(data_utils.EOS_ID)
            encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
            encoder_lens = [len(token_ids)]
            token_ids = list(token_ids) + encoder_pad
            encoder_inputs = []
            encoder_inputs.append([idx] for idx in token_ids)

            decoder_inputs = [[data_utils.GO_ID]]
                
            for x in range(FLAGS.feature_size):
                _, outputs, log_prob = model.dynamic_decode_G(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs, feature[x], gloveA_emb, gloveB_emb)
                outputs = [output_ids[0] for output_ids in outputs]
                if data_utils.EOS_ID in outputs:
                    outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                print(feature[x],':'," ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                
            sys.stdout.write('> ')
            sys.stdout.flush()
            sentence = sys.stdin.readline()


def disctest():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    gloveA = Glove.load(modelA)
    gloveA.add_dictionary(corpus.dictionary)
    gloveB = Glove.load(modelB)
    gloveB.add_dictionary(corpus.dictionary)
    
    # glove embeddings
    gloveA_emb = gloveA.word_vectors[:FLAGS.vocab_size,:]
    gloveB_emb = gloveB.word_vectors[:FLAGS.vocab_size,:]

    with tf.Session() as sess:
        model = GAN(
            'D_test',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.feature_size, 
            FLAGS.baseline,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            batch_size=FLAGS.batch_size,
            D_lambda = FLAGS.lambda_dis,
            G_lambda = (FLAGS.lambda_one, FLAGS.lambda_two),
            dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)

        sys.stdout.write('> ')
        sys.stdout.flush()
        sentence = sys.stdin.readline()
        while sentence:
            if sentence.strip() == 'exit()':
                break
            token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
            token_ids.append(data_utils.EOS_ID)
            encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
            encoder_lens = [len(token_ids)]
            token_ids = list(token_ids) + encoder_pad
            encoder_inputs = []
            encoder_inputs.append([idx] for idx in token_ids)
                
            rf, c = model.dynamic_decode_D(sess, encoder_inputs, encoder_lens, gloveA_emb)
            
            print('rf : ', rf)
            print('c  : ', c)

            sys.stdout.write('> ')
            sys.stdout.flush()
            sentence = sys.stdin.readline()

def filetest():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    gloveA = Glove.load(modelA)
    gloveA.add_dictionary(corpus.dictionary)
    gloveB = Glove.load(modelB)
    gloveB.add_dictionary(corpus.dictionary)
    
    # glove embeddings
    gloveA_emb = gloveA.word_vectors[:FLAGS.vocab_size,:]
    gloveB_emb = gloveB.word_vectors[:FLAGS.vocab_size,:]

    if not os.path.exists('./fileTest_log/'):
        os.makedirs('./fileTest_log/')
    
    with tf.Session() as sess:
        model = GAN(
            'G_test',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.feature_size, 
            FLAGS.baseline,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            batch_size=FLAGS.batch_size,
            D_lambda = FLAGS.lambda_dis,
            G_lambda = (FLAGS.lambda_one, FLAGS.lambda_two),
            dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
         
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)


        with open(FLAGS.test_path, 'r') as sentences:
            # step
            number = 0 
            feature = []
            output_file = []
            for idx in range(FLAGS.feature_size):
                output_file.append('./fileTest_log/%s.txt' %idx)
            
            output_list = []
            for f in range(FLAGS.feature_size):
                feature.append([[3 if x == f else 0 for x in range(FLAGS.feature_size)]])
                output_list.append([])
            
            for id, sentence in enumerate(sentences.readlines()):
                token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                token_ids.append(data_utils.EOS_ID)
                if len(token_ids) > _buckets[-1][0]:
                    continue
                encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                encoder_lens = [len(token_ids)]
                token_ids = list(token_ids) + encoder_pad
                encoder_inputs = []
                encoder_inputs.append([idx] for idx in token_ids)
                decoder_inputs = [[data_utils.GO_ID]]
                
                for x in range(FLAGS.feature_size):
                    A, outputs, log_prob = model.dynamic_decode_G(sess, encoder_inputs, encoder_lens, \
                                                             decoder_inputs, feature[x], gloveA_emb, gloveB_emb)
                    outputs = [output_ids[0] for output_ids in outputs]
                    if data_utils.EOS_ID in outputs:
                        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                    output_list[x].append(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))
                
                if number % 1000 ==0:
                    print('fileTest: generating line %d' %number)
                if number == 1000:
                    break
                number+=1
            for x in range(FLAGS.feature_size):
                with open(output_file[x], 'w') as op:
                    for line in output_list[x]:
                        op.write(line)
                        op.write('\n')

def multibleutest():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    if not os.path.exists("./multiBleu_log/"):
        os.makedirs("./multiBleu_log/")
    
    gloveA = Glove.load(modelA)
    gloveA.add_dictionary(corpus.dictionary)
    gloveB = Glove.load(modelB)
    gloveB.add_dictionary(corpus.dictionary)
    
    # glove embeddings
    gloveA_emb = gloveA.word_vectors[:FLAGS.vocab_size,:]
    gloveB_emb = gloveB.word_vectors[:FLAGS.vocab_size,:]
    
    with tf.Session() as sess:
        model = GAN(
            'G_test',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.feature_size, 
            FLAGS.baseline,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic = None,
            use_attn = FLAGS.use_attn,
            output_sample = True,
            input_embed = True,
            batch_size = FLAGS.batch_size,
            D_lambda = FLAGS.lambda_dis,
            G_lambda = (FLAGS.lambda_one, FLAGS.lambda_two),
            dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        
        prompt = None
        cheatsheetMAP = {}
        with open(FLAGS.test_path, 'r') as source:
            for line in source.readlines():
                line = line.strip()
                if line == "":
                    prompt = None
                    continue
                elif prompt == None:
                    cheatsheetMAP[line] = []
                    prompt = line
                else:
                    cheatsheetMAP[prompt].append(line)
        
        answer = []
        with open("./multiBleu_log/"+FLAGS.file_head+"_ref.txt", 'w') as ffop:
            with open("./multiBleu_log/"+FLAGS.file_head+"_Q.txt", 'w') as fop:
                feature, output_file, output_list = [], [], []
                for i in range(FLAGS.feature_size):
                    output_file.append("./multiBleu_log/"+FLAGS.file_head+"_{}.txt".format(i))
                    feature.append([[3 if x==i else 0 for x in range(FLAGS.feature_size)]])
                    output_list.append([])
                
                for p, refs in cheatsheetMAP.items():
                    check = False
                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(p), vocab, normalize_digits=False)
                    token_ids.append(data_utils.EOS_ID)
                    if len(token_ids) > _buckets[-1][0]:
                        continue
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    encoder_lens = [len(token_ids)]
                    token_ids = list(token_ids) + encoder_pad
                    encoder_inputs = []
                    for idx in token_ids:
                        encoder_inputs.append([idx])
                    decoder_inputs = [[data_utils.GO_ID]]

                    outputs_list = []
                    for x in range(FLAGS.feature_size):
                        A, outputs, log_prob = model.dynamic_decode_G(sess, encoder_inputs, encoder_lens, \
                                                                decoder_inputs, feature[x], gloveA_emb, gloveB_emb)
                        outputs = [output_ids[0] for output_ids in outputs]
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                        if data_utils.UNK_ID in outputs:
                            check = True
                            break
                        outputs_list.append(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))

                    if check:
                        continue
                    else:
                        fop.write(p)
                        fop.write('\n')
                        for x in refs:
                            ffop.write(x+'\n')
                        ffop.write('\n')
                        answer.append(refs)
                        for x in range(FLAGS.feature_size):
                            output_list[x].append(outputs_list[x])

            for x in range(FLAGS.feature_size):
                with open(output_file[x], 'w')as op:
                    bleu = []
                    for i, line in enumerate(output_list[x]):
                        op.write(line)
                        op.write("\n")
                        score = sentence_bleu(answer[i], line)
                        bleu.append(score)
                    op.write("My BLEU: {}".format(sum(bleu)/len(bleu)))
                    op.write('\n')

def maxbleutest():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    if not os.path.exists("./maxBleu_log/"):
        os.makedirs("./maxBleu_log/")
    
    gloveA = Glove.load(modelA)
    gloveA.add_dictionary(corpus.dictionary)
    gloveB = Glove.load(modelB)
    gloveB.add_dictionary(corpus.dictionary)
    
    # glove embeddings
    gloveA_emb = gloveA.word_vectors[:FLAGS.vocab_size,:]
    gloveB_emb = gloveB.word_vectors[:FLAGS.vocab_size,:]
    
    with tf.Session() as sess:
        model = GAN(
            'G_test',
            FLAGS.size,
            FLAGS.num_layers,
            FLAGS.vocab_size,
            _buckets,
            FLAGS.feature_size, 
            FLAGS.baseline,
            FLAGS.lr,
            FLAGS.lr_decay,
            FLAGS.grad_norm,
            critic=None,
            use_attn=FLAGS.use_attn,
            output_sample=True,
            input_embed=True,
            batch_size=FLAGS.batch_size,
            D_lambda = FLAGS.lambda_dis,
            G_lambda = (FLAGS.lambda_one, FLAGS.lambda_two),
            dtype=tf.float32)

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        model.saver.restore(sess, ckpt.model_checkpoint_path)
        print ('read in model from {}'.format(ckpt.model_checkpoint_path))
        
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d" % FLAGS.vocab_size)
        vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
        group_data, _ = data_utils.training_data_grouping(FLAGS.train_path, FLAGS.feature_path, FLAGS.feature_size)
        
        with open("./maxBleu_log/"+FLAGS.file_head+"_ref_msg.txt", 'w') as fop:
            with open(FLAGS.test_path, 'r')as sentences:
                number = 0
                feature, output_file, output_list = [], [], []
                for i in range(FLAGS.feature_size):
                    output_file.append("./maxBleu_log/"+FLAGS.file_head+"_{}.txt".format(i))
                    feature.append([[3 if x==i else 0 for x in range(FLAGS.feature_size)]])
                    output_list.append([])

                for sentence in sentences.readlines():
                    if number%100 == 0:
                        print("maxBleuTesting: parsing line {}".format(number))

                    if number == 500:
                        break

                    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab, normalize_digits=False)
                    token_ids.append(data_utils.EOS_ID)
                    if len(token_ids) > _buckets[-1][0]:
                        continue
                    encoder_pad = [data_utils.PAD_ID] * (_buckets[-1][0] - len(token_ids))
                    encoder_lens = [len(token_ids)]
                    token_ids = list(token_ids) + encoder_pad
                    encoder_inputs = []
                    encoder_inputs.append([idx] for idx in token_ids)
                    decoder_inputs = [[data_utils.GO_ID]]
                    
                    check = False
                    outputs_list = []
                    for x in range(FLAGS.feature_size):
                        A, outputs, log_prob = model.dynamic_decode_G(sess, encoder_inputs, encoder_lens, \
                                                                decoder_inputs, feature[x], gloveA_emb, gloveB_emb)
                        outputs = [output_ids[0] for output_ids in outputs]
                        if data_utils.EOS_ID in outputs:
                            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
                        if data_utils.UNK_ID in outputs:
                            check = True
                            break
                        outputs_list.append(" ".join([tf.compat.as_str(rev_vocab[output]) for output in outputs]))

                    if check:
                        continue
                    else:
                        number += 1
                        fop.write(sentence)
                        for x in range(FLAGS.feature_size):
                            output_list[x].append(outputs_list[x])

            for x in range(FLAGS.feature_size):
                with open(output_file[x], 'w')as op:
                    pos_lst = []
                    for i, line in enumerate(output_list[x]):
                        op.write(line)
                        op.write("\n")
                        grps_bleu = []
                        for _, sentence_lst in group_data.items():
                            bleu = -1
                            for ref in sentence_lst:
                                score = sentence_bleu(ref, line)
                                if score > bleu:
                                    bleu = score
                            grps_bleu.append(bleu)

                        max_bleu = max(grps_bleu)
                        #predicted persona: max_candidates
                        max_candidates = [i for i, x in enumerate(grps_bleu) if x == max_bleu]
                        if len(max_candidates) > 1:
                            pos_lst.append(-1)
                        else:
                            pos_lst.append(max_candidates[0])

                    tranc_pos_lst = [pos_lst[i] for i in range(len(pos_lst)) if pos_lst[i] != -1]
                    op.write("MaxBLEU Acc: {}".format(tranc_pos_lst.count(x)/len(tranc_pos_lst)))
                
def read_data_with_buckets(data_path, max_size=None):
    if FLAGS.option == 'MIXER':
        buckets = [_buckets[-1]]
    else:
        buckets = _buckets
    dataset = [[] for _ in buckets]

    index = 0
    with tf.gfile.GFile(data_path, mode='r') as data_file:
        source = data_file.readline()
        target = data_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            source_ids = [int(x) for x in source.split()]
            target_ids = [int(x) for x in target.split()]
            source_ids.append(data_utils.EOS_ID)
            target_ids.append(data_utils.EOS_ID)
            for bucket_id, (source_size, target_size) in enumerate(buckets):
                if len(source_ids) < source_size and len(target_ids) < target_size:
                    dataset[bucket_id].append([source_ids, target_ids, index, index+1])
                    break
            source = data_file.readline()
            target = data_file.readline()
            index = index+2
    return dataset

def get_batch_with_buckets(feature_size, data, batch_size, bucket_id, size=None):
    # data should be [whole_data_length x (source, target)] 
    # decoder_input should contain "GO" symbol and target should contain "EOS" symbol
    encoder_size, decoder_size     = _buckets[bucket_id]
    decoder_size_b, encoder_size_b = _buckets[bucket_id]

    encoder_inputs, decoder_inputs, seq_len = [], [], []
    encoder_inputs_b, decoder_inputs_b, seq_len_b = [], [], []

    batch_feature_inputs = []
    batch_feature_inputs_b = []
    
    for i in range(batch_size):
        encoder_input, decoder_input, id1, id2 = random.choice(data[bucket_id])
        encoder_input_b = decoder_input
        decoder_input_b = encoder_input

        encoder_pad   = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_pad_b = [data_utils.PAD_ID] * (encoder_size_b - len(encoder_input_b))

        # feature in my implementation
        encoder_inputs.append(list(encoder_input) + encoder_pad)
        encoder_inputs_b.append(list(encoder_input_b) + encoder_pad_b)

        seq_len.append(len(encoder_input))
        seq_len_b.append(len(encoder_input_b))

        decoder_pad = [data_utils.PAD_ID] * (decoder_size - len(decoder_input))
        decoder_pad_b = [data_utils.PAD_ID] * (decoder_size_b - len(decoder_input_b))

        decoder_inputs.append([data_utils.GO_ID] + decoder_input + decoder_pad)
        decoder_inputs_b.append([data_utils.GO_ID] + decoder_input_b + decoder_pad_b)

        feature = [0 for _ in range(feature_size)]
        batch_feature_inputs_b.append(feature)
        batch_feature_inputs.append(feature)


    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []
    batch_encoder_inputs_b, batch_decoder_inputs_b, batch_weights_b = [], [], []
    
    # make batch for encoder inputs
    for length_idx in range(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))

    for length_idx in range(encoder_size_b):
        batch_encoder_inputs_b.append(
            np.array([encoder_inputs_b[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))

    # make batch for decoder inputs
    for length_idx in range(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))
        batch_weight = np.ones(batch_size, dtype = np.float32)
        for batch_idx in range(batch_size):
            if length_idx < decoder_size - 1:
                target = decoder_inputs[batch_idx][length_idx + 1]
            if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
                batch_weight[batch_idx] = 0.0
        batch_weights.append(batch_weight)

    for length_idx in range(decoder_size_b):
        batch_decoder_inputs_b.append(
            np.array([decoder_inputs_b[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))
        batch_weight_b = np.ones(batch_size, dtype = np.float32)
        for batch_idx in range(batch_size):
            if length_idx < decoder_size_b - 1:
                target = decoder_inputs_b[batch_idx][length_idx + 1]
            if length_idx == decoder_size_b - 1 or target == data_utils.PAD_ID:
                batch_weight_b[batch_idx] = 0.0
        batch_weights_b.append(batch_weight_b)

    return batch_feature_inputs, batch_encoder_inputs, batch_decoder_inputs, batch_weights, seq_len, encoder_inputs, \
           batch_feature_inputs_b, batch_encoder_inputs_b, batch_decoder_inputs_b, batch_weights_b, seq_len_b, encoder_inputs_b

def get_gan_data(feature, feature_size, data, batch_size, bucket_id, size=None):
    # data should be [whole_data_length x (source, target)] 
    # decoder_input should contain "GO" symbol and target should contain "EOS" symbol
    encoder_size, decoder_size     = _buckets[bucket_id]
    decoder_size_b, encoder_size_b = _buckets[bucket_id]

    batch_feature_inputs, encoder_inputs, decoder_inputs, seq_len, real_seq_len = [], [], [], [], []
    batch_feature_inputs_b, decoder_inputs_b  = [], []
    real_feature_inputs, real_inputs = [], []
    
    for i in range(batch_size):
        encoder_input, _, id1, _ = random.choice(data[bucket_id])
        _, real_data, _, id2 = random.choice(data[bucket_id])

        encoder_pad   = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
        encoder_inputs.append(list(encoder_input) + encoder_pad)

        seq_len.append(len(encoder_input))
        real_seq_len.append(len(real_data))

        real_pad   = [data_utils.PAD_ID] * (decoder_size - len(real_data))
        real_inputs.append(list(real_data) + real_pad)

        decoder_input_b = encoder_input
        decoder_pad_b = [data_utils.PAD_ID] * (decoder_size_b - len(decoder_input_b))
        decoder_inputs_b.append([data_utils.GO_ID] + decoder_input_b + decoder_pad_b)
        
        decoder_inputs.append([data_utils.GO_ID] + [data_utils.PAD_ID]*(decoder_size-1))
        batch_feature_inputs_b.append(feature[id1])
        batch_feature_inputs.append(feature[id2])
        real_feature_inputs.append(feature[id2])

    batch_encoder_inputs, batch_weights_b = [], []
    batch_decoder_inputs = []

    batch_decoder_inputs_b = []
    batch_real_inputs = []
    
    # make batch for encoder inputs
    for length_idx in range(encoder_size):
        batch_encoder_inputs.append(
            np.array([encoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))

    # make batch for decoder inputs
    for length_idx in range(decoder_size):
        batch_real_inputs.append(
            np.array([real_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))

    for length_idx in range(decoder_size_b):
        batch_decoder_inputs_b.append(
            np.array([decoder_inputs_b[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))
        batch_weight_b = np.ones(batch_size, dtype = np.float32)
        for batch_idx in range(batch_size):
            if length_idx < decoder_size_b - 1:
                target = decoder_inputs_b[batch_idx][length_idx + 1]
            if length_idx == decoder_size_b - 1 or target == data_utils.PAD_ID:
                batch_weight_b[batch_idx] = 0.0
        batch_weights_b.append(batch_weight_b)
    
    for length_idx in range(decoder_size):
        batch_decoder_inputs.append(
            np.array([decoder_inputs[batch_idx][length_idx]
                      for batch_idx in range(batch_size)],
                     dtype = np.int32))

    return batch_feature_inputs, batch_encoder_inputs, batch_decoder_inputs, seq_len, \
           batch_feature_inputs_b, batch_decoder_inputs_b, batch_weights_b, \
           batch_real_inputs, real_feature_inputs, real_seq_len


if __name__ == '__main__':
    if FLAGS.test_type == 'Pretrain':
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        with open('{}/model.conf'.format(FLAGS.model_dir),'w') as f:
            for key, value in vars(FLAGS).items():
                f.write("{}={}\n".format(key, value))
        train_Pretrain() 
    elif FLAGS.test_type == 'GAN':
        if not os.path.exists(FLAGS.model_dir):
            os.makedirs(FLAGS.model_dir)
        with open('{}/model.conf'.format(FLAGS.model_dir),'w') as f:
            for key, value in vars(FLAGS).items():
                f.write("{}={}\n".format(key, value))
        train_GAN()
    elif FLAGS.test_type == 'D_test':
        disctest()
    elif FLAGS.test_type == 'series_test':
        seriestest()
    elif FLAGS.test_type == 'file_test':
        filetest()
    elif FLAGS.test_type == 'maxbleu_test':
        maxbleutest()
    elif FLAGS.test_type == 'bleu_test':
        bleutest()
    elif FLAGS.test_type == 'multibleu_test':
        multibleutest()
    else:
        test()
    #TODO: add style training


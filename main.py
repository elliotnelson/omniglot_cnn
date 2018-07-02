from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from random import shuffle
from scipy.ndimage import imread
from scipy.spatial.distance import cdist

import numpy as np
import tensorflow as tf

import os
import argparse
import sys
import tempfile

# FLAGS = NONE

# this ensures that full non-truncated numpy arrays are printed
np.set_printoptions(threshold='nan')


def main():
    
    # Initialize the command line parser
    parser = argparse.ArgumentParser()

    parser.add_argument('command',type=str,help='command to execute') 

    parser.add_argument('-n_pixel',type=int,default=105,help='number of pixels')

    # Training parameters:

    #o read this off from path_save string:
    parser.add_argument('-n_char',type=int,default=100,help='number of characters/classes')
    parser.add_argument('-images_per_char',type=int,default=18,help='number of training images per character')
    ## increasing:
    parser.add_argument('-learning_rate',type=float,default=1e-3,help='learning rate for training')
    ## (increasing:)
    parser.add_argument('-beta',type=float,default=0.0005,help='L2 regularization coefficient')
    parser.add_argument('-batch_size',type=int,default=18,help='batch size for training')
    # unused unless optimizer depends on momentum:
    parser.add_argument('-momentum',type=float,default=0.8,help='momentum')
    ## increasing:
    parser.add_argument('-dropout_rate',type=float,default=0.7,help='dropout rate')

    # this terminates training if early stopping never kicks in:
    parser.add_argument('-n_epochs',type=int,default=12,help='number of passes through training set')
    parser.add_argument('-early_stopping',type=bool,default=True,help='Boolean for early stopping with validation data')
    ## -
    parser.add_argument('-validation_freq',type=int,default=10,help='frequency with which we calculate the validation cost (which seems costly to compute); should increase with n_characters')
    parser.add_argument('-n_vcost',type=int,default=10,help='timescale over which to average the validation cost, as it decreases')

    #o ideally, when used in oneshot(), read this off of path_save string:
    parser.add_argument('-size',type=int,default=4,help='overall size of convnet')

    # One-shot testing parameters
    parser.add_argument('-path_save',type=str,default='n100size4_dropout7/cnn',help='path from which to load saved model')
    # Parameters for few-shot learning:
    parser.add_argument('-learn_rate_fewshot',type=float,default=0e-4,help='learning rate for few-shot learning')
    parser.add_argument('-learn_steps_fewshot',type=int,default=10,help='number of gradient steps per pair of matching images')

    # Parse the arguments
    args = parser.parse_args()

    if args.command == 'train':
        train(args)
    
    if args.command == 'oneshot':
        oneshot(args)


def cnn(x,args):
    "convolutional neural network"

    n_pixel = args.n_pixel  ## fix this to refer to main()
    # note that n_class only affects the logits layer
    n_class = args.n_char  ## fix this to refer to main(), to n_characters

    beta_cnn = 1.0 # this is redundant with beta in main() and so is set to 1.0 currently

    dropout_rate = args.dropout_rate

    size = args.size  # Lake et. al. CNN size = 10

    n_kernel = 5  # should be 5 for 28x28
    n_kernel_2 = 5
    n_features_1 = 12*size
    n_features_2 = 30*size
    n_pool = 5  # was 5
    n_stride = 5  # was 5
    n_pool_2 = 2
    n_stride_2 = 2
    n_dense = 300*size

    # doesn't have to be an integer?
    n_dense_input = (n_pixel // n_stride) // n_stride_2  # divide by n_stride not n_pool right?
    n_dense_input = n_features_2 * n_dense_input * n_dense_input

    reg = tf.contrib.layers.l2_regularizer(scale=beta_cnn, scope=None)

    # downsample first:
    pool1 = tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[n_pool, n_pool], strides=n_stride)

    # default strides=(1,1)
    conv1 = tf.layers.conv2d(
        inputs=pool1,
        filters=n_features_1,
        kernel_size=[n_kernel, n_kernel],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=reg)

    conv2 = tf.layers.conv2d(
        inputs=conv1,
        filters=n_features_2,
        kernel_size=[n_kernel_2, n_kernel_2],
        padding="same",
        activation=tf.nn.relu,
        kernel_regularizer=reg)

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[n_pool_2, n_pool_2], strides=n_stride_2)

    pool2_flat = tf.reshape(pool2, [-1, n_dense_input])

    dense = tf.layers.dense(
        inputs=pool2_flat,
        units=n_dense,
        activation=tf.nn.relu,
        kernel_regularizer=reg)

    dropout = tf.layers.dropout(inputs=dense, rate=dropout_rate)
    # omitting (define mode): training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=n_class)

    return logits, dense
    # Generate predictions (for PREDICT and EVAL mode)
    # Configure the Training Op (for TRAIN mode)
    # evaluation metrics ...


# a function for few-shot learning of the final layer of cnn
def cnn_logit(features, n_class):

    logits = tf.layers.dense(inputs=features, units=n_class)
    return logits

# quantifies overlap between feature representation vectors
def score(f, f2):

    cosine_sq = np.tensordot(f,f2) * np.tensordot(f,f2) / np.tensordot(f,f) / np.tensordot(f2,f2)
    return cosine_sq
    #o be sure this is operating correctly on the scalar elements of the np.tensordot() arrays; at least it outputs a scalar

    # ALTERNATE:
    # if f and f2 are logits, we can return a score of 1 if they prefer the same label:
#    if np.argmax(f)==np.argmax(f2): return 1
#    else: return 0


def image_array(filename):
    # load png image from filename and return numpy array for centered image

    pixels = imread(filename,flatten=True)  # is there a faster way?
    pixels = np.array(pixels,dtype=bool)
    pixels = np.logical_not(pixels)

    (n_pixel, _) = np.shape(pixels)  # this works assuming square images

    [center_x, center_y] = center_of_mass(pixels)
    center_x = int(n_pixel/2 - center_x)  #o might want to check whether int() and // round in same way, or are 1 pixel off
    center_y = int(n_pixel/2 - center_y)

    pixels = np.roll(pixels, (center_x, center_y), axis=(0,1))  # ** not axis=(1,0) ?

    pixels = np.reshape(pixels, (n_pixel, n_pixel, 1))
    return pixels.astype(int)


def one_hot_label_training(filename, n_class):
    # returns one-hot label from training data filename for image of character, assuming n_class different classes

    str_label = filename[-11:-7]
    target = int(str_label)  # starts at 1, since background image filenames start with e.g. 0001_01.png
    return np.eye(n_class)[target-1]

def one_hot_label_test(filename, n_class):
    # returns one-hot label from test data filename for image of character, assuming n_class different classes

    str_label = filename[9:11]
    target = int(str_label)  # starts at 1, since filenames start with e.g. 0001_01.png
    return np.eye(n_class)[target-1]


# taken from demo_classification.py
# input should be convert_to_inked()
def ModHausdorffDistance(itemA,itemB):
	# Modified Hausdorff Distance
	#
	# Input
	#  itemA : [n x 2] coordinates of "inked" pixels
	#  itemB : [m x 2] coordinates of "inked" pixels
	#
	#  M.-P. Dubuisson, A. K. Jain (1994). A modified hausdorff distance for object matching.
	#  International Conference on Pattern Recognition, pp. 566-568.
	#
	D = cdist(itemA,itemB)
	mindist_A = D.min(axis=1)
	mindist_B = D.min(axis=0)
	mean_A = np.mean(mindist_A)
	mean_B = np.mean(mindist_B)
	return max(mean_A,mean_B)

# taken from demo_classification.py
def convert_to_inked(filename):
	# Load image file and return coordinates of 'inked' pixels in the binary image
	# Output:
	#  D : [n x 2] rows are coordinates

        pixels = imread(filename,flatten=True)  # is there a faster way?
        pixels = np.array(pixels,dtype=bool)
        I = np.logical_not(pixels)

	(row,col) = I.nonzero()
	D = np.array([row,col])
	D = np.transpose(D)
	D = D.astype(float)
	n = D.shape[0]
	mean = np.mean(D,axis=0)
	for i in range(n):
		D[i,:] = D[i,:] - mean
	return D

def center_of_mass(I):
        # modified from convert_to_inked()
	# Load image file and return coordinates of center-of-mass pixel
	# Input should come from np.logical_not(pixels)

	(row,col) = I.nonzero()
	D = np.array([row,col])
	D = np.transpose(D)
	D = D.astype(float)
	n = D.shape[0]
	return np.mean(D,axis=0)

def train(args):

    n_characters = args.n_char
    images_per_character = args.images_per_char
    train_size = n_characters * images_per_character

    # import data
    train_dir = '/Users/elliot/Python/omniglot/python/images_bg/'
    filenames = os.listdir(train_dir)

    # choose the first train_size training images and train over a subset:
    filenames = filenames[0:train_size]

    # OPTIMIZATION PARAMETERS
    learn_rate = args.learning_rate
    optimizer = tf.train.AdamOptimizer(learn_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learn_rate)
    beta = args.beta
    validation_freq = args.validation_freq
    n_vcost = args.n_vcost
    vcost_ave_prev = pow(10,10)  #o a hack to initialize at ~infinity, for now; this ensures code below runs correctly

    n_pixel = args.n_pixel

    bsize = args.batch_size
    n_epochs = args.n_epochs
    n_batches = train_size // bsize
    n_batch_iterations = n_batches * n_epochs
    early_stopping = args.early_stopping
    use_stopping_ratio = False

    x = tf.placeholder(tf.float32, [None, n_pixel, n_pixel, 1])

    y_ = tf.placeholder(tf.float32, [None, n_characters])

    y_conv = cnn(x,args)

    with tf.name_scope('loss'):
        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=y_,
            logits=y_conv[0])  #o recall y_conv = (logits, dense)

    loss = tf.reduce_mean(loss)
    loss_0 = loss  # loss_0 is just the unregulated cross-entropy; use this for validation
    reg_loss = tf.losses.get_regularization_loss(scope=None, name='total_regularization_loss')
    # this is including kernel_regularization arguments in cnn layers because reg_loss varies with the scale in regularizer in cnn()
    loss += beta * reg_loss
#   PROBLEM "No weights to regularize" ... I think weights have not been added to GraphKeys.WEIGHTS collection ... cf. RBM_CDL_copy directory
#    l2_regularizer = tf.contrib.layers.l2_regularizer(scale=1.0, scope=None)
#    loss += beta * tf.contrib.layers.apply_regularization(l2_regularizer)

    with tf.name_scope('optimizer'):
        train_step = optimizer.minimize(loss)

    # VALIDATION
    validate_dir = '/Users/elliot/Python/omniglot/python/images_validate/'
    filenames_v = os.listdir(validate_dir)
    validation_size = n_characters * 2
    filenames_v = filenames_v[0:validation_size]

    x_v = np.zeros([validation_size,n_pixel,n_pixel,1])
    y_v = np.zeros([validation_size,n_characters])

    for k in range(validation_size):
        filename = validate_dir + filenames_v[k]
        x_v[k] = image_array(filename)
        y_v[k] = one_hot_label_training(filename, n_characters)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    bcount = 0
    i_epochs = 0
    mode = 'w'  # this ensures that we write data to an empty file
    vcost_list = []  # store validation cost over time
    for i in range(n_batch_iterations):

        if (i * bsize) % train_size == 0:  # checked that this is working correctly
            bcount = 0
            shuffle(filenames)
            print('%d epochs completed.' % i_epochs)
            i_epochs += 1

        # confirmed: this divides up filenames into lists for each batch
        batch_filenames=filenames[bcount*bsize: bcount*bsize + bsize]

        batch_x = np.zeros([bsize,n_pixel,n_pixel,1])  # extra "1" argument needed b/c 1 black/white channel? (tf.layers expect this?)
        batch_y = np.zeros([bsize,n_characters])

        # prepare (x,y)'s for the current batch
        for j in range(bsize):
            filename = '/Users/elliot/Python/omniglot/python/images_bg/' + batch_filenames[j]
            batch_x[j] = image_array(filename)
            batch_y[j] = one_hot_label_training(filename, n_characters)

            ### OPTIONAL: check whether correct labels are being learned:
            # logits_x = sess.run(y_conv, feed_dict={x: [batch_x[j]]})
            # if np.argmax(logits_x)==np.argmax(one_hot_label): print('Correct label')
            # else: print('Wrong label')

        sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
        
        if i % validation_freq == 0:
            vcost = sess.run(loss_0, feed_dict={x: x_v, y_: y_v})
            vcost_list.append(vcost)
            print('vcost = ')
            print(vcost)
            file = open('vcost.txt',mode) 
            file.write('{' + str(i) + ',' + str(vcost) + '},')
            file.close()
            if (i / validation_freq) > n_vcost:  # only do the following if we have enough vcost datapoints to average over
                n = len(vcost_list)
                n_min = n - n_vcost  # use this below to truncate to the last n_vcost elements
                vcost_ave = sum(vcost_list[n_min:n]) / n_vcost
                #o currently this does NOT any previous contents of the file (b/c of mode = 'a' below)
                file = open('vcost_ave.txt',mode)
                file.write('{' + str(i) + ',' + str(vcost_ave) + '},')
                file.close()
                # EARLY STOPPING:
                if early_stopping==True and vcost_ave > vcost_ave_prev:
                    print('EARLY STOPPING NOW')
                    break
                vcost_ave_prev = vcost_ave  # on next loop, compare the new vcost_ave to this quantity
            # next line (REDUNDANT) ensures that we don't erase data we've already written to file
            mode = 'a'

        # store the initial validation cost
        if i==0 and bcount==0: initial_vcost = vcost
        # naive stopping criterion with the current minibatch training cost:
        if use_stopping_ratio==True:
            if vcost/initial_vcost<stopping_ratio: break

        bcount += 1
        print('Batch %d' % bcount)

    # ** can replace 'cnn' string with parameter-dependent string...
    saver = tf.train.Saver()
    save_path = saver.save(sess, 'cnn')
    print("Model saved in path: %s" % save_path)


def oneshot(args):

    eval_dir = '/Users/elliot/Python/omniglot/python/images_evaluation/'
    alphabet_list = os.listdir(eval_dir)

    n_alphabets = 20
    n_questions = 50  # number of questions per alphabet; note error ~? 1/sqrt(N) where N=n_alphabets*n_questions
    n_choices = 20

    learn_rate = args.learn_rate_fewshot
    oneshot_steps = args.learn_steps_fewshot
    optimizer = tf.train.AdamOptimizer(learn_rate)

    #o these are currently just copied from train()
    n_pixel = args.n_pixel
    size = args.size
    n_dense = 300*size

    x = tf.placeholder(tf.float32, [None, n_pixel, n_pixel, 1])
    y_conv = cnn(x,args)  #o change to cnn_out ?

    # IF NEW SESSION OR INITIALIZATION, BE SURE TRAINED WEIGHTS ARE UNCHANGED
    #o is this redundant with above?:
    # sess = tf.Session()
    #o just global vars needed?
    # sess.run(tf.global_variables_initializer())

    sess = tf.Session()

    # load the cnn parameters from saved file
    saver = tf.train.Saver()
    path_save = args.path_save
    saver.restore(sess, path_save)
    print("Model loaded from files: " + path_save)

    # OPTIONAL - TEST IMAGE, y(x) as proxy for weight values
    # image_ones = np.ones([1,n_pixel,n_pixel,1])  # extra "1" argument needed b/c 1 black/white channel? (tf.layers expect this?)
    # print('a black image yields logits:')
    # print(sess.run(y_conv, feed_dict={x: image_ones}))
    # OR, use actual DATA as a TEST IMAGE:
    # filename_test = '/Users/elliot/Python/omniglot/python/images_bg/0001_01.png'
    # image_test = image_array(filename_test)
    # label_test = one_hot_label_training(filename_test, n_characters)
    # logit_test = sess.run(y_conv[0], feed_dict={x: [image_test]})
    # if np.argmax(logit_test)==np.argmax(label_test): print("Test image CORRECTLY labelled")
    # else: print("Test image FAIL")

    points = 0
    points_available = 0

    # confirmed: this correctly picks an image as filename_x,
    # with image of same character as last element of char_options_list, with file name = filename_choice when i=n_choices-1
    for i in range(n_alphabets):

        path_alphabet = eval_dir + alphabet_list[i] + '/'  ## previously: '/Users/elliot/Python/omniglot/python/images_background/Tifinagh/'
        print('Evaluating with images from:')
        print(path_alphabet)
        character_list = os.listdir(path_alphabet)
        n_char_eval = len(character_list)

        ## n_characters_eval = 10  ## number of characters in evaluation set
        ## if n_characters_eval<len(character_list):
        ## character_list = character_list[0:n_characters_eval] ## truncate the alphabet for now
        ## if n_choices>n_characters_eval:

#        scopename = 'one_shot' + '_' + str(i)

        # build the final layer graph for the current alphabet
#        with tf.variable_scope(scopename):  # initialization below worked with enough variables included under this scope
#            f_ = tf.placeholder(tf.float32, [None, 1, n_dense])
#            y = cnn_logit(f_, n_char_eval)
#            y_ = tf.placeholder(tf.float32, [None, n_char_eval])
            # LOSS to minimize for the current alphabet
            #o with tf.name_scope('loss'):
#            loss = tf.nn.softmax_cross_entropy_with_logits(
#                labels=y_,
#                logits=y)
#            train_step = optimizer.minimize(loss)  # train_step must be under scope referred to in vars_oneshot below
        # I don't think we need to specify var_list=vars_oneshot, since as long as we input features f_ we're only training the final layer anyways

        # restrict initialization to just the final layer, cnn_logit()
#        vars_oneshot = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scopename)
#        sess.run(tf.variables_initializer(vars_oneshot))

        # to confirm that y_conv dense layer output from disk is unchanged, confirming that weights from disk are being used
        # print('a black image yields logits:')
        # print(sess.run(y_conv, feed_dict={x: image_ones}))

        # ask a number of questions with each chosen alphabet
        for nq in range(n_questions):

            # choose n_choices random characters:
            char_choice_list = np.random.choice(character_list, n_choices, replace=False)
    
            # choose the character under question to be the LAST in the list:
            char = char_choice_list[n_choices-1]
            path_character = path_alphabet + char + '/'
            image_list = os.listdir(path_character)

            # choose 2 images for this character
            pair = np.random.choice(image_list, 2, replace=False)

            # one of them will be the input x
            filename_x = path_character + pair[0]
            ### inked_x = convert_to_inked(filename_x)
            image_x = image_array(filename_x)

            #o turn this (and similar code below) into a function analogous to one_hot_label_training():
            ## TEMPORARY: CHECK LABELLING PERFORMANCE; this worked fine for a small training set of 10 characters (and small network)
            # note that now we label within the current alphabet
            y_x = one_hot_label_test(char, n_char_eval) ## we truncated character_list above to 1st n_characters for evaluation
            # logits_x = sess.run(y_conv, feed_dict={x: [image_x]})  # if in EVAL section: x:[image_x]
            # (confirmed): this correctly checks whether highest logit and one-hot-label are in the same slot
            # if np.argmax(logits_x)==np.argmax(y_x): print('Correct label EVAL')
            # else: print('Wrong label EVAL')

            char_options_list = []

            # the rest of the chosen characters contribute "wrong answers" to char_options_list
            for i in range(n_choices-1):
                other_char = char_choice_list[i]
                path_other_char = path_alphabet + other_char + '/'
                image_list = os.listdir(path_other_char)
                char_options_list.append(np.random.choice(image_list))

            # correct match will be the LAST in char_options_list
            char_options_list.append(pair[1])

            match_choices = []
            ### inked_choices = []
            
            #o combine this with the loop above?
            # convert possible match images to np arrays, and compile in list
            for i in range(n_choices):
                filename_choice = path_alphabet + char_choice_list[i] + '/' + char_options_list[i]
                # confirmed that when i=n_choices-1, filename_choice is the correct match
                ### inked_choices.append(convert_to_inked(filename_choice))
                image_choice = image_array(filename_choice)
                match_choices.append(image_choice)

            # label for the matching image
            char_match =  char_choice_list[n_choices-1]
            y_match = one_hot_label_test(char_match, n_char_eval) ## we truncated character_list above to 1st n_characters for evaluation

            _,f = sess.run(y_conv, feed_dict={x: [image_x]})  # confirmed: object type of y & f = np.ndarray
            #o this and line below unnecessarily attempt to classify into training characters

            ### OPTIONAL: check whether image_x is labelled correctly
            # if np.argmax(y)==np.argmax(label_x): print("Correct label...")
            # else: print("Wrong label...")

            scores = []
            ### inked_scores = []
            for i in range(n_choices):  # i ranges up to n_choices-1
                _,f2 = sess.run(y_conv, feed_dict={x: [match_choices[i]]})  #o fix up the match_choices shape...
                ### inked_scores.append(ModHausdorffDistance(inked_x,inked_choices[i]))
                scores.append(score(f,f2))  # confirmed: for np.tensordot(), this is dotting f and f2 as expected
            scores = scores / sum(scores)
            print(scores)
            guess = np.argmax(scores)  # confirmed: this correctly picks the index of the highest element in scores
            ### guess = np.argmin(inked_scores)
            points_available +=1
            if guess==n_choices-1:
                points += 1  # correct guess is the LAST element in char_options_list, pair[1] above
                print('Correct!')
            else: print('Wrong')
            error_rate = 1-points/points_available

            print('%d questions asked so far' % nq)
            print('cumulative ERROR RATE for ALL alphabets = %f' % error_rate)

            # ** cost_eval = sess.run(loss, feed_dict= ... )
            
            # y_x_predict = sess.run(y, feed_dict={f_: f})
            # y_match_predict = sess.run(y, feed_dict={f_: f2})  # note f2 should now be stored for the matching image
            

            # FEW-SHOT LEARNING after EACH QUESTION:
            # train logit layer on the pair of matching images

#            for _ in range(oneshot_steps):
#                sess.run(train_step, feed_dict={f_: [f,f2], y_: [y_x,y_match]})

            # WHAT MAKES SENSE? WHAT TO EXPECT? IS THIS THE RIGHT TEST??
            # ***** I would like a loop for each character, iterating over M match pairs and learning each time


            # OPTIONAL: print filenames of relevant images
            # print('Image:')
            # print(filename_x)
            # print('Correct Match:')
            # print(path_alphabet + char_choice_list[n_choices-1] + '/' + char_options_list[n_choices-1])
            # print('Guess for Match:')
            # print(path_alphabet + char_choice_list[guess] + '/' + char_options_list[guess])


if __name__ == "__main__":
    main()
    # tf.app.run()


# SCRAP CODE:
# returns scores of the candidate match images
#def scores(x_, x_list):

#    n_choices = len(x_list)
#    print('n_choices = %d' % n_choices)

    # make a list? not tensor?
#    scores = []

#    for i in range(n_choices):
#        score = tf.tensordot(cnn(x_,eval=True), cnn(x_list[i],eval=True), 2)  # 3rd argument = axes = 2?
#        scores.append(score)

#    return scores

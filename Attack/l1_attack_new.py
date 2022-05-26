## l2_attack.py -- attack a network optimizing for l_2 distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

BINARY_SEARCH_STEPS = 10  # number of times to adjust the constant with binary search
MAX_ITERATIONS = 1000  # number of iterations to perform gradient descent
ABORT_EARLY = True       # if we stop improving, abort gradient descent early
LEARNING_RATE = 0.0001   # larger values converge faster to less accurate results
TARGETED = True          # should we target one specific class? or just be wrong?
CONFIDENCE = 0           # how strong the adversarial example should be
INITIAL_CONST = 1  # the initial constant c to pick as a first guess

class EADL1:
    def __init__(self, sess, model, batch_size=1, confidence = CONFIDENCE,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 binary_search_steps = BINARY_SEARCH_STEPS, max_iterations = MAX_ITERATIONS,
                 abort_early = ABORT_EARLY, 
                 initial_const = INITIAL_CONST,
                 boxmin = -0.5, boxmax = 0.5):
        """
        The L_2 optimized attack. 

        This attack is the most efficient and should be used as the primary 
        attack to evaluate potential defenses.

        Returns adversarial examples for the supplied model.

        confidence: Confidence of adversarial examples: higher produces examples
          that are farther away, but more strongly classified as adversarial.
        batch_size: Number of attacks to run simultaneously.
        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        binary_search_steps: The number of times we perform binary search to
          find the optimal tradeoff-constant between distance and confidence. 
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. If binary_search_steps is large,
          the initial constant is not important.
        boxmin: Minimum pixel value (default -0.5).
        boxmax: Maximum pixel value (default 0.5).
        """

        image_size, num_channels, num_labels = 512,3, 40
        self.sess = sess
        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.BINARY_SEARCH_STEPS = binary_search_steps
        self.ABORT_EARLY = abort_early
        self.CONFIDENCE = confidence
        self.initial_const = initial_const
        self.batch_size = batch_size
        self.model = model
        self.repeat = binary_search_steps >= 10

        self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = False

        shape = (1,image_size,num_channels)
        
        # the variable we're going to optimize over
        self.modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        # these are variables to be more efficient in sending data to tf
        self.timg = tf.Variable(np.zeros(shape), dtype=tf.float32)
        self.tlab = tf.Variable(np.zeros((batch_size,num_labels)), dtype=tf.float32)
        self.const = tf.Variable(np.zeros(batch_size), dtype=tf.float32)

        # and here's what we use to assign them
        self.assign_timg = tf.placeholder(tf.float32, shape)
        self.assign_tlab = tf.placeholder(tf.float32, (batch_size,num_labels))
        self.assign_const = tf.placeholder(tf.float32, [batch_size])
        
       
        #self.newimg = tf.tanh(modifier + self.timg) * self.boxmul + self.boxplus
        self.newimg = self.modifier + self.timg 
        # prediction BEFORE-SOFTMAX of the model
        self.output = model(self.newimg)
        
        # distance to the input data
        self.l2dist = tf.reduce_sum(tf.square(self.newimg-self.timg ),[1,2])
        self.l1dist = tf.reduce_sum(tf.abs(self.newimg-self.timg ),[1,2])
        # compute the probability of the label class versus the maximum other
        real = tf.reduce_sum((self.tlab)*self.output,1)
        other = tf.reduce_max((1-self.tlab)*self.output - (self.tlab),1)

        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0, other-real)
        else:
            # if untargeted, optimize for making this class least likely.
            #loss1 = tf.maximum(0.0, real-other+self.CONFIDENCE)
            loss1 = tf.maximum(0.0, real)

        # sum up the losses
        self.loss2 = tf.reduce_sum(self.l1dist)
        #self.loss2 = tf.reduce_sum(self.l2dist)
        self.loss1 = tf.reduce_sum(self.const*loss1)
        self.loss = self.loss1+self.loss2
        
        # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        self.train = optimizer.minimize(self.loss, var_list=[self.modifier])
        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        # these are the variables to initialize when we run
        self.setup = []
        self.setup.append(self.timg.assign(self.assign_timg))
        self.setup.append(self.tlab.assign(self.assign_tlab))
        self.setup.append(self.const.assign(self.assign_const))
        
        self.init = tf.variables_initializer(var_list=[self.modifier]+new_vars)

    def attack(self, imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        dist = []
        print('go up to',len(imgs))
        for i in range(0,len(imgs),self.batch_size):
            #if i <1:
            m,d = self.attack_batch(imgs[i:i+self.batch_size], targets[i:i+self.batch_size])
            if np.isnan(m).sum() >0:
                pass
            else:
                print('tick',i)
                r.extend(m)
                dist.extend(d)
        return dist

    def attack_batch(self, imgs, labs):
        f = open('./cw9layer-1.txt', "a+")
        print('start here #############',file = f)
        """
        Run the attack on a batch of images and labels.
        """
        def compare(x,y):
            if not isinstance(x, (float, int, np.int64)):
                x = np.copy(x)
                if self.TARGETED:
                    x[y] -= self.CONFIDENCE
                else:
                    x[y] += self.CONFIDENCE
                x = np.argmax(x)
            if self.TARGETED:
                return x == y
            else:
                return x != y

        batch_size = self.batch_size

        # convert to tanh-space
        #imgs = (imgs - self.boxplus) / self.boxmul 

        # set the lower and upper bounds accordingly
        lower_bound = np.zeros(batch_size)
        CONST = np.ones(batch_size)*self.initial_const
        upper_bound = np.ones(batch_size)*1e10

        # the best l2, score, and image attack
        o_bestl1 = [100]*batch_size
        o_bestscore = [-1]*batch_size
        o_bestattack = [np.full(imgs[0].shape, np.nan)]*batch_size
        
        for outer_step in range(self.BINARY_SEARCH_STEPS):
            print(o_bestl1)
            # completely reset adam's internal state.
            self.sess.run(self.init)
            batch = imgs[:batch_size]
            batchlab = labs[:batch_size]
    
            bestl1 = [100]*batch_size
            bestscore = [-1]*batch_size

            # The last iteration (if we run many steps) repeat the search once.
            if self.repeat == True and outer_step == self.BINARY_SEARCH_STEPS-1:
                CONST = upper_bound
            self.sess.run(tf.assign(self.modifier,tf.truncated_normal([1,512,3], mean=0, stddev=0.0000001)))
            # set the variables so that we don't have to send them over again
            self.sess.run(self.setup, {self.assign_timg: batch,
                                       self.assign_tlab: batchlab,
                                       self.assign_const: CONST})
            
            prev = 1e6
            for iteration in range(self.MAX_ITERATIONS):
                # perform the attack 
                _, l, l1s, scores, nimg,oimg = self.sess.run([self.train, self.loss, 
                                                         self.l1dist, self.output, 
                                                         self.newimg,self.timg])

                if np.all(scores>=-.0001) and np.all(scores <= 1.0001):
                    if np.allclose(np.sum(scores,axis=1), 1.0, atol=1e-3):
                        if not self.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK:
                            raise Exception("The output of model.predict should return the pre-softmax layer. It looks like you are returning the probability vector (post-softmax). If you are sure you want to do that, set attack.I_KNOW_WHAT_I_AM_DOING_AND_WANT_TO_OVERRIDE_THE_PRESOFTMAX_CHECK = True")
                
                # print out the losses every 10%
                if iteration%(self.MAX_ITERATIONS//10) == 0:
                    print(iteration,self.sess.run((self.loss,self.loss1,self.loss2)))
                '''
                # check if we should abort search if we're getting nowhere.
                if self.ABORT_EARLY and iteration%(self.MAX_ITERATIONS//10) == 0:
                    if l > prev*.9999:
                        break
                    prev = l
                '''
                

                # adjust the best result found so far
                for e,(l1,sc,ii) in enumerate(zip(l1s,scores,nimg)):
                    if l1 < bestl1[e] and compare(sc, np.argmax(batchlab[e])):
                        bestl1[e] = l1
                        bestscore[e] = np.argmax(sc)
                    if l1 < o_bestl1[e] and compare(sc, np.argmax(batchlab[e])):
                        o_bestl1[e] = l1
                        o_bestscore[e] = np.argmax(sc)
                        o_bestattack[e] = ii

            # adjust the constant as needed
            for e in range(batch_size):
                if compare(bestscore[e], np.argmax(batchlab[e])) and bestscore[e] != -1:
                    # success, divide const by two
                    upper_bound[e] = min(upper_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                else:
                    # failure, either multiply by 10 if no solution found yet
                    #          or do binary search with the known upper bound
                    lower_bound[e] = max(lower_bound[e],CONST[e])
                    if upper_bound[e] < 1e9:
                        CONST[e] = (lower_bound[e] + upper_bound[e])/2
                    else:
                        CONST[e] *= 5

        # return the best solution found
        
        o_bestl1 = np.array(o_bestl1)
        print('o_bestl1 is',o_bestl1)
        print('o_bestl1 is',o_bestl1,file =f )
        return oimg-o_bestattack,o_bestl1
def pgd_attack(self,clean_images,tar):
        #max_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            #reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(clean_images))
        # max_X contains adversarial examples and is updated after each restart
        #max_X = clean_images
        loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Start restart loop
        X = tf.convert_to_tensor(np.expand_dims(clean_images,axis=0),dtype=tf.float32)
        random_delta = 2 * 0.0001 * tf.random.uniform(shape=X.shape) - 0.001
        X = X + random_delta
        mini = 100
        for i in range(10):
            # Get random perturbation uniformly in l infinity epsilon ball
            target = tf.convert_to_tensor(np.expand_dims(tar,axis=0),dtype=tf.float32)
            # Start projective gradient descent from X
            for j in range(1000):
                # Track gradients
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    # Only gradients w.r.t. X are taken NOT model parameters
                    tape.watch(X)
                    pred = self.model(X)
                    loss = -tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target)
                    #loss = -loss_fun(target[0],pred[0])

                # Get gradients of loss w.r.t X
                gradients = tf.gradients(loss, X)
                # Compute perturbation as step size times sign of gradients
                perturbation = 0.001 * tf.sign(gradients)
                # Update X by adding perturbation
                X = X + perturbation
                X = tf.reshape(X,(1,clean_images.shape[0],3))
                
                if tf.argmax(self.model(X)[0]).eval() == np.argmax(tar):
                    delta = tf.sqrt(tf.sum((X-clean_images)**2)).eval()
                    if delta < mini:
                        mini = delta
                        print(mini)
                
            if mini == 100:
                
                return None
            else:
                return mini

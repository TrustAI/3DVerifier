## li_attack.py -- attack a network optimizing for l_infinity distance
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import sys
import tensorflow as tf
import numpy as np

DECREASE_FACTOR = 0.9   # 0<f<1, rate at which we shrink tau; larger is more accurate
MAX_ITERATIONS = 1000   # number of iterations to perform gradient descent
ABORT_EARLY = True      # abort gradient descent upon first valid solution
INITIAL_CONST = 1e-4    # the first value of c to start at
LEARNING_RATE = 5e-3    # larger values converge faster to less accurate results
LARGEST_CONST = 20    # the largest value of c to go up to before giving up
REDUCE_CONST = True    # try to lower c each iteration; faster to set to false
TARGETED = True         # should we target one specific class? or just be wrong?
CONST_FACTOR = 2.0      # f>1, rate at which we increase constant, smaller better
NUM_POINTS = 512
NUM_CLASS = 40
class CarliniLi:
    def __init__(self, sess, model,
                 targeted = TARGETED, learning_rate = LEARNING_RATE,
                 max_iterations = MAX_ITERATIONS, abort_early = ABORT_EARLY,
                 initial_const = INITIAL_CONST, largest_const = LARGEST_CONST,
                 reduce_const = REDUCE_CONST, decrease_factor = DECREASE_FACTOR,
                 const_factor = CONST_FACTOR):
        """
        The L_infinity optimized attack. 

        Returns adversarial examples for the supplied model.

        targeted: True if we should perform a targetted attack, False otherwise.
        learning_rate: The learning rate for the attack algorithm. Smaller values
          produce better results but are slower to converge.
        max_iterations: The maximum number of iterations. Larger values are more
          accurate; setting too small will require a large learning rate and will
          produce poor results.
        abort_early: If true, allows early aborts if gradient descent gets stuck.
        initial_const: The initial tradeoff-constant to use to tune the relative
          importance of distance and confidence. Should be set to a very small
          value (but positive).
        largest_const: The largest constant to use until we report failure. Should
          be set to a very large value.
        reduce_const: If true, after each successful attack, make const smaller.
        decrease_factor: Rate at which we should decrease tau, less than one.
          Larger produces better quality results.
        const_factor: The rate at which we should increase the constant, when the
          previous constant failed. Should be greater than one, smaller is better.
        """
        self.model = model
        self.sess = sess

        self.TARGETED = targeted
        self.LEARNING_RATE = learning_rate
        self.MAX_ITERATIONS = max_iterations
        self.ABORT_EARLY = abort_early
        self.INITIAL_CONST = initial_const
        self.LARGEST_CONST = largest_const
        self.DECREASE_FACTOR = decrease_factor
        self.REDUCE_CONST = reduce_const
        self.const_factor = const_factor
        

        

    def gradient_descent(self, sess, model,oimgs, labs, starts, tt, CONST):
        def compare(x,y):
            if self.TARGETED:
                return x == y
            else:
                return x != y
        shape = (1,NUM_POINTS ,3)
    
        # the variable to optimize over
        modifier = tf.Variable(np.zeros(shape,dtype=np.float32))

        tau = tf.placeholder(tf.float32, [])
        simg = tf.placeholder(tf.float32, shape)
        timg = tf.placeholder(tf.float32, shape)
        tlab = tf.placeholder(tf.float32, (1,NUM_CLASS ))
        const = tf.placeholder(tf.float32, [])
        
        newimg = (modifier + simg)
        
        output = model(newimg)
        orig_output = model(timg)
    
        real = tf.reduce_sum((tlab)*output)
        other = tf.reduce_max((1-tlab)*output - (tlab*10000))
    
        if self.TARGETED:
            # if targetted, optimize for making the other class most likely
            loss1 = tf.maximum(0.0,other-real)
        else:
            # if untargeted, optimize for making this class least likely.
            loss1 = tf.maximum(0.0,real-other)

        # sum up the losses
        loss2 = tf.reduce_sum(tf.maximum(0.0,tf.abs(newimg-timg)-tau))
        loss = const*loss1+loss2
    
        # setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        train = optimizer.minimize(loss, var_list=[modifier])

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        init = tf.variables_initializer(var_list=[modifier]+new_vars)
    
        #def doit(oimgs, labs, starts, tt, CONST):
            # convert to tanh-space
        imgs = np.array(oimgs)
        starts = np.array(starts)
    
        # initialize the variables
        sess.run(init)
        while CONST < self.LARGEST_CONST:
            # try solving for each value of the constant
            print('try const', CONST)
            for step in range(self.MAX_ITERATIONS):
                    feed_dict={timg: imgs, 
                               tlab:labs, 
                               tau: tt,
                               simg: starts,
                               const: CONST}
                    if step%(self.MAX_ITERATIONS//10) == 0:
                        print(step,sess.run((loss,loss1,loss2),feed_dict=feed_dict))

                    # perform the update step
                    _, works = sess.run([train, loss], feed_dict=feed_dict)
    
                    # it worked
                    if works < .00001*CONST and (self.ABORT_EARLY or step == CONST-1):
                        get = sess.run(output, feed_dict=feed_dict)
                        works = compare(np.argmax(get), np.argmax(labs))
                        if works:
                            scores, origscores, nimg = sess.run((output,orig_output,newimg),feed_dict=feed_dict)
                            

                            
                            return scores, origscores, nimg, CONST

                # we didn't succeed, increase constant and try again
            CONST *= self.const_factor
        return None
    
    def attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        i= 0
        for img,target in zip(imgs, targets):
            #if i <1: 
            out_tau = self.attack_single(img, target)
            if out_tau==100:
                pass
            else:
                r.append(out_tau)
            i+=1
        return np.array(r)
    def pgd_attack(self, imgs, targets):
        """
        Perform the L_0 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        i= 0
        for img,target in zip(imgs, targets):
            out_tau = self.PGDattack(img, target)
            if out_tau==None:
                pass
            else:
                r.append(out_tau)
            i+=1
        return np.array(r)

    def attack_single(self, img, target):
        """
        Run the attack on a single image and label
        """
        f = open('./cw20layer-i.txt', "a+")
        print('start here !!',file = f)
        # the previous image
        prev = np.copy(img).reshape((1,NUM_POINTS,3))
        tau = 1.0
        mintau = 100
        const = self.INITIAL_CONST
        m =0
        for i in range(10):
            # try to solve given this tau value
            print('before Tau',tau)
            failed = False

            res = self.gradient_descent(self.sess,self.model,[np.copy(img)], [target], np.copy(prev), tau, const)
            #last = res
            if res == None:
                # the attack failed, we return this as our final answer
                return tau
            else:
                _, _, nimg, const = res
                if self.REDUCE_CONST: const /= 2

                # the attack succeeded, reduce tau and try again
    
                actualtau = np.max(np.abs(nimg-img))
                #actualtau = np.sum(np.abs(nimg-img))
                #actualtau = np.sum((nimg-img)**2)
                if actualtau < mintau:
                    mintau = actualtau
    
                prev = nimg
                tau *= 0.9
            
            
        print('tau is',mintau,file = f)
        return mintau
    def pgd_attack(self,clean_images,tar):
        #max_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            #reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(clean_images))
        # max_X contains adversarial examples and is updated after each restart
        #max_X = clean_images
        f = open('./cw20layer-i-pgr.txt', "a+")
        loss_fun = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # Start restart loop
        X = tf.convert_to_tensor(np.expand_dims(clean_images,axis=0),dtype=tf.float32)
        mini = 100
        for i in range(10):
            # Get random perturbation uniformly in l infinity epsilon ball
            random_delta = 2 * 0.0001 * tf.random.uniform(shape=X.shape) - 0.001
            X = X + random_delta

            target = tf.convert_to_tensor(np.expand_dims(tar,axis=0),dtype=tf.float32)
            # Start projective gradient descent from X
            for j in range(1000):
                # Track gradients

                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    # Only gradients w.r.t. X are taken NOT model parameters
                    tape.watch(X)
                    pred = self.model(X)
                    #loss = -tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=target)
                    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
                    loss = -cce(target,pred)
                    #loss = -loss_fun(target[0],pred[0])

                # Get gradients of loss w.r.t X
                gradients = tf.gradients(loss, X)
                # Compute perturbation as step size times sign of gradients
                perturbation = 0.001 * tf.sign(gradients)
                # Update X by adding perturbation
                X = X + perturbation
                X = tf.reshape(X,(1,clean_images.shape[0],3))
                delta = tf.reduce_max(tf.abs(X-clean_images)).eval()
                print(delta)
                if delta == np.argmax(tar):
                    if delta < mini:
                        mini = delta
                        print(mini,file =f)
                
            if mini == 100:
                
                return None
            else:
                return mini
                # Make sure X did not leave L infinity epsilon ball around clean_images
            '''
            # Get crossentroby loss for each image in X
            loss_vector = tf.keras.losses.SparseCategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE)(true_labels, self.model(X))

            # mask is 1D tensor where true values are the rows of images that have higher loss than previous restarts
            mask = tf.greater(loss_vector, max_loss)
            # Update max_loss
            max_loss = tf.where(mask, loss_vector, max_loss)
            """
            we cannot do max_X[mask] = X[mask] like in numpy. We need mask that fits shape of max_X.
            Keep in mind that we want to select the rows that are True in the 1D tensor mask.
            We can simply stack the mask along the dimensions of max_X to select each desired row later.
            """
            # Create 2D mask of shape (max_X.shape[0],max_X.shape[1])
            multi_mask = tf.stack(max_X.shape[1] * [mask], axis=-1)
            # Create 3D mask of shape (max_X.shape[0],max_X.shape[1], max_X.shape[2])
            multi_mask = tf.stack(max_X.shape[2] * [multi_mask], axis=-1)
            # Create 4D mask of shape (max_X.shape[0],max_X.shape[1], max_X.shape[2], max_X.shape[3])
            multi_mask = tf.stack(max_X.shape[3] * [multi_mask], axis=-1)

            # Replace adversarial examples max_X[i] that have smaller loss than X[i] with X[i]
            max_X = tf.where(multi_mask, X, max_X)
            
            '''
            
       
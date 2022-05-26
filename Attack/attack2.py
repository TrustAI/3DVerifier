import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

#import tf_nndistance
#import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')

parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=64, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='perturbation', help='dump folder path [perturbation]')

parser.add_argument('--lr_attack', type=float, default=0.01, help='learning rate for optimization based attack')

parser.add_argument('--initial_weight', type=float, default=10, help='initial value for the parameter lambda')
parser.add_argument('--upper_bound_weight', type=float, default=80, help='upper_bound value for the parameter lambda')
parser.add_argument('--step', type=int, default=10, help='binary search step')
parser.add_argument('--num_iter', type=int, default=1000, help='number of iterations for each binary search step')

FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point

GPU_INDEX = FLAGS.gpu
#MODEL = importlib.import_module(FLAGS.model) # import network module
#if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

LR_ATTACK=FLAGS.lr_attack
#WEIGHT=FLAGS.weight

#attacked_data_all=joblib.load(os.path.join(DATA_DIR,'attacked_data.z'))

INITIAL_WEIGHT=FLAGS.initial_weight
UPPER_BOUND_WEIGHT=FLAGS.upper_bound_weight
#ABORT_EARLY=False
BINARY_SEARCH_STEP=FLAGS.step
NUM_ITERATIONS=FLAGS.num_iter

def attack_2(sess,model,imgs, targets):
            pointclouds_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NUM_POINT, 3))
            labels_pl = tf.placeholder(tf.float32, (BATCH_SIZE,40))
            is_training_pl = tf.placeholder(tf.bool, shape=())

            #pert=tf.get_variable(name='pert',shape=[BATCH_SIZE,NUM_POINT,3],initializer=tf.truncated_normal_initializer(stddev=0.01))
            pert=tf.get_variable(name='pert',shape=[BATCH_SIZE,NUM_POINT,3])

            pointclouds_input=pointclouds_pl+pert
            
            pred= model(pointclouds_input)
            real = tf.reduce_sum((labels_pl)*pred,1)
            other = tf.reduce_max((1-labels_pl)*pred - (labels_pl*10000),1)

            #if self.TARGETED:
                # if targetted, optimize for making the other class most likely
            adv_loss = tf.maximum(0.0, other-real)
            #else:
                # if untargeted, optimize for making this class least likely.
                #adv_loss = tf.maximum(0.0, real-other)
           
            
            #perturbation l2 constraint
            pert_norm=tf.sqrt(tf.reduce_sum(tf.square(pert),[1,2]))

            dist_weight=tf.placeholder(shape=[BATCH_SIZE],dtype=tf.float32)
            lr_attack=tf.placeholder(dtype=tf.float32)
            attack_optimizer = tf.train.AdamOptimizer(lr_attack)
            attack_op = attack_optimizer.minimize( adv_loss+ tf.reduce_mean(tf.multiply(dist_weight,pert_norm)),var_list=[pert])
            
            vl=tf.global_variables()
            vl=[x for x in vl if 'pert' not in x.name]
      
            sess.run(tf.global_variables_initializer())


            ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pointclouds_input':pointclouds_input,
               'dist_weight':dist_weight,
               'pert': pert,
               #'pre_max':end_points['pre_max'],
               #'post_max':end_points['post_max'],
               'pred': pred,
               'adv_loss': adv_loss,
               'pert_norm':pert_norm,
               #'total_loss':tf.reduce_mean(tf.multiply(dist_weight,pert_norm))+adv_loss,
               'lr_attack':lr_attack,
               'attack_op':attack_op
               }

   
            r = []
            dist = []
            for i in range(0,len(imgs),BATCH_SIZE):
              if i <1:
                print('tick',i)
                
                m ,d= attack_one_batch(sess,ops,imgs[i:i+BATCH_SIZE], targets[i:i+BATCH_SIZE],i)
                if np.isnan(m).sum() >0:
                    pass
                else:
                    r.extend(m)
                    dist.extend(d)
            return np.array(r),np.array(dist)
def attack_cw(imgs, targets):
        """
        Perform the L_2 attack on the given images for the given targets.

        If self.targeted is true, then the targets represents the target labels.
        If self.targeted is false, then targets are the original class labels.
        """
        r = []
        print('go up to',len(imgs))
        
def attack_one_batch(sess,ops,attacked_data,attacked_label,index):

    ###############################################################
    ### a simple implementation
    ### Attack all the data in variable 'attacked_data' into the same target class (specified by TARGET)
    ### binary search is used to find the near-optimal results
    ### part of the code is adpated from https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py
    ###############################################################

    is_training = False
    f = open('./cw-2.txt', "a+")
    print('index is',index,file = f)


    #the bound for the binary search
    lower_bound=np.zeros(BATCH_SIZE)
    WEIGHT = np.ones(BATCH_SIZE) * INITIAL_WEIGHT
    upper_bound=np.ones(BATCH_SIZE) * UPPER_BOUND_WEIGHT

   
    o_bestdist = [1e2] * BATCH_SIZE
    o_bestscore = [-1] * BATCH_SIZE
    o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_POINT,3))

    feed_dict = {ops['pointclouds_pl']: attacked_data,
        ops['labels_pl']:attacked_label,
         ops['is_training_pl']: is_training,
         ops['lr_attack']:LR_ATTACK,
         ops['dist_weight']:WEIGHT}

    for out_step in range(BINARY_SEARCH_STEP):

        feed_dict[ops['dist_weight']]=WEIGHT

        sess.run(tf.assign(ops['pert'],tf.truncated_normal([BATCH_SIZE,NUM_POINT,3], mean=0, stddev=1)))

        bestdist = [1e10] * BATCH_SIZE
        bestscore = [-1] * BATCH_SIZE  

        prev = 1e6      

        for iteration in range(NUM_ITERATIONS):
            _= sess.run([ops['attack_op']], feed_dict=feed_dict)

            adv_loss_val,dist_val,pred_val,input_val = sess.run([ops['adv_loss'],
                ops['pert_norm'],ops['pred'],ops['pointclouds_input']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            loss=dist_val+np.average(adv_loss_val*WEIGHT)
            if iteration % ((NUM_ITERATIONS // 10) or 1) == 0:
                print((" Iteration {} of {}: loss={} adv_loss:{} " +
                               "distance={}")
                              .format(iteration, NUM_ITERATIONS,
                                loss, adv_loss_val,np.mean(dist_val)))


            # check if we should abort search if we're getting nowhere.
            '''
            if ABORT_EARLY and iteration % ((MAX_ITERATIONS // 10) or 1) == 0:
                
                if loss > prev * .9999999:
                    msg = "    Failed to make progress; stop early"
                    print(msg)
                    break
                prev = loss
            '''

            for e, (dist, pred, ii) in enumerate(zip(dist_val, pred_val, input_val)):
                if dist < bestdist[e] and pred == np.argmax(attacked_label):
                    bestdist[e] = dist
                    bestscore[e] = pred
                if dist < o_bestdist[e] and pred == np.argmax(attacked_label):
                    o_bestdist[e]=dist
                    o_bestscore[e]=pred
                    o_bestattack[e] = ii

        # adjust the constant as needed
        for e in range(BATCH_SIZE):
            if bestscore[e]==np.argmax(attacked_label)and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e] :
                # success
                lower_bound[e] = max(lower_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
                #print('new result found!')
            else:
                # failure
                upper_bound[e] = min(upper_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
        #bestdist_prev=deepcopy(bestdist)

    print(" Successfully generated adversarial exampleson {} of {} instances." .format(sum(lower_bound > 0), BATCH_SIZE))
    print('o_bestdist is',o_bestdist,file = f)
    print('o_bestdist is',o_bestdist)
    return attacked_data-o_bestattack,o_bestdist
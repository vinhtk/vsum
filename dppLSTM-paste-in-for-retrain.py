                                                                                                                                           
#                              dddddddd                                                                                                      
#                              d::::::d                                                                                        tttt          
#                              d::::::d                                                                                     ttt:::t          
#                              d::::::d                                                                                     t:::::t          
#                              d:::::d                                                                                      t:::::t          
#   aaaaaaaaaaaaa      ddddddddd:::::d   aaaaaaaaaaaaa      mmmmmmm    mmmmmmm      ooooooooooo   ppppp   ppppppppp   ttttttt:::::ttttttt    
#   a::::::::::::a   dd::::::::::::::d   a::::::::::::a   mm:::::::m  m:::::::mm  oo:::::::::::oo p::::ppp:::::::::p  t:::::::::::::::::t    
#   aaaaaaaaa:::::a d::::::::::::::::d   aaaaaaaaa:::::a m::::::::::mm::::::::::mo:::::::::::::::op:::::::::::::::::p t:::::::::::::::::t    
#            a::::ad:::::::ddddd:::::d            a::::a m::::::::::::::::::::::mo:::::ooooo:::::opp::::::ppppp::::::ptttttt:::::::tttttt    
#     aaaaaaa:::::ad::::::d    d:::::d     aaaaaaa:::::a m:::::mmm::::::mmm:::::mo::::o     o::::o p:::::p     p:::::p      t:::::t          
#   aa::::::::::::ad:::::d     d:::::d   aa::::::::::::a m::::m   m::::m   m::::mo::::o     o::::o p:::::p     p:::::p      t:::::t          
#  a::::aaaa::::::ad:::::d     d:::::d  a::::aaaa::::::a m::::m   m::::m   m::::mo::::o     o::::o p:::::p     p:::::p      t:::::t          
# a::::a    a:::::ad:::::d     d:::::d a::::a    a:::::a m::::m   m::::m   m::::mo::::o     o::::o p:::::p    p::::::p      t:::::t    tttttt
# a::::a    a:::::ad::::::ddddd::::::dda::::a    a:::::a m::::m   m::::m   m::::mo:::::ooooo:::::o p:::::ppppp:::::::p      t::::::tttt:::::t
# a:::::aaaa::::::a d:::::::::::::::::da:::::aaaa::::::a m::::m   m::::m   m::::mo:::::::::::::::o p::::::::::::::::p       tt::::::::::::::t
#  a::::::::::aa:::a d:::::::::ddd::::d a::::::::::aa:::am::::m   m::::m   m::::m oo:::::::::::oo  p::::::::::::::pp          tt:::::::::::tt
#   aaaaaaaaaa  aaaa  ddddddddd   ddddd  aaaaaaaaaa  aaaammmmmm   mmmmmm   mmmmmm   ooooooooooo    p::::::pppppppp              ttttttttttt  
#                                                                                                  p:::::p                                   
#                                                                                                  p:::::p                                   
#                                                                                                 p:::::::p                                  
#                                                                                                 p:::::::p                                  
#                                                                                                 p:::::::p                                  
#                                                                                                 ppppppppp                                  
                                                                                                                                         
import numpy
import theano
import theano.tensor as T
import time


def adam_opt(model, train_set, valid_set, model_save_dir,
                          minibatch=64, valid_period=1, total_period = 0,
                          disp_period = 1, n_iters=10000000, lr=0.001,
                          beta1=0.1, beta2=0.001, epsilon=1e-8, gamma=1-1e-8):
    """
    Adam optimizer (ICLR 2015)
    """
    # initialize learning rate
    lr_file = open(model_save_dir+'lr.txt', 'w')
    lr_file.write(str(lr))
    lr_file.close()
    lr = theano.shared(numpy.array(lr).astype('float64'))

    updates = []
    all_grads = theano.grad(model.costs[0], model.params)
    i = theano.shared(numpy.float32(1))
    i_t = i + 1.
    fix1 = 1. - (1. - beta1)**i_t
    fix2 = 1. - (1. - beta2)**i_t
    beta1_t = 1-(1-beta1)*gamma**(i_t-1)
    lr_t = lr * (T.sqrt(fix2) / fix1)

    for p, g in zip(model.params, all_grads):
        m = theano.shared(
            numpy.zeros(p.get_value().shape, dtype='float64'))
        v = theano.shared(
            numpy.zeros(p.get_value().shape, dtype='float64'))

        m_t = (beta1_t * g) + ((1. - beta1_t) * m)
        v_t = (beta2 * g**2) + ((1. - beta2) * v)
        g_t = m_t / (T.sqrt(v_t) + epsilon)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    grad_and_cost = all_grads
    grad_and_cost.append(model.costs[0])
    train_grad_f = theano.function(model.inputs, grad_and_cost, on_unused_input='warn')
    train_update_params_f = theano.function(grad_and_cost[0:-1], None, updates=updates)
    if valid_set != None:
        valid_f = theano.function(model.inputs, model.costs, on_unused_input='warn')

    # create log file
    log_file = open(model_save_dir + 'log.txt', 'w')
    log_file.write('adam_optimizer\n')
    log_file.write('lr=%f, beta1=%f, beta2=%f, epsilon=%f, gamma=%f\n' % (lr.get_value(), beta1, beta2, epsilon, gamma))
    log_file.close()

    print( '... training with Adam optimizer')
    cap_count = 0
    train_cost = []
    t0 = time.perf_counter()
    try:
        for u in range(n_iters):
            if u % 10 == 0:
                # refresh lr
                try:
                    lr_file = open(model_save_dir+'_lr.txt', 'r')
                    lr.set_value(float(lr_file.readline().rstrip()))
                    lr_file.close()
                except IOError:
                    pass

            grads = [numpy.zeros_like(p).astype('float64') for p in model.params]
            mb_cost = []
            for i in train_set.iterate(True):
                tmp = train_grad_f(*i)
                new_grads = tmp[0:-1]
                mb_cost.append(tmp[-1])
                grads = [g1+g2 for g1, g2 in zip(grads, new_grads)]
            grads = [g/numpy.array(minibatch) for g in grads]
            train_update_params_f(*grads)
            train_cost.append(numpy.mean(mb_cost))

            # output some information
            if u % disp_period == 0 and u > 0:
                p_now = numpy.concatenate([p.get_value().flatten() for p in model.params])
                if u < 4*disp_period:
                    p_last = numpy.zeros_like(p_now)
                    delta_last = numpy.zeros_like(p_now)
                delta_now = p_now - p_last
                angle = numpy.arccos(numpy.dot(delta_now, delta_last) / numpy.linalg.norm(delta_now) / numpy.linalg.norm(delta_last))
                angle = angle / numpy.pi * 180
                p_last = p_now
                delta_last = delta_now
                t1 = time.perf_counter()
                print ('period=%d, update=%d, mb_cost=[%.4f], |delta|=[%.2e], angle=[%.1f], lr=[%.6f], t=[%.2f]sec' % \
                      (u/valid_period, u, numpy.mean(train_cost), numpy.mean(abs(delta_now[0:10000])), angle, lr.get_value(), (t1-t0)))
                t0 = time.perf_counter()
                train_cost = []

            if u % valid_period == 0 and u > 0:
                model.save_to_file(model_save_dir, total_period + (u)/valid_period)
                valid_loss = []
                valid_acc = []
                train_loss = []
                train_acc = []
                for i in valid_set.iterate(True):
                    loss, acc = valid_f(*i)
                    valid_loss.append(loss)
                    valid_acc.append(acc)
                for i in train_set.iterate(True):
                    loss, acc = valid_f(*i)
                    train_loss.append(loss)
                    train_acc.append(acc)
                cap_count += valid_period*minibatch
                output_info = 'period %i, valid loss=[%.4f], valid acc=[%.4f], train loss=[%.4f], train acc=[%.4f]' % \
                              (u/valid_period, numpy.mean(valid_loss), numpy.mean(valid_acc), numpy.mean(train_loss), numpy.mean(train_acc))
                print (output_info)
                log_file = open(model_save_dir + 'log.txt', 'a')
                log_file.write(output_info+'\n')
                log_file.close()
    except KeyboardInterrupt:
        print( 'Training interrupted.')

                                                 
                                                   
#                         lllllll                    
#                         l:::::l                    
#                         l:::::l                    
#                         l:::::l                    
#    mmmmmmm    mmmmmmm    l::::lppppp   ppppppppp   
#  mm:::::::m  m:::::::mm  l::::lp::::ppp:::::::::p  
# m::::::::::mm::::::::::m l::::lp:::::::::::::::::p 
# m::::::::::::::::::::::m l::::lpp::::::ppppp::::::p
# m:::::mmm::::::mmm:::::m l::::l p:::::p     p:::::p
# m::::m   m::::m   m::::m l::::l p:::::p     p:::::p
# m::::m   m::::m   m::::m l::::l p:::::p     p:::::p
# m::::m   m::::m   m::::m l::::l p:::::p    p::::::p
# m::::m   m::::m   m::::ml::::::lp:::::ppppp:::::::p
# m::::m   m::::m   m::::ml::::::lp::::::::::::::::p 
# m::::m   m::::m   m::::ml::::::lp::::::::::::::pp  
# mmmmmm   mmmmmm   mmmmmmllllllllp::::::pppppppp    
#                                 p:::::p            
#                                 p:::::p            
#                                p:::::::p           
#                                p:::::::p           
#                                p:::::::p           
#                                ppppppppp           
                                                   


import h5py
import numpy
import theano
import theano.tensor as T



class mlp(object):
    """
    An implement of mlp. Support two initialization methods:
    1. create a new one from the given settings
    2. read the model parameters from a given file
    """
    # def __init__(self, layers=[-1, -1], model_file=None, layer_name='mlp', inputs=None, net_type='tanh'):
    def __init__(self, layers=[0, 0], model_file=None, layer_name='mlp', inputs=None, net_type='tanh'):
        self.layer_name = layer_name
        self.layers = layers
        print(layers)
        # create a new model
        if model_file == None:
            self.W = []
            self.b = []
            for i in range(len(layers)-1):
                self.W.append(theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (layers[i], layers[i+1])).astype('float64')))
                self.W[-1].name = self.layer_name + '_W' + str(i)
                self.b.append(theano.shared(numpy.zeros(layers[i+1]).astype('float64')))
                self.b[-1].name = self.layer_name + '_b' + str(i)
        # read from model_file
        else:
            # read model file, in hdf5 format
            f = h5py.File(model_file)
            self.W = []
            self.b = []
            self.layers = []
            i = 0
            while True:
                if self.layer_name+'_W'+str(i) in f:
                    self.W.append(theano.shared(numpy.array(f[self.layer_name+'_W'+str(i)]).astype('float64')))
                    self.W[-1].name = self.layer_name + '_W' + str(i)
                    self.layers.append(self.W[-1].get_value().shape[0])
                    self.b.append(theano.shared(numpy.array(f[self.layer_name+'_b'+str(i)]).astype('float64')))
                    self.b[-1].name = self.layer_name + '_b' + str(i)
                    i += 1
                else:
                    self.layers.append(self.W[-1].get_value().shape[1])
                    break
            # close the hdf5 model file
            f.close()

        self.params = []
        self.params.extend(self.W)
        self.params.extend(self.b)

        assert inputs != None
        x = inputs[0]
        self.inputs = [x]
        self.h = [x]

        def relu(x):
            return x * (x > 0)

        for i in range(len(self.layers)-2):
            self.h.append(T.nnet.sigmoid(T.dot(self.h[i], self.W[i]) + self.b[i]))

        if net_type == 'tanh':
            self.h.append(T.tanh(T.dot(self.h[-1], self.W[-1]) + self.b[-1]))
        elif net_type == 'softmax':
            self.h.append(T.nnet.softmax(T.dot(self.h[-1], self.W[-1]) + self.b[-1]))
        elif net_type == 'sigmoid':
            self.h.append(T.nnet.sigmoid(T.dot(self.h[-1], self.W[-1]) + self.b[-1]))
        elif net_type == 'linear':
            self.h.append(T.dot(self.h[-1], self.W[-1]) + self.b[-1])
        else:
            print('Unsupported MLP Type !!')

    def save_to_file(self, file_dir, file_index=-1):
        file_name = file_dir + self.layer_name + '.h5'
        if file_index >= 0:
            file_name = file_name + '.' + str(file_index)
        f = h5py.File(file_name)
        for p in self.params:
            f[p.name] = p.get_value()
        f.close()


                                                                                                                                                                                              
#                                                                        dddddddd                                                                                                               
#                                                                        d::::::d                                       lllllll                           tttt                                  
#                                                                        d::::::d                                       l:::::l                        ttt:::t                                  
#                                                                        d::::::d                                       l:::::l                        t:::::t                                  
#                                                                        d:::::d                                        l:::::l                        t:::::t                                  
#     ssssssssss   uuuuuu    uuuuuu     mmmmmmm    mmmmmmm       ddddddddd:::::dppppp   ppppppppp   ppppp   ppppppppp    l::::l     ssssssssss   ttttttt:::::ttttttt       mmmmmmm    mmmmmmm   
#   ss::::::::::s  u::::u    u::::u   mm:::::::m  m:::::::mm   dd::::::::::::::dp::::ppp:::::::::p  p::::ppp:::::::::p   l::::l   ss::::::::::s  t:::::::::::::::::t     mm:::::::m  m:::::::mm 
# ss:::::::::::::s u::::u    u::::u  m::::::::::mm::::::::::m d::::::::::::::::dp:::::::::::::::::p p:::::::::::::::::p  l::::l ss:::::::::::::s t:::::::::::::::::t    m::::::::::mm::::::::::m
# s::::::ssss:::::su::::u    u::::u  m::::::::::::::::::::::md:::::::ddddd:::::dpp::::::ppppp::::::ppp::::::ppppp::::::p l::::l s::::::ssss:::::stttttt:::::::tttttt    m::::::::::::::::::::::m
#  s:::::s  ssssss u::::u    u::::u  m:::::mmm::::::mmm:::::md::::::d    d:::::d p:::::p     p:::::p p:::::p     p:::::p l::::l  s:::::s  ssssss       t:::::t          m:::::mmm::::::mmm:::::m
#    s::::::s      u::::u    u::::u  m::::m   m::::m   m::::md:::::d     d:::::d p:::::p     p:::::p p:::::p     p:::::p l::::l    s::::::s            t:::::t          m::::m   m::::m   m::::m
#       s::::::s   u::::u    u::::u  m::::m   m::::m   m::::md:::::d     d:::::d p:::::p     p:::::p p:::::p     p:::::p l::::l       s::::::s         t:::::t          m::::m   m::::m   m::::m
# ssssss   s:::::s u:::::uuuu:::::u  m::::m   m::::m   m::::md:::::d     d:::::d p:::::p    p::::::p p:::::p    p::::::p l::::l ssssss   s:::::s       t:::::t    ttttttm::::m   m::::m   m::::m
# s:::::ssss::::::su:::::::::::::::uum::::m   m::::m   m::::md::::::ddddd::::::ddp:::::ppppp:::::::p p:::::ppppp:::::::pl::::::ls:::::ssss::::::s      t::::::tttt:::::tm::::m   m::::m   m::::m
# s::::::::::::::s  u:::::::::::::::um::::m   m::::m   m::::m d:::::::::::::::::dp::::::::::::::::p  p::::::::::::::::p l::::::ls::::::::::::::s       tt::::::::::::::tm::::m   m::::m   m::::m
#  s:::::::::::ss    uu::::::::uu:::um::::m   m::::m   m::::m  d:::::::::ddd::::dp::::::::::::::pp   p::::::::::::::pp  l::::::l s:::::::::::ss          tt:::::::::::ttm::::m   m::::m   m::::m
#   sssssssssss        uuuuuuuu  uuuummmmmm   mmmmmm   mmmmmm   ddddddddd   dddddp::::::pppppppp     p::::::pppppppp    llllllll  sssssssssss              ttttttttttt  mmmmmm   mmmmmm   mmmmmm
#                                                                                p:::::p             p:::::p                                                                                    
#                                                                                p:::::p             p:::::p                                                                                    
#                                                                               p:::::::p           p:::::::p                                                                                   
#                                                                               p:::::::p           p:::::::p                                                                                   
#                                                                               p:::::::p           p:::::::p                                                                                   
#                                                                               ppppppppp           ppppppppp                                                                                   
                                                                                                                                                                                              
import h5py
import numpy
import theano
import theano.tensor as T
# from mlp import mlp


class summ_dppLSTM(object):
    """
    bidirectional LSTM units: h_backwards + h_forwards to MLP
    """
    # def __init__(self, nx=-1, nh=-1, nout=-1, model_file=None, layer_name='sumLSTM_bid', inputs=None):
    def __init__(self, nx=0, nh=0, nout=0, model_file=None, layer_name='sumLSTM_bid', inputs=None):
        self.layer_name = layer_name
        # input video
        if inputs == None:
            video = T.matrix('video')
            label = T.vector('label')
            labelS = T.ivector('labelS')
            dpp_weight = T.dscalar('dpp_weight')
        else:
            video = inputs[0]
            label = inputs[1]
            labelS = inputs[2]
            dpp_weight = inputs[3]

        # create a new model
        if model_file == None:
            # image feature projection
            self.c_init_mlp = mlp(layers=[nx, nh], layer_name='c_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            self.h_init_mlp = mlp(layers=[nx, nh], layer_name='h_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            # input gate
            self.Wi = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype('float64'))
            self.Wi.name = self.layer_name + '_Wi'
            self.bi = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype('float64'))
            self.bi.name = self.layer_name + '_bi'
            # input modulator
            self.Wc = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype('float64'))
            self.Wc.name = self.layer_name + '_Wc'
            self.bc = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype('float64'))
            self.bc.name = self.layer_name + '_bc'
            # forget gate
            self.Wf = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype('float64'))
            self.Wf.name = self.layer_name + '_Wf'
            self.bf = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype('float64'))
            self.bf.name = self.layer_name + '_bf'
            # output gate
            self.Wo = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype('float64'))
            self.Wo.name = self.layer_name + '_Wo'
            self.bo = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype('float64'))
            self.bo.name = self.layer_name + '_bo'
        # read from model_file
        else:
            # read model file, in hdf5 format
            f = h5py.File(model_file)
            # image feature projection
            self.c_init_mlp = mlp(model_file=model_file, layer_name='c_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            self.h_init_mlp = mlp(model_file=model_file, layer_name='h_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            # input gate
            self.Wi = theano.shared(numpy.array(f[self.layer_name+'_Wi']).astype('float64'))
            self.Wi.name = self.layer_name + '_Wi'
            self.bi = theano.shared(numpy.array(f[self.layer_name+'_bi']).astype('float64'))
            self.bi.name = self.layer_name + '_bi'
            # input modulator
            self.Wc = theano.shared(numpy.array(f[self.layer_name+'_Wc']).astype('float64'))
            self.Wc.name = self.layer_name + '_Wc'
            self.bc = theano.shared(numpy.array(f[self.layer_name+'_bc']).astype('float64'))
            self.bc.name = self.layer_name + '_bc'
            # forget gate
            self.Wf = theano.shared(numpy.array(f[self.layer_name+'_Wf']).astype('float64'))
            self.Wf.name = self.layer_name + '_Wf'
            self.bf = theano.shared(numpy.array(f[self.layer_name+'_bf']).astype('float64'))
            self.bf.name = self.layer_name + '_bf'
            # output gate
            self.Wo = theano.shared(numpy.array(f[self.layer_name+'_Wo']).astype('float64'))
            self.Wo.name = self.layer_name + '_Wo'
            self.bo = theano.shared(numpy.array(f[self.layer_name+'_bo']).astype('float64'))
            self.bo.name = self.layer_name + '_bo'
            # close the hdf5 model file
            f.close()

        # record the size information
        self.nx = self.c_init_mlp.W[0].get_value().shape[0]
        self.nh = self.c_init_mlp.b[-1].get_value().shape[0]
        # add all above into params
        self.params = [self.Wi, self.bi,
                       self.Wc, self.bc,
                       self.Wf, self.bf,
                       self.Wo, self.bo]
        self.params.extend(self.c_init_mlp.params)
        self.params.extend(self.h_init_mlp.params)

        # initializing memory cell and hidden state
        self.c0 = self.c_init_mlp.h[-1]
        self.h0 = self.h_init_mlp.h[-1]

        # go thru the sequence
        # forwards
        ([self.c, self.h], updates) = theano.scan(fn=self.one_step, sequences=[video], outputs_info=[self.c0, self.h0])
        self.c0_back = self.c_init_mlp.h[-1]
        self.h0_back = self.h_init_mlp.h[-1]
        # backwards
        ([self.c_back, self.h_back], updates) = theano.scan(fn=self.one_step, sequences=[video[::-1, :]], outputs_info=[self.c0_back, self.h0_back])
        self.h = T.concatenate([self.h, self.h_back[::-1, :]], axis=1)
        self.h = T.concatenate([video, self.h], axis=1)
        # predicted probobility
        if model_file == None:
            self.classify_mlp = mlp(layers=[self.nx + 2*self.nh, nh, 1],
                                    layer_name='classify_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')

            self.kernel_mlp = mlp(layers=[self.nx + 2*self.nh, nh, nout],
                                    layer_name='kernel_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')
        else:
            self.classify_mlp = mlp(model_file=model_file,
                                    layer_name='classify_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')

            self.kernel_mlp = mlp(model_file=model_file,
                                    layer_name='kernel_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')

        self.nout = self.classify_mlp.b[-1].get_value().shape[0]
        self.params.extend(self.classify_mlp.params)
        self.pred = self.classify_mlp.h[-1]

        self.nout_k = self.kernel_mlp.b[-1].get_value().shape[0]
        self.params.extend(self.kernel_mlp.params)
        self.pred_k = self.kernel_mlp.h[-1]

        kv = self.pred_k # kv means kernel_vector
        qv = self.pred
        K_mat = T.dot(kv, kv.T)
        Q_mat = T.outer(qv, qv)
        L = K_mat * Q_mat
        Ly = L[labelS, :][:, labelS]

        dpp_loss = (- (T.log(T.nlinalg.Det()(Ly)) - T.log(T.nlinalg.Det()(L + T.identity_like(L)))))
        if not T.isnan(dpp_loss):
            loss = T.mean(T.sqr(self.pred.flatten() - label)) + dpp_weight * dpp_loss
        else:
            loss = T.mean(T.sqr(self.pred.flatten() - label)) + dpp_weight * T.nlinalg.Det()(Ly + T.identity_like(Ly)) # when the dpp_loss is nan, just randomly fill in a number

        acc = T.log(T.nlinalg.Det()(L + T.identity_like(L)))

        self.inputs = [video, label, labelS, dpp_weight]
        self.costs = [loss, acc]

    def one_step(self, x_t, c_tm1, h_tm1):
        x_and_h = T.concatenate([x_t, h_tm1], axis=0)
        i_t = T.nnet.sigmoid(T.dot(x_and_h, self.Wi) + self.bi)
        c_tilde = T.tanh(T.dot(x_and_h, self.Wc) + self.bc)
        f_t = T.nnet.sigmoid(T.dot(x_and_h, self.Wf) + self.bf)
        o_t = T.nnet.sigmoid(T.dot(x_and_h, self.Wo) + self.bo)
        c_t = i_t * c_tilde + f_t * c_tm1
        h_t = o_t * T.tanh(c_t)
        return [c_t, h_t]

    def save_to_file(self, file_dir, file_index=-1):
        file_name = file_dir + self.layer_name + '.h5'
        if file_index >= 0:
            file_name = file_name + '.' + str(file_index)
        f = h5py.File(file_name)
        for p in self.params:
            f[p.name] = p.get_value()
        f.close()


#  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄       ▄            ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄ 
# ▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌     ▐░▌          ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
# ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌          ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌     ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌       ▐░▌
# ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌     ▐░▌          ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌
# ▐░▌       ▐░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░░░░░░░░░░░▌     ▐░▌          ▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌
# ▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌     ▐░▌          ▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀█░▌▐░▌       ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀█░█▀▀ 
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌     ▐░▌          ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌     ▐░▌  
# ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌     ▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌      ▐░▌ 
# ▐░░░░░░░░░░▌ ▐░▌       ▐░▌     ▐░▌     ▐░▌       ▐░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░▌       ▐░▌
#  ▀▀▀▀▀▀▀▀▀▀   ▀         ▀       ▀       ▀         ▀       ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀▀▀▀▀▀▀▀▀▀▀  ▀         ▀ 
                                                                                                                                       
import sys
import h5py
import json
import numpy
# import theano 

#An edit
# trying to replace theano
# replace float32 with float64
import scipy.io as sio

def extend_set(s, feature, label, label_tmp, weight):
    s[0].extend(feature)
    s[1].extend(label)
    s[2].extend(label_tmp)
    s[3].extend(weight)

def load_data(data_dir = '../data/SOY/', dataset_testing = 'TVSum', model_type = 1):
    
    train_set = [[], [], [], []]
    
    # [feature, label, weight] = load_dataset_h5(data_dir, 'OVP', model_type)
    # label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
    # train_set[0].extend(feature)
    # train_set[1].extend(label)
    # train_set[2].extend(label_tmp)
    # train_set[3].extend(weight)

    # [feature, label, weight] = load_dataset_h5(data_dir, 'Youtube', model_type)
    # label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
    # train_set[0].extend(feature)
    # train_set[1].extend(label)
    # train_set[2].extend(label_tmp)
    # train_set[3].extend(weight)

    val_set = [[], [], [], []]
    val_idx = []
    test_set = [[], [], [], []]
    te_idx = []

    if dataset_testing == 'TVSum':
        [feature, label, weight] = load_dataset_h5('for.training/datasets/eccv16_dataset_tvsum_google_pool5.h5', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        extend_set(test_set, feature, label, label_tmp, weight)
        te_idx.extend(range(50))
        [feature, label, weight] = load_dataset_h5('for.training/datasets/merged.tvsum.summe_google_pool.h5', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        rand_idx = numpy.random.permutation(25)
        for i in range(25):
            if i <= 15:
                add = train_set
            else:
                add = val_set
                val_idx.append(rand_idx[i])
            extend_set(add, feature, label, label_tmp, weight)


    elif dataset_testing == 'SumMe':
        [feature, label, weight] = load_dataset_h5(data_dir, 'SumMe', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        extend_set(test_set, feature, label, label_tmp, weight)

        te_idx.extend(range(25))
        [feature, label, weight] = load_dataset_h5(data_dir, 'TVSum', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        rand_idx = numpy.random.permutation(50)
        for i in range(50):
            
            if i <= 30:
                add = train_set
            else:
                add = val_set
                val_idx.append(rand_idx[i])
            extend_set(add, feature, label, label_tmp, weight)


    for s in [train_set, val_set, test_set]:
        for i in range(len(s[0])):
            s[0][i] = numpy.transpose(s[0][i])
            s[1][i] = s[1][i].flatten().astype('float64')
            s[2][i] = s[2][i].flatten().astype('int32')


    return train_set, val_set, val_idx, test_set, te_idx

def load_dataset_h5(file_name, label_type): #An EDIT: Don't hardcode file name in functions
    # loading data from a dataset in the format of hdf5 file
    feature = []
    label = []
    weight = []
    f = h5py.File(file_name)
    print(f)
    for i in f:
        i = f[i]
        feature.append(i['features'][:])
        label.append(i['gtscore'][:])
        weight.append(numpy.array(label_type - 1.0).astype('float64'))

        break
    f.close()

    return feature, label, weight

def load_data_mat(data_dir, idx):

    mat_contents = sio.loadmat(data_dir)
    feature = mat_contents['fea' + idx]
    label = mat_contents['fea' + idx]
    return feature, label

class SequenceDataset:
  '''Slices, shuffles and manages a small dataset for the HF optimizer.'''

  def __init__(self, data, batch_size, number_batches, minimum_size=10):
    '''SequenceDataset __init__

  data : list of lists of numpy arrays
    Your dataset will be provided as a list (one list for each graph input) of
    variable-length tensors that will be used as mini-batches. Typically, each
    tensor is a sequence or a set of examples.
  batch_size : int or None
    If an int, the mini-batches will be further split in chunks of length
    `batch_size`. This is useful for slicing subsequences or provide the full
    dataset in a single tensor to be split here. All tensors in `data` must
    then have the same leading dimension.
  number_batches : int
    Number of mini-batches over which you iterate to compute a gradient or
    Gauss-Newton matrix product.
  minimum_size : int
    Reject all mini-batches that end up smaller than this length.'''
    self.current_batch = 0
    self.number_batches = number_batches
    self.items = []

    for i_sequence in range(len(data[0])):
      if batch_size is None:
        self.items.append([data[i][i_sequence] for i in range(len(data))])
      else:
        for i_step in range(0, len(data[0][i_sequence]) - minimum_size + 1, batch_size):
          self.items.append([data[i][i_sequence][i_step:i_step + batch_size] for i in range(len(data))])

    self.shuffle()

  def shuffle(self):
    numpy.random.shuffle(self.items)

  def iterate(self, update=True):
    for b in range(self.number_batches):
      yield self.items[(self.current_batch + b) % len(self.items)]
    if update: self.update()

  def update(self):
    if self.current_batch + self.number_batches >= len(self.items):
      self.shuffle()
      self.current_batch = 0
    else:
      self.current_batch += self.number_batches


#  ▄▄       ▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄ 
# ▐░░▌     ▐░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌      ▐░▌
# ▐░▌░▌   ▐░▐░▌▐░█▀▀▀▀▀▀▀█░▌ ▀▀▀▀█░█▀▀▀▀ ▐░▌░▌     ▐░▌
# ▐░▌▐░▌ ▐░▌▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌▐░▌    ▐░▌
# ▐░▌ ▐░▐░▌ ▐░▌▐░█▄▄▄▄▄▄▄█░▌     ▐░▌     ▐░▌ ▐░▌   ▐░▌
# ▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌     ▐░▌     ▐░▌  ▐░▌  ▐░▌
# ▐░▌   ▀   ▐░▌▐░█▀▀▀▀▀▀▀█░▌     ▐░▌     ▐░▌   ▐░▌ ▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌     ▐░▌     ▐░▌    ▐░▌▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌ ▄▄▄▄█░█▄▄▄▄ ▐░▌     ▐░▐░▌
# ▐░▌       ▐░▌▐░▌       ▐░▌▐░░░░░░░░░░░▌▐░▌      ▐░░▌
#  ▀         ▀  ▀         ▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀        ▀▀ 
                                                    

"""
Video Summarization with Long Short-term Memory
4 datasets used in our task: OVP, Youtube, SumMe, TVSum: Pre-train on OVP, Youtube and SumMe/TVSum, test on TVSum/SumMe
Ke Zhang
Apr, 2017
"""

import os
import numpy
import theano
import h5py
from  datetime import datetime
# from tools import data_loader
# from optimizer.adam_opt import adam_opt
# from layers.summ_dppLSTM import summ_dppLSTM


def train(model_idx, train_set, val_set, lr=0.001, n_iters=100, minibatch=10, valid_period=1, model_saved = ''):

    print('... training')
    model_save_dir = 'dppLST_retrain_'+ str(datetime.now()).replace(' ', 'T') + model_idx + '/'
    if os.path.exists(model_save_dir):
        os.system('rm -r %s' % model_save_dir)
    os.mkdir(model_save_dir)

    # build model
    print('... building model')
    model = summ_dppLSTM(model_file = model_saved)
    train_seq = SequenceDataset(train_set, batch_size=None, number_batches=minibatch)
    valid_seq = SequenceDataset(val_set, batch_size=None, number_batches=len(val_set[0]))

    # train model
    adam_opt(model, train_seq, valid_seq, model_save_dir = model_save_dir, minibatch = minibatch,
             valid_period = valid_period, n_iters=n_iters, lr=lr)

def inference(model_file, model_idx, test_set, test_dir, te_idx):

    print('... inference')
    if os.path.exists(model_file) == 0:
        print('model doesn\'t exist')
        return

    model = summ_dppLSTM(model_file = model_file)
    res = test_dir + '/' + model_idx + '_inference.h5'
    f = h5py.File(res, 'w')
    h_func = theano.function(inputs=[model.inputs[0]], outputs=model.classify_mlp.h[-1])
    h_func_k = theano.function(inputs=[model.inputs[0]], outputs=model.kernel_mlp.h[-1])
    cFrm = []
    for i in range(len(test_set[0])):
        cFrm.append(test_set[0][i].shape[0])
    xf = h5py.File(model_file, 'r')
    xf.keys()

    pred = []
    pred_k = []
    for seq in test_set[0]:
        pred.append(h_func(seq))
        pred_k.append(h_func_k(seq))
    pred = numpy.concatenate(pred, axis=0)
    pred_k = numpy.concatenate(pred_k, axis=0)

    f['pred'] = pred
    f['pred_k'] = pred_k
    f['cFrm'] = cFrm
    f['idx'] = te_idx

    f.close()

if __name__ == '__main__':

    dataset_testing = 'TVSum' # testing dataset: SumMe or TVSum
    model_type = 2 # 1 for vsLSTM and 2 for dppLSTM, please refer to the readme file for more detail
    model_idx = 'dppLSTM_' + dataset_testing + '_' + model_type.__str__()

    # load data
    print('... loading data')
    train_set, val_set, val_idx, test_set, te_idx = load_data(data_dir = '../data/', dataset_testing = dataset_testing, model_type = model_type)
    # model_file = 'ddpLSTM_retrain_test_with_' + dataset_testing

    """
    Uncomment the following line if you want to train the model
    """
    model_file = None
    train(model_idx = model_idx, train_set = train_set, val_set = val_set, model_saved = model_file)

    inference(model_file=model_file, model_idx = model_idx, test_set=test_set, test_dir='./res_LSTM/', te_idx=te_idx)

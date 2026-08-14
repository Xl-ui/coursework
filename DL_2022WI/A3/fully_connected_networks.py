"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from a3_helper import softmax_loss
from eecs598 import Solver


def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from fully_connected_networks.py!')


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in out.  #
        # You will need to reshape the input into rows.                      #
        ######################################################################
        # Replace "pass" statement with your code
        N=x.shape[0]
        X=x.view(N,-1)
        out=torch.addmm(b,X,w)
        """notes:
        torch.addmm(实际上Pytorch中的运算都)要求参数b,X,w在同一个设备上(且数据类型相同);
        forward函数中不必担心此问题;应该由模型初始化时控制device和dtype;
        详见TwoLayerNet中的notes
        """
        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # TODO: Implement the linear backward pass.      #
        ##################################################
        # Replace "pass" statement with your code
        """notes:
        (1)dx
        dL=tr(dout.T X w)=<dout w.T ,X>
        dX=torch.mm(dout ,w.T)
        转置:矩阵使用.t()或T,或者transpose(0,1)--transpose必须指明维度
        dx=dX.view(x.shape) 
        (2)dw
        dL=tr(dout.T X dw)=<X.T dout , dw>
        dw=torch.mm(X.T,dout)
        (3)db
        out=XW+Ib I为单位列向量 
        实现两个向量a,b的外积/秩一矩阵:将a作为列向量按列复制b.shape,
        将b作为行向量复制a.shape次,将两个得到的矩阵进行逐元素相乘.
        dout=Idb, dL=tr(dout.T I db)=<I.T dout, db>
        db=I.T dout
        代码:I=torch.ones(N), db=I@dout 或者db=torch.matmul(I,dout)
        torch.matmul和@会自动识别左侧的一维向量为行向量,返回一维
        torch.mv()会自动识别右侧的一维向量为列向量
        由于I是单位向量,完全可以通过torch.sum()来实现
        """
        X=x.view(x.shape[0],-1)
        dX=torch.mm(dout,w.t())
        dx=dX.view(x.shape)
        dw=torch.mm(X.T,dout)
        db=torch.sum(dout,dim=0)
        ##################################################
        #                END OF YOUR CODE                #
        ##################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor with an  #
        # in-place operation.                             #
        ###################################################
        # Replace "pass" statement with your code
        """notes:
        (1)不能原地修改x的原因:cache中绑定了x,backward时会使用到
        (2)Pytorch中也不推荐原地修改张量,否则可能引发autograd的
        RuntimeError
        """
        out=torch.clamp(x,min=0)
        """
        #alternative:
        (1)out=torch.maximum(x,torch.tensor(0.0))
        (2)out=torch.clamp(x,min=0)  min=,max= 把x每个元素的值限制在[min,max]
        (3)
        mask=x<0
        out=x.clone()
        out[mask]=0
        """

        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # TODO: Implement the ReLU backward pass.           #
        # You should not change the input tensor with an    #
        # in-place operation.                               #
        #####################################################
        # Replace "pass" statement with your code
        dx=dout * (x>=0)
        """notes:
        原本实现:
        dx=torch.zeros(x.shape,dtype=dout.dtype)
        dx[x>=0]=dout[x>=0] 
        x>=0返回与x同形状的布尔数组,每个元素由True(1)和False(0)组成
        布尔数组索引的读写都返回一维张量(行优先顺序),只需要两侧元素个数相同
        注意需要dtype=dout.dtype,否则会引发RunTimeError
        因为这里dout是torch.float64(double)但dx=torch.zeros默认为torch.float32
        dx=dout * (x>=0)的实现更加简洁
        """

        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: Initialize the weights and biases of the two-layer net.   #
        # Weights should be initialized from a Gaussian centered at       #
        # 0.0 with standard deviation equal to weight_scale, and biases   #
        # should be initialized to zero. All weights and biases should    #
        # be stored in the dictionary self.params, with first layer       #
        # weights and biases using the keys 'W1' and 'b1' and second layer#
        # weights and biases using the keys 'W2' and 'b2'.                #
        ###################################################################
        # Replace "pass" statement with your code
        dims=[input_dim,hidden_dim,num_classes]
        self.dims=dims  #增加一个属性以便后续使用
        keys_w=['W%i' %i for i in range(1,len(dims))]
        keys_b=['b%i' %i for i in range(1,len(dims))]
        for i in range(len(dims)-1):
            self.params[keys_w[i]]=torch.randn(dims[i],dims[i+1],dtype=dtype,device=device)*weight_scale
            self.params[keys_b[i]]=torch.zeros(dims[i+1],dtype=dtype,device=device)
        """notes:
        (1)dtype与device
        模型的__init__方法中,需要设置dtype和device两个参数;
        device : 因为Pytorch中的张量必须在同一种设备(cpu或cuda)上计算,因为二者内存是
        物理隔离的,数据传输开销很大;
        dtype : Pytorch要求使用者明确控制数据类型(精度与内存开销的权衡),
        不像numpy一样在类型不同的运算时会做隐式类型提升.
        e.g. float32与float64占用内存相差一倍,在深度学习中差距很大.
        (2)W与b的初始化要明确dtype和device
        与cs231n中numpy实现不同,使用torch初始化参数时,务必记得张量可以在cpu/cuda上,
        要设置device(模型初始化时也会传入该参数); 否则像Linear.forward中使用的
        torch.addmm()等函数会报错,因为它们要求参数在同一个设备上.
        e.g. RuntimeError: Expected all tensors to be on the same device, 
        but got mat1 is on cuda:0, different from other tensors on cpu 
        (when checking argument in method wrapper_CUDA_addmm)
        """
        ###############################################################
        #                            END OF YOUR CODE                 #
        ###############################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################
        # Replace "pass" statement with your code
        a1,cache1=Linear_ReLU.forward(X,self.params['W1'],self.params['b1'])
        scores,cache2=Linear.forward(a1,self.params['W2'],self.params['b2'])

        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the two-layer net.        #
        # Store the loss in the loss variable and gradients in the grads  #
        # dictionary. Compute data loss using softmax, and make sure that #
        # grads[k] holds the gradients for self.params[k]. Don't forget   #
        # to add L2 regularization!                                       #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and       #
        # you pass the automated tests, make sure that your L2            #
        # regularization does not include a factor of 0.5.                #
        ###################################################################
        # Replace "pass" statement with your code
        loss,ds=softmax_loss(scores,y)
        l2_term=sum([torch.sum(self.params['W%i'%i]**2) for i in range(1,len(self.dims))])
        loss+=self.reg*l2_term

        da1,dW2,db2=Linear.backward(ds,cache2)
        dX,dW1,db1=Linear_ReLU.backward(da1,cache1)
        keys=['W1','W2','b1','b2']
        values=[dW1,dW2,db1,db2]
        for i in range(len(keys)):
            grads[keys[i]]=values[i]
            if i<2: #前两个权重矩阵
                grads[keys[i]]+=self.reg*2*self.params[keys[i]]
                #dW还需要加上正则化的梯度2*W*reg;注意由文档正则化没有乘以0.5
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu - [dropout]} x (L - 1) - linear - softmax

    where dropout is optional, and the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0.0, reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving the drop probability
          for networks with dropout. If dropout=0 then the network
          should not use dropout.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - seed: If not None, then pass this random seed to the dropout
          layers. This will make the dropout layers deteriminstic so we
          can gradient check the model.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.use_dropout = dropout != 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all         #
        # values in the self.params dictionary. Store weights and biases      #
        # for the first layer in W1 and b1; for the second layer use W2 and   #
        # b2, etc. Weights should be initialized from a normal distribution   #
        # centered at 0 with standard deviation equal to weight_scale. Biases #
        # should be initialized to zero.                                      #
        #######################################################################
        # Replace "pass" statement with your code
        dims=[input_dim]+hidden_dims+[num_classes]
        for i in range(1,len(dims)):
            self.params['W%d'%i]=torch.randn(dims[i-1],dims[i],dtype=dtype,device=device)*weight_scale
            self.params['b%d'%i]=torch.zeros(dims[i],dtype=dtype,device=device)


        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################

        # When using dropout we need to pass a dropout_param dictionary
        # to each dropout layer so that the layer knows the dropout
        # probability and the mode (train / test). You can pass the same
        # dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'use_dropout': self.use_dropout,
          'dropout_param': self.dropout_param,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.use_dropout = checkpoint['use_dropout']
        self.dropout_param = checkpoint['dropout_param']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param
        # since they behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        scores = None
        ##################################################################
        # TODO: Implement the forward pass for the fully-connected net,  #
        # computing the class scores for X and storing them in the       #
        # scores variable.                                               #
        #                                                                #
        # When using dropout, you'll need to pass self.dropout_param     #
        # to each dropout forward pass.                                  #
        ##################################################################
        # Replace "pass" statement with your code
        """notes:
        1.hidden layers->output layer(->softmax loss)
        input -> Linear.forward ->output,cache
        原实现:
        input=X
        num_layers=self.num_layers
        cache_list=[None]*(num_layers+1)
        for k in torch.arange(1,num_layers):
            input,cache_list[k]=Linear_ReLU.forward(input,self.params['W%d'%k],self.params['b%d'%k])
        scores,cache_list[num_layers]=Linear.forward(input,self.params['W%d'%num_layers],self.params['b%d'%num_layers])
        """
        input=X
        #hidden layers
        num_layers=self.num_layers
        cache_list = [None] * (num_layers + 1)  # 第一个None不使用
        if self.use_dropout:
            #hidden_layers
            for k in range(1,num_layers):
                input,cache1=Linear_ReLU.forward(input,self.params['W%d'%k],self.params['b%d'%k])
                input,cache2=Dropout.forward(input,self.dropout_param)
                cache_list[k]=(cache1,cache2)  #可以忽略此处Pycharm的静态类型检查
            #output_layer
            scores,cache_list[num_layers]=Linear.forward(input,self.params['W%d'%num_layers],self.params['b%d'%num_layers])
        else:

            #hidden layer:input->Linear.forward(input)->output,cache[k]
            forward_funcs=[Linear_ReLU.forward,Linear.forward]
            for k in range(1,num_layers+1):
                input ,cache_list[k] = forward_funcs[k//num_layers](
                    input,self.params['W%d'%k],self.params['b%d'%k]
                )
            scores = input
            """notes:
            1.forward_funcs:
            除了最后一层(即k<num_layers),都使用forward_funcs[0];
            k<num_layers时k//num_layers==0;否则最后输出层时k==num_layers,索引为1
            边界情况: num_layers==1时也正确使用Linear作为输出层
            2.for k in range(1,num_layers+1)
            不建议使用for k in torch.arange(1,num_layers+1) 
            无论是意图表达还是性能开销/正确性; 
            意图上只是要循环,使用range即可
            代码正确性/性能上,与for k in np.arange()中k是int不同,
            由于torch.arange()返回一维张量,k得到的是 0维张量 而不是Python int
            cache_list[k]碰巧正确只是因为Pytorch实现了__index__
            'W%d'%k 用%d格式化0维张量 正确也只是因为Pytorch实现了__int__
            """



        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #####################################################################
        # TODO: Implement the backward pass for the fully-connected net.    #
        # Store the loss in the loss variable and gradients in the grads    #
        # dictionary. Compute data loss using softmax, and make sure that   #
        # grads[k] holds the gradients for self.params[k]. Don't forget to  #
        # add L2 regularization!                                            #
        # NOTE: To ensure that your implementation matches ours and you     #
        # pass the automated tests, make sure that your L2 regularization   #
        # includes a factor of 0.5 to simplify the expression for           #
        # the gradient.                                                     #
        #####################################################################
        # Replace "pass" statement with your code
        loss,dout=softmax_loss(scores,y)
        if self.use_dropout:
            #由于init中只传入了一个p,这里默认不对输入层进行dropout
            #output layer
            W_key='W%d'%num_layers
            dout,grads[W_key],grads['b%d'%num_layers]=Linear.backward(dout,cache_list[num_layers])
            loss += 0.5 * self.reg * torch.sum(self.params[W_key] ** 2)
            grads[W_key] += self.reg * self.params[W_key]
            #hidden layers
            for k in range(num_layers-1,0,-1):
                cache1,cache2=cache_list[k]
                dout=Dropout.backward(dout,cache2)
                W_key='W%d'%k
                dout,grads[W_key],grads['b%d'%k]=Linear_ReLU.backward(dout,cache1)
                loss += 0.5 * self.reg * torch.sum(self.params[W_key] ** 2)
                grads[W_key] += self.reg * self.params[W_key]
        else:
            backward_funcs=[Linear_ReLU.backward,Linear.backward]
            for k in range(num_layers,0,-1): #注意反向传播时的循环反向.
                W_key='W%d'%k
                dout ,grads[W_key] ,grads['b%d'%k]=backward_funcs[k//num_layers](
                                                   dout,cache_list[k])
                loss+=0.5*self.reg*torch.sum(self.params[W_key]**2)
                grads[W_key]+=self.reg*self.params[W_key]
        """notes:
        0.建议统一采用if self.use_dropout中的写法(分离hidden layer和output layer)
        在循环内部添加if self.use_dropout即可,该条件判断的多次循环并不是问题;
        这样代码不必大量重复
        1.cache_list使用list, Python list是动态数组,连续存储引用(Pyobject*)
        可以根据计算指针偏移量来访问元素,时间是O(1);
        也可以使用字典,键为整数k本身(不需要转换为'k'),字典访问元素也是O(1),
        但在这里还要hash(k),对于连续的整数来说使用list会略好一些
        2.反向传播的循环必须要从后往前,range容易写错为range(1,num_layers+1)
        逆序时也要记得加上step=-1,即range(num_layers,0,-1)
        可以写多个遍历num_layers的循环,将计算loss和梯度的逻辑分离清楚,
        num_layers一般不会太大,不用担心多次循环的效率        
        """
        ###########################################################
        #                   END OF YOUR CODE                      #
        ###########################################################

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, device=device)
    #############################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that   #
    # achieves at least 50% accuracy on the validation set.     #
    #############################################################
    solver = None
    # Replace "pass" statement with your code
    solver=Solver(model,data_dict,optim_config={'learning_rate':3e-2},lr_decay=0.95,num_epochs=20,device=device)
    ##############################################################
    #                    END OF YOUR CODE                        #
    ##############################################################
    return solver


def get_three_layer_network_params():
    ###############################################################
    # TODO: Change weight_scale and learning_rate so your         #
    # model achieves 100% training accuracy within 20 epochs.     #
    ###############################################################
    weight_scale = 5e-1   # Experiment with this!
    learning_rate = 1e-2  # Experiment with this!
    # Replace "pass" statement with your code

    ################################################################
    #                             END OF YOUR CODE                 #
    ################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ################################################################
    # TODO: Change weight_scale and learning_rate so your          #
    # model achieves 100% training accuracy within 20 epochs.      #
    ################################################################
    learning_rate = 2e-3  # Experiment with this!
    weight_scale = 3e-1   # Experiment with this!
    # Replace "pass" statement with your code

    #打印参数配置
    print('weight_scale为',weight_scale)
    print('learning_rate为',learning_rate)

    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################
    return weight_scale, learning_rate


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.
    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.
    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to
      store a moving average of the gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', torch.zeros_like(w))
    next_w = None
    ##################################################################
    # TODO: Implement the momentum update formula. Store the         #
    # updated value in the next_w variable. You should also use and  #
    # update the velocity v.                                         #
    ##################################################################
    # Replace "pass" statement with your code
    v = config['momentum'] * v - config['learning_rate'] * dw
    w += v
    next_w = w
    """notes:
    1.v,w都原地修改
    2.typically momentum=0.9 or 0.99
    3.
    通常说的sgd_momentum,指的是标准的动量法;与adam中使用的EMA不同
    alternative: 
    v = config['momentum']*v + dw
    w -= config['learning_rate']*v
    (w的更新结果相同,但v不同,该实现中的v只是单纯的梯度)
    
    adam中使用EMA:
    v= config['momentum']* v + (1-config['momentum'])*dw
    w -= config['learning_rate']*v
    但由于本题参考实现不同,该实现无法通过测试(计算的v,更新后的w与上述实现的不同,记为v',w')
    关系为:v'=(1-config['momentum']) * v / (-config['learning_rate']) 相差乘积的系数
         在更新w'时,δw'=next_w'-w'= (1-config['momentum']) * δw   
    4.sgd+momentum解决的问题:
    sgd的振荡问题:Loss曲面较陡方向上来回走动,较缓方向上虽方向一致但改变缓慢 
    (切换坐标轴到合适位置;想象二维参数,此时Loss等高线是二维平面上的椭圆.)
    sgd+momentum通过利用历史梯度信息,使得梯度在来回走动的较陡方向上相互抵消,
    在改变缓慢的较平坦方向上累加梯度从而加速走动.
    """
    ###################################################################
    #                           END OF YOUR CODE                      #
    ###################################################################
    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.
    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', torch.zeros_like(w))

    next_w = None
    ###########################################################################
    # TODO: Implement the RMSprop update formula, storing the next value of w #
    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #
    ###########################################################################
    # Replace "pass" statement with your code
    config['cache']=config['decay_rate']*config['cache']+(1-config['decay_rate'])*dw**2
    w -= config['learning_rate']* (dw / (torch.sqrt(config['cache'])+config['epsilon']))
    next_w=w
    """notes:
    1.注意给sqrt(squared_grad)加上epsilon以避免除0错误
    """
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.
    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', torch.zeros_like(w))
    config.setdefault('v', torch.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None
    ##########################################################################
    # TODO: Implement the Adam update formula, storing the next value of w in#
    # the next_w variable. Don't forget to update the m, v, and t variables  #
    # stored in config.                                                      #
    #                                                                        #
    # NOTE: In order to match the reference output, please modify t _before_ #
    # using it in any calculations.                                          #
    ##########################################################################
    # Replace "pass" statement with your code
    config['t']+=1
    config['m']= config['beta1'] * config['m'] + (1-config['beta1'])*dw
    config['v']=config['beta2'] * config['v']+(1-config['beta2'])*(dw**2)
    m_bias=config['m']/(1-config['beta1']**config['t'])
    v_bias=config['v'] / (1-config['beta2']**config['t'])
    w -= config['learning_rate']* m_bias / (v_bias.sqrt()+config['epsilon'])
    next_w=w

    """notes:
    1.注意与momentum中的v区分: momentum中v指梯度的移动平均
    但在Adam中,v指梯度平方的移动平均,m才是梯度的移动平均
    2.m是梯度的一阶矩估计;v是梯度的二阶矩估计 (一阶矩:Ex;二阶矩:E(x**2))
    引入偏差项/(1-beta**t)的原因:初始化t=0时刻的m,v为0(直观上相当于认为历史信息为0),会引入系统偏差;
    导致E(m_t)与E(v_t)是真实期望的(1-beta**t)倍,
    这只有在t比较大的时候才近似于无偏估计;
    因此除以(1-beta**t)就得到了无偏估计.
    随机性来源于mini-batch中的抽样
    beta1=0.9 beta2=0.999 lr=1e-4,5e-4,1e-3
    3.in practice
    Adam is a good default choice in many cases 
    SGD+Momentum can outperform Adam but may require more tuning
    """
    #########################################################################
    #                              END OF YOUR CODE                         #
    #########################################################################

    return next_w, config


class Dropout(object):

    @staticmethod
    def forward(x, dropout_param):
        """
        Performs the forward pass for (inverted) dropout.
        Inputs:
        - x: Input data: tensor of any shape
        - dropout_param: A dictionary with the following keys:
          - p: Dropout parameter. We *drop* each neuron output with
            probability p.
          - mode: 'test' or 'train'. If the mode is train, then
            perform dropout;
          if the mode is test, then just return the input.
          - seed: Seed for the random number generator. Passing seed
            makes this
            function deterministic, which is needed for gradient checking
            but not in real networks.
        Outputs:
        - out: Tensor of the same shape as x.
        - cache: tuple (dropout_param, mask). In training mode, mask
          is the dropout mask that was used to multiply the input; in
          test mode, mask is None.
        NOTE: Please implement **inverted** dropout, not the vanilla
              version of dropout.
        See http://cs231n.github.io/neural-networks-2/#reg for more details.
        NOTE 2: Keep in mind that p is the probability of **dropping**
                a neuron output; this might be contrary to some sources,
                where it is referred to as the probability of keeping a
                neuron output.
        """
        p, mode = dropout_param['p'], dropout_param['mode']
        if 'seed' in dropout_param:
            torch.manual_seed(dropout_param['seed'])
        """notes:
        1.每次forward时都通过torch.manual_seed()重置Pytorch的全局随机数生成器(RNG)
        的种子(决定生成的随机数序列),从而每次torch.rand()等随机数生成都会得到相同的结果
        2.dropout作为一个操作(与Linear_ReLU,Linear等一样)存在,实现
        forward,backward,这样就能方便地拼接在输入数据和隐藏层之后
        """
        mask = None
        out = None

        if mode == 'train':
            ##############################################################
            # TODO: Implement training phase forward pass for            #
            # inverted dropout.                                          #
            # Store the dropout mask in the mask variable.               #
            ##############################################################
            # Replace "pass" statement with your code
            mask=torch.rand_like(x)<(1-p)  #(1-p)的概率保留
            out=x/(1-p)*mask
            """notes:
            1.注意不要直接使用torch.rand(x.shape) 
            否则得到的张量与x的dtype和device可能不同,导致运行时错误
            torch.rand()默认在CPU上创建张量,
            如果x是在GPU上,就会导致设备间数据运输,而Pytorch会对此报错.
            
            2.torch.rand_like(x)等_like(x)方法,会继承x的
            shape,dtype,device
            """

            ##############################################################
            #                   END OF YOUR CODE                         #
            ##############################################################
        elif mode == 'test':
            ##############################################################
            # TODO: Implement the test phase forward pass for            #
            # inverted dropout.                                          #
            ##############################################################
            # Replace "pass" statement with your code
            out =x
            #mask=None  前面已经初始化
            """notes:
            注意不要直接return x;最后已经有返回语句return out,cache了
            直接return x会破坏返回值签名的一致性;
            forward函数本身是不能显式地判断mode=train/test的,
            而是通过dropout_param['mode']判断的
            """
            ##############################################################
            #                      END OF YOUR CODE                      #
            ##############################################################

        cache = (dropout_param, mask)

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Perform the backward pass for (inverted) dropout.
        Inputs:
        - dout: Upstream derivatives, of any shape
        - cache: (dropout_param, mask) from Dropout.forward.
        """
        dropout_param, mask = cache
        mode = dropout_param['mode']

        dx = None
        if mode == 'train':
            ###########################################################
            # TODO: Implement training phase backward pass for        #
            # inverted dropout                                        #
            ###########################################################
            # Replace "pass" statement with your code
            p=dropout_param['p']
            dx=dout/(1-p) * mask
            ###########################################################
            #                     END OF YOUR CODE                    #
            ###########################################################
        elif mode == 'test':
            dx = dout
        return dx

"""
Implements convolutional networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from click.core import batch
from tornado.gen import moment

from a3_helper import softmax_loss
from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU


def hello_convolutional_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from convolutional_networks.py!')


class Conv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A naive implementation of the forward pass for a convolutional layer.
        The input consists of N data points, each with C channels, height H and
        width W. We convolve each input with F different filters, where each
        filter spans all C channels and has height HH and width WW.

        Input:
        - x: Input data of shape (N, C, H, W)
        - w: Filter weights of shape (F, C, HH, WW)
        - b: Biases, of shape (F,)
        - conv_param: A dictionary with the following keys:
          - 'stride': The number of pixels between adjacent receptive fields
            in the horizontal and vertical directions.
          - 'pad': The number of pixels that is used to zero-pad the input.

        During padding, 'pad' zeros should be placed symmetrically (i.e equally
        on both sides) along the height and width axes of the input. Be careful
        not to modfiy the original input x directly.

        Returns a tuple of:
        - out: Output data of shape (N, F, H', W') where H' and W' are given by
          H' = 1 + (H + 2 * pad - HH) / stride
          W' = 1 + (W + 2 * pad - WW) / stride
        - cache: (x, w, b, conv_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the convolutional forward pass.                  #
        # Hint: you can use function torch.nn.functional.pad for padding.  #
        # You are NOT allowed to use anything in torch.nn in other places. #
        ####################################################################
        # Replace "pass" statement with your code
        """notes:
        1.torch.nn.functional.pad(input,pad=,mode='constant',value=0)
        mode表示填充的方式,默认为填充常数,且常数默认为0(value代表constant模式下填充的值);
        该函数的pad参数需要传入一个元组,从最后一个维度开始往之前的维度填写,
        每个维度依次填写要在前(原本的0索引),后(原本最后一个元素)分别添加多少个索引
        pad_size=conv_param['pad'] 
        x_pad=torch.nn.functional.pad(x,pad=(pad_size,)*4)
        注意不要原地修改x; 填充最后两个维度,分别指明前后填充尺寸,因此元组长度为4而不是2;
        2.计算H',W'时,对于无法整除stride的情况,通常向下取整,丢弃边缘信息(Pytorch默认实现)
        3.b.shape返回torchSize对象,因此不能直接b.view(b.shape,1,1)
        可以用-1自动补全:b.view(-1,1,1)
        """
        N,F=x.shape[0],w.shape[0]
        (H,W),(HH,WW)=x.shape[2:],w.shape[2:]
        pad=conv_param['pad']
        stride=conv_param['stride']
        H1=(H+2*pad-HH)//stride+1
        W1=(W+2*pad-WW)//stride+1
        out=torch.empty(N,F,H1,W1,dtype=w.dtype,device=w.device)
        x_pad=torch.nn.functional.pad(x,pad=(pad,)*4)
        for i in range(H1):
            for j in range(W1):
                x_window=x_pad[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]
                out[:,:,i,j]=torch.tensordot(x_window,w,dims=([1,2,3],[1,2,3]))
        out+=b.view(-1,1,1)
        #####################################################################
        #                          END OF YOUR CODE                         #
        #####################################################################
        cache = (x, w, b, conv_param)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a convolutional layer.
          Inputs:
        - dout: Upstream derivatives.
        - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

        Returns a tuple of:
        - dx: Gradient with respect to x
        - dw: Gradient with respect to w
        - db: Gradient with respect to b
        """
        dx, dw, db = None, None, None
        ###############################################################
        # TODO: Implement the convolutional backward pass.            #
        ###############################################################
        # Replace "pass" statement with your code
        """notes:
        1.张量内积本身就是线性的,因此对于y=A*X,dy/dX=A, 即dy=A*dX;
        2.考虑第k个filter w_k; 只影响out_k=x[:,k,:,:],三维张量;
        w_k->out_k  dout_k=dout[:,k,:,:],
        只考虑第n个样本,设dout_k_n=dout_k[n,k,:,:]
        for i ,j in out_k 行,列:
            dw_k+=dout_k_n[i][j]x[n]_window
        3.x=(N,C,H,W)  w=(F,C,HH,WW)  out=(N,F,H',W')
        x'=(N,H',W',C,HH,WW),(p,q)代表(H',W')坐标 
        dw=(F,HH,WW)
        """
        x,w,b,conv_param=cache
        N, F = x.shape[0], w.shape[0]
        (H, W), (HH, WW) = x.shape[2:], w.shape[2:]
        pad = conv_param['pad']
        stride = conv_param['stride']
        H1 = (H + 2 * pad - HH) // stride + 1
        W1 = (W + 2 * pad - WW) // stride + 1
        x_pad = torch.nn.functional.pad(x, pad=(pad,) * 4)
        dw=torch.zeros_like(w)
        dx_pad = torch.zeros_like(x_pad)
        for i in range(H1):
            for j in range(W1):
                x_window = x_pad[:, :, i * stride:i * stride + HH, j * stride:j * stride + WW]
                dw+=torch.tensordot(dout[:,:,i,j], x_window,dims=([0],[0]))
                #或dw+=torch.einsum('nf,nchw->fchw',dout[:,:,i,j],x_window)
                dx_pad[:,:,i*stride:i*stride+HH,j*stride:j*stride+WW]+=torch.tensordot(dout[:,:,i,j],w,dims=([1],[0]))

        db=torch.sum(dout,dim=[0,2,3])
        dx=dx_pad[:,:,pad:pad+H,pad:pad+W]
        """notes:
        1.计算dx_pad时,由于各个位置元素会被多次用到不同(i,j)的out上,难以分离逻辑,
        但是对于给定的(i,j),out使用了哪些位置的dx_pad是非常清晰的.
        因此可以考虑反向梯度流动,遍历父节点并让父节点向子节点累加梯度.
        : ,i*stride : i*stride+HH, j*stride : j*stride+WW
        取得了x_pad对应的三维窗口,第一个维度的:取了所有channel
        for i, j:  给定index为k的filter
        dw[k]+=torch.tensordot(dout[:,k,i,j],x_window,dims=([0],[0]))
        =>  dw+=torch.tensordot(dout[:,:,i,j],x_window,dims=([0],[0]))
        
        alternative:
        #dw暂时存储所有样本,最后再求和
        dw=torch.zeros((N,)+w.shape,dtype=w.dtype,device=w.device)
        for i, j:
            dw[:,:]+=dout[:,:,i,j]*x_window
         dw=torch.sum(dw,dim=0)
        
        """

        ###############################################################
        #                       END OF YOUR CODE                      #
        ###############################################################
        return dx, dw, db


class MaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        """
        A naive implementation of the forward pass for a max-pooling layer.

        Inputs:
        - x: Input data, of shape (N, C, H, W)
        - pool_param: dictionary with the following keys:
          - 'pool_height': The height of each pooling region
          - 'pool_width': The width of each pooling region
          - 'stride': The distance between adjacent pooling regions
        No padding is necessary here.

        Returns a tuple of:
        - out: Output of shape (N, C, H', W') where H' and W' are given by
          H' = 1 + (H - pool_height) / stride
          W' = 1 + (W - pool_width) / stride
        - cache: (x, pool_param)
        """
        out = None
        ####################################################################
        # TODO: Implement the max-pooling forward pass                     #
        ####################################################################
        # Replace "pass" statement with your code
        N,C,H,W=x.shape
        pool_height=pool_param['pool_height']
        pool_width=pool_param['pool_width']
        stride=pool_param['stride']
        H1=1+(H-pool_height)//stride
        W1=1+(W-pool_width)//stride
        out=torch.empty(N,C,H1,W1,dtype=x.dtype,device=x.device)
        indices=torch.empty(N,C,H1,W1,dtype=torch.long,device=x.device)
        for i in range(H1):
            for j in range(W1):
                window=x[:,:,i*stride:i*stride+pool_height,j*stride:j*stride+pool_width]
                flat=window.flatten(2)
                out[:,:,i,j],indices[:,:,i,j]=torch.max(flat,2)
        ####################################################################
        #                         END OF YOUR CODE                         #
        ####################################################################
        cache = (x, pool_param,indices)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        A naive implementation of the backward pass for a max-pooling layer.
        Inputs:
        - dout: Upstream derivatives
        - cache: A tuple of (x, pool_param) as in the forward pass.
        Returns:
        - dx: Gradient with respect to x
        """
        dx = None
        #####################################################################
        # TODO: Implement the max-pooling backward pass                     #
        #####################################################################
        # Replace "pass" statement with your code
        x,pool_param,indices=cache
        dx=torch.zeros_like(x)
        N, C, H, W = x.shape
        pool_height = pool_param['pool_height']
        pool_width = pool_param['pool_width']
        stride = pool_param['stride']
        H1 = 1 + (H - pool_height) // stride
        W1 = 1 + (W - pool_width) // stride

        for i in range(H1):
            for j in range(W1):
                idx=indices[:,:,i,j]  #(N,C)
                row=idx//pool_width  #(N,C)
                col=idx%pool_width
                abs_row=row+i*stride  #(N,C)
                abs_col=col+j*stride
                n_id=torch.arange(N)[:,None].expand(N,C)
                c_id=torch.arange(C)[None,:].expand(N,C)
                dx[n_id,c_id,abs_row,abs_col]+=dout[:,:,i,j]
        """notes:
        1.由于每个样本,每个channe的(i,j)窗口中最大值元素的索引一般不同,
        不能像前面的实现一样通过切片来为dx赋值
        """
        ####################################################################
        #                          END OF YOUR CODE                        #
        ####################################################################
        return dx


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:
    conv - relu - 2x2 max pool - linear - relu - linear - softmax
    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=32,
                 filter_size=7,
                 hidden_dim=100,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in convolutional layer
        - hidden_dim: Number of units to use in fully-connected hidden layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you
          should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ######################################################################
        # TODO: Initialize weights，biases for the three-layer convolutional #
        # network. Weights should be initialized from a Gaussian             #
        # centered at 0.0 with standard deviation equal to weight_scale;     #
        # biases should be initialized to zero. All weights and biases       #
        # should be stored in the dictionary self.params. Store weights and   #
        # biases for the convolutional layer using the keys 'W1' and 'b1';   #
        # use keys 'W2' and 'b2' for the weights and biases of the hidden    #
        # linear layer, and key 'W3' and 'b3' for the weights and biases of  #
        # the output linear layer                                            #
        #                                                                    #
        # IMPORTANT: For this assignment, you can assume that the padding    #
        # and stride of the first convolutional layer are chosen so that     #
        # **the width and height of the input are preserved**. Take a        #
        # look at the start of the loss() function to see how that happens.  #
        ######################################################################
        # Replace "pass" statement with your code
        C,H,W=input_dims
        filter_shape=(num_filters,C,filter_size,filter_size)
        self.params['W1']=torch.randn(filter_shape,dtype=self.dtype,device=device)*weight_scale
        self.params['b1']=torch.zeros(num_filters,dtype=self.dtype,device=device)
        hidden_input_dim=num_filters*H*W//4
        #注意经过max_pool后才进入隐藏层
        self.params['W2']=torch.randn(hidden_input_dim,hidden_dim,dtype=self.dtype,device=device)*weight_scale
        self.params['b2']=torch.zeros(hidden_dim,dtype=self.dtype,device=device)
        self.params['W3']=torch.randn(hidden_dim,num_classes,dtype=self.dtype,device=device)*weight_scale
        self.params['b3']=torch.zeros(num_classes,dtype=self.dtype,device=device)


        ######################################################################
        #                            END OF YOUR CODE                        #
        ######################################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = checkpoint['dtype']
        self.reg = checkpoint['reg']
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        Input / output: Same API as TwoLayerNet.
        """
        X = X.to(self.dtype)
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        """notes:
        H'=1+(2*pad+H-filter_size)//stride
        令stride=1,则2*pad+1=filter_size,从而pad=(filter_size-1)//2
        """
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ######################################################################
        # TODO: Implement the forward pass for three-layer convolutional     #
        # net, computing the class scores for X and storing them in the      #
        # scores variable.                                                   #
        #                                                                    #
        # Remember you can use functions defined in your implementation      #
        # above                                                              #
        ######################################################################
        # Replace "pass" statement with your code
        z1,cache1=Conv_ReLU_Pool.forward(X,W1,b1,conv_param,pool_param)
        z2,cache2=Linear_ReLU.forward(z1,W2,b2)
        scores,cache3=Linear.forward(z2,W3,b3)

        ######################################################################
        #                             END OF YOUR CODE                       #
        ######################################################################

        if y is None:
            return scores

        loss, grads = 0.0, {}
        ####################################################################
        # TODO: Implement backward pass for three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables.  #
        # Compute data loss using softmax, and make sure that grads[k]     #
        # holds the gradients for self.params[k]. Don't forget to add      #
        # L2 regularization!                                               #
        #                                                                  #
        # NOTE: To ensure that your implementation matches ours and you    #
        # pass the automated tests, make sure that your L2 regularization  #
        # does not include a factor of 0.5                                 #
        ####################################################################
        # Replace "pass" statement with your code
        """notes:
        1.import的modules:
        from a3_helper import softmax_loss
        from fully_connected_networks import Linear_ReLU, Linear, Solver, adam, ReLU
        """
        loss,dout=softmax_loss(scores,y)
        for w in [W1,W2,W3]:
            loss+=self.reg*torch.sum(w*w)
        backward_funcs=[None,Conv_ReLU_Pool.backward,Linear_ReLU.backward,Linear.backward]
        caches=[None,cache1,cache2,cache3]
        for k in range(3,0,-1):
            dout,grads['W%d'%k],grads['b%d'%k]=backward_funcs[k](dout,caches[k])
            grads['W%d'%k]+=self.reg*2*self.params['W%d'%k]

        ###################################################################
        #                             END OF YOUR CODE                    #
        ###################################################################

        return loss, grads


class DeepConvNet(object):
    """
    A convolutional neural network with an arbitrary number of convolutional
    layers in VGG-Net style. All convolution layers will use kernel size 3 and
    padding 1 to preserve the feature map size, and all pooling layers will be
    max pooling layers with 2x2 receptive fields and a stride of 2 to halve the
    size of the feature map.

    The network will have the following architecture:

    {conv - [batchnorm?] - relu - [pool?]} x (L - 1) - linear

    Each {...} structure is a "macro layer" consisting of a convolution layer,
    an optional batch normalization layer, a ReLU nonlinearity, and an optional
    pooling layer. After L-1 such macro layers, a single fully-connected layer
    is used to predict the class scores.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """
    def __init__(self,
                 input_dims=(3, 32, 32),
                 num_filters=[8, 8, 8, 8, 8],
                 max_pools=[0, 1, 2, 3, 4],
                 batchnorm=False,
                 num_classes=10,
                 weight_scale=1e-3,
                 reg=0.0,
                 weight_initializer=None,
                 dtype=torch.float,
                 device='cpu'):
        """
        Initialize a new network.

        Inputs:
        - input_dims: Tuple (C, H, W) giving size of input data
        - num_filters: List of length (L - 1) giving the number of
          convolutional filters to use in each macro layer.
        - max_pools: List of integers giving the indices of the macro
          layers that should have max pooling (zero-indexed).
        - batchnorm: Whether to include batch normalization in each macro layer
        - num_classes: Number of scores to produce from the final linear layer.
        - weight_scale: Scalar giving standard deviation for random
          initialization of weights, or the string "kaiming" to use Kaiming
          initialization instead
        - reg: Scalar giving L2 regularization strength. L2 regularization
          should only be applied to convolutional and fully-connected weight
          matrices; it should not be applied to biases or to batchnorm scale
          and shifts.
        - dtype: A torch data type object; all computations will be performed
          using this datatype. float is faster but less accurate, so you should
          use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.num_layers = len(num_filters)+1
        self.max_pools = max_pools
        self.batchnorm = batchnorm
        self.reg = reg
        self.dtype = dtype

        if device == 'cuda':
            device = 'cuda:0'

        #####################################################################
        # TODO: Initialize the parameters for the DeepConvNet. All weights, #
        # biases, and batchnorm scale and shift parameters should be        #
        # stored in the dictionary self.params.                             #
        #                                                                   #
        # Weights for conv and fully-connected layers should be initialized #
        # according to weight_scale. Biases should be initialized to zero.  #
        # Batchnorm scale (gamma) and shift (beta) parameters should be     #
        # initilized to ones and zeros respectively.                        #
        #####################################################################
        # Replace "pass" statement with your code
        C,H,W=input_dims
        channels=[C]+num_filters
        kernel_size=3  #VGG-Net style
        #conv layers
        for k in range(1,len(channels)):
            Cin,Cout=channels[k-1],channels[k]
            #初始化权重:weight_scale='Kaiming'时使用Kaiming初始化,否则使用普通weight_scale
            if weight_scale=='kaiming':
                self.params['W%d'%k]=kaiming_initializer(Cin,Cout,kernel_size,dtype=dtype,device=device)
            else:
                self.params['W%d'%k]=torch.randn(Cout,Cin,kernel_size,kernel_size,
                                             dtype=dtype,device=device)*weight_scale

            self.params['b%d'%k]=torch.zeros(Cout,dtype=dtype,device=device)
            if self.batchnorm:
                self.params['gamma%d'%k]=torch.ones(Cout,dtype=dtype,device=device)
                self.params['beta%d'%k]=torch.zeros(Cout,dtype=dtype,device=device)
        #linear layer
        #每次max_pool都会将H,W减半,普通conv会保持H,W
        H_final,W_final=H//2**len(max_pools),W//2**len(max_pools)
        linear_input_dim = channels[-1] * H_final * W_final
        if weight_scale == 'kaiming':
            self.params['W%d' % len(channels)] = kaiming_initializer(
                linear_input_dim, num_classes, relu=False,
                dtype=dtype, device=device)
        else:
            self.params['W%d' % len(channels)] = torch.randn(
                linear_input_dim, num_classes, dtype=dtype, device=device
            ) * weight_scale
        self.params['b%d'%len(channels)]=torch.zeros(num_classes,dtype=dtype,device=device)
        """notes:
        1.实现BN时,β替代了bias的作用,所以可以舍弃参数bias;
        但是本题中需要保留bias的实现(下面参数数量检查中计算了bias)
        2.BN中的γ初始化为1(代表std=1),β初始化为0(代表mean=0)
        """
        ################################################################
        #                      END OF YOUR CODE                        #
        ################################################################

        # With batch normalization we need to keep track of running
        # means and variances, so we need to pass a special bn_param
        # object to each batch normalization layer. You should pass
        # self.bn_params[0] to the forward pass of the first batch
        # normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.batchnorm:
            self.bn_params = [{'mode': 'train'}
                              for _ in range(len(num_filters))]

        # Check that we got the right number of parameters
        if not self.batchnorm:
            params_per_macro_layer = 2  # weight and bias
        else:
            params_per_macro_layer = 4  # weight, bias, scale, shift
        num_params = params_per_macro_layer * len(num_filters) + 2
        msg = 'self.params has the wrong number of ' \
              'elements. Got %d; expected %d'
        msg = msg % (len(self.params), num_params)
        assert len(self.params) == num_params, msg

        # Check that all parameters have the correct device and dtype:
        for k, param in self.params.items():
            msg = 'param "%s" has device %r; should be %r' \
                  % (k, param.device, device)
            assert param.device == torch.device(device), msg
            msg = 'param "%s" has dtype %r; should be %r' \
                  % (k, param.dtype, dtype)
            assert param.dtype == dtype, msg

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
          'max_pools': self.max_pools,
          'batchnorm': self.batchnorm,
          'bn_params': self.bn_params,
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']
        self.max_pools = checkpoint['max_pools']
        self.batchnorm = checkpoint['batchnorm']
        self.bn_params = checkpoint['bn_params']

        for p in self.params:
            self.params[p] = \
                self.params[p].type(dtype).to(device)

        for i in range(len(self.bn_params)):
            for p in ["running_mean", "running_var"]:
                self.bn_params[i][p] = \
                    self.bn_params[i][p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the deep convolutional
        network.
        Input / output: Same API as ThreeLayerConvNet.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params since they
        # behave differently during training and testing.
        if self.batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None

        # pass conv_param to the forward pass for the
        # convolutional layer
        # Padding and stride chosen to preserve the input
        # spatial size
        filter_size = 3
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        #########################################################
        # TODO: Implement the forward pass for the DeepConvNet, #
        # computing the class scores for X and storing them in  #
        # the scores variable.                                  #
        #                                                       #
        # You should use the fast versions of convolution and   #
        # max pooling layers, or the convolutional sandwich     #
        # layers, to simplify your implementation.              #
        #########################################################
        # Replace "pass" statement with your code
        caches=[None]*self.num_layers
        out=X
        if self.batchnorm:
            for k in range(self.num_layers-1):
                if k in self.max_pools:
                    out,caches[k]=Conv_BatchNorm_ReLU_Pool.forward(out,self.params['W%d'%(k+1)],self.params['b%d'%(k+1)],
                                                                   self.params['gamma%d'%(k+1)],self.params['beta%d'%(k+1)],
                                                                   conv_param,self.bn_params[k],pool_param)
                else :
                    out,caches[k]=Conv_BatchNorm_ReLU.forward(out,self.params['W%d'%(k+1)],self.params['b%d'%(k+1)],
                                                                   self.params['gamma%d'%(k+1)],self.params['beta%d'%(k+1)],
                                                                   conv_param,self.bn_params[k])
        else:
            for k in range(self.num_layers-1):
                if k in self.max_pools:
                    out,caches[k]=Conv_ReLU_Pool.forward(out,self.params['W%d'%(k+1)],self.params['b%d'%(k+1)],
                                                         conv_param,pool_param)
                else :
                    out,caches[k]=Conv_ReLU.forward(out,self.params['W%d'%(k+1)],self.params['b%d'%(k+1)],
                                                         conv_param)
        scores,caches[self.num_layers-1]=Linear.forward(out,self.params['W%d'%self.num_layers],
                                                        self.params['b%d'%self.num_layers])
        #####################################################
        #                 END OF YOUR CODE                  #
        #####################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the DeepConvNet,          #
        # storing the loss and gradients in the loss and grads variables. #
        # Compute data loss using softmax, and make sure that grads[k]    #
        # holds the gradients for self.params[k]. Don't forget to add     #
        # L2 regularization!                                              #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and you   #
        # pass the automated tests, make sure that your L2 regularization #
        # does not include a factor of 0.5                                #
        ###################################################################
        # Replace "pass" statement with your code
        loss,dout=softmax_loss(scores,y)
        #Linear layer
        loss+=self.reg*torch.sum(self.params['W%d'%self.num_layers]**2)
        dout,grads['W%d'%self.num_layers],grads['b%d'%self.num_layers]=Linear.backward(dout,caches[self.num_layers-1])
        grads['W%d'%self.num_layers]+=2*self.reg*self.params['W%d'%self.num_layers]
        #Conv layers
        if self.batchnorm:
            for k in range(self.num_layers-2,-1,-1):
                W_key='W%d'%(k+1)
                b_key='b%d'%(k+1)
                gamma_key='gamma%d'%(k+1)
                beta_key='beta%d'%(k+1)
                if k in self.max_pools:
                    dout,grads[W_key],grads[b_key],grads[gamma_key],grads[beta_key]=\
                            Conv_BatchNorm_ReLU_Pool.backward(dout,caches[k])
                else:
                    dout,grads[W_key],grads[b_key],grads[gamma_key],grads[beta_key]=\
                            Conv_BatchNorm_ReLU.backward(dout,caches[k])
                loss+=self.reg*torch.sum(self.params[W_key]**2)
                grads[W_key]+=2*self.reg*self.params[W_key]

        else:
            for k in range(self.num_layers-2,-1,-1):
                W_key = 'W%d' % (k + 1)
                b_key = 'b%d' % (k + 1)
                if k in self.max_pools:
                    dout,grads[W_key],grads[b_key]=Conv_ReLU_Pool.backward(dout,caches[k])
                else:
                    dout,grads[W_key],grads[b_key]=Conv_ReLU.backward(dout,caches[k])
                loss += self.reg * torch.sum(self.params[W_key] ** 2)
                grads[W_key]+=2*self.reg*self.params[W_key]

        #############################################################
        #                       END OF YOUR CODE                    #
        #############################################################

        return loss, grads


def find_overfit_parameters():
    weight_scale =5e-3   # Experiment with this!
    learning_rate = 1e-5  # Experiment with this!

    ###########################################################
    # TODO: Change weight_scale and learning_rate so your     #
    # model achieves 100% training accuracy within 30 epochs. #
    ###########################################################
    # Replace "pass" statement with your code
    weight_scale,learning_rate= 7e-2, 1e-2
    ###########################################################
    #                       END OF YOUR CODE                  #
    ###########################################################
    return weight_scale, learning_rate


def create_convolutional_solver_instance(data_dict, dtype, device):
    model = None
    solver = None
    #########################################################
    # TODO: Train the best DeepConvNet that you can on      #
    # CIFAR-10 within 60 seconds.                           #
    #########################################################
    # Replace "pass" statement with your code
    model = DeepConvNet(
        input_dims=data_dict['X_train'].shape[1:],
        num_filters=[32, 64, 128],
        max_pools=[0, 1, 2],
        batchnorm=True,
        weight_scale='kaiming',
        reg=1e-4,
        dtype=dtype,
        device=device,
    )
    solver = Solver(
        model,
        data_dict,
        batch_size=128,
        num_epochs=10,
        update_rule=adam,
        optim_config={'learning_rate': 1e-3},
        lr_decay=0.95,
        print_every=20,
        device=device,
    )
    #########################################################
    #                  END OF YOUR CODE                     #
    #########################################################
    return solver


def kaiming_initializer(Din, Dout, K=None, relu=True, device='cpu',
                        dtype=torch.float32):
    """
    Implement Kaiming initialization for linear and convolution layers.

    Inputs:
    - Din, Dout: Integers giving the number of input and output dimensions
      for this layer
    - K: If K is None, then initialize weights for a linear layer with
      Din input dimensions and Dout output dimensions. Otherwise if K is
      a nonnegative integer then initialize the weights for a convolution
      layer with Din input channels, Dout output channels, and a kernel size
      of KxK.
    - relu: If ReLU=True, then initialize weights with a gain of 2 to
      account for a ReLU nonlinearity (Kaiming initializaiton); otherwise
      initialize weights with a gain of 1 (Xavier initialization).
    - device, dtype: The device and datatype for the output tensor.

    Returns:
    - weight: A torch Tensor giving initialized weights for this layer.
      For a linear layer it should have shape (Din, Dout); for a
      convolution layer it should have shape (Dout, Din, K, K).
    """
    gain = 2. if relu else 1.
    weight = None
    if K is None:
        ###################################################################
        # TODO: Implement Kaiming initialization for linear layer.        #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din).                           #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        scale = (gain / Din) ** 0.5
        weight = torch.randn(Din, Dout, dtype=dtype, device=device) * scale
        ###################################################################
        #                            END OF YOUR CODE                     #
        ###################################################################
    else:
        ###################################################################
        # TODO: Implement Kaiming initialization for convolutional layer. #
        # The weight scale is sqrt(gain / fan_in),                        #
        # where gain is 2 if ReLU is followed by the layer, or 1 if not,  #
        # and fan_in = num_in_channels (= Din) * K * K                    #
        # The output should be a tensor in the designated size, dtype,    #
        # and device.                                                     #
        ###################################################################
        # Replace "pass" statement with your code
        fan_in = Din * K * K
        scale = (gain / fan_in) ** 0.5
        weight = torch.randn(
            Dout, Din, K, K, dtype=dtype, device=device
        ) * scale
        ###################################################################
        #                         END OF YOUR CODE                        #
        ###################################################################
    return weight


class BatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance
        are computed from minibatch statistics and used to normalize the
        incoming data. During training we also keep an exponentially decaying
        running mean of the mean and variance of each feature, and these
        averages are used to normalize data at test-time.

        At each timestep we update the running averages for mean and
        variance using an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different
        test-time behavior: they compute sample mean and variance for
        each feature using a large number of training images rather than
        using a running average. For this implementation we have chosen to use
        running averages instead since they do not require an additional
        estimation step; the PyTorch implementation of batch normalization
        also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean
            of features
          - running_var Array of shape (D,) giving running variance
            of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        mode = bn_param['mode']
        eps = bn_param.get('eps', 1e-5)
        momentum = bn_param.get('momentum', 0.9)

        N, D = x.shape
        running_mean = bn_param.get('running_mean',
                                    torch.zeros(D,
                                                dtype=x.dtype,
                                                device=x.device))
        running_var = bn_param.get('running_var',
                                   torch.zeros(D,
                                               dtype=x.dtype,
                                               device=x.device))

        out, cache = None, None
        if mode == 'train':
            ##################################################################
            # TODO: Implement the training-time forward pass for batch norm. #
            # Use minibatch statistics to compute the mean and variance, use #
            # these statistics to normalize the incoming data, and scale and #
            # shift the normalized data using gamma and beta.                #
            #                                                                #
            # You should store the output in the variable out.               #
            # Any intermediates that you need for the backward pass should   #
            # be stored in the cache variable.                               #
            #                                                                #
            # You should also use your computed sample mean and variance     #
            # together with the momentum variable to update the running mean #
            # and running variance, storing your result in the running_mean  #
            # and running_var variables.                                     #
            #                                                                #
            # Note that though you should be keeping track of the running    #
            # variance, you should normalize the data based on the standard  #
            # deviation (square root of variance) instead!                   #
            # Referencing the original paper                                 #
            # (https://arxiv.org/abs/1502.03167) might prove to be helpful.  #
            ##################################################################
            # Replace "pass" statement with your code
            sample_mean=torch.sum(x,dim=0)/N
            sample_var=torch.sum((x-sample_mean)**2,dim=0)/N

            running_mean=momentum*running_mean+(1-momentum)*sample_mean
            running_var=momentum*running_var+(1-momentum)*sample_var

            var_denominator=1/torch.sqrt(sample_var+eps)

            x_norm=(x-sample_mean)*var_denominator
            out=x_norm*gamma+beta
            cache=(var_denominator,x_norm,gamma)
            ################################################################
            #                           END OF YOUR CODE                   #
            ################################################################
        elif mode == 'test':
            ################################################################
            # TODO: Implement the test-time forward pass for               #
            # batch normalization. Use the running mean and variance to    #
            # normalize the incoming data, then scale and shift the        #
            # normalized data using gamma and beta. Store the result       #
            # in the out variable.                                         #
            ################################################################
            # Replace "pass" statement with your code
            x_norm=(x-running_mean)/torch.sqrt(running_var+eps)
            out=x_norm*gamma+beta
            ################################################################
            #                      END OF YOUR CODE                        #
            ################################################################
        else:
            raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

        # Store the updated running means back into bn_param
        bn_param['running_mean'] = running_mean.detach()
        bn_param['running_var'] = running_var.detach()

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for batch normalization.

        For this implementation, you should write out a
        computation graph for batch normalization on paper and
        propagate gradients backward through intermediate nodes.

        Inputs:
        - dout: Upstream derivatives, of shape (N, D)
        - cache: Variable of intermediates from batchnorm_forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (N, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        dx, dgamma, dbeta = None, None, None
        #####################################################################
        # TODO: Implement the backward pass for batch normalization.        #
        # Store the results in the dx, dgamma, and dbeta variables.         #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167) #
        # might prove to be helpful.                                        #
        # Don't forget to implement train and test mode separately.         #
        #####################################################################
        # Replace "pass" statement with your code

        var_denominator,x_norm,gamma=cache
        dgamma=torch.sum(x_norm*dout,dim=0)
        dbeta=torch.sum(dout,dim=0)
        N=dout.shape[0]
        dx_norm=dout*gamma
        dx_var=-1/N *x_norm*torch.sum(dx_norm*x_norm,dim=0)
        dx_mean=-1/N *torch.sum(dx_norm,dim=0)
        dx=var_denominator*(dx_var+dx_mean+dx_norm)
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        return dx, dgamma, dbeta

    @staticmethod
    def backward_alt(dout, cache):
        """
        Alternative backward pass for batch normalization.
        For this implementation you should work out the derivatives
        for the batch normalizaton backward pass on paper and simplify
        as much as possible. You should be able to derive a simple expression
        for the backward pass. See the jupyter notebook for more hints.

        Note: This implementation should expect to receive the same
        cache variable as batchnorm_backward, but might not use all of
        the values in the cache.

        Inputs / outputs: Same as batchnorm_backward
        """
        dx, dgamma, dbeta = None, None, None
        ###################################################################
        # TODO: Implement the backward pass for batch normalization.      #
        # Store the results in the dx, dgamma, and dbeta variables.       #
        #                                                                 #
        # After computing the gradient with respect to the centered       #
        # inputs, you should be able to compute gradients with respect to #
        # the inputs in a single statement; our implementation fits on a  #
        # single 80-character line.                                       #
        ###################################################################
        # Replace "pass" statement with your code
        var_denominator,x_norm,gamma=cache
        dgamma = torch.sum(x_norm * dout, dim=0)
        dbeta = torch.sum(dout, dim=0)
        N = dout.shape[0]
        dx_norm = dout * gamma
        dx = var_denominator * (-1 / N * x_norm * torch.sum(dx_norm * x_norm, dim=0) +
                                -1 / N * torch.sum(dx_norm, dim=0) + dout * gamma)
        #################################################################
        #                        END OF YOUR CODE                       #
        #################################################################

        return dx, dgamma, dbeta


class SpatialBatchNorm(object):

    @staticmethod
    def forward(x, gamma, beta, bn_param):
        """
        Computes the forward pass for spatial batch normalization.

        Inputs:
        - x: Input data of shape (N, C, H, W)
        - gamma: Scale parameter, of shape (C,)
        - beta: Shift parameter, of shape (C,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance. momentum=0
            means that old information is discarded completely at every
            time step, while momentum=1 means that new information is never
            incorporated. The default of momentum=0.9 should work well
            in most situations.
          - running_mean: Array of shape (C,) giving running mean of
            features
          - running_var Array of shape (C,) giving running variance
            of features

        Returns a tuple of:
        - out: Output data, of shape (N, C, H, W)
        - cache: Values needed for the backward pass
        """
        out, cache = None, None

        ################################################################
        # TODO: Implement the forward pass for spatial batch           #
        # normalization.                                               #
        #                                                              #
        # HINT: You can implement spatial batch normalization by       #
        # calling the vanilla version of batch normalization you       #
        # implemented above. Your implementation should be very short; #
        # ours is less than five lines.                                #
        ################################################################
        # Replace "pass" statement with your code
        N,C,H,W=x.shape
        X=x.permute(0,2,3,1).reshape(-1,C)
        out,cache=BatchNorm.forward(X,gamma,beta,bn_param)
        out=out.reshape(N,H,W,C).permute(0,3,1,2)
        """notes:
        1.重点在于理解内存布局以及permute和reshape的机制:
        permute只改变stride(根据给定顺序来重排列stride),
        不改变物理内存,因此permute后张量不再连续
        此时调用reshape,等价于contiguous().view()
        reshape: 张量连续->相当于view; 不连续->先使其物理内存顺序
        按照此时的逻辑顺序排列,再进行view
        view(或连续时的reshape)是只改变stride和shape的.
        在这个例子中,
        即先按照permute后的逻辑顺序,重新布局内存,使得
        物理内存顺序与permute后逻辑顺序一致,
        (设contiguous()后的张量为X')
        再通过reshape/view来改变stride.
        2.对于连续的张量,改变形状再复原是简单的.
        例如,对于X', X'.reshape(-1,C),
        再Y=X'.reshape(N,H,W,C)
        是可以保证Y与X'的对应元素相等的,
        即X'[n,h,w,c]=Y[n,h,w,c],任意合法坐标(n,h,w,c)
        """
        ################################################################
        #                       END OF YOUR CODE                       #
        ################################################################

        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for spatial batch normalization.
        Inputs:
        - dout: Upstream derivatives, of shape (N, C, H, W)
        - cache: Values from the forward pass
        Returns a tuple of:
        - dx: Gradient with respect to inputs, of shape (N, C, H, W)
        - dgamma: Gradient with respect to scale parameter, of shape (C,)
        - dbeta: Gradient with respect to shift parameter, of shape (C,)
        """
        dx, dgamma, dbeta = None, None, None

        #################################################################
        # TODO: Implement the backward pass for spatial batch           #
        # normalization.                                                #
        #                                                               #
        # HINT: You can implement spatial batch normalization by        #
        # calling the vanilla version of batch normalization you        #
        # implemented above. Your implementation should be very short;  #
        # ours is less than five lines.                                 #
        #################################################################
        # Replace "pass" statement with your code
        N, C, H, W = dout.shape
        dout_flat = dout.permute(0, 2, 3, 1).reshape(-1, C)
        dx_flat, dgamma, dbeta = BatchNorm.backward(dout_flat, cache)
        dx = dx_flat.reshape(N, H, W, C).permute(0, 3, 1, 2)
        ##################################################################
        #                       END OF YOUR CODE                         #
        ##################################################################

        return dx, dgamma, dbeta

##################################################################
#           Fast Implementations and Sandwich Layers             #
##################################################################


class FastConv(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        N, C, H, W = x.shape
        F, _, HH, WW = w.shape
        stride, pad = conv_param['stride'], conv_param['pad']
        layer = torch.nn.Conv2d(C, F, (HH, WW), stride=stride, padding=pad)
        layer.weight = torch.nn.Parameter(w)
        layer.bias = torch.nn.Parameter(b)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, w, b, conv_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, _, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
            dw = layer.weight.grad.detach()
            db = layer.bias.grad.detach()
            layer.weight.grad = layer.bias.grad = None
        except RuntimeError:
            dx, dw, db = torch.zeros_like(tx), \
                         torch.zeros_like(layer.weight), \
                         torch.zeros_like(layer.bias)
        return dx, dw, db


class FastMaxPool(object):

    @staticmethod
    def forward(x, pool_param):
        N, C, H, W = x.shape
        pool_height, pool_width = \
            pool_param['pool_height'], pool_param['pool_width']
        stride = pool_param['stride']
        layer = torch.nn.MaxPool2d(kernel_size=(pool_height, pool_width),
                                   stride=stride)
        tx = x.detach()
        tx.requires_grad = True
        out = layer(tx)
        cache = (x, pool_param, tx, out, layer)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        try:
            x, _, tx, out, layer = cache
            out.backward(dout)
            dx = tx.grad.detach()
        except RuntimeError:
            dx = torch.zeros_like(tx)
        return dx


class Conv_ReLU(object):

    @staticmethod
    def forward(x, w, b, conv_param):
        """
        A convenience layer that performs a convolution
        followed by a ReLU.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for the
          convolutional layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        out, relu_cache = ReLU.forward(a)
        cache = (conv_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu convenience layer.
        """
        conv_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Conv_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, conv_param, pool_param):
        """
        A convenience layer that performs a convolution,
        a ReLU, and a pool.
        Inputs:
        - x: Input to the convolutional layer
        - w, b, conv_param: Weights and parameters for
          the convolutional layer
        - pool_param: Parameters for the pooling layer
        Returns a tuple of:
        - out: Output from the pooling layer
        - cache: Object to give to the backward pass
        """
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        s, relu_cache = ReLU.forward(a)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the conv-relu-pool
        convenience layer
        """
        conv_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        da = ReLU.backward(ds, relu_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db


class Linear_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, bn_param):
        """
        Convenience layer that performs an linear transform,
        batch normalization, and ReLU.
        Inputs:
        - x: Array of shape (N, D1); input to the linear layer
        - w, b: Arrays of shape (D2, D2) and (D2,) giving the
          weight and bias for the linear transform.
        - gamma, beta: Arrays of shape (D2,) and (D2,) giving
          scale and shift parameters for batch normalization.
        - bn_param: Dictionary of parameters for batch
          normalization.
        Returns:
        - out: Output from ReLU, of shape (N, D2)
        - cache: Object to give to the backward pass.
        """
        a, fc_cache = Linear.forward(x, w, b)
        a_bn, bn_cache = BatchNorm.forward(a, gamma, beta, bn_param)
        out, relu_cache = ReLU.forward(a_bn)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-batchnorm-relu
        convenience layer.
        """
        fc_cache, bn_cache, relu_cache = cache
        da_bn = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = BatchNorm.backward(da_bn, bn_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma,
                                                beta, bn_param)
        out, relu_cache = ReLU.forward(an)
        cache = (conv_cache, bn_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache = cache
        dan = ReLU.backward(dout, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta


class Conv_BatchNorm_ReLU_Pool(object):

    @staticmethod
    def forward(x, w, b, gamma, beta, conv_param, bn_param, pool_param):
        a, conv_cache = FastConv.forward(x, w, b, conv_param)
        an, bn_cache = SpatialBatchNorm.forward(a, gamma, beta, bn_param)
        s, relu_cache = ReLU.forward(an)
        out, pool_cache = FastMaxPool.forward(s, pool_param)
        cache = (conv_cache, bn_cache, relu_cache, pool_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        conv_cache, bn_cache, relu_cache, pool_cache = cache
        ds = FastMaxPool.backward(dout, pool_cache)
        dan = ReLU.backward(ds, relu_cache)
        da, dgamma, dbeta = SpatialBatchNorm.backward(dan, bn_cache)
        dx, dw, db = FastConv.backward(da, conv_cache)
        return dx, dw, db, dgamma, dbeta

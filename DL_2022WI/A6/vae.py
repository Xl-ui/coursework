from __future__ import print_function

import math

import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F


def hello_vae():
    print("Hello from vae.py!")


class VAE(nn.Module):
    def __init__(self, input_size, latent_size=15):
        super(VAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.hidden_dim = 256  # H_d
        self.encoder=None
        self.mu_layer=None
        self.logvar_layer=None
        self.decoder=None


        ###########################################################################
        # TODO: Implement the fully-connected encoder architecture described in   #
        # the notebook. Specifically, self.encoder should be a network that       #
        # inputs a batch of input images of shape (N, 1, H, W) into a batch of    #
        # hidden features of shape (N, H_d). Set up self.mu_layer and             #
        # self.logvar_layer to be a pair of linear layers that map the hidden     #
        # features into estimates of the mean and log-variance of the posterior   #
        # over the latent vectors; the mean and log-variance estimates will both  #
        # be tensors of shape (N, Z).                                             #
        ###########################################################################
        # Replace "pass" statement with your code
        self.encoder = nn.Sequential(nn.Flatten(),
                                     nn.Linear(input_size, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU())
        self.mu_layer = nn.Linear(self.hidden_dim, latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim, latent_size)
        ###########################################################################
        # TODO: Implement the fully-connected decoder architecture described in   #
        # the notebook. Specifically, self.decoder should be a network that inputs#
        # a batch of latent vectors of shape (N, Z) and outputs a tensor of       #
        # estimated images of shape (N, 1, H, W).                                 #
        ###########################################################################
        # Replace "pass" statement with your code
        self.decoder = nn.Sequential(nn.Linear(latent_size, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, input_size),
                                     nn.Sigmoid(),
                                     nn.Unflatten(dim=1,unflattened_size=(1, int(math.sqrt(input_size)), int(math.sqrt(input_size))))
                                     )
        """notes:
        1.明确任务与建模: 
        数据集:
        当前数据集是MNIST,每张图像只有单通道,经过transform(ToTensor())后,
        每个pixel都是[0,1]中的实数.
        模型建立:
        (1)其中一种建模解释——sigmoid建模pixel的值:
        建模p(x|z)时(x代表(H,W)的张量)时,由于x中每个pixel的值是[0,1]间实数
        所以可以使用Sigmoid来建模pixel的值;
        但这个解释实际上与标准VAE中的概率建模不符合;仔细回忆,我们希望decoder建模概率分布p(x|z)
        但decoder输出的实际上是这个分布的参数,而不是直接输出x的值;
        如果把Sigmoid的输出解释为pixel的值,那么pixel取这个值的概率/概率密度是什么?
        (2)按照VAE框架的概率建模解释——sigmoid建模P(pixel=1)
        我们实际上按照二分类问题来处理,给每个pixel建模一个伯努利分布:
        假设pixel只能取0或1,Sigmoid()输出建模的是pixel=1的概率
        decoder输出的是伯努利分布的期望(即pixel=1的概率)
        当然,实际上,MNIST图像的pixel值是[0,1]间的连续值；
        如果我们先把图片进行二值化,那么以上概率建模就是严格成立的,采取BCE损失等价于MLE;
        但是我们通常不会进行二值化,因此从严格的概率建模角度看,以上解释是不成立的;
        但是从训练的目标函数来看,BCE依旧是实践中实用有效的损失函数;
        
        在(2)的解释下,采取BCE作为损失函数可以看成是实践中的近似,
        我们以严格概率建模框架下得到的BCE为指导,来作为实际任务中的损失函数,
        用它来训练模型
        
        而在重建图像阶段,我们不会只给每个pixel赋值1或0,而是会把Sigmoid输出的值
        直接作为重建图像中pixel的值;这相当于取伯努利分布的期望来作为重建pixel的值;
        也就是说,在生成阶段的解释实际上更像(1).
        """
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################

    def forward(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)

        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z),
          with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the input batch through the encoder model to get posterior     #
        #     mu and logvariance                                                  #
        # (2) Reparametrize to compute  the latent vector z                       #
        # (3) Pass z through the decoder to resconstruct x                        #
        ###########################################################################
        # Replace "pass" statement with your code
        imm=self.encoder(x)
        mu=self.mu_layer(imm)
        logvar=self.logvar_layer(imm)
        latent_code=reparametrize(mu,logvar)
        x_hat=self.decoder(latent_code)
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


class CVAE(nn.Module):
    def __init__(self, input_size, num_classes=10, latent_size=15):
        super(CVAE, self).__init__()
        self.input_size = input_size  # H*W
        self.latent_size = latent_size  # Z
        self.num_classes = num_classes  # C
        self.hidden_dim = 128  # H_d
        self.encoder = None
        self.mu_layer = None
        self.logvar_layer = None
        self.decoder = None

        ###########################################################################
        # TODO: Define a FC encoder as described in the notebook that transforms  #
        # the image--after flattening and now adding our one-hot class vector (N, #
        # H*W + C)--into a hidden_dimension (N, H_d) feature space, and a final   #
        # two layers that project that feature space to posterior mu and posterior#
        # log-variance estimates of the latent space (N, Z)                       #
        ###########################################################################
        # Replace "pass" statement with your code
        """
        此处encoder不再负责flatten();
        但decoder依旧负责unflatten()
        """
        self.encoder = nn.Sequential(nn.Linear(input_size+num_classes,self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim,self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     )
        self.mu_layer = nn.Linear(self.hidden_dim,self.latent_size)
        self.logvar_layer = nn.Linear(self.hidden_dim,self.latent_size)
        ###########################################################################
        # TODO: Define a fully-connected decoder as described in the notebook that#
        # transforms the latent space (N, Z + C) to the estimated images of shape #
        # (N, 1, H, W).                                                           #
        ###########################################################################
        # Replace "pass" statement with your code
        H = int(math.sqrt(input_size))  # H==W
        self.decoder = nn.Sequential(nn.Linear(latent_size + num_classes, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, self.hidden_dim),
                                     nn.ReLU(),
                                     nn.Linear(self.hidden_dim, input_size),
                                     nn.Sigmoid(),
                                     nn.Unflatten(dim=1, unflattened_size=(1, H, H))
                                     )
        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################

    def forward(self, x, c):
        """
        Performs forward pass through FC-CVAE model by passing image through
        encoder, reparametrize trick, and decoder models

        Inputs:
        - x: Input data for this timestep of shape (N, 1, H, W)
        - c: One hot vector representing the input class (0-9) (N, C)

        Returns:
        - x_hat: Reconstructed input data of shape (N, 1, H, W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent
          space dimension
        - logvar: Matrix representing estimated variance in log-space (N, Z),  with
          Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ###########################################################################
        # TODO: Implement the forward pass by following these steps               #
        # (1) Pass the concatenation of input batch and one hot vectors through   #
        #     the encoder model to get posterior mu and logvariance               #
        # (2) Reparametrize to compute the latent vector z                        #
        # (3) Pass concatenation of z and one hot vectors through the decoder to  #
        #     resconstruct x                                                      #
        ###########################################################################
        # Replace "pass" statement with your code

        #1. Flatten and concatenate
        imm = self.encoder(torch.cat([x.flatten(start_dim=1),c],dim=1))
        """notes:
        1.注意函数式API(torch.xxx)与模块式API(nn.xxx)的行为差异
        nn.Flatten()默认start_dim=1 , 把0维当作batch维度
        torch.flatten()默认start_dim=0,展平整个张量
        因此此处需要指明start_dim=1
        2.torch.cat中的tensors需要传入list或者tuple,不能直接用逗号分隔多个要拼接的张量
        """
        mu = self.mu_layer(imm)
        logvar = self.logvar_layer(imm)
        latent_code = reparametrize(mu,logvar)
        x_hat = self.decoder(torch.cat([latent_code,c],dim=1))

        ###########################################################################
        #                                      END OF YOUR CODE                   #
        ###########################################################################
        return x_hat, mu, logvar


def reparametrize(mu, logvar):
    """
    Differentiably sample random Gaussian data with specified mean and variance
    using the reparameterization trick.

    Suppose we want to sample a random number z from a Gaussian distribution with
    mean mu and standard deviation sigma, such that we can backpropagate from the
    z back to mu and sigma. We can achieve this by first sampling a random value
    epsilon from a standard Gaussian distribution with zero mean and unit variance,
    then setting z = sigma * epsilon + mu.

    For more stable training when integrating this function into a neural network,
    it helps to pass this function the log of the variance of the distribution from
    which to sample, rather than specifying the standard deviation directly.

    Inputs:
    - mu: Tensor of shape (N, Z) giving means
    - logvar: Tensor of shape (N, Z) giving log-variances

    Returns:
    - z: Estimated latent vectors, where z[i, j] is a random value sampled from a
      Gaussian with mean mu[i, j] and log-variance logvar[i, j].
    """
    z = None
    ###############################################################################
    # TODO: Reparametrize by initializing epsilon as a normal distribution and    #
    # scaling by posterior mu and sigma to estimate z                             #
    ###############################################################################
    # Replace "pass" statement with your code
    epsilon=torch.randn(logvar.shape,device=mu.device)
    z=epsilon*torch.exp(logvar/2)+mu
    """notes:
    注意确保设置torch.randn的device=mu.device
    mu与logvar由model前向计算而得,它们由model训练时
    设置model的device来统一管理
    """
    ###############################################################################
    #                              END OF YOUR CODE                               #
    ###############################################################################
    return z


def loss_function(x_hat, x, mu, logvar):
    """
    Computes the negative variational lower bound loss term of the VAE (refer to
    formulation in notebook).

    Inputs:
    - x_hat: Reconstruced input data of shape (N, 1, H, W)
    - x: Input data for this timestep of shape (N, 1, H, W)
    - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space
      dimension
    - logvar: Matrix representing estimated variance in log-space (N, Z), with Z
      latent space dimension

    Returns:
    - loss: Tensor containing the scalar loss for the negative variational
      lowerbound
    """
    loss = None
    ###############################################################################
    # TODO: Compute negative variational lowerbound loss as described in the      #
    # notebook                                                                    #
    ###############################################################################
    # Replace "pass" statement with your code
    D_kl=torch.sum((mu**2+torch.exp(logvar)-logvar-1)/2)
    E=F.binary_cross_entropy(x_hat,x,reduction="sum")
    #逐元素计算BCE (input,target)
    # reduction="sum"代表将张量中所有元素的BCE结果求和,
    # 默认是"mean",相当于sum后除以张量的元素数量,取平均值
    loss = (D_kl+E)/x.shape[0]
    """
    notes:
    1.p(x|z)各分量的分布建模为,每个pixel相互独立地遵循Bernoulli分布,
    从而一张图片x的损失是每个pixel的BCE的总和 
    Loss_per_sample = ∑i Binary_cross_entropy(pixel_i,real_pixel_i)
    整个训练集的损失=所有图片损失的总和/batch_size(回忆DL的建模过程,平均值作为期望的无偏估计)
    2.
    由于q(z|x)与p(z)分别是对角高斯分布与单位高斯分布,各分量独立,
    因此联合分布KL散度=各分量边缘分布KL散度之和
    D_{KL} (q(z|x)||p(z))=∑_{i=1}^n D_{KL} (q_i(z_i|x) || p_i(z_i))
    而各分量边缘分布KL可以通过高斯分布的数字特征计算出来(无需用pdf的积分计算)
    """

    ###############################################################################
    #                            END OF YOUR CODE                                 #
    ###############################################################################
    return loss

import torch

# Type hints.
from typing import List, Tuple
from torch import Tensor


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up on Google Colab.
    """
    print('Hello from pytorch101.py!')
    #print('自动看见这行说明reload成功了')
    """
    使用方法:先加载完reload相关的cell;先正常运行一次包含
    from pytorch101 import hello 
    hello()
    的cell;然后打开该文件,取消掉第二行print注释,ctrl+s保存
    （判断修改文件后是否被保存,可以观察文件名前面是否有*；
    有*则说明还未被保存）
    再直接重新运行该cell,如果成功打印,则成功reload
    """


def create_sample_tensor() -> Tensor:
    """
    Return a torch Tensor of shape (3, 2) which is filled with zeros, except
    for element (0, 1) which is set to 10 and element (1, 0) which is set to
    100.

    Returns:
        Tensor of shape (3, 2) as described above.
    """
    x = None
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    x=torch.tensor([[0,10],[100,0],[0,0]])
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return x


def mutate_tensor(
    x: Tensor, indices: List[Tuple[int, int]], values: List[float]
) -> Tensor:
    """
    Mutate the tensor x according to indices and values. Specifically, indices
    is a list [(i0, j0), (i1, j1), ... ] of integer indices, and values is a
    list [v0, v1, ...] of values. This function should mutate x by setting:

    x[i0, j0] = v0
    x[i1, j1] = v1

    and so on.

    If the same index pair appears multiple times in indices, you should set x
    to the last one.

    Args:
        x: A Tensor of shape (H, W)
        indices: A list of N tuples [(i0, j0), (i1, j1), ..., ]
        values: A list of N values [v0, v1, ...]

    Returns:
        The input tensor x
    """
    ##########################################################################
    #                     TODO: Implement this function                      #
    ##########################################################################
    # Replace "pass" statement with your code
    for i in range(len(indices)):
        x[indices[i]]=values[i]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def count_tensor_elements(x: Tensor) -> int:
    """
    Count the number of scalar elements in a tensor x.

    For example, a tensor of shape (10,) has 10 elements; a tensor of shape
    (3, 4) has 12 elements; a tensor of shape (2, 3, 4) has 24 elements, etc.

    You may not use the functions torch.numel or x.numel. The input tensor
    should not be modified.

    Args:
        x: A tensor of any shape

    Returns:
        num_elements: An integer giving the number of scalar elements in x
    """
    num_elements = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    #   You CANNOT use the built-in functions torch.numel(x) or x.numel().   #
    ##########################################################################
    # Replace "pass" statement with your code
    shape=x.shape
    num_elements=1
    for dim in shape:
        num_elements*=dim
    """notes:
    1.x.numel()返回x的元素数量
    2.当x是0维时,shape为空,for循环无效,num_elements=1仍正确
    """
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return num_elements


def create_tensor_of_pi(M: int, N: int) -> Tensor:
    """
    Returns a Tensor of shape (M, N) filled entirely with the value 3.14

    Args:
        M, N: Positive integers giving the shape of Tensor to create

    Returns:
        x: A tensor of shape (M, N) filled with the value 3.14
    """
    x = None
    ##########################################################################
    #         TODO: Implement this function. It should take one line.        #
    ##########################################################################
    # Replace "pass" statement with your code
    x=torch.full((M,N),3.14)
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def multiples_of_ten(start: int, stop: int) -> Tensor:
    """
    Returns a Tensor of dtype torch.float64 that contains all of the multiples
    of ten (in order) between start and stop, inclusive. If there are no
    multiples of ten in this range then return an empty tensor of shape (0,).

    Args:
        start: Beginning ot range to create.
        stop: End of range to create (stop >= start).

    Returns:
        x: float64 Tensor giving multiples of ten between start and stop
    """
    assert start <= stop
    x = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    real_start = start + (10 - start % 10) % 10
    if real_start>stop:
        x=torch.zeros(0,dtype=torch.float64)
    else:
        x = torch.arange(real_start, stop + 1, step=10, dtype=torch.float64)

    """notes:
    1.原本实现:list comprehension+tensor 创建
    对于空列表,自动创建shape=(0,)的tensor
    lst=[n for n in range(start,stop+1) if n%10==0]
    x=torch.tensor(lst,dtype=torch.float64)
    2.real_start=start + (10-start % 10) % 10
    x=torch.arange(real_start,stop+1,step=10,dtype=torch.float64)
    这个写法在real_start>stop时arange函数会报错;
    需要先进行一个条件判断
    
    """
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def slice_indexing_practice(x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given a two-dimensional tensor x, extract and return several subtensors to
    practice with slice indexing. Each tensor should be created using a single
    slice indexing operation.

    The input tensor should not be modified.

    Args:
        x: Tensor of shape (M, N) -- M rows, N columns with M >= 3 and N >= 5.

    Returns:
        A tuple of:
        - last_row: Tensor of shape (N,) giving the last row of x. It should be
          a one-dimensional tensor.
        - third_col: Tensor of shape (M, 1) giving the third column of x. It
          should be a two-dimensional tensor.
        - first_two_rows_three_cols: Tensor of shape (2, 3) giving the data in
          the first two rows and first three columns of x.
        - even_rows_odd_cols: Two-dimensional tensor containing the elements in
          the even-valued rows and odd-valued columns of x.
    """
    assert x.shape[0] >= 3
    assert x.shape[1] >= 5
    last_row = None
    third_col = None
    first_two_rows_three_cols = None
    even_rows_odd_cols = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    last_row=x[-1]
    third_col=x[:,2]
    first_two_rows_three_cols=x[:2,:3]
    even_rows_odd_cols=x[0::2,1::2]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    out = (
        last_row,
        third_col,
        first_two_rows_three_cols,
        even_rows_odd_cols,
    )
    return out


def slice_assignment_practice(x: Tensor) -> Tensor:
    """
    Given a two-dimensional tensor of shape (M, N) with M >= 4, N >= 6, mutate
    its first 4 rows and 6 columns so they are equal to:

    [0 1 2 2 2 2]
    [0 1 2 2 2 2]
    [3 4 3 4 5 5]
    [3 4 3 4 5 5]

    Note: the input tensor shape is not fixed to (4, 6).

    Your implementation must obey the following:
    - You should mutate the tensor x in-place and return it
    - You should only modify the first 4 rows and first 6 columns; all other
      elements should remain unchanged
    - You may only mutate the tensor using slice assignment operations, where
      you assign an integer to a slice of the tensor
    - You must use <= 6 slicing operations to achieve the desired result

    Args:
        x: A tensor of shape (M, N) with M >= 4 and N >= 6

    Returns:
        x
    """
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    """notes:
    根据要求,每次只能使用一个整数来赋值,0~5共6个整数
    """
    x[:2,0]=0
    x[:2,1]=1
    x[:2,2:6]=2
    x[2:4,0:4:2]=3
    x[2:4,1:5:2]=4
    x[2:4,4:6]=5

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return x


def shuffle_cols(x: Tensor) -> Tensor:
    """
    Re-order the columns of an input tensor as described below.

    Your implementation should construct the output tensor using a single
    integer array indexing operation. The input tensor should not be modified.

    Args:
        x: A tensor of shape (M, N) with N >= 3

    Returns:
        A tensor y of shape (M, 4) where:
        - The first two columns of y are copies of the first column of x
        - The third column of y is the same as the third column of x
        - The fourth column of y is the same as the second column of x
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    y=x[:,[0,0,2,1]]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def reverse_rows(x: Tensor) -> Tensor:
    """
    Reverse the rows of the input tensor.

    Your implementation should construct the output tensor using a single
    integer array indexing operation. The input tensor should not be modified.

    Your implementation may not use torch.flip.

    Args:
        x: A tensor of shape (M, N)

    Returns:
        y: Tensor of shape (M, N) which is the same as x but with the rows
            reversed - the first row of y should be equal to the last row of x,
            the second row of y should be equal to the second to last row of x,
            and so on.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    y=x[torch.arange(x.shape[0]-1,-1,-1)]
    """notes:
    1.不要混淆list的切片与range/torch.arange()
    range()/arange(start,stop,step)是整数序列,即start,stop都可以是负数
    且此时负数就是指负数本身的含义,而不是"倒数第几个数",因为它不代表序列的索引
    
    对于list lst, lst[::-1]可以反转序列,
    是因为此时(step<0)默认start=len(lst)-1,stop=-len(lst)-1,
    即从序列最后一个元素开始往回走,同样不包含stop,包含start
    """
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def take_one_elem_per_col(x: Tensor) -> Tensor:
    """
    Construct a new tensor by picking out one element from each column of the
    input tensor as described below.

    The input tensor should not be modified, and you should only access the
    data of the input tensor using a single indexing operation.

    Args:
        x: A tensor of shape (M, N) with M >= 4 and N >= 3.

    Returns:
        A tensor y of shape (3,) such that:
        - The first element of y is the second element of the first column of x
        - The second element of y is the first element of the second column of x
        - The third element of y is the fourth element of the third column of x
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    """notes:
    1.第n个元素(日常语义,即1-based),索引为n-1
    2.返回值y的形状为(3,)=>可以取一维整数数组索引
    """
    y=x[[1,0,3],[0,1,2]]
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def make_one_hot(x: List[int]) -> Tensor:
    """
    Construct a tensor of one-hot-vectors from a list of Python integers.

    Your implementation should not use any Python loops (including
    comprehensions).

    Args:
        x: A list of N integers

    Returns:
        y: Tensor of shape (N, C) and where C = 1 + max(x) is one more than the
            max value in x. The nth row of y is a one-hot-vector representation
            of x[n]; in other words, if x[n] = c then y[n, c] = 1; all other
            elements of y are zeros. The dtype of y should be torch.float32.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    c=max(x)+1
    n=len(x)
    y=torch.zeros(n,c,dtype=torch.float32)
    y[torch.arange(n),x]=1
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def sum_positive_entries(x: Tensor) -> Tensor:
    """
    Return the sum of all the positive values in the input tensor x.

    For example, given the input tensor

    x = [[ -1   2   0 ],
         [  0   5  -3 ],
         [  8  -9   0 ]]

    This function should return sum_positive_entries(x) = 2 + 5 + 8 = 15

    Your output should be a Python integer, *not* a PyTorch scalar.

    Your implementation should not modify the input tensor, and should not use
    any explicit Python loops (including comprehensions). You should access
    the data of the input tensor using only a single comparison operation and a
    single indexing operation.

    Args:
        x: A tensor of any shape with dtype torch.int64

    Returns:
        pos_sum: Python integer giving the sum of all positive values in x
    """
    pos_sum = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    mask=x>0
    pos_sum=torch.sum(x[mask]).item()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return pos_sum


def reshape_practice(x: Tensor) -> Tensor:
    """
    Given an input tensor of shape (24,), return a reshaped tensor y of shape
    (3, 8) such that

    y = [[x[0], x[1], x[2],  x[3],  x[12], x[13], x[14], x[15]],
         [x[4], x[5], x[6],  x[7],  x[16], x[17], x[18], x[19]],
         [x[8], x[9], x[10], x[11], x[20], x[21], x[22], x[23]]]

    You must construct y by performing a sequence of reshaping operations on
    x (view, t, transpose, permute, contiguous, reshape, etc). The input
    tensor should not be modified.

    Args:
        x: A tensor of shape (24,)

    Returns:
        y: A reshaped version of x of shape (3, 8) as described above.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    """
    view(2,3,4) 两个矩阵
    fancy indexing: 0,3;1,4;2,5
    y=x.view(2,3,4)
    y=y[:,[0,3,1,4,2,5]]
    """
    pass

    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def zero_row_min(x: Tensor) -> Tensor:
    """
    Return a copy of the input tensor x, where the minimum value along each row
    has been set to 0.

    For example, if x is:
    x = torch.tensor([
          [10, 20, 30],
          [ 2,  5,  1]])

    Then y = zero_row_min(x) should be:
    torch.tensor([
        [0, 20, 30],
        [2,  5,  0]
    ])

    Your implementation shoud use reduction and indexing operations. You should
    not use any Python loops (including comprehensions). The input tensor
    should not be modified.

    Args:
        x: Tensor of shape (M, N)

    Returns:
        y: Tensor of shape (M, N) that is a copy of x, except the minimum value
            along each row is replaced with 0.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    y=torch.clone(x)
    indices=x.argmin(dim=1)
    y[torch.arange(y.shape[0]),indices]=0
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def batched_matrix_multiply(
    x: Tensor, y: Tensor, use_loop: bool = True
) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    Depending on the value of use_loop, this calls to either
    batched_matrix_multiply_loop or batched_matrix_multiply_noloop to perform
    the actual computation. You don't need to implement anything here.

    Args:
        x: Tensor of shape (B, N, M)
        y: Tensor of shape (B, M, P)
        use_loop: Whether to use an explicit Python loop.

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    if use_loop:
        return batched_matrix_multiply_loop(x, y)
    else:
        return batched_matrix_multiply_noloop(x, y)


def batched_matrix_multiply_loop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    This implementation should use a single explicit loop over the batch
    dimension B to compute the output.

    Args:
        x: Tensor of shaper (B, N, M)
        y: Tensor of shape (B, M, P)

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    # Replace "pass" statement with your code
    """notes:
    仅需要一个特定形状的张量z存储结果时,使用torch.empty()
    用法与torch.zeros,torch.ones(),torch.rand(),torch.randn()等构造函数类似;
    可以指定dtype
    区别在于,torch.empty()只分配内存,不初始化(值随机);而其它上述函数既分配内存又初始化值
    因此torch.empty()会更快一些
    """
    B,N,_=x.shape
    P=y.shape[2]
    z=torch.empty(B,N,P,dtype=x.dtype)
    for i in range(B):
        z[i]=torch.mm(x[i],y[i])

    ###########################################################################
    #                           END OF YOUR CODE                              #
    ###########################################################################
    return z


def batched_matrix_multiply_noloop(x: Tensor, y: Tensor) -> Tensor:
    """
    Perform batched matrix multiplication between the tensor x of shape
    (B, N, M) and the tensor y of shape (B, M, P).

    This implementation should use no explicit Python loops (including
    comprehensions).

    Hint: torch.bmm

    Args:
        x: Tensor of shaper (B, N, M)
        y: Tensor of shape (B, M, P)

    Returns:
        z: Tensor of shape (B, N, P) where z[i] of shape (N, P) is the result
            of matrix multiplication between x[i] of shape (N, M) and y[i] of
            shape (M, P). The output z should have the same dtype as x.
    """
    z = None
    ###########################################################################
    #                      TODO: Implement this function                      #
    ###########################################################################
    # Replace "pass" statement with your code
    #NotImplementedError?
    #z=torch.bmm(x,y,out_dtype=x.dtype)
    B,N,_=x.shape
    P=y.shape[2]
    z=torch.empty(B,N,P,dtype=x.dtype)
    torch.bmm(x,y,out=z)
    """notes:
    签名:
    torch.bmm(input, mat2, out_dtype=None, *, out=None)
    input,mat2相当于本题中的x,y.
    out_dtype和out是可选参数
    e.g. torch.bmm(x,y,out=z)
    (0)该函数不会进行广播,因此形状必须严格准确
    (1) out
    out代表结果所存储的的对象;准确地说,会原地修改out所绑定的对象为计算结果.  *
    z.shape必须与结果相同;且dtype与结果的dtype兼容,可以通过设定out_dtype来确保这一点.
    out的默认值为None;此时函数会返回一个存储计算结果的新对象,
    因此result=torch.bmm(x,y)可以得到计算结果.
    以下代码是无法达到预期效果的(不能让z得到计算结果):
    z=None torch.bmm(x,y,out=z)  因为此时等价于out=None   (根据*来理解,区分名字和对象)
    torch.bmm(x,y,out=z)虽然返回了计算结果,但是没有将其与任何变量绑定.
    
    (2) out_dtype 不建议使用;专门为 GPU 混合精度计算设计的参数,有严格限制
        out_dtype (dtype, optional) – the dtype of the output tensor, 
        Supported only on CUDA and for torch.float32 (只允许指定输出dtype为torch.float32)
        given torch.float16/torch.bfloat16 input dtypes (限制输入dtype)
    两种方式:
    (1)先预分配内存,让z绑定到合适形状的张量后,再使用out=z
        z=torch.empty(B,N,P)
        torch.bmm(x,y,out_dtype=x.dtype,out=z)
    (2)直接将名字z绑定到返回值(默认参数out=None)
        z=torch.bmm(x,y)
        z.to(x.dtype)  #类型转换以符合本题要求
    注意:
        z=torch.bmm(x,y,out_dtype=x.dtype)等使用out_dtype=x.dtype的实现方式,
        在本题中会报NotImplementedError,大意是指CPU backend没有相应的算子实现
        因为违反了out_dtype的限制:
        (1)在CPU上运行; out_dtype只支持CUDA等backend.
        (2)潜在报错原因:
            Pytorch中默认类型为torch.float32,
            如果输入x,y的dtype为torch.float32,那么不符合输入限制
    
    """
    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
    return z


def normalize_columns(x: Tensor) -> Tensor:
    """
    Normalize the columns of the matrix x by subtracting the mean and dividing
    by standard deviation of each column. You should return a new tensor; the
    input should not be modified.

    More concretely, given an input tensor x of shape (M, N), produce an output
    tensor y of shape (M, N) where y[i, j] = (x[i, j] - mu_j) / sigma_j, where
    mu_j is the mean of the column x[:, j].

    Your implementation should not use any explicit Python loops (including
    comprehensions); you may only use basic arithmetic operations on tensors
    (+, -, *, /, **, sqrt), the sum reduction function, and reshape operations
    to facilitate broadcasting. You should not use torch.mean, torch.std, or
    their instance method variants x.mean, x.std.

    Args:
        x: Tensor of shape (M, N).

    Returns:
        y: Tensor of shape (M, N) as described above. It should have the same
            dtype as the input x.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    M=x.shape[0]
    sums=torch.sum(x,dim=0)
    means=sums/M
    stds=torch.sum((x-means)**2,dim=0) #unnormalized
    stds/=(M-1)  #normalized
    stds=torch.sqrt(stds)  #square root
    y=(x-means)/stds
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def mm_on_cpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on CPU.

    You don't need to implement anything for this function.

    Args:
        x: Tensor of shape (A, B), on CPU
        w: Tensor of shape (B, C), on CPU

    Returns:
        y: Tensor of shape (A, C) as described above. It should not be in GPU.
    """
    y = x.mm(w)
    return y


def mm_on_gpu(x: Tensor, w: Tensor) -> Tensor:
    """
    Perform matrix multiplication on GPU.

    Specifically, given two input tensors this function should:
    (1) move each input tensor to the GPU;
    (2) perform matrix multiplication between the GPU tensors;
    (3) move the result back to CPU

    When you move the tensor to GPU, use the "your_tensor.cuda()" operation

    Args:
        x: Tensor of shape (A, B), on CPU
        w: Tensor of shape (B, C), on CPU

    Returns:
        y: Tensor of shape (A, C) as described above. It should not be in GPU.
    """
    y = None
    ##########################################################################
    #                      TODO: Implement this function                     #
    ##########################################################################
    # Replace "pass" statement with your code
    x_gpu=x.cuda()
    w_gpu=w.cuda()
    y = x_gpu.mm(w_gpu)
    y = y.cpu()
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_mean_tensors(xs: List[Tensor], ls: Tensor) -> Tensor:
    """
    Compute mean of each tensor in a given list of tensors.

    Specifically, the input is a list of N tensors, (1 <= N <= 10000). The i-th
    tensor in this list has length Ki, (1 <= Ki <= 10000). Return a tensor of
    shape (N, ) whose i-th element gives the mean of i-th tensor in input list.
    You may assume that all tensors are on the same device (CPU or GPU).

    Your implementation should not use any explicit Python loops (including
    comprehensions).

    Args:
        xs: List of N 1-dimensional tensors.
        ls: Length of tensors in `xs`. An int64 Tensor of same length as `xs`
            with `ls[i]` giving the length of `xs[i]`.

    Returns:
        y: Tensor of shape (N, ) with `y[i]` giving the mean of `xs[i]`.
    """

    y = None
    ##########################################################################
    # TODO: Implement this function without using `for` loops and store the  #
    # mean values as a tensor in `y`.                                        #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return y


def challenge_get_uniques(x: torch.Tensor) -> Tuple[Tensor, Tensor]:
    """
    Get unique values and first occurrence from an input tensor.

    Specifically, the input is 1-dimensional int64 Tensor with length N. This
    tensor contains K unique values (not necessarily consecutive). Your
    implementation must return two tensors:
    1. uniques: int64 Tensor of shape (K, ) - giving K uniques from input.
    2. indices: int64 Tensor of shape (K, ) - giving indices of the first
       occurring unique values.

    Concretely, this should hold True: x[indices[i]] = uniques[i] 

    Your implementation should not use any explicit Python loops (including
    comprehensions), and should not require more than O(N) memory. Creating
    new tensors larger than input tensor is not allowed. If you wish to
    create new tensors like input tensor, use `input.clone()`.

    You may use `torch.unique`, but you will receive half credit for that.

    Args:
        x: Tensor of shape (N, ) with K <= N unique values.

    Returns:
        uniques and indices: Se description above.
    """

    uniques, indices = None, None
    ##########################################################################
    # TODO: Implement this function without using `for` loops and within     #
    # O(N) memory.                                                           #
    ##########################################################################
    # Replace "pass" statement with your code
    pass
    ##########################################################################
    #                            END OF YOUR CODE                            #
    ##########################################################################
    return uniques, indices

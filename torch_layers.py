


# Python file that contains various functions, coded
# using functionalities of the torch.tensor class. 
# These function are 
# - softmax
# - cross_entropy
# - reframe_1d: essential step in the 1d convolution
# - reframe_2d: essential step in the 2d convolution
# - Conv2d
# - pos_encoding
# - LayerNorm
# - masked_multi_head_attention
# - DecoderBlock: a decoder block for a transformer with masked attention



import torch
import numpy as np


class softmax(torch.nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        """softmax applied to tensor x along the dim dimension, we implement 
        the usual regularization technics that substracts the max along dim dimension,
        -- implemented using pytorch functionalities"""
        max_x = torch.max(x, dim = self.dimension, keepdim=True).values
        soft =  torch.exp(x - max_x) / torch.exp(
            x - max_x).sum(dim = self.dimension, keepdim=True)
        return soft


def cross_entropy(x, y):
    """x is a (n_mini_batch, n_classes) tensors, while y is a (n_minibatch,) tensor
       whose entries are the values of the class of each element of the mini_batch
       -- implemented using pytorch functionalities"""
    y_one_hot = torch.nn.functional.one_hot(y, x.size(1)).float()
    return - y_one_hot.matmul(torch.log(softmax(x, 1)).t()).diagonal()



class pos_encoding(torch.nn.Module):
    """positional encoding: N controls the frequencies, d_model is the input
       size dimension"""
    def __init__(self, N, d_model):
        super().__init__()
        self.N = N
        self.d_model = d_model

    def forward(self, t):
        """t is a (dim_t,) tensor, the call returns a dim (dim_t, d_model) tensor"""
        position = torch.zeros(t.size()[0], self.d_model)
        for k in range(0, int(np.floor(self.d_model / 2)) + 1):
            omega = 1 / self.N**(2*((k + 1)/self.d_model))
            if 2*k <= self.d_model - 1:
                position[:, 2*k] = torch.cos(t*omega)
            if 2*k + 1 <= self.d_model - 1:
                position[:, 2*k + 1] = torch.sin(t*omega)
        return position
    

class LayerNorm(torch.nn.Module):
    """
    Defines a layer norm
    Args:
            - in: torch.tensor(batch_size, seq_len, dim)
    Returns:
            - out: torch.tensor(batch_size, seq_len, dim)
    The normalization is along dimension = dim     
    """
    def __init__(self, dimension, d_model):
        super().__init__()
        self.dim = dimension
        self.d_model = d_model
        self.gamma = torch.nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.beta = torch.nn.Parameter(
            torch.zeros(self.d_model, dtype=torch.float32))

    def forward(self, x):  
        eps = 1e-05  
        mean = x.mean(dim=self.dim, keepdim=True)
        sigma = torch.sqrt(
            x.var(dim=self.dim, keepdim=True, correction=0) + eps)
        if x.shape[-1] == 1:
            return x * self.gamma + self.beta
        else:
            return ((x - mean) / sigma) * self.gamma + self.beta
        

class masked_multi_head_attention(torch.nn.Module):
    """
    Defines a masked multi head attention that acts on an input a tensor 
    x: torch.tensor(batch_size, seq_len, d_model) and ouputs a tensor 
    out: torch.tensor(batch_size, seq_len, d_model)
    We impose headsize = d_model / num_heads. This is to make sure the FLOP cost
    is the same as a single headed attention with headsize = d_model.
    
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert self.d_model % self.num_heads == 0
        self.headsize = d_model // self.num_heads
        self.lq = torch.nn.Linear(self.d_model, self.d_model)
        self.lk = torch.nn.Linear(self.d_model, self.d_model)
        self.lv = torch.nn.Linear(self.d_model, self.d_model)
        self.lout = torch.nn.Linear(self.d_model, self.d_model)

    def mask(self, x):
        """
        Args:
            - any N-dim torch tensor with N >= 2
        Returns:
            - out: lower triangular part of the tensor, w.r.t last two indices,
                   other indices are set to -1e9. 
        """
        mask = torch.triu(torch.ones(x.size(), dtype=bool), diagonal=1)
        x[mask] = - 1e9
        return x

    def masked_attention(self, q, k, v):
        """
        Args:
            - q: torch.tensor(batch_size, num_heads, seq_len, headsize)
            - k: torch.tensor(batch_size, num_heads, seq_len, headsize)
            - v: torch.tensor(batch_size, num_heads, seq_len, headsize)
        Returns:
            - out: torch.tensor(batch_size, num_heads, seq_len, headsize)
        """
        # att_matrix: torch.tensor(batch_size, num_heads, seq_len, seq_len)
        att_matrix = torch.matmul(q, k.transpose(-1, -2))
        headsize = q.shape[-1]
        att_matrix = att_matrix / torch.sqrt(torch.tensor(headsize)).item()
        att_matrix = self.mask(att_matrix)
        weights = softmax(-1)(att_matrix)
        return torch.matmul(weights, v)

    def forward(self, x):
        q = self.lq(x)
        k = self.lk(x)
        v = self.lv(x)
        batch_size, seq_length, d_model = x.shape[0], x.shape[1], x.shape[2]
        q = q.reshape(
            *q.size()[:-1], self.num_heads, self.headsize).transpose(-3, -2)
        k = k.reshape(
            *k.size()[:-1], self.num_heads, self.headsize).transpose(-3, -2)
        v = v.reshape(
            *v.size()[:-1], self.num_heads, self.headsize).transpose(-3, -2)
        out = self.masked_attention(q, k, v)
        out = out.transpose(-3, -2)
        out = out.reshape(*out.size()[:-2], d_model)
        out = self.lout(out)
    
        return out
    

class DecoderBlock(torch.nn.Module):
    def __init__(self, num_heads, d_model, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.mmh = masked_multi_head_attention(self.d_model, self.num_heads)
        self.LN1 = LayerNorm(-1, self.d_model)
        self.LN2 = LayerNorm(-1, self.d_model)
        self.linear1 = torch.nn.Linear(self.d_model, 4*self.d_model)
        self.linear2 = torch.nn.Linear(4*self.d_model, self.d_model)
        self.ReLu = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(dropout)
    
    def forward(self, x):
        """
        x: torch.tensor(batch_size, seq_len, d_model)
        """
        #masked multi head attention + add and norm
        x = self.LN1(self.mmh(x) + x)
        #Feed forward
        x = self.LN2(x + self.linear2(self.ReLu(self.linear1(self.drop(x)))))
        return x
    

class Transformer(torch.nn.Module):
    def __init__(self, N, num_heads, d_model, n_blocks, vocab_size, dropout):
        super().__init__()
        self.N = N
        self.num_heads = num_heads
        self.d_model = d_model 
        self.n_blocks = n_blocks
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_enc = pos_encoding(N, d_model)
        self.DecoderBlocks = torch.nn.Sequential(
            *[DecoderBlock(num_heads, d_model, dropout) for k in range(n_blocks)]
        )
        self.LN = LayerNorm(-1, d_model)
        self.linear_output = torch.nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, x):
        """
        x: torch.tensor(batch_size, seq_length)
        elements of x are integers in range(vocab_size)
        out: torch.tensor(batch_size, seq_length, vocab_size)
        ouput = proba over vocabulary
        """
        
        seq_length = x.size()[-1]
        # Embedding
        x = self.embed(x) # output: torch.tensor(batch_size, seq_length, d_model)
        # Positional encoding
        p = self.pos_enc(torch.arange(0, seq_length))
        x = x + p
        # Application n_blocks * (attention + feed forward)
        x = self.DecoderBlocks(x)
        # Layer norm
        x = self.LN(x)
        # Linear layer d_model -> vocab_size
        x = self.linear_output(x)
        return x
    

    


        



import time
from typing import Dict, List, Tuple
import jax
import jax.numpy as jnp
import optax
from jax import value_and_grad
from flax import nnx
import numpy as np
import matplotlib.pyplot as plt # type: ignore
from datetime import datetime
import argparse


## load and preprocess the dataset
class Processor():

    def __init__(
            self,
            path: str = 'input.txt'
        ):

        with open(path,'r',encoding='utf-8') as f:
            self.data = f.read()

        # print('-- Dataset preview first 100 characters--')
        # print()
        # print(self.data[:100])

        ## get a list of unique characters in the dataset for forming character level token
        chars = sorted(list(set(self.data)))
        print(f'vocab size {len(chars)}')

        ## map characters to integrs
        self.char_to_int = {v:idx for idx,v in enumerate(chars)}
        ## map int to chars
        self.int_to_char = {idx:v for idx,v in enumerate(chars)}

    ## encode the text string into 
    def encode(
            self,
            text:str
    ):
        
        return [self.char_to_int[s] for s in text]
    
    def decode(
            self,
            tokens : list
    ):
        return ''.join([self.int_to_char[t] for t in tokens])

    
    def get_batch(
        self,
        prng : jax.random.key = jax.random.key(42),
        batch_size : int = 4,
        block_size : int = 8
        ):

        ## encode the dataset into integers
        encoded_data = jnp.array(self.encode(self.data), dtype= jnp.int16)
        ## get train/val split
        train_split = int(len(encoded_data) * 0.9)
        train_data = encoded_data[:train_split]
        val_data = encoded_data[train_split:]

        ## get the starting indices of random batches
        train_idx = jax.random.randint(prng,(batch_size,),0,len(train_data)-block_size)
        val_idx = jax.random.randint(prng,(batch_size,),0,len(val_data)-block_size)

        
        x_train = jnp.vstack([train_data[idx:idx+block_size] for idx in train_idx])
        y_train = jnp.vstack([train_data[idx+1:idx+block_size+1] for idx in train_idx])
        x_val = jnp.vstack([val_data[idx:idx+block_size] for idx in val_idx])
        y_val = jnp.vstack([val_data[idx+1:idx+block_size+1] for idx in val_idx])

        return x_train,y_train,x_val,y_val

    def generate_text(
        self,
        model : jnp.array, 
        seq_len : int =100
        ):

        key = jax.random.key(int(time.time()))
        key,key2 = jax.random.split(key)
        sos_token = jax.random.randint(key=key2,shape=(1,),minval=0,maxval=len(self.char_to_int)).item()
        text_tokens = [sos_token]
        next_token = sos_token
        for i in range(seq_len):
            # get the raw logits prediction
            # apply softmax to convert into probabilty distribution and get the next token pred sampled from the distribution
            key,key3 = jax.random.split(key)
            prob = jax.nn.softmax(model[next_token],axis=0)
            next_token = jax.random.choice(key=key3,a=len(prob),p=prob).item()
            text_tokens.append(next_token)

        return self.decode(text_tokens)
    
    
def embedding_model(
        key : jax.random,
        vocab_size : int = 65,
        embed_size: int = 128,
        ) :
    
    model = jax.random.normal(key,shape=(vocab_size,embed_size))
    return model

def positional_encoding(
        key: jax.random,
        seq_len: int = 8,
        embed_size: int = 128
) -> jnp.array :
    
    pos_encoding = jax.random.normal(key,shape=(seq_len,embed_size))

    return pos_encoding

def get_qkv_projections(
        q_key : jax.random.key,
        k_key : jax.random.key,
        v_key : jax.random.key,
        embed_size: int,
        attn_head_size : int, 
        params : dict = None,
) -> Dict :

    # w_q = jax.random.normal(q_key,(embed_size,attn_head_size))
    # w_k = jax.random.normal(k_key,(embed_size,attn_head_size))
    # w_v = jax.random.normal(v_key,(embed_size,attn_head_size))

    if params:
        w_q, w_k, w_v = params['w_q'], params['w_k'], params['w_v']
    else:
    # moving from normal to uniform distribution
        d_dist = 1.0 / jnp.sqrt(embed_size)
        w_q = jax.random.uniform(q_key, (embed_size, attn_head_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
        w_k = jax.random.uniform(k_key, (embed_size, attn_head_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
        w_v = jax.random.uniform(v_key, (embed_size, attn_head_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
        params = {
            'w_q' : w_q,
            'w_k' : w_k,
            'w_v' : w_v
        }

    return params


def get_multihead_projections(
    key: jax.random.key,
    embed_size: int,
    number_of_heads: int
) -> Tuple:
    
    attn_head_size = embed_size // number_of_heads
    projections = []
    
    # First split the main key into number_of_heads subkeys
    keys = jax.random.split(key, num=number_of_heads+1)
    
    for i in range(number_of_heads):
        # For each head, split the key into 3 subkeys for q, k, v
        head_key = keys[i]
        q_key, k_key, v_key = jax.random.split(head_key, num=3)
        
        # Get projections for this head
        head_projections = get_qkv_projections(q_key, k_key, v_key, embed_size, attn_head_size)
        projections.append(head_projections)

    d_dist = 1.0 / jnp.sqrt(embed_size)
    w_m = jax.random.uniform(keys[-1], (embed_size, embed_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_m = jax.random.uniform(keys[-1], (embed_size,), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    
    return projections, w_m, bias_m



def single_head_attention(
        x_embed : jnp.array, # shape sequence_length, embedding_size
        params : dict ,
) -> jnp.array :
    
    w_q, w_k, w_v = params['w_q'], params['w_k'], params['w_v']
    q = jnp.matmul(x_embed,w_q)  # shape sequence_length, attn_head_size
    k = jnp.matmul(x_embed,w_k)  # shape sequence_length, attn_head_size
    v = jnp.matmul(x_embed,w_v)  # shape sequence_length, attn_head_size

    wei = jnp.matmul(q,k.T) # shape sequence_length, sequence_length
    #print(f'shape of weight matrix: {wei.shape}')
    attn_head_size = q.shape[1]
    wei = wei / jnp.sqrt(attn_head_size)

    seq_len = wei.shape[0]
    tril = jnp.tril(jnp.ones((seq_len,seq_len)))
    # ref - https://github.com/google/flax/discussions/2915
    big_neg = jnp.finfo(jnp.float32).min
    wei = jnp.where(tril,wei,big_neg)
    # print('weight matrix before softmax')
    # print(wei)
    wei = jax.nn.softmax(wei,axis=-1)
    # print('weight matrix after softmax')
    # print(wei)

    w_attn = jnp.matmul(wei,v) #shape sequence_length, attn_head_size
    # print('final attention weights matrix')
    # print(w_attn)

    return w_attn


def multihead_attention(
    x_embed : jnp.array, # shape sequence_length, embedding_size
    multihead_params : List ,
    w_m : jnp.array,
    bias_m: jnp.array
) -> Tuple :

    out = jnp.concat([single_head_attention(x_embed,params) for params in multihead_params],axis =1)
    out = jnp.matmul(out,w_m) + bias_m
    out = nnx.Dropout(0.5,rngs=nnx.Rngs(42))(out)

    return out, multihead_params, w_m, bias_m

 # calculate the layer norm
def layer_norm(
        x_embed : jnp.array, # shape sequence_length, embedding_size
        gamma : jnp.array, # scaling factor - learnable parameter
        beta : jnp.array, # shifting factor - learnable parameter
 ) -> Tuple :
    
    # takes an embedding matrix of sequence length, embedding_szie
    # calculates the mean for each token in the sequence,i.e, sum of all embed values for a single token divided by embed size
    # and the variance similarly 

    mean = jnp.expand_dims(jnp.mean(x_embed,axis=-1),-1)
    
    assert mean.shape == (x_embed.shape[0],1), f'check for the shape of mean in layernorm'
    var = jnp.expand_dims(jnp.var(x_embed,axis=-1),-1)
    assert var.shape == (x_embed.shape[0],1), f'check for the shape of var in layernorm'

    epsilon = 1e-8
    x = (x_embed - mean)/(jnp.sqrt(var + epsilon))
    #print(f'layer norm shapes x = {x.shape},gamma = {gamma.shape},beta = {beta.shape}')
    x = jnp.multiply(gamma,x) + beta
    assert x.shape == x_embed.shape, f'check for the shape of final matrix in layernorm'

    return x


# define helper function to get feedforward parameters 
def feedforward_weights_initialization(
        embed_size : int,
        linear_1_key: jax.random.key,
        linear_2_key: jax.random.key,
        ) :
    d_dist = 1.0 / jnp.sqrt(embed_size)
    linear_1 = jax.random.uniform(linear_1_key, (embed_size, 4*embed_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_1 = jax.random.uniform(linear_1_key, (4*embed_size,), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    linear_2 = jax.random.uniform(linear_2_key, (4*embed_size, embed_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_2 = jax.random.uniform(linear_2_key, (embed_size,), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    params ={
        'linear_1': linear_1,
        'bias_1': bias_1,
        'linear_2': linear_2,
        'bias_2': bias_2
    }

    return params

#define classification layer weights initialization
def classification_weights_initialization(
        embed_size : int,
        vocab_size: int,
        key: jax.random.key,
        ) :
    d_dist = 6.0 / jnp.sqrt(embed_size + vocab_size)
    linear_cls = jax.random.uniform(key, (embed_size, vocab_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_cls = jax.random.uniform(key, (vocab_size,), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    params ={
        'linear_cls': linear_cls,
        'bias_cls': bias_cls,
    }

    return params
    
# define the feed forward network 
def feed_forward_block(
    input_weights: jnp.array,
    params: Dict = None
    ) -> Tuple :

    if params:
        linear_1, bias_1, linear_2, bias_2 = params['linear_1'], params['bias_1'], params['linear_2'], params['bias_2']
    
    # make the feedforward calculations
    x = jnp.matmul(input_weights,linear_1) + bias_1
    x = jax.nn.relu(x)
    x = jnp.matmul(x,linear_2) + bias_2
    x = nnx.Dropout(0.5,rngs=nnx.Rngs(42))(x)

    return x, params
        


# define the transformers block
def block(
    x_embed: jnp.array,
    params: Dict
):
    x = layer_norm(x_embed,  
                     params['gamma_1'],
                     params['beta_1']
                     )
    self_attention,params['multihead_params'], params['w_m'], params['bias_m']  = multihead_attention(    
                                                        x,
                                                        params['multihead_params'],
                                                        params['w_m'],
                                                        params['bias_m']
                                                            ) 
    assert self_attention.shape == x_embed.shape, f'multihead attention out shape != x_embed shape'
    x = self_attention + x_embed
    x = layer_norm(x,  
                params['gamma_2'],
                params['beta_2']
                 )
    x, params = feed_forward_block(x,params)
    assert self_attention.shape == x_embed.shape, f'feed-forward layer out shape != x_embed shape'
    x = x + x_embed

    return x, params



 # ref used for implementing the embedding forwards pass :
# https://stackoverflow.com/questions/72817730/what-is-the-recommended-way-to-do-embeddings-in-jax
def get_model_params(embed_size: int,
            vocab_size: int,
            seq_len: int ,
            number_of_heads: int,
            ) -> Dict :
    
    # vmap to support batch operation in x_train 
    # ref : https://dinocausevic.com/2023/06/13/jax-vmap/
    # in_axes as  None means that axes will not be iterated over will stay fixed during the iteration process
    # add detailed comment
    params = {}
    params['embed_model'] = embedding_model(jax.random.key(0),vocab_size,embed_size)
    params['pos_encoding'] = positional_encoding(jax.random.key(1),seq_len,embed_size)

    key = jax.random.key(42)
    linear_1_key, linear_2_key, linear_cls_key = jax.random.split(key,num=3)
    params['multihead_params'], params['w_m'], params['bias_m'] = get_multihead_projections(key,embed_size,number_of_heads)
    
    params['gamma_1'] = params['gamma_2'] = params['gamma_cls'] = jnp.ones((seq_len,embed_size))
    params['beta_1'] = params['beta_2'] = params['beta_cls'] = jnp.zeros((seq_len,embed_size))

    feedforward_params = feedforward_weights_initialization(embed_size,linear_1_key,linear_2_key)
    params.update(feedforward_params)

    cls_params = classification_weights_initialization(embed_size,vocab_size,linear_cls_key)
    params.update(cls_params)

    return params

    
# ref used for implementing the embedding forwards pass :
# https://stackoverflow.com/questions/72817730/what-is-the-recommended-way-to-do-embeddings-in-jax
def transformer(
            x : jnp.array, # single x of shape (batch_size,block_size)
            params: Dict,
            ) -> jnp.array :
    
    # vmap to support batch operation in x_train 
    # ref : https://dinocausevic.com/2023/06/13/jax-vmap/
    # in_axes as  None means that axes will not be iterated over will stay fixed during the iteration process
    # add detailed comment
    #token_emb = jax.vmap(lambda embed_model,x: embed_model[x], in_axes=[None,0],out_axes=0)(params['embed_model'],x)
    vocab_size = params['embed_model'].shape[0]
    embed_size = params['embed_model'].shape[1]
    token_emb = params['embed_model'][x]
    output = token_emb + params['pos_encoding']

    #print(f'output vmapped embed model + pos embed -> shape = {output.shape}')
    assert output.shape == (x.shape[0],embed_size), f'transformer block -> token_embed + pos_emb layer out shape {output.shape} != (seq_len,embed_size) {(x.shape[0],embed_size)} '

    output, params = block(output,params)
    assert output.shape == (x.shape[0],embed_size), f'transformer block -> attention block layer out shape {output.shape} != (seq_len,embed_size) {(x.shape[0],embed_size)} '

    output = layer_norm(output,params['gamma_cls'],params['beta_cls'])
    assert output.shape == (x.shape[0],embed_size), f'transformer block -> classification layer normalization out shape {output.shape} != (seq_len,embed_size) {(x.shape[0],embed_size)} '

    output = jnp.matmul(output,params['linear_cls']) + params['bias_cls']
    assert output.shape == (x.shape[0],vocab_size), f'transformer block -> classification output layer shape {output.shape} != (seq_len,vocab_size) {(x.shape[0],vocab_size)} '


    return output



def compute_loss(
        x_train : jnp.array, # single x of shape (batch_size,block_size)
        targets : jnp.array, # shape: (batch_size*block_size)
        params: Dict
) :
    logits = jax.vmap(transformer,in_axes=[0,None])(x_train,params)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits,targets).mean()

    return loss

@jax.jit
def train_step(
    x_train : jnp.array,  # shape (batch,block)
    y_train : jnp.array, # shape (batch*block)
    params: Dict,
    lr : float = 1e-1,
    ) -> Tuple:

    # use jax value_and_grad to compute the gradients
    loss, grads = value_and_grad(compute_loss,argnums=2)(x_train,y_train,params)

    # update the model parameters
    params_update = jax.tree.map(lambda params, grads: params - lr*grads,params,grads)
    # return the loss and the updated embedding model array
    return loss,params_update

@jax.jit
def val_step(
    x_val : jnp.array,  # shape (batch,block)
    y_val : jnp.array, # shape (batch*block)
    params: Dict,
    ) -> Tuple:
    # use jax value_and_grad to compute the gradients
    loss = compute_loss(x_val,y_val,params)

    return loss

def train(steps=100):

    batch_size = 16 
    block_size = 32 
    max_iters = steps
    eval_interval = 5
    learning_rate = 1e-4
    eval_iters = 200
    embed_size = 64
    number_of_heads = 4
    vocab_size = 65

    # print(f'train input shape: {x_train.shape}, target shape: {y_train.shape}')
    # print(f'val input shape: {x_val.shape}, target shape: {y_val.shape}')
   
    params = get_model_params(embed_size=embed_size,
                              vocab_size=vocab_size,
                              seq_len=block_size,
                              number_of_heads=number_of_heads)


    processor = Processor()
    #print('Model Output with seqeunce length of 100 after initialization')
    #print(processor.generate_text(model,seq_len=1000))

    train_losses = []
    val_losses = []
    steps_list = []

    for step in range(max_iters): 
        x_train,y_train,x_val,y_val = processor.get_batch(batch_size=batch_size,block_size=block_size)
        loss, params = train_step(x_train,y_train,params,learning_rate)
        print(f'step: {step+1} || train loss: {loss}',flush=True)
        
        train_losses.append(loss)
        steps_list.append(step)

        if (step + 1) % eval_interval == 0:
            val_loss = val_step(x_val,y_val,params)
            print(f'step: {step+1} || val loss: {val_loss}',flush=True)
            val_losses.append(val_loss)

    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, train_losses, label='Training Loss')
    val_steps = steps_list[eval_interval-1::eval_interval]
    plt.plot(val_steps, val_losses, label='Validation Loss')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Save the plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'training_loss_{timestamp}.jpg')
    plt.close()


    

if __name__ == '__main__' :

    args = argparse.ArgumentParser()
    args.add_argument('--steps', type=int, default=100, help='number of training steps')
    args = args.parse_args()

    print('Start training',flush=True)
    train(args.steps)


import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt # type: ignore

import jax
from functools import partial
import jax.numpy as jnp

from optax import softmax_cross_entropy_with_integer_labels, adamw, apply_updates
from flax.nnx import Dropout, Rngs


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
    
    def generate_text_from_transfomer(
        self,
        params: Dict,
        num_tokens_to_generate: int = 100,
        number_of_layers: int = 4,
    ):
        model_block_size = params['pos_encoding'].shape[0]
        actual_vocab_size = len(self.char_to_int)

        key = jax.random.key(int(time.time()))
        key, sos_key, loop_key = jax.random.split(key, 3)

        sos_token = jax.random.randint(key=sos_key, shape=(1,), minval=0, maxval=actual_vocab_size).item()
        
        current_context_list = [0] * (model_block_size -1) + [sos_token]
        generated_tokens_list = []

        for _ in range(num_tokens_to_generate):
            loop_key, choice_key = jax.random.split(loop_key)
            
            input_sequence = jnp.array(current_context_list[-model_block_size:], dtype=jnp.int32)
            
            
            logits_all_positions = transformer(input_sequence, params, number_of_layers, training=False)
            
            next_token_logits = logits_all_positions[-1] 
            
            probabilities = jax.nn.softmax(next_token_logits, axis=-1)
            next_token = jax.random.choice(key=choice_key, a=actual_vocab_size, p=probabilities).item()
            
            generated_tokens_list.append(next_token)
            
            current_context_list.append(next_token)
            if len(current_context_list) > model_block_size:
                current_context_list.pop(0)

        return self.decode(generated_tokens_list)
    
    
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
        d_dist = jnp.sqrt(6.0 / (embed_size + attn_head_size))
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

    d_dist = jnp.sqrt(6.0 / (embed_size + embed_size))
    w_m = jax.random.uniform(keys[-1], (embed_size, embed_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_m = jnp.zeros((embed_size,), dtype=jnp.float32)
    
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
    bias_m: jnp.array,
    block_number: int = 1,
    training: bool = True
) -> Tuple :

    out = jnp.concat([single_head_attention(x_embed,params) for params in multihead_params],axis =1)
    out = jnp.matmul(out,w_m) + bias_m
    out = Dropout(0.5,rngs=Rngs(block_number),deterministic= not training)(out)

    return out

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
        block_num: int, # integer to track which transformer block ffn
        linear_1_key: jax.random.key,
        linear_2_key: jax.random.key,
        ) :
    d_dist = jnp.sqrt(6.0 / (embed_size + 4*embed_size))
    linear_1 = jax.random.uniform(linear_1_key, (embed_size, 4*embed_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_1 = jnp.zeros((4*embed_size,), dtype=jnp.float32)
    linear_2 = jax.random.uniform(linear_2_key, (4*embed_size, embed_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_2 = jnp.zeros((embed_size,), dtype=jnp.float32)
    params ={
        f'linear_{block_num}_1': linear_1,
        f'bias_{block_num}_1': bias_1,
        f'linear_{block_num}_2': linear_2,
        f'bias_{block_num}_2': bias_2
    }

    return params

#define classification layer weights initialization
def classification_weights_initialization(
        embed_size : int,
        vocab_size: int,
        key: jax.random.key,
        ) :
    d_dist = jnp.sqrt(6.0 / (embed_size + vocab_size))
    linear_cls = jax.random.uniform(key, (embed_size, vocab_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    bias_cls = jnp.zeros((vocab_size,), dtype=jnp.float32)
    params ={
        'linear_cls': linear_cls,
        'bias_cls': bias_cls,
    }

    return params
    
# define the feed forward network 
def feed_forward_block(
    input_weights: jnp.array,
    block_num: int,
    params: Dict = None,
    training: bool = True
    ) -> Tuple :

    if params:
        linear_1, bias_1, linear_2, bias_2 = params[f'linear_{block_num}_1'], params[f'bias_{block_num}_1'], params[f'linear_{block_num}_2'], params[f'bias_{block_num}_2']
    
    # make the feedforward calculations
    x = jnp.matmul(input_weights,linear_1) + bias_1
    x = jax.nn.relu(x)
    x = jnp.matmul(x,linear_2) + bias_2
    x = Dropout(0.5,rngs=Rngs(block_num),deterministic=not training)(x)

    return x
        


# define the transformers block
def block(
    x_embed: jnp.array,
    block_num: int, # integer to track which transformer block ffn
    params: Dict,
    training: bool = True
):
    x = layer_norm(x_embed,  
                     params[f'gamma_{block_num}_1'],
                     params[f'beta_{block_num}_1']
                     )
    self_attention  = multihead_attention(    
                                            x,
                                            params[f'multihead_params_{block_num}'],
                                            params[f'w_m_{block_num}'],
                                            params[f'bias_m_{block_num}'],
                                            block_number = block_num,
                                            training= training
                                        ) 
    assert self_attention.shape == x_embed.shape, f'multihead attention out shape != x_embed shape'
    x = self_attention + x_embed
    x = layer_norm(x,  
                params[f'gamma_{block_num}_2'],
                params[f'beta_{block_num}_2']
                 )
    x = feed_forward_block(x,block_num,params,training)
    assert self_attention.shape == x_embed.shape, f'feed-forward layer out shape != x_embed shape'
    x = x + x_embed

    return x



 # ref used for implementing the embedding forwards pass :
# https://stackoverflow.com/questions/72817730/what-is-the-recommended-way-to-do-embeddings-in-jax
def get_model_params(embed_size: int,
            vocab_size: int,
            seq_len: int ,
            number_of_heads: int,
            number_of_blocks: int =None
            ) -> Dict :
    
    # vmap to support batch operation in x_train 
    # ref : https://dinocausevic.com/2023/06/13/jax-vmap/
    # in_axes as  None means that axes will not be iterated over will stay fixed during the iteration process
    # add detailed comment
    params = {}
    params['embed_model'] = embedding_model(jax.random.key(0),vocab_size,embed_size)
    params['pos_encoding'] = positional_encoding(jax.random.key(1),seq_len,embed_size)

    key = jax.random.key(42)
    splits = jax.random.split(key,num=2)
    attention_block_keys = jax.random.split(splits[0],num=number_of_blocks)
    linear_1_key, linear_2_key, linear_cls_key = jax.random.split(splits[1],num=3)
    linear_1_block_keys = jax.random.split(linear_1_key,num=number_of_blocks)
    linear_2_block_keys = jax.random.split(linear_2_key,num=number_of_blocks)
    
    for i in range(number_of_blocks): 
        params[f'multihead_params_{i}'], params[f'w_m_{i}'], params[f'bias_m_{i}'] = get_multihead_projections(attention_block_keys[i],embed_size,number_of_heads)
    
        params[f'gamma_{i}_1'] = params[f'gamma_{i}_2'] = jnp.ones((seq_len,embed_size))
        params[f'beta_{i}_1'] = params[f'beta_{i}_2'] = jnp.zeros((seq_len,embed_size))

        feedforward_params = feedforward_weights_initialization(embed_size,i,linear_1_block_keys[i],linear_2_block_keys[i])
        params.update(feedforward_params)
    
    params['gamma_cls'] = jnp.ones((seq_len,embed_size))
    params['beta_cls'] = jnp.zeros((seq_len,embed_size))

    cls_params = classification_weights_initialization(embed_size,vocab_size,linear_cls_key)
    params.update(cls_params)

    return params

    
# ref used for implementing the embedding forwards pass :
# https://stackoverflow.com/questions/72817730/what-is-the-recommended-way-to-do-embeddings-in-jax
def transformer(
            x : jnp.array, # single x of shape (batch_size,block_size)
            params: Dict,
            number_of_layers : int = 1,
            training: bool = True
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

    #change here to n blocks
    for n in range(number_of_layers):
        output = block(output,n,params,training)
    assert output.shape == (x.shape[0],embed_size), f'transformer block -> attention block layer out shape {output.shape} != (seq_len,embed_size) {(x.shape[0],embed_size)} '

    output = layer_norm(output,params['gamma_cls'],params['beta_cls'])
    assert output.shape == (x.shape[0],embed_size), f'transformer block -> classification layer normalization out shape {output.shape} != (seq_len,embed_size) {(x.shape[0],embed_size)} '

    output = jnp.matmul(output,params['linear_cls']) + params['bias_cls']
    assert output.shape == (x.shape[0],vocab_size), f'transformer block -> classification output layer shape {output.shape} != (seq_len,vocab_size) {(x.shape[0],vocab_size)} '


    return output



def compute_loss(
        x_train : jnp.array, # single x of shape (batch_size,block_size)
        targets : jnp.array, # shape: (batch_size*block_size)
        params: Dict,
        number_of_layers: int = 1,
        training: bool = True
) :
    logits = jax.vmap(transformer,in_axes=[0,None,None,None])(x_train,params,number_of_layers,training)
    loss = softmax_cross_entropy_with_integer_labels(logits,targets).mean()

    return loss

@partial(jax.jit, static_argnames=['optimizer','number_of_layers'])
def train_step(
    x_train : jnp.array,  # shape (batch,block)
    y_train : jnp.array, # shape (batch*block)
    params: Dict,
    number_of_layers: int,
    optimizer,
    optimizer_state,
    ) -> Tuple:

    # use jax value_and_grad to compute the gradients
    loss, grads = jax.value_and_grad(compute_loss,argnums=2)(x_train,y_train,params,number_of_layers)

    # update the model parameters
    #params_update = jax.tree.map(lambda params, grads: params - lr*grads,params,grads)
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params_update = apply_updates(params, updates)
    # return the loss and the updated params dict
    return loss,optimizer_state,params_update

@partial(jax.jit, static_argnames=['number_of_layers'])
def val_step(
    x_val : jnp.array,  # shape (batch,block)
    y_val : jnp.array, # shape (batch*block)
    params: Dict,
    number_of_layers: int
    ) -> Tuple:
    # use jax value_and_grad to compute the gradients
    loss = compute_loss(x_val,y_val,params,
                        number_of_layers,training=False)

    return loss

def train(steps=100):

    # batch_size = 16
    # block_size = 32
    # max_iters = steps
    # eval_interval = 5
    # learning_rate = 1e-3
    # eval_iters = 200
    # embed_size = 64
    # number_of_heads = 4
    # vocab_size = 65
    # number_of_layers = 4

    batch_size = 32
    block_size = 128
    max_iters = steps
    eval_interval = 5
    learning_rate = 1e-3
    eval_iters = 200
    embed_size = 256
    number_of_heads = 8
    vocab_size = 65
    number_of_layers = 6

    # print(f'train input shape: {x_train.shape}, target shape: {y_train.shape}')
    # print(f'val input shape: {x_val.shape}, target shape: {y_val.shape}')
   
    params = get_model_params(embed_size=embed_size,
                              vocab_size=vocab_size,
                              seq_len=block_size,
                              number_of_heads=number_of_heads,
                              number_of_blocks=number_of_layers)


    processor = Processor()
    #print('Model Output with seqeunce length of 100 after initialization')
    #print(processor.generate_text(model,seq_len=1000))

    train_losses = []
    val_losses = []
    steps_list = []
    optimizer = adamw(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    train_key = jax.random.key(42)

    for step in range(max_iters): 
        train_key, batch_key = jax.random.split(train_key)
        x_train,y_train,x_val,y_val = processor.get_batch(prng=batch_key,batch_size=batch_size,block_size=block_size)
        loss, optimizer_state, params = train_step(x_train,y_train,params,number_of_layers,
                                                   optimizer,optimizer_state)
        print(f'step: {step+1} || train loss: {loss}',flush=True)
        
        train_losses.append(loss)
        steps_list.append(step)

        if (step + 1) % eval_interval == 0:
            val_loss = val_step(x_val,y_val,params,number_of_layers)
            print(f'step: {step+1} || val loss: {val_loss}',flush=True)
            val_losses.append(val_loss)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try: 
        print('saving the model weights as jnp arrays')
        jnp.savez(f'model_weights_multilayer_{number_of_layers}',**params)
    except Exception as e:
        print('error in model saving weight',e)

    plt.figure(figsize=(10, 6))
    plt.plot(steps_list, train_losses, label='Training Loss')
    val_steps = steps_list[eval_interval-1::eval_interval]
    plt.plot(val_steps, val_losses, label='Validation Loss')
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'training_loss_with_8_layers_diff_lr.jpg')
    plt.close()

    

if __name__ == '__main__' :

    args = argparse.ArgumentParser()
    args.add_argument('--generate',  action='store_true', help='if true starts the training loop else generates text from a trained model')
    args.add_argument('--steps', type=int, default=100, help='number of training steps')
    args = args.parse_args()

    if args.generate :
        
        print(f'generate from trained weights',flush=True)
        processor = Processor()
        
        model_file_path = 'weights/model_weights_multilayer_6.npz'
        loaded_data = np.load(model_file_path, allow_pickle=True)
        
        params = {}
        for key in loaded_data.files:

            if 'multihead_params_' in key:
                multihead_weights=[]
                for n in range(len(loaded_data[key])):
                    ar = jax.tree_util.tree_map(jnp.asarray,loaded_data[key][n])
                    multihead_weights.append(ar)
                params.update({key:multihead_weights})
            else:
                params.update({key:jnp.asarray(loaded_data[key])})
        #print(params.keys())


        generated_text = processor.generate_text_from_transfomer(params, 
                                                                    num_tokens_to_generate=500, 
                                                                    number_of_layers=4)
        print("\n--- Generated Text ---")
        print(generated_text)
    else:
        print(f'train nanoGPT in jax for {args.steps} steps',flush=True)
        train(args.steps)
        



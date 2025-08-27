import argparse
from typing import Dict, List, Tuple

import jax
from functools import partial
import jax.numpy as jnp
import matplotlib.pyplot as plt # type: ignore

from optax import softmax_cross_entropy_with_integer_labels, adamw, apply_updates, multi_transform
from flax.nnx import Dropout, Rngs
from muon import muon
from utils import Processor,label_lf, plot_and_save_loss, load_model_params

# helper fx to initialize the character level embedding model 
# uses a normal distribution for initialization
def embedding_model(
        key : jax.random,
        vocab_size : int = 65,
        embed_size: int = 128,
        )-> jax.Array :
    
    model = jax.random.normal(key,shape=(vocab_size,embed_size))
    return model

# helper fx to initialize the positional encodings
# uses a normal distribution for initialization
def positional_encoding(
        key: jax.random,
        seq_len: int = 8,
        embed_size: int = 128
) -> jax.Array :
    
    pos_encoding = jax.random.normal(key,shape=(seq_len,embed_size))

    return pos_encoding

# helper fx to initialize the Q,K,V weight matrices
def get_qkv_weights(
        q_key : jax.random.key,
        k_key : jax.random.key,
        v_key : jax.random.key,
        embed_size: int,
        attn_head_size : int, 
) -> Dict :
    # uses Xavier initialization 
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

# helper fx to initialize the multihead weight matrices for n heads
def get_multihead_weights(
    key: jax.random.key,
    embed_size: int,
    number_of_heads: int
) -> Tuple:
    
    attn_head_size = embed_size // number_of_heads
    projections = []
    
    # split the main key into number_of_heads subkeys
    keys = jax.random.split(key, num=number_of_heads+1)
    
    for i in range(number_of_heads):
        # for each head, split the key into 3 subkeys for q, k, v
        head_key = keys[i]
        q_key, k_key, v_key = jax.random.split(head_key, num=3)
        
        # get weights for this head
        head_projections = get_qkv_weights(q_key, k_key, v_key, embed_size, attn_head_size)
        projections.append(head_projections)

    # Xavier initilization for the projection matrix 
    # (embed_size, embed_size) -> (seq_len, number_of_heads*attn_head_size == embed_size) 
    d_dist = jnp.sqrt(6.0 / (embed_size + embed_size))
    w_m = jax.random.uniform(keys[-1], (embed_size, embed_size), 
                                minval=-d_dist, maxval=d_dist, dtype=jnp.float32)
    # bias is zero initialized
    bias_m = jnp.zeros((embed_size,), dtype=jnp.float32)
    
    return projections, w_m, bias_m



def single_head_attention(
        x_embed : jax.Array, # shape sequence_length, embedding_size
        params : dict ,
) -> jax.Array :
    
    w_q, w_k, w_v = params['w_q'], params['w_k'], params['w_v']
    q = jnp.matmul(x_embed,w_q)  # shape sequence_length, attn_head_size
    k = jnp.matmul(x_embed,w_k)  # shape sequence_length, attn_head_size
    v = jnp.matmul(x_embed,w_v)  # shape sequence_length, attn_head_size

    wei = jnp.matmul(q,k.T) # shape sequence_length, sequence_length
    attn_head_size = q.shape[1]
    # scale attention logits by sqrt(d_k) to stabilize magnitudes
    wei = wei / jnp.sqrt(attn_head_size)
    # number of tokens in the sequence
    seq_len = wei.shape[0]
    # generate lower-triangular causal mask that will mask all future tokens
    # matrix with 0 where mask is true and 1 everywhere else
    tril = jnp.tril(jnp.ones((seq_len,seq_len)))
    # this generated the lowest negative number possible for jax dtype float32 (basically negative inf)
    big_neg = jnp.finfo(jnp.float32).min
    # mask the weight matrix with a large negative number such that softmax return 0 
    wei = jnp.where(tril,wei,big_neg)
    wei = jax.nn.softmax(wei,axis=-1)
    w_attn = jnp.matmul(wei,v) #shape sequence_length, attn_head_size
   
    return w_attn


def multihead_attention(
    x_embed : jax.Array, # shape sequence_length, embedding_size
    multihead_params : List ,
    w_m : jax.Array,
    bias_m: jax.Array,
    block_number: int = 1, # integer to track which transformer block ffn
    training: bool = True
) -> jax.Array :

    # concat all n_heads into a single matrix of shape (seq_len, embed_size)
    out = jnp.concat([single_head_attention(x_embed,params) for params in multihead_params],axis =1)
    out = jnp.matmul(out,w_m) + bias_m
    out = Dropout(0.5,rngs=Rngs(block_number),deterministic= not training)(out)

    return out

 # helper fx that calculates the layer norm
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


# helper fx to get feedforward parameters 
# uses Xavier initialization for linear layers and zero initialization for bias
def feedforward_weights_initialization(
        embed_size : int,
        block_num: int, # integer to track which transformer block ffn
        linear_1_key: jax.random.key,
        linear_2_key: jax.random.key,
        ) -> Dict :
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

# helper fx for classification layer weights initialization
def classification_weights_initialization(
        embed_size : int,
        vocab_size: int,
        key: jax.random.key,
        )-> Dict :
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
    input_weights: jax.Array,
    block_num: int,
    params: Dict = None,
    training: bool = True
    ) -> jax.Array :

    linear_1, bias_1, linear_2, bias_2 = (params[f'linear_{block_num}_1'], params[f'bias_{block_num}_1'],
                                            params[f'linear_{block_num}_2'], params[f'bias_{block_num}_2'])
    # make the feedforward calculations
    x = jnp.matmul(input_weights,linear_1) + bias_1
    x = jax.nn.relu(x)
    x = jnp.matmul(x,linear_2) + bias_2
    x = Dropout(0.5,rngs=Rngs(block_num),deterministic=not training)(x)

    return x
        


# define the transformers block
def block(
    x_embed: jax.Array,
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

# helper fx that initializes all the parameters and returns a paramter Dict
def get_model_params(embed_size: int,
            vocab_size: int,
            seq_len: int ,
            number_of_heads: int,
            number_of_blocks: int =None
            ) -> Dict :
    
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
        params[f'multihead_params_{i}'], params[f'w_m_{i}'], params[f'bias_m_{i}'] = get_multihead_weights(attention_block_keys[i],embed_size,number_of_heads)
    
        params[f'gamma_{i}_1'] = params[f'gamma_{i}_2'] = jnp.ones((seq_len,embed_size))
        params[f'beta_{i}_1'] = params[f'beta_{i}_2'] = jnp.zeros((seq_len,embed_size))

        feedforward_params = feedforward_weights_initialization(embed_size,i,linear_1_block_keys[i],linear_2_block_keys[i])
        params.update(feedforward_params)
    
    params['gamma_cls'] = jnp.ones((seq_len,embed_size))
    params['beta_cls'] = jnp.zeros((seq_len,embed_size))

    cls_params = classification_weights_initialization(embed_size,vocab_size,linear_cls_key)
    params.update(cls_params)

    return params

# main transformer block with attention and linear layers 
def transformer(
            x : jax.Array, # single x of shape (batch_size,block_size)
            params: Dict,
            number_of_layers : int = 1,
            training: bool = True
            ) -> jax.Array :
    
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


# loss function
def compute_loss(
        x_train : jnp.array, # single x of shape (batch_size,block_size)
        targets : jnp.array, # shape: (batch_size*block_size)
        params: Dict,
        number_of_layers: int = 1,
        training: bool = True
)-> jax.Array :
    # in_axes as  None means that axes will not be iterated over and stay fixed during the iteration process
    logits = jax.vmap(transformer,in_axes=[0,None,None,None])(x_train,params,number_of_layers,training)
    loss = softmax_cross_entropy_with_integer_labels(logits,targets).mean()

    return loss

# uses functools.partial to preconfigure jax.jit so it can be applied as a decorator with static arguments
# static_argnames marks these as compile-time constants
# here in this case JIT compiles once and caches, does not recompile for each update
# It recompiles only if static args state changes ('optimizer', 'number_of_layers') or input shapes/dtypes/tree change.
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
    updates, optimizer_state = optimizer.update(grads, optimizer_state, params)
    params_update = apply_updates(params, updates)
    return loss,optimizer_state,params_update

@partial(jax.jit, static_argnames=['number_of_layers'])
def val_step(
    x_val : jax.Array,  # shape (batch,block)
    y_val : jax.Array, # shape (batch*block)
    params: Dict,
    number_of_layers: int
    ) -> jax.Array:

    loss = compute_loss(x_val,y_val,params,
                        number_of_layers,training=False)

    return loss

# train the model, log train/val loss and save the model weights
def train(steps=100,adamw_lr=1e-3,muon_lr=2e-3):

    batch_size = 64
    block_size = 256
    max_iters = steps
    eval_interval = 5
    adamw_lr = adamw_lr
    muon_lr = muon_lr
    embed_size = 384
    number_of_heads = 8
    vocab_size = 65
    number_of_layers = 6

    # print(f'train input shape: {x_train.shape}, target shape: {y_train.shape}')
    # print(f'val input shape: {x_val.shape}, target shape: {y_val.shape}')
   
   # get the initial values of the model paramters 
    params = get_model_params(embed_size=embed_size,
                              vocab_size=vocab_size,
                              seq_len=block_size,
                              number_of_heads=number_of_heads,
                              number_of_blocks=number_of_layers)
    
    # label the paramters Pytree based on which sould be muon/adam updated
    params_label = label_lf(params)

    processor = Processor()
    train_losses = []
    val_losses = []
    steps_list = []
    # multi_transform enables to partition the optimizer updated with labels
    optimizer = multi_transform(
                   {'adam': adamw(adamw_lr), 'muon': muon(muon_lr,polynomial='quintic')}, params_label)
    # optimizer = adamw(learning_rate=adamw_lr)
    optimizer_state = optimizer.init(params)

    train_key = jax.random.key(42)

    for step in range(max_iters): 
        train_key, batch_key = jax.random.split(train_key)
        # generate batch train/val data
        x_train,y_train,x_val,y_val = processor.get_batch(prng=batch_key,batch_size=batch_size,block_size=block_size)
        # do one forward pass and update parameters
        loss, optimizer_state, params = train_step(x_train,y_train,params,number_of_layers,
                                                   optimizer,optimizer_state)
        print(f'step: {step+1} || train loss: {loss}',flush=True)
        
        train_losses.append(float(loss))
        steps_list.append(step)
        # log validation every 5 steps
        if (step + 1) % eval_interval == 0:
            val_loss = val_step(x_val,y_val,params,number_of_layers)
            print(f'step: {step+1} || val loss: {val_loss}',flush=True)
            val_losses.append(float(val_loss))
    
    try: 
        print('saving the model weights as jnp arrays')
        jnp.savez(f'weights/gpt_quintic',**params)
    except Exception as e:
        print('error in model saving weight',e)
        
    # plot the results
    plot_and_save_loss(
        steps_list=steps_list,
        train_losses=train_losses,
        val_losses=val_losses,
        eval_interval=eval_interval,
        plot_path='plots/training_loss_with_quintic.jpg',
        log_path='logs/loss_data_with_quintic.json'
    )


if __name__ == '__main__' :

    args = argparse.ArgumentParser()
    args.add_argument('--mode',  type=str,default='train', help='if train starts the training loop else generates text from a trained model')
    args.add_argument('--steps', type=int, default=100, help='number of training steps')
    args.add_argument('--adamw_lr', type=float, default=2e-3, help='adamw learning rate')
    args.add_argument('--muon_lr', type=float, default=2e-3, help='muon learning rate')
    args = args.parse_args()

    if args.mode == 'train' :
        print(f'train nanoGPT in jax for {args.steps} steps',flush=True)
        train(args.steps,adamw_lr=args.adamw_lr,muon_lr=args.muon_lr)
        
    else:
        print(f'generate from trained weights',flush=True)
        processor = Processor()
        model_file_path = 'weights/gpt_muon.npz'
        params = load_model_params(model_file_path)
        
        generated_text = processor.generate_text_from_transfomer(params, transformer,
                                                                    num_tokens_to_generate=500, 
                                                                    number_of_layers=4)
        print("\n--- Generated Text ---")
        print(generated_text)
        
        



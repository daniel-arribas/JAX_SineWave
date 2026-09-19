import jax   # importamos JAX y JAX.numpy
import jax.numpy as jnp

def init_params(key, layers):
    '''
    Initiolization of the weights and bias for each MLP layer.
    layers: List with dimensions example[1, 32, 32, 1]
    '''
    keys = jax.random.split(key,len(layers)-1)
    params = []

    for in_dim, out_dim, k in zip(layers[:-1], layers[1:], keys):
        #Xavier/Glorot initialization
        limit = jnp.sqrt(6.0/(in_dim + out_dim))
        w = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,))
        
        params.append({'w': w, 'b': b})

    return params

def forward(params, x):
    '''
    Evaluates the NN for a given x input
    '''

    # Processing hidden layers with senoidal activation
    for layer in params [:-1]:
        x = jnp.dot(x, layer['w']) + layer['b']
        x = jnp.sin(x) # Activacion periodica

    # output layer
    final_layer = params[-1]
    out = jnp.dot(x, final_layer['w']) + final_layer['b']

    return out

if __name__ == "__main__": # model.py test code
    # Fast test of the module
    key = jax.random.PRNGKey(0)
    layer_sizes = [1, 64, 64, 1]
    
    # 1. Create params
    params = init_params(key, layer_sizes)
    
    # 2. try a fictional batch of 10 data
    dummy_x = jnp.ones((10, 1))
    preds = forward(params, dummy_x)
    
    print("Shape of predictions:", preds.shape)
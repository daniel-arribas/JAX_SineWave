# JAX_SineWave

This pryect is a tutorial of NNs in native JAX consinsting in aproximating the sine function using MLP architecture.

## Structure of the proyect

```txt
JAX_SineWave                                                                       
    ├── config.json                                                                    
    ├── dataset.py                                                                     
    ├── loss.py                                                                        
    ├── model.py                                                                       
    ├── README.md                                                                      
    ├── test.py                                                                        
    └── train.py       

```

## Model definition (model.py)

The model is defined by the following functions in the model.py file.

### Parameter initializaion

```python
def init_params(key, layers):
    '''
    Initiolization of the weights and bias for each MLP layer.
    layers: List with dimensions example[1, 32, 32, 1]
    '''
    keys = jax.random.split(key,len(layers)-1)
    params = []

    for in_dim, out_dim, k in zip(layers[:-1], layers[1:], keys):
        #Xavier/Glorot initialization
        limit = jnp.sqrt(60/(in_dim + out_dim))
        w = jax.random.uniform(k, (in_dim, out_dim), minval=-limit, maxval=limit)
        b = jnp.zeros((out_dim,))
        
        params.append({'w': w, 'b': b})

    return params
```

### NN evalutaion

```python
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
```





## Dataset management (dataset.py)


## Loss function (loss.py)

## Training (train.py)

## Testing (test.py)


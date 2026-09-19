import jax
import jax.numpy as jnp
from model import forward


def mse_loss(params, x, y):
  """Calculates the Mean Squared Error (MSE) between predictions and the ground

  truth sine function.

  Args:
      params (list[dict]): PyTree of model parameters.
      x (jnp.ndarray): Input tensor of shape (batch_size, 1).
      y (jnp.ndarray): Tensor of ground truth values sin(x) of shape
        (batch_size, 1).

  Returns:
      float: A scalar value representing the error.
  """
  # 1. Forward pass to get predictions
  preds = forward(params, x)

  # 2. Mean squared error: Average of (ŷ - y)^2
  loss = jnp.mean((preds - y) ** 2)

  return loss


# ==============================================================================
# JAX TRANSFORMATION: Function that computes Loss AND its Gradients
# ==============================================================================
# jax.value_and_grad takes a function and returns a NEW function that
# simultaneously computes: (return_value, gradients_wrt_first_argument)
loss_and_grad_fn = jax.value_and_grad(mse_loss)


if __name__ == "__main__":
  # Quick module test
  from model import init_params

  key = jax.random.PRNGKey(0)
  params = init_params(key, [1, 32, 32, 1])

  # Dummy data
  x_dummy = jnp.array([[0.5], [1.0]])
  y_dummy = jnp.sin(x_dummy)

  # Compute loss and gradients
  loss_val, grads = loss_and_grad_fn(params, x_dummy, y_dummy)

  print("Initial loss:", loss_val)
  print("Are gradient shapes identical to params?:", len(grads) == len(params))
  print("Gradient shape of W1:", grads[0]["w"].shape)
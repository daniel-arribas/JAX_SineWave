import jax
import jax.numpy as jnp
from model import forward


def mse_loss(params, x, y):
  """Calcula el Error Cuadrático Medio (MSE) entre las predicciones y el seno real.

  Args:
      params (list[dict]): PyTree de parámetros del modelo.
      x (jnp.ndarray): Tensor de entradas con forma (batch_size, 1).
      y (jnp.ndarray): Tensor de valores reales sin(x) con forma (batch_size,
        1).

  Returns:
      float: Un número escalar con el valor del error.
  """
  # 1. Pasada hacia adelante para obtener las predicciones
  preds = forward(params, x)

  # 2. Error cuadrático medio: Promedio de (ŷ - y)^2
  loss = jnp.mean((preds - y) ** 2)

  return loss


# ==============================================================================
# TRANSFORMACIÓN DE JAX: Función que calcula el Loss Y sus Gradientes
# ==============================================================================
# jax.value_and_grad toma una función y devuelve una NUEVA función que
# calcula simultáneamente: (valor_de_retorno, gradientes_respecto_al_primer_argumento)
loss_and_grad_fn = jax.value_and_grad(mse_loss)


if __name__ == "__main__":
  # Test rápido del módulo
  from model import init_params

  key = jax.random.PRNGKey(0)
  params = init_params(key, [1, 32, 32, 1])

  # Datos ficticios
  x_dummy = jnp.array([[0.5], [1.0]])
  y_dummy = jnp.sin(x_dummy)

  # Calculamos loss y gradientes
  loss_val, grads = loss_and_grad_fn(params, x_dummy, y_dummy)

  print("Loss inicial:", loss_val)
  print("¿Forma de gradientes igual a params?:", len(grads) == len(params))
  print("Forma del gradiente de W1:", grads[0]["w"].shape)
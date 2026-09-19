import jax
import jax.numpy as jnp


def generate_full_dataset(
    num_samples=1000, x_min=-2 * jnp.pi, x_max=2 * jnp.pi
):
  """Genera un conjunto de datos uniforme y ordenado en el rango [x_min, x_max].

  Ideal para evaluar el modelo y hacer gráficas limpias.
  """
  x = jnp.linspace(x_min, x_max, num_samples).reshape(-1, 1)
  y = jnp.sin(x)
  return x, y


def get_batch(
    key, batch_size=32, x_min=-2 * jnp.pi, x_max=2 * jnp.pi, noise_std=0.0
):
  """Genera un mini-batch aleatorio de puntos dentro del rango deseado.

  Args:
      key: PRNGKey de JAX para el muestreo aleatorio.
      batch_size: Número de ejemplos a generar.
      x_min, x_max: Rango del dominio de entrada.
      noise_std: Desviación estándar del ruido gaussiano a añadir (0.0 = sin
        ruido).

  Returns:
      x, y: Tensores de dimensión (batch_size, 1).
  """
  key_x, key_noise = jax.random.split(key)

  # Muestreo aleatorio uniforme dentro del intervalo
  x = jax.random.uniform(
      key_x, shape=(batch_size, 1), minval=x_min, maxval=x_max
  )

  # Cálculo de la función seno con opción a ruido
  noise = jax.random.normal(key_noise, shape=(batch_size, 1)) * noise_std
  y = jnp.sin(x) + noise

  return x, y


if __name__ == "__main__":
  # Código de prueba rápido
  key = jax.random.PRNGKey(42)

  # Prueba de batch aleatorio para entrenamiento
  x_batch, y_batch = get_batch(key, batch_size=10)
  print("Batch aleatorio X shape:", x_batch.shape)
  print("Batch aleatorio Y shape:", y_batch.shape)

  # Prueba de dataset completo para evaluación
  x_eval, y_eval = generate_full_dataset(num_samples=100)
  print("Dataset completo X shape:", x_eval.shape)
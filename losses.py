import tensorflow as tf

from tensorflow.python.framework import ops

_trace_norm = tf.load_op_library('trace_norm.so')

@ops.RegisterGradient("TraceNorm")
def _trace_norm_grad(op, grad, g_u, g_v):
    """The gradients for `trace_norm`.

    Args:
    op: The `trace_norm` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `trace_norm` op.

    Returns:
    Gradients with respect to the input of `trace_norm`.
    """

    # TODO: Ensure that we are only using the gradient of the trace norm.
    # and not the `u' and `v' matrices.

    _, u, v = op.outputs
    trace_grad = tf.matmul(u, v, transpose_b=True)

    return [grad * trace_grad]

def regularize(inputs, regularisation):
    return inputs + tf.ones_like(inputs) * regularisation

def correlation_cost(source, target, source_regularisation=1e-6, target_regularisation=1e-6):
    num_source_samples = source.get_shape().as_list()[0]
    num_target_samples = target.get_shape().as_list()[0]
    
    # assert num_source_samples == num_target_samples
    
    num_samples = tf.to_float(tf.shape(target)[0])
    
    source -= tf.reduce_mean(source, 0)
    target -= tf.reduce_mean(target, 0)
    
    correlation_matrix = tf.matmul(source, target, transpose_a=True) / (num_samples - 1)

    source_covariance = regularize(tf.matmul(source, source, transpose_a=True) / (num_samples-1), source_regularisation) 
    target_covariance = regularize(tf.matmul(target, target, transpose_a=True) / (num_samples-1), source_regularisation)
    
    # source_covariance = (tf.transpose(source_covariance) + source_covariance) / 2.
    
    root_source_covariance = tf.cholesky(source_covariance)
    inv_root_source_covariance = tf.matrix_inverse(root_source_covariance)

    # target_covariance = (tf.transpose(target_covariance) + target_covariance) / 2.
    
    root_target_covariance = tf.cholesky(target_covariance)
    inv_root_target_covariance = tf.matrix_inverse(root_target_covariance)

    canonical_correlation = tf.matmul(inv_root_source_covariance, correlation_matrix)
    canonical_correlation = tf.matmul(canonical_correlation, inv_root_target_covariance)
    
    loss = _trace_norm.trace_norm(canonical_correlation)[0]

    return - loss, source_covariance
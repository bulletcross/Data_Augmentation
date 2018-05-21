# Authored by: bulletcross@gmail.com

import tensorflow as tf
import numpy as np

NR_CLASS = 2
INPUT_SHAPE = [128, 128, 3]
LEARNING_RATE = 0.0001

def separable_conv2d_(input, filters):
    layer = tf.layers.separable_conv2d(
        inputs = input,
        filters =  filters,
        kernel_size = [3,3],
        strides = [1,1],
        padding = "same",
        depth_multiplier = 1,
        activation = tf.nn.relu,
        use_bias = True,
        depthwise_initializer = tf.truncated_normal_initializer,
        pointwise_initializer = tf.truncated_normal_initializer,
        bias_initializer = tf.zero_initializer,
        name = "depthwise_seperable"
    )
    layer = tf.layers.max_pooling2d(layer, pool_size=2, strides=2)
    return layer


def model_arch(input):
    input = tf.reshape(input, INPUT_SHAPE)
    layer1 = separable_conv2d_(input, 32)
    layer2 = separable_conv2d_(layer1, 128)
    layer3 = separable_conv2d_(layer2, 256)
    layer4 = separable_conv2d_(layer3, 256)
    layer5 = separable_conv2d_(layer4, 512)
    layer6 = tf.reduce_mean(layer5, axis=[1,2])
    layer7 = tf.nn.avg_pool(layer6, [1,1,1,512])
    layer8 = tf.contrib.layers.flatten(layer7)
    logits = tf.layers.dense(layer8, units = NR_CLASS)
    out_probability = tf.nn.softmax(logits)
    ans = tf.argmax(out_probability, axis = 1)
    return out_probability, ans

def model_estimator(input, labels, mode, params):
    out, ans = model_arch(input)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels = labels, logits = out
    )
    loss = tf.reduce_mean(entropy)
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE)
    train_op = optimizer.minimize(loss)
    metrics = {
        "accuracy": tf.metrics.accuracy(labels, ans)
    }
    estimator_specs = tf.estimator.EstimatorSpec(
        mode = mode,
        loss = loss,
        train_op = train_op,
        eval_metric_ops = metrics
    )
    return estimator_specs

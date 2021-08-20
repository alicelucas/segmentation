import tensorflow.keras.backend as K
from tensorflow.keras.losses import sparse_categorical_crossentropy
import tensorflow as tf
import numpy

def weightedSparseCategoricalCrossEntropyLoss(weights):
    """
    From https://stackoverflow.com/a/51851303
    :param weights: weights list
    :return: weighted sparse categorical loss
    """

    # # Copyright 2021 The TensorFlow Authors. All Rights Reserved.
    # #
    # # Licensed under the Apache License, Version 2.0 (the "License");
    # # you may not use this file except in compliance with the License.
    # # You may obtain a copy of the License at
    # #
    # #     http://www.apache.org/licenses/LICENSE-2.0
    # #
    # # Unless required by applicable law or agreed to in writing, software
    # # distributed under the License is distributed on an "AS IS" BASIS,
    # # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # # See the License for the specific language governing permissions and
    # # limitations under the License.
    #
    # """Weighted sparse categorical cross-entropy losses."""
    #
    # import tensorflow as tf
    #
    # def _adjust_labels(labels, predictions):
    #     """Adjust the 'labels' tensor by squeezing it if needed."""
    #     labels = tf.cast(labels, tf.int32)
    #     if len(predictions.shape) == len(labels.shape):
    #         labels = tf.squeeze(labels, [-1])
    #     return labels, predictions
    #
    # def _validate_rank(labels, predictions, weights):
    #     if weights is not None and len(weights.shape) != len(labels.shape):
    #         raise RuntimeError(
    #             ("Weight and label tensors were not of the same rank. weights.shape "
    #              "was %s, and labels.shape was %s.") %
    #             (predictions.shape, labels.shape))
    #     if (len(predictions.shape) - 1) != len(labels.shape):
    #         raise RuntimeError(
    #             ("Weighted sparse categorical crossentropy expects `labels` to have a "
    #              "rank of one less than `predictions`. labels.shape was %s, and "
    #              "predictions.shape was %s.") % (labels.shape, predictions.shape))


    def lossFunc(true, pred, weights=weights):

        axis = -1  # if channels last
        # axis=  1 #if channels first

        # argmax returns the index of the element with the greatest value
        # done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)
        # if your loss is sparse, use only true as classSelectors

        # considering weights are ordered by class, for each class
        # true(1) if the class index is equal to the weight index
        one64 = numpy.ones(1, dtype=numpy.int64)
        classSelectors = [K.equal(one64[0] * i, classSelectors) for i in range(len(weights))]

        # classSelectors = [K.equal(i, classSelectors) for i in range(len(weights))]

        # casting boolean to float for calculations
        # each tensor in the list contains 1 where ground true class is equal to its index
        # if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        # for each of the selections above, multiply their respective weight
        weights = [sel * w for sel, w in zip(classSelectors, weights)]

        # sums all the selections
        # result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]

        # make sure your originalLossFunc only collapses the class axis
        # you need the other axes intact to multiply the weights tensor
        loss = sparse_categorical_crossentropy(
            true, pred, from_logits=True)
        loss = loss * weightMultiplier

        return loss


    # def loss(labels, predictions, weights=weights, from_logits=True):
    #     """Calculate a per-batch sparse categorical crossentropy loss.
    #     This loss function assumes that the predictions are post-softmax.
    #     Args:
    #       labels: The labels to evaluate against. Should be a set of integer indices
    #         ranging from 0 to (vocab_size-1).
    #       predictions: The network predictions. Should have softmax already applied.
    #       weights: An optional weight array of the same shape as the 'labels' array.
    #         If None, all examples will be used.
    #       from_logits: Whether the input predictions are logits.
    #     Returns:
    #       A loss scalar.
    #     Raises:
    #       RuntimeError if the passed tensors do not have the same rank.
    #     """
    #     # When using these functions with the Keras core API, we will need to squeeze
    #     # the labels tensor - Keras adds a spurious inner dimension.
    #     labels, predictions = _adjust_labels(labels, predictions)
    #     # _validate_rank(labels, predictions, weights)
    #
    #     example_losses = tf.keras.losses.sparse_categorical_crossentropy(
    #         labels, predictions, from_logits=from_logits)
    #
    #     if weights is None:
    #         return tf.reduce_mean(example_losses)
    #     weights = tf.cast(weights, predictions.dtype)
    #     return tf.math.divide_no_nan(
    #         tf.reduce_sum(example_losses * weights), tf.reduce_sum(weights))
    #

    return lossFunc


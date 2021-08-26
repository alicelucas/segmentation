import tensorflow.keras.backend as K

def weighted_categorical_crossentropy(weights):
    """
    From https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d#file-keras_weighted_categorical_crossentropy-py-L22
    :param weights: weights to assign to each class
    :return:
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss
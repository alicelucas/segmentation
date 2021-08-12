import tensorflow as tf


def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

    #TODO experiment with leaving the batch normalization in vs not
  # result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def unet_model(input_shape):
    """
    Simple segmentation from tensorflow tutorial: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/segmentation.ipynb
    It's a Unet model with MobileVnet as encoder backbone
    """

    output_channels = 3

    #Instantiate model using stored weights
    base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=input_shape, include_top=False,
                                                                weights="./weights/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5") #Instantiate architecture

    # Use the activations of these layers
    layer_names = [
        'block_1_expand_relu',   # 112x112
        'block_3_expand_relu',   # 56x56
        'block_6_expand_relu',   # 28x28
        'block_13_expand_relu',  # 14x14
        'block_16_project',      # 7x7
    ]
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

    # Create the feature extraction model
    down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

    #FIXME:should you set this to False?
    down_stack.trainable = False


    #The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples.
    up_stack = [
        upsample(512, 3),  # 7x7 -> 14x14
        upsample(256, 3),  # 14x14 -> 28x28
        upsample(128, 3),  # 28x28 -> 56x56
        upsample(64, 3),  # 56x56 -> 112x112
    ]

    inputs = tf.keras.layers.Input(shape=[224, 224, 3])

    #Use mobile net's way of input input
    normalized = tf.keras.applications.mobilenet_v2.preprocess_input(inputs) #Use mobilenet's input (maps to [-1, 1] range)

    # Downsampling through the model
    skips = down_stack(normalized)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
      output_channels, 3, strides=2,
      padding='same')  #112x112 -> 224x224

    x = last(x)

    # outputs = tf.keras.layers.Conv2D(3, 1, padding="same", activation="sigmoid")(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
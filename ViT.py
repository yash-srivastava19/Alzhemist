import numpy
import tensorflow as tf


image_size = 50
patch_size = 5
projection_dim = 64
num_heads = 4
num_classes = 100
input_shape = (32,32,3)
image_size = 50
num_patches = (image_size//patch_size)**2

transformer_units = [projection_dim*2, projection_dim]
transformer_layers = 8
mlp_head_units = [2048, 1024]

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = tf.keras.layers.Dense(units, activation=tf.nn.gelu)(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


class Patches(tf.keras.layers.Layer):
    def __init__(self, patch_size) -> None:
        super(Patches, self).__init__()
        self._patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images = images,
            sizes = [1, self._patch_size, self._patch_size, 1],
            strides = [1, self._patch_size, self._patch_size, 1],
            rates = [1,1,1,1],
            padding = 'VALID',
        ) 
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches 


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projections_dim) -> None:
        super(PatchEncoder, self).__init__()
        self._num_patches = num_patches
        self._projection = tf.keras.layers.Dense(units=projections_dim)

        self._pos_embedding = tf.keras.layers.Embedding(input_dim=num_patches, output_dim=projections_dim)
        self._num_patches = num_patches

    def call(self, patch):
        positions = tf.range(start=0, limit=self._num_patches, delta=1)
        encoded = self._projection(patch) + self._pos_embedding(positions)


class VITLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, patch_size, num_patches, projection_dim, trans_layers, num_heads) -> None:
        super(VITLayer, self).__init__()
        self._inp_shape  = input_shape
        self._patch_size = patch_size
        self._num_patches = num_patches
        self._pro_dim = projection_dim
        self._trans_layers = trans_layers
        self._num_heads = num_heads
        self._trans_units = [self._pro_dim*2, self._pro_dim]
        self._mlp_head_units = [2048, 1024]
    
    def call(self):
        inputs = tf.keras.Input(self._inp_shape)
        patches = Patches(self._patch_size)(inputs)
        enc_patches = PatchEncoder(self._num_patches, self._pro_dim)(patches)
    
        for _ in range(self._trans_layers):
            x1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)(enc_patches)
            att_opt = tf.keras.layers.MultiHeadAttention(num_heads=self._num_heads, key_dim=self._pro_dim)(x1, x1)
            x2 = tf.keras.layers.Add()([att_opt, enc_patches])
            x3 = tf.keras.layers.LayerNormalization(epsilon=1e-4)(x2)
            x3 = mlp(x3, hidden_units=self._trans_units, dropout_rate=0.1)
            enc_patches = tf.keras.layers.Add()([x3, x2])
        

        reprs = tf.keras.layers.LayerNormalization(epsilon=1e-4)(enc_patches)
        reprs = tf.keras.layers.Flatten()(reprs)
        reprs = tf.keras.layers.Dropout(0.1)(reprs)

        features = mlp(reprs, hidden_units=self._mlp_head_units, dropout_rate=0.5)

        logits = tf.keras.layers.Dense(num_classes)(features)

        return tf.keras.Model(inputs=inputs, outputs=logits)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_shape': self._inp_shape,
            'num_patches': self._num_patches,
            'projection_dims': self._pro_dim,
            'transformer_layers': self._trans_layers,
            'transformer_units': self._trans_units,
            'num_heads': self._num_heads,
            'mlp_head_units': self._mlp_head_units,
        })

        return config





(x_train, _), (_, _) = tf.keras.datasets.cifar100.load_data()

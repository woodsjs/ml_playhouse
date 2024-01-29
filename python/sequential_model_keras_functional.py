from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# used to let the model know we have two sets of inputs
from tensorflow.keras.layers import Concatenate

inputs = Input(shape=(10,))
# these go to the second dense layer
bypass_inputs = Input(shape=(5,))

# layers
layer1 = Dense(64, activation='relu')
concat_layer = Concatenate()
layer2 = Dense(64, activation='relu')

#connect the layers
layer1_outputs = layer1(inputs)
layer2_inputs = concat_layer([layer1_outputs, bypass_inputs])
layer2_outputs = layer2(layer1_outputs)

# create the model
model = Model(inputs=[inputs, bypass_inputs], outputs=layer2_outputs)
model.summary()

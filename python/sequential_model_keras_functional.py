from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

inputs = Input(shape=(10,1))

# layers
layer1 = Dense(64, activation='relu')
layer2 = Dense(64, activation='relu')

#connect the layers
layer1_outputs = layer1(inputs)
layer2_outputs = layer2(layer1_outputs)

# create the model
model = Model(inputs=inputs, outputs=layer2_outputs)
model.summary()

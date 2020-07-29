import keras, json;
import numpy as np;

# NN
def CreateModel(layers, learning_rate, actions):
	if type(layers) != list: return None;
	
	model = keras.models.Sequential();
	
	if len(layers) == 0:
		model.add(keras.layers.Dense(actions, input_dim=3, activation="linear", name=("lr_{}".format(learning_rate))));
	else:
		model.add(keras.layers.Dense(layers[0], input_dim=3, name=("lr_{}".format(learning_rate))));
		model.add(keras.layers.Activation("tanh"));
		# model.add(keras.layers.LeakyReLU(alpha=0.3));
	
		for neuronAmount in layers[1:]:
			model.add(keras.layers.Dense(neuronAmount));
			model.add(keras.layers.Activation("tanh"));
			# model.add(keras.layers.LeakyReLU(alpha=0.3));
		
		model.add(keras.layers.Dense(actions, activation="linear"));
		
	model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate), metrics=["accuracy"]);
	return model;

if __name__ == "__main__":
	model_name = input("Model name -> ");
	layers = json.loads(input("Neurons in hidden layers as list (e.g. [6, 3, 3]) -> "));
	actions = int(input("Output neuron count -> "));
	lr = float(input("Learning rate -> "));

	model = CreateModel(layers, lr, actions);
	if not model: raise Exception("Invalid model, did you use correct param format?");

	model_dir = "models/" + model_name + "/";
	import os;
	if not os.path.exists(model_dir):
		os.makedirs(model_dir);

	model.save(model_dir + "model.h5");
	keras.utils.plot_model(model, show_shapes=True, to_file=model_dir + "model.png");
	
	model.summary();
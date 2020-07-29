import keras, random, json;
import numpy as np;

class DQNAgent(object):
	def __init__(self, model_location, epsilon=1.0):
		self.model_location = model_location;
		self.model = keras.models.load_model(model_location);
		self.target_model = keras.models.load_model(model_location);
		self.action_space = self.model.layers[-1].output_shape[1];
		self.memory = [];
		self.replays = 0;
		self.max_memory = 50000;

		self.gamma = 0.99;							# long term value
		self.epsilon = max(epsilon, 0.0);			# exploration rate
		self.epsilon = min(self.epsilon, 1.0);
		self.epsilon_min = 0.05;
		self.epsilon_decay = 0.9996;

		self.lastrandom = False;
		self.lastrandoms = [];

		tmp = np.zeros(self.model.layers[0].input_shape[1:]);
		tmp = np.reshape(tmp, [1, tmp.shape[0]])
		self.model.predict_on_batch(tmp);
		self.target_model.predict_on_batch(tmp);

		tmptargets = np.zeros((1, self.action_space))
		self.model.fit(tmp, tmptargets, verbose=0)
		self.target_model.fit(tmp, tmptargets, verbose=0)

	def remember(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done));

		if len(self.memory) > self.max_memory: self.memory = self.memory[-self.max_memory:];

	def merge_models(self):
		self.target_model.set_weights(self.model.get_weights());
		
	def act(self, state):
		if random.random() <= self.epsilon:
			return random.randint(0, self.action_space-1);
		
		act_values = self.model.predict_on_batch(state);
		return np.argmax(act_values[0]);

	def replay(self, batch_size):
		# tb._SYMBOLIC_SCOPE.value = True
		
		# select random samples from memory
		minibatch = random.sample(self.memory, batch_size);
		
		states = []
		actions = []
		rewards = []
		next_states = []
		dones = []
		for state, action, reward, next_state, done in minibatch:
			states.append(state[0])
			actions.append(action)
			rewards.append(reward)
			next_states.append(next_state[0])
			dones.append(done)

		states = np.array(states, dtype=np.float32);
		next_states = np.array(next_states, dtype=np.float32);

		target_r = self.target_model.predict_on_batch(next_states)
		targetn_r = [np.amax(x) for x in target_r]
		current_r = self.model.predict_on_batch(states)
		ytrain = []

		for (i, reward) in enumerate(rewards):
			current_r[i][actions[i]] = reward + ((targetn_r[i]*self.gamma) if not dones[i] else 0.0)
			ytrain.append(current_r[i])

		xtrain = states

		if len(xtrain) > 0 and len(xtrain) == len(ytrain):
			xtrain = np.array(xtrain, dtype=np.float32);
			ytrain = np.array(ytrain, dtype=np.float32);
			#self.model.fit(xtrain, ytrain, epochs=1, batch_size=batch_size, verbose=0);
			self.model.train_on_batch(xtrain, ytrain);
			self.replays += 1;

		# update epsilon to minimize exploration on the long term
		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay;
	
	def save(self, model_name=None):
		try:
			location = self.model_location.replace("model_trained.h5", "model_trained.h5" if not model_name else model_name+".h5");
			location = location.replace("model.h5", "model_trained.h5" if not model_name else model_name+".h5");
			self.model.save(location);
			
		except Exception as e:
			print("Exception", str(e));

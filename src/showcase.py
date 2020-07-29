import json, time;
import numpy as np;
import keras, gym, agent;

def mean(values):
	return round(sum(values) / len(values), 2);
	
def calculate_u(action_space, action):
	u = 0.0;
	u_step = 4.0 / (action_space-1);

	if action_space % 2 == 1:
		middle = action_space//2
		if action == middle: return u

		if action < middle: u -= (middle-action)*u_step
		else: u += (action-middle)*u_step
	else:
		middle = action_space/2.0 - 0.5
		
		if action < middle: u -= (middle-action+0.5)*u_step
		else: u += (action-middle+0.5)*u_step

	return u

if __name__ == "__main__":
	env = gym.make("Pendulum-v0");
	state_size = env.observation_space.shape[0];
	
	model_location = input("Model location -> ");
	model_name = input("Model name -> ");
	my_model = "models/{}/{}.h5".format(model_location, model_name);
	epsilon = float(input("Epsilon -> "));
	
	print("Loading", my_model, "with epsilon", epsilon);
	agent = agent.DQNAgent(my_model, epsilon);

	episode_count = int(input("Episode count -> "));
	done = False;
	
	for e in range(episode_count):		
		# at each episode, reset environment to starting position
		state = env.reset();
		
		while not done:
			# show game graphics
			env.render();
			state = np.reshape(state, [1, state_size]);

			# select action, observe environment, calculate reward
			action = agent.act(state);
			u = calculate_u(agent.action_space, action);
			state, reward, done, _ = env.step([u]);
		
		done = False;
		
		print("episode: {}/{}".format(e+1, episode_count));

	# print("Showcase time:", round((time.time()-first_start)/60, 2), "minutes");
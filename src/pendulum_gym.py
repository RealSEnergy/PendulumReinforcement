import json, time;
import numpy as np;
import keras, gym, agent;

def sign(value):
	return round(val/abs(value)) if value != 0 else 0;

def mean(values):
	return round(sum(values) / len(values), 2) if type(values) == list else 0.0;

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
	
	model_name = input("Model name -> ");
	load_trained = input("Load trained (y/n)? ");
	load_trained = True if load_trained.lower() == "y" else False;
	
	my_model_location = "models/" + model_name + "/";
	my_model = my_model_location + ("model_trained.h5" if load_trained else "model.h5");

	epsilon = float(input("Epsilon -> ")); # if load_trained else 1.0;
	
	print("Loading", my_model, "with epsilon", epsilon);
	agent = agent.DQNAgent(my_model, epsilon);
	
	try: agent.memory = json.load(my_model.replace(".h5", ".json"));
	except: agent.memory = [];

	episode_count = int(input("Episode count -> "));
	batch_size = 16;
	
	max_score = None;
	highest_score = None;
	scores = [];
	rewards = [];
	
	start = time.time();
	first_start = start;
	
	for e in range(episode_count):		
		# at each episode, reset environment to starting position
		state = env.reset();
		state = np.reshape(state, [1, state_size]);
		score = 0;

		done = False;
		rewards.append(0.0);
		firststeps = 20;
		
		while not done and (score < max_score if max_score else True):
			# show game graphics
			env.render();

			# select action, observe environment, calculate reward
			action = agent.act(state);
			u = calculate_u(agent.action_space, action);
			next_state, reward, done, _ = env.step([u]);
			next_state = np.reshape(next_state, [1, state_size]);
			
			if firststeps > 0: firststeps -= 1;
			else:
				angle = np.angle([next_state[0][0] + 1j * next_state[0][1]], True);
				score += abs(angle[0]);
			
			# save experience and update current state
			agent.remember(state, action, reward, next_state, done);
			state = next_state;
			rewards[-1] += reward;
			
			# dynamic batch_size and max_memory
			# batch_size = round((highest_score/500) * 80) + 48;
			# max_memory = round(highest_score*20 + 250) if highest_score != 500 else 9500;
			
			if len(agent.memory) > batch_size:
				agent.replay(batch_size);
		
		score /= 180.0; # average angle in episode
		scores.append(score);
		
		if len(scores) > 20: scores = scores[-20:];
		if len(rewards) > 20: rewards = rewards[-20:];
		
		if highest_score == None or score < highest_score: highest_score = score;
		
		print("episode: {}/{}, average angle: {}, best average angle: {}, last 20 average: {}, e: {}, in memory: {}, batch size: {}"
				.format(e+1, episode_count, round(score, 2), round(highest_score, 2), round(mean(scores), 2), round(agent.epsilon, 3), len(agent.memory), batch_size));

		if len(scores) >= 5 and score <= 10 and sum(scores[-5:]) <= 50:
			agent.save();
		
		if len(scores) >= 15 and sum(scores[-15:]) <= 75:
			print("training successfull!");
			agent.save("final");
			break;
		
		if (e+1) % 5 == 0:
			print("Took", round((time.time()-start)/60, 2), "minutes\n");
			start = time.time();
			agent.merge_models();

	agent.save();
	print("Total training time:", round((time.time()-first_start)/60, 2), "minutes");
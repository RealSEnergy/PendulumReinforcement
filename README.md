# Pendulum control using Reinforcement Learning
This project contains several scripts used to build, train and showcase a (deep) neural network on a Pendulum problem.

## Models
All trained and untrained models are stored in the *models* directory by default. The name of the model is specified by the name of the subdirectory, which contains H5 files of untrained and trained model, and it's structure as an image.

## Model Builder
Included is also an easy to modify model builder. Feel free to tweak network hyperparameters.

## Agent
*DQNAgent* is a simple class that can load or save a specific model, decide upon the action, and remember and replay steps in the environment.

## Pendulum Gym
The gym loads a specific model - either untrained or trained - as an agent and trains it on the pendulum environment.

## Showcase
It is also possible to showcase a specific model in the pendulum environment without training it using a showcase script.
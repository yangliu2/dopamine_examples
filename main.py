import numpy as np
import os
from dopamine.agents.dqn import dqn_agent
from dopamine.atari import run_experiment
from dopamine.colab import utils as colab_utils
from absl import flags
import seaborn as sns
import matplotlib.pyplot as plt

BASE_PATH = '/tmp/colab_dope_run'
GAME = 'Asterix'

experimental_data = colab_utils.load_baselines('/content')

# Create an agent based on DQN, but choosing actions randomly.# @titl 

LOG_PATH = os.path.join(BASE_PATH, 'random_dqn', GAME)

class MyRandomDQNAgent(dqn_agent.DQNAgent):
  def __init__(self, sess, num_actions):
    """This maintains all the DQN default argument values."""
    super(MyRandomDQNAgent, self).__init__(sess, num_actions)
    
  def step(self, reward, observation):
    """Calls the step function of the parent class, but returns a random action.
    """
    _ = super(MyRandomDQNAgent, self).step(reward, observation)
    return np.random.randint(self.num_actions)

def create_random_dqn_agent(sess, environment):
  """The Runner class will expect a function of this type to create an agent."""
  return MyRandomDQNAgent(sess, num_actions=environment.action_space.n)

def main():
  # Create the runner class with this agent. We use very small numbers of steps
  # to terminate quickly, as this is mostly meant for demonstrating how one can
  # use the framework. We also explicitly terminate after 110 iterations (instead
  # of the standard 200) to demonstrate the plotting of partial runs.
  random_dqn_runner = run_experiment.Runner(LOG_PATH,
                                            create_random_dqn_agent,
                                            game_name=GAME,
                                            num_iterations=200,
                                            training_steps=10,
                                            evaluation_steps=10,
                                            max_steps_per_episode=100)

  # @title Train MyRandomDQNAgent.# @titl 
  print('Will train agent, please be patient, may be a while...')
  random_dqn_runner.run_experiment()
  print('Done training!')


  # @title Load the training logs.# @titl 
  random_dqn_data = colab_utils.read_experiment(LOG_PATH, verbose=True)
  random_dqn_data['agent'] = 'MyRandomDQN'
  random_dqn_data['run_number'] = 1
  print(experimental_data.__dict__)
  experimental_data[GAME] = experimental_data[GAME].merge(random_dqn_data,
                                                          how='outer')

  # @title Plot training results.
  fig, ax = plt.subplots(figsize=(16,8))
  sns.tsplot(data=experimental_data[GAME], time='iteration', unit='run_number',
            condition='agent', value='train_episode_returns', ax=ax)
  plt.title(GAME)
  plt.savefig('game.png')

if __name__ == '__main__':
  main()
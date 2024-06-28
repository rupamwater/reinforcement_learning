## Reinforcement Learning Projects

This project lists the Reinfrocment learning agents developed using Q learning or Deep Q learning. The following are the project details

1. Power Controller - This is a reinforcement learning agent developed using Q learning using Open AI gym to simulate the agent. The agent gets the power consumption from the consumer and available power from Solar Panel, Battery and Grid to optimise the usage of power. Highest reward is awarded to the agent for using power from Solar Panel followed by Battery and the a penalty for using power from Grid.  Random combination of connections are chosen from a possible 7 connection configuration. There are  3 possible source of power namely, a. Solar Panel, b. Battery and c. Grid and two possible consumers namely, a. House or Office and b. Battery for charging.
 
To give a simple example of Connection Configuration is that the  any combination of source of power supply and consumer can be connected to each other with the following contraints.
  a. Battery cannot be charged and supply power at the same time.
  b. Battery can be charged only from Grid or Solar panel at any point of time.
  c. The power at Home or Office would be consumed from Solar Panel, Battery and then Grid in that order of preference



  

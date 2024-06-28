## Reinforcement Learning Projects

This project lists the Reinfrocment learning agents developed using Q learning or Deep Q learning. The following are the project details

1. Power Controller - This is a reinforcement learning agent developed using Q learning using Open AI gym to simulate the agent. The agent gets the power consumption from the consumer and available power from Solar Panel, Battery and Grid to optimise the usage of power. Highest reward is awarded to the agent for using power from Solar Panel followed by Battery and the a penalty for using power from Grid.  Random combination of connections are chosen from a possible 7 connection configuration. There are  3 possible source of power namely, a. Solar Panel, b. Battery and c. Grid and two possible consumers namely, a. House or Office and b. Battery for charging.
 
To give a simple example of Connection Configuration is that the  any combination of source of power supply and consumer can be connected to each other with the following contraints.
  a. Battery cannot be charged and supply power at the same time.
  b. Battery can be charged only from Grid or Solar panel at any point of time.
  c. The power at Home or Office would be consumed from Solar Panel, Battery and then Grid in that order of preference


### Project Structure

The project has the following  classes to facilitate the learning of the agent

1. SmartWatt - This is the Reinforcement Learning agent that interact with the environment which has been implemented using the supporting classes Home, Battery
   and Panel that simulate the demand, supply and storage state at any point of time during the simulation period.
   
2. Home -     This is the supporting class that simulates the consumer of power and implements functions that returns the power demand at any partucular time of the day. This class is used in the Agent to get the power demand from the consumer
   
3. Battery -   This is a the supporting class that simulates the available charge in the battery storage and the agent then draws power based on the state of the environment and the policy of the agent.
   
4. Panel -     This is a the supporting class that simulates the available supply from the Solar Panel at any particualr time of the day.

5. Grid   -    This is a the supporting class that simulates the available supply from the Groid at any particualr time of the day.




  


### Requirements

To run the project you need to have the following softwares installed and follow the instruction in the following section on Installation

1. Anaconda
2. Git bash
   
  
### Environment setup

To setup the environment create a project directory and run the following command.

    conda create -n gymenv
    conda activate gymenv

    conda install python=3.11

    pip install gymnasium[classic_control]
    pip install gymnasium[toy-text]
    pip install gymnasium[mujoco]
    pip install gymnasium[atari]
    pip install gymnasium[accept-rom-license]
    pip install gymnasium[box2d]
    
    conda install swig
    
    pip install Theano
    pip install numpy scipy ipython pandas matplotlib
    pip install tensorflow==2.12.0 --upgrade

### How to run

To run the Power controller project following the following steps

1. Open the project directory in command prompt ie. PowerShell
2. Activate the conda environment setup in the previous steps with all the pre requisite libraries installed

   conda activate gymenv

3. python power_controller.py

   


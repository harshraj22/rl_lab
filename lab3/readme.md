## Policy And Value Iteration


#### Problem Formulation
For the current lab, we consider the following Grid world environment as MDP:


| | | | |
|-|-|-|-|
|0 |0 |0 |1 |
|0 |0 |0 |-100 |
|0 |XXX|0 |0 |
|0 |0 |0 |0 |

The cells represent reward associated with the states, `xxx` represents the state that is
blocked. The states with non zero reward are absorbing states. An agent reaching an absorbing
state can not move out of the state and the episode ends once an agent reaches the absorbing
state.


If an agent selects an action a, there is 80% probability of the agent moving in that direction,
and 10% each probability of moving in the direction orthogonal to the selected action a.

Even though this is a finite horizon setting, We formulate this as a discounted reward MDP (with
discount factor 0.9), encouraging the agent to choose the actions that make them reach the
optimal state in the least possible number of steps.
## Policy And Value Iteration


#### 1. Problem Formulation
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

#### 2. Observations and conclusion:
##### 2.1 Value Iteration

![value_functions](https://user-images.githubusercontent.com/46635452/159168728-d378144b-0a5b-4f35-8010-687f930b8723.gif)

Value function corresponding to each state per iteration


<img width="1367" alt="image" src="https://user-images.githubusercontent.com/46635452/159168604-5c491440-b8bb-47cc-8c9a-c4a6b0ea73db.png">
Here we show the average value of difference in sum of value functions of states across each
update. As can be observed from the given plot, the difference between value functions is large
initially, and gradually it starts to decrease giving hints that the value function is approaching the
optimal value function of the given task.
We use the following equation to obtain the average value function difference:

```python
avg_diff = (old_val_func - new_val_func).abs().sum()
```


<img width="1367" alt="image" src="https://user-images.githubusercontent.com/46635452/159168688-08868776-bb5b-4596-8130-11caf47f695f.png">
Here we show the maximum difference between the value of function of states across each update.
As can be observed from the given plot, the largest difference between any state is decreasing
across the iterations. This also gives strong hints that the value function has approached the
optimal value function.
We use the following equation to obtain the max value function difference:

```python
max_diff = (old_val_func - new_val_func).abs().max()
```

##### 2.2 Policy Iteration
![policy_iteration_agent_gif](https://user-images.githubusercontent.com/46635452/159168809-4bddbbc2-0e3f-4763-8922-0894fb06929a.gif)

Policies corresponding to each state per iteration

<strong>Note</strong>: Some states have almost equal values after convergence. This justifies the fact that even
after convergence, policy of some state is altering between two directions, as both the next states
have almost the same values.

<img width="1367" alt="image" src="https://user-images.githubusercontent.com/46635452/159168841-83e8d63f-bbaf-4327-aefe-1e7ef6900191.png">

Here we show the average value of difference in sum of value functions of states across each
update. As can be observed from the given plot, the difference between value functions decreases
and suddenly jumps at times. This could be understood by the fact that the policy evaluation
process makes the value function converge to the optimal value function corresponding to the
current policy. Then policy improvement comes, and the difference grows again as the optimal value
function corresponding to the new policy is different from that of the old policy. Gradually the policy
converges to optimal policy, and we do not see any more jumps in the difference of value function.
We use the following equation to obtain the average value function difference:

```python
avg_diff = (old_val_func - new_val_func).abs().sum()

```


<img width="1367" alt="image" src="https://user-images.githubusercontent.com/46635452/159168888-2c0a4056-e87f-48d0-afb2-4cedb41af21a.png">

Here we show the maximum difference between the value of function of states across each update.
A trend similar to what above is observed.
We use the following equation to obtain the max value function difference:

```python
max_diff = (old_val_func - new_val_func).abs().max()

```

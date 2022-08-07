# mancala.ai

I used the MCTS/DNN approach from AlphaGo Zero to create a mancala-playing AI.

## Technique of AlphaGo Zero

### Deep Neural Network (DNN)

AlphaGo Zero uses a DNN $f_{\theta}$ with parameters $\theta$.

$f_\theta$ takes in a raw game board representation $s$ and outputs:
  1. For each possible action $a$ from $s$, the probability $p_a=Pr(a|s)$ of the current player selecting $a$ from $s$.
  2. The probability $v$ of the current player winner from position $s$.

In other words:
$$f_\theta(s)=(\mathbf p, v)$$
where $\mathbf p$ is a vector of probabilities $p_a=Pr(a|s)$ for each action $a$ from $s$.

### Monte Carlo Tree Search (MCTS)

To improve the parameters $\theta$ of the DNN, AlphaGo Zero uses MCTS to gain experience playing the game.



# DeepQLearning - Library
Implementation of all the QLearning variations.

## Features
- [X] Vanilla RAM/Convolution Network
- [X] Dueling Network Architecture
- [ ] Noisy Network
- [ ] Distributional RL
- [X] Preprocessor
- [X] Stack
- [X] Stack Memory
- [ ] Prioritised Memory/Prioritised Experience Replay
- [ ] SARSA
- [ ] Q(0)/TD(0)
- [ ] Q(λ)/TD(λ)
- [ ] Soft Target Updates

Currently thinking about the best way to solve this.
In the end the user should simply be able to pass arguments and the appropriate environment, algorithm, network and features should be chosen.
Args[algorithm = "q/sarsa",lr = X, channels = 0/X , frame_stack = X, observation_space= X,(X,X), action_space = X, action_offset = X, loss = "huber/mse", reduction = "elementwise_mean/none", dueling = True/False, optim = "sgd/rmsprop/...", memory = "none/stack/prioritized", memory_size = X, batch_size = X, warmup_steps = X, target = "none/target/double", soft_update = 0.0-1.0, replace_epochs = X, td_lambda = X, ...]
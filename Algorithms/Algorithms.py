
#Need to find a clever way to let the algorithms run
#Should i create an extra class for every algorithm type or just check and set appropriate methods

class SARSA:
    # TO BE IMPLEMENTED
    def runfor(self,env,agent,epochs,max_timestep):
        for e in range(epochs):
            state = env.reset()

            for f in range(max_timestep):
                env.render()
                action = agent()

class QLearning:
    # TO BE IMPLEMENTED
    pass
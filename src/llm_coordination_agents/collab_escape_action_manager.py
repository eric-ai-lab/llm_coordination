from llm_coordination_agents.collab_escape_agent import LLMAgent

class CollabEscapeActionManager():
    def __init__(self, player_id, mdp):
        self.mdp = mdp 
        self.llm_agent = LLMAgent(player_id, mdp)
    
    #def get_next_move(self, state)

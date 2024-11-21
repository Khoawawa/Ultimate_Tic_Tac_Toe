import numpy as np
import time
from math import *
from copy import *

def select_move(cur_state, remain_time):

    # valid_moves = cur_state.get_valid_moves
    # if len(valid_moves) != 0:
    #     return np.random.choice(valid_moves)
    # return None
    mcts = MCTS(cur_state.player_to_move)
    return mcts.getMove(cur_state)


class Node():
    def __init__(self, parent, state):
        self.total_simulations = 0
        self.score = 0
        self.parent = parent
        self.children = []
        self.state = state

    def expand(self):
        next_moves = self.state.get_valid_moves

        for move in next_moves:
        # Deep copy then play a move to get a new state
            new_state = deepcopy(self.state)
            new_state.act_move(move)

            new_node = Node(self, new_state)
            self.children.append(new_node)

    
    def backPropagate(self, game_result):
        self.total_simulations += 1
        self.score += game_result

        # If not root, backPropagate
        if self.parent != None:
            self.parent.backPropagate(game_result)
    
    # UCB1 = Exploitation Term + Exploration Term

    def getExplorationTerm(self):
        """ Encourages the algorithm to explore nodes that have been visited fewer times """
        if self.total_simulations == 0:
            return inf
        return sqrt(log(self.parent.total_simulations) / (self.total_simulations))

    def getExploitationTerm(self):
        """ Focuses on the quality of the node based on the average score from its simulations """
        if self.total_simulations == 0:
            return inf
        return self.score / (self.total_simulations)

class MCTS():
    def __init__(self, player_turn, C = sqrt(2)):
        self.player_turn = player_turn
        self.move_time = 3
        self.C = C

    def selection(self, current_node, turn):
        current_state = current_node.state

        # If selection reach leaf node have finished or not expanded
        if current_state.game_over or len(current_node.children) == 0:
            return current_node

        if turn == self.player_turn:
            sorted_children = sorted(current_node.children, key=lambda child: child.getExploitationTerm() + self.C*child.getExplorationTerm(), reverse=True)
        else:
            sorted_children = sorted(current_node.children, key=lambda child: -child.getExploitationTerm() - self.C*child.getExplorationTerm(), reverse=True)
        return self.selection(sorted_children[0], turn * -1)
    
    def simulate(self, state):
        state.game_result(state.global_cells.reshape(3,3))
        if not state.game_over:
            moves = state.get_valid_moves

            if len(moves) == 0: # Game_Result bug ?
                # score = state.global_cells.sum()
                # return score * 0.1 - 0.1
                return 0
            
            # Randmoly choose the next move
            random_move = np.random.choice(moves)
            state.act_move(random_move)

            return self.simulate(state)
        
        else:
            final_result = state.game_result(state.global_cells.reshape(3,3))
            if final_result == self.player_turn:
                return 1
            else:
                return -1
        
    def getMove(self, state):
        
        root_node = Node(None, deepcopy(state))
        i = 0
        start_time = time.time()
        while time.time() - start_time < self.move_time:
            selected_node = self.selection(root_node, self.player_turn)

            if selected_node.total_simulations == 0 and selected_node != root_node:
                game_result = self.simulate(deepcopy(selected_node.state))
                selected_node.backPropagate(game_result)
            else:
                selected_node.expand()
            i = i+1

        print(i)
        if len(root_node.children) == 0:
            return None
        
        # winning_children = [child for child in root_node.children if child.state.game_over == True and child.state.player_to_move == self.player_turn]
        # if len(winning_children) != 0:
        #     return winning_children[0].state.previous_move
        
        sorted_children = sorted(root_node.children, key=lambda child: child.getExploitationTerm(), reverse=True)
        return sorted_children[0].state.previous_move


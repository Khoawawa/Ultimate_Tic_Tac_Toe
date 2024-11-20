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
        next_moves = [move for move in next_moves if move not in self.children]
        move = np.random.choice(next_moves)

        # Deep copy then play a move to get a new state
        new_state = deepcopy(self.state)
        new_state.act_move(move)

        new_node = Node(self, new_state)
        self.children.append(new_node)

        return new_node
    
    def backPropagate(self, game_result):
        self.total_simulations += 1
        self.score += game_result

        # If not root, backPropagate
        if self.parent != None:
            self.parent.backPropagate(game_result)
    
    # UCB1 = Exploitation Term + Exploration Term

    def getExplorationTerm(self):
        """ Encourages the algorithm to explore nodes that have been visited fewer times """
        return sqrt(log(self.parent.total_simulations)) / (self.total_simulations or 1)

    def getExploitationTerm(self):
        """ Focuses on the quality of the node based on the average score from its simulations """
        return self.score / (self.total_simulations or 1)

class MCTS():
    def __init__(self, player_turn, C = sqrt(2)):
        self.player_turn = player_turn
        self.move_time = 10
        self.C = C

    def selection(self, current_node, turn):
        current_state = current_node.state

        # If selection reach leaf node have finished or not expanded
        if current_state.game_over or len(current_node.children) == 0:
            return current_node

        if turn == self.player_turn:
            sorted_children = sorted(current_node.children, key=lambda child: child.getExploitationTerm() + self.C*child.getExplorationTerm(), reverse=True)
        else:
            sorted_children = sorted(current_node.children, key=lambda child: -child.getExploitationTerm() + self.C*child.getExplorationTerm(), reverse=True)
        return self.selection(sorted_children[0], turn * -1)
    
    def simulate(self, state):
        
        if not state.game_over:
            moves = state.get_valid_moves

            # Randmoly choose the next move
            random_move = np.random.choice(moves)
            state.act_move(random_move)

            return self.simulate(state)
        
        else:
            if state.game_result != None and state.game_result == self.player_turn:
                return 1
            elif state.game_result != None and state.game_result != self.player_turn:
                return -1
            else:
                return 0
        
    def getMove(self, state):
        
        root_node = Node(None, deepcopy(state))
        i = 0
        start_time = time.time()
        while time.time() - start_time < self.move_time:
            selected_node = self.selection(root_node, self.player_turn)
            child_selected_node = selected_node.expand()
            game_result = self.simulate(child_selected_node.state)
            child_selected_node.backPropagate(game_result)
            i = i+1

        print(i)
        if root_node.children == None:
            return None
        sorted_children = sorted(root_node.children, key=lambda child: child.getExploitationTerm(), reverse=True)
        return sorted_children.previous_move


import numpy as np

def select_move(cur_state, remain_time):
    valid_moves = cur_state.get_valid_moves
    if len(valid_moves) != 0:
        return np.random.choice(valid_moves)
    return None

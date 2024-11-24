import numpy as np
from state import *
from copy import deepcopy

GAME_SCORE = float('inf')
BOX_SCORE = 100
BLOCK_SCORE = 150
BLOCK_SMALL_SCORE = 20
CONSECUTIVE_SCORE = 200
CONSECUTIVE_SMALL_SCORE = 5
NO_BENEFIT = - 150
NO_BENEFIT_SMALL = -20
DEPTH = 5

def block(cur_state,player,block_idx,score_method):
    r = block_idx // 3
    c = block_idx % 3
    
    global_cells = cur_state.global_cells
    global_cells_3x3 = np.array(global_cells).reshape(3, 3)
    return block_small(global_cells_3x3,r,c,player,score_method)
    
def block_small(board,r,c,player,score_method):
    return check_two_in_a_row_small(board,r,c,-player,score_method,False)

def check_two_in_a_row_small(board, r, c, player, score_method, is_limit_one):
    """
    Check for two-in-a-row scenarios when placing a move on a 3x3 board.

    Args:
        board (list[list]): The game board as a 3x3 matrix.
        r (int): Row index of the current cell.
        c (int): Column index of the current cell.
        player (str): The player symbol to check ('X' or 'O').
        score_method (int): A multiplier for the calculated score.
        is_limit_one (bool): If True, stop after detecting one match per direction.
                             If False, require two matches for scoring.

    Returns:
        int: The calculated score for the move.
    """
    def count_matches(cells, target, is_limit_one):
        """Helper function to count matches in a line."""
        matches = 0
        for cell in cells:
            if cell == target:
                matches += 1
                if is_limit_one:
                    return 1
        return 1 if matches >=2 else 0  # Score only if two matches found when not limited.

    score = 0

    # Check the row
    score += count_matches([cell for i, cell in enumerate(board[r]) if i != c], player, is_limit_one)

    # Check the column
    score += count_matches([cell for j, cell in enumerate(np.transpose(board)[c]) if j != r], player, is_limit_one)

    # Check the main diagonal
    if r == c:
        score += count_matches([board[i][i] for i in range(3) if i != r], player, is_limit_one)

    # Check the anti-diagonal
    if r + c == 2:
        score += count_matches([board[i][2 - i] for i in range(3) if i != r], player, is_limit_one)

    return score * score_method

def check_two_in_a_row(cur_state: State, player, block_idx,score_method):
    r = block_idx // 3
    c = block_idx % 3
    
    global_cells = cur_state.global_cells
    global_cells_3x3 = np.array(global_cells).reshape(3, 3)
    
    return check_two_in_a_row_small(global_cells_3x3,r,c,player,score_method,True)
def global_check_blocked_win(cur_state,player, blk_idx, score_method):
    r = blk_idx // 3
    c = blk_idx % 3
    
    global_cells = cur_state.global_cells
    global_cells_3x3 = np.array(global_cells).reshape(3, 3)
    
    return check_blocked_win(global_cells_3x3,player,r,c,score_method)

def check_blocked_win(board, player, r, c, score_method):
    def _check_block(cells,target):
        player = opponent = 1
        for cell in cells:
            if cell == target:
                player -= 1
            elif cell == opponent:
                opponent -=1
                
        return 1 if player == 0 and opponent == 0 else 0
    score = 0
    
    score += _check_block([cell for i, cell in enumerate(board[r]) if i != c],player)
    score += _check_block([cell for j, cell in enumerate(np.transpose(board)[c]) if j != r],player)
    if r== c:
        score += _check_block([board[i][i] for i in range(3) if i != r],player)
    if r + c == 2:
        score += _check_block([board[i][2 - i] for i in range(3) if i != r], player)
    
    return score * score_method
    

def evaluate_search(cur_state: State, player):
    heuristic = 0
    global_board_3x3 = cur_state.global_cells.reshape(3,3)
    ### check for game win
    game_result = cur_state.game_result(global_board_3x3)
    if game_result is not None:
        return float('inf')*game_result*player
    ### check for small board win
    move = cur_state.previous_move
    blk_idx = move.index_local_board
    blk_r = blk_idx // 3
    blk_c = blk_idx % 3
    
    board_result = cur_state.game_result(cur_state.blocks[blk_idx])
    # WINNING A BOARD
    if board_result:
        heuristic += BOX_SCORE*board_result*player
        
        if board_result == player:
            # WINNING A BOARD RESULT IN MULTIPLE BOARD WIN
            heuristic += check_two_in_a_row_small(global_board_3x3,blk_r,blk_c,player,CONSECUTIVE_SCORE,True)
            # WINNING a BOARD RESULTS IN BLOCKING
            heuristic += block_small(global_board_3x3,blk_r,blk_c, player,BLOCK_SCORE)
            # WINNING a BOARD BLOCK BY OPPONENT
            heuristic += check_blocked_win(global_board_3x3,player,blk_r,blk_c,NO_BENEFIT)
    #NOT WINNING THE BOARD      
    else:
        #MAKING 2 MARKS IN A ROW ON LOCAL BOARD
        heuristic += check_two_in_a_row_small(cur_state.blocks[blk_idx],move.x,move.y,player,CONSECUTIVE_SMALL_SCORE,True)
        heuristic += block_small(cur_state.blocks[blk_idx],move.x,move.y,player,BLOCK_SMALL_SCORE)
        heuristic += check_blocked_win(cur_state.blocks[blk_idx],player,move.x,move.y,NO_BENEFIT_SMALL)
        
    return heuristic
            
            
def minimax(cur_state:State, depth,maximize_player,player, alpha=float('-inf'), beta=float('inf')):
    if depth == 0 or cur_state.game_over or cur_state.get_valid_moves is None:
        return evaluate_search(cur_state, player), None
    
    best_move = None
    if maximize_player:
        max_eval = float('-inf')
        for move in cur_state.get_valid_moves:
            # print('The current state is: ',cur_state.blocks)
            # print('The current move is: ',move)
            new_state = deepcopy(cur_state) 
            # if not new_state.is_valid_move(move):
            #     continue
            new_state.act_move(move)
            # print(new_state)
            eval, _ = minimax(new_state,depth - 1, not maximize_player,  player,alpha, beta)
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    
    else:
        min_eval = float('inf')
        for move in cur_state.get_valid_moves:
            # print('The current state is: ',cur_state.blocks)
            # print('The current move is: ',move)
            new_state = deepcopy(cur_state)
            # if not new_state.is_valid_move(move):
            #     continue
            new_state.act_move(move)
            eval, _ = minimax(new_state,depth - 1, not maximize_player,player,alpha, beta)
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta,min_eval)
            if beta <= alpha:
                break
        return min_eval, best_move
            
def select_move(cur_state, remain_time):
    _,move = minimax(cur_state,DEPTH,True,cur_state.player_to_move)
    print(remain_time)
    return move

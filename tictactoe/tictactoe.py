"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x = 0
    o = 0
    for row in board:
        for cell in row:
            if cell == X:
                x += 1
            elif cell == O:
                o += 1
    if x == o:
        return X
    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()

    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                moves.add((i, j))
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action

    if i < 0 or i > 2 or j < 0 or j > 2:
        raise Exception("Invalid move")

    if board[i][j] != EMPTY:
        raise Exception("Invalid move")
    board_copy = copy.deepcopy(board)
    board_copy[i][j] = player(board)
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not EMPTY:
            return board[i][0]

    for j in range(3):
        if board[0][j] == board[1][j] == board[2][j] and board[0][j] is not EMPTY:
            return board[0][j]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not EMPTY:
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not EMPTY:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False

    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    if player(board) == X:
        best_score = -math.inf
        best_action = None
        for action in actions(board):
            score = min_value(
                result(board, action)
            )
            if score > best_score:
                best_score = score
                best_action = action
    else:
        best_score = math.inf
        best_action = None
        for action in actions(board):
            score = max_value(
                result(board, action)
            )
            if score < best_score:
                best_score = score
                best_action = action

    return best_action


def max_value(board):
    if terminal(board):
        return utility(board)

    value = -math.inf

    for action in actions(board):
        value = max(value, min_value(result(board, action)))

    return value


def min_value(board):
    if terminal(board):
        return utility(board)

    value = math.inf

    for action in actions(board):
        value = min(value, max_value(result(board, action)))

    return value

"""
chess_game.py
Simple GUI chess: local 1v1 (pass-and-play) + play vs AI (minimax w/ alpha-beta using python-chess).

Requirements:
    pip install pygame python-chess

Run:
    python chess_game.py

Controls:
 - Click a piece to select it, click a square to move.
 - Buttons: New Game, Undo, Toggle AI (AI plays Black), AI Depth (- / +)
 - Promotion auto-queens (simple).
"""

import pygame
import sys
import threading
import time
import chess

# ---------- Config ----------
WIDTH, HEIGHT = 640, 640
SQUARE = WIDTH // 8
PANEL_WIDTH = 220
FPS = 30
AI_THINK_DELAY = 0.15

# Colors
LIGHT = (240, 217, 181)
DARK = (181, 136, 99)
PANEL_BG = (40, 40, 40)
TEXT = (240, 240, 240)
HIGHLIGHT = (80, 200, 120, 110)

# Piece values for simple evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        # if side to move is checkmated it's bad for them
        return -999999 if board.turn else 999999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    score = 0
    for ptype, val in PIECE_VALUES.items():
        score += val * (len(board.pieces(ptype, chess.WHITE)) - len(board.pieces(ptype, chess.BLACK)))
    # mobility
    score += 5 * (len(list(board.legal_moves)) if board.turn else -len(list(board.legal_moves)))
    return score

def minimax(board: chess.Board, depth: int, alpha: int, beta: int, maximizing: bool):
    if depth == 0 or board.is_game_over():
        return evaluate(board), None
    best_move = None
    if maximizing:
        max_eval = -10**9
        for mv in board.legal_moves:
            board.push(mv)
            val, _ = minimax(board, depth-1, alpha, beta, False)
            board.pop()
            if val > max_eval:
                max_eval = val
                best_move = mv
            alpha = max(alpha, val)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = 10**9
        for mv in board.legal_moves:
            board.push(mv)
            val, _ = minimax(board, depth-1, alpha, beta, True)
            board.pop()
            if val < min_eval:
                min_eval = val
                best_move = mv
            beta = min(beta, val)
            if beta <= alpha:
                break
        return min_eval, best_move

# ---------- Pygame setup ----------
pygame.init()
screen = pygame.display.set_mode((WIDTH + PANEL_WIDTH, HEIGHT))
pygame.display.set_caption("Mini Chess — 1v1 & Bot")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 20)
bigfont = pygame.font.SysFont(None, 36)

# Unicode pieces (using text icons)
UNICODE_PIECES = {
    chess.PAWN:   "♙",
    chess.KNIGHT: "♘",
    chess.BISHOP: "♗",
    chess.ROOK:   "♖",
    chess.QUEEN:  "♕",
    chess.KING:   "♔"
}
UNICODE_PIECES_BLACK = {
    chess.PAWN:   "♟",
    chess.KNIGHT: "♞",
    chess.BISHOP: "♝",
    chess.ROOK:   "♜",
    chess.QUEEN:  "♛",
    chess.KING:   "♚"
}

# Game state
board = chess.Board()
selected = None
highlight_squares = []
move_history = []
ai_enabled = False   # when True: AI plays Black
ai_depth = 2
ai_thinking = False

def coord_to_square(pos):
    x, y = pos
    if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
        return None
    file = x // SQUARE
    rank = 7 - (y // SQUARE)
    return chess.square(file, rank)

def square_to_coord(sq):
    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    x = file * SQUARE
    y = (7 - rank) * SQUARE
    return x, y

def draw_board():
    # draw squares
    for r in range(8):
        for f in range(8):
            rect = pygame.Rect(f*SQUARE, r*SQUARE, SQUARE, SQUARE)
            color = LIGHT if (r + f) % 2 == 0 else DARK
            pygame.draw.rect(screen, color, rect)
    # highlights
    for sq in highlight_squares:
        x, y = square_to_coord(sq)
        s = pygame.Surface((SQUARE, SQUARE), pygame.SRCALPHA)
        s.fill(HIGHLIGHT)
        screen.blit(s, (x, y))
    # pieces
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            x, y = square_to_coord(sq)
            # draw a circle bg to make piece visible
            pygame.draw.circle(screen, (120,120,120) if p.color==chess.BLACK else (220,220,220),
                               (x + SQUARE//2, y + SQUARE//2), SQUARE//2 - 8)
            if p.color == chess.WHITE:
                text = UNICODE_PIECES[p.piece_type]
            else:
                text = UNICODE_PIECES_BLACK[p.piece_type]
            txt = bigfont.render(text, True, (0,0,0))
            rect = txt.get_rect(center=(x + SQUARE//2, y + SQUARE//2 + 2))
            screen.blit(txt, rect)

def draw_panel():
    panel_rect = pygame.Rect(WIDTH, 0, PANEL_WIDTH, HEIGHT)
    pygame.draw.rect(screen, PANEL_BG, panel_rect)
    # Buttons text (simple)
    lines = [
        ("New Game", (WIDTH + 20, 18)),
        ("Undo", (WIDTH + 20, 58)),
        (f"AI: {'On' if ai_enabled else 'Off'}", (WIDTH + 20, 98)),
        ("Toggle AI", (WIDTH + 20, 128)),
        ("AI Depth:", (WIDTH + 20, 168)),
        (str(ai_depth), (WIDTH + 140, 168)),
        ("Depth -", (WIDTH + 20, 200)),
        ("Depth +", (WIDTH + 120, 200)),
    ]
    for text, pos in lines:
        screen.blit(font.render(text, True, TEXT), pos)
    # status area
    status = "White to move" if board.turn else "Black to move"
    if board.is_check():
        status += "  (CHECK)"
    if board.is_checkmate():
        status = "Checkmate! " + ("Black wins" if board.turn else "White wins")
    if board.is_stalemate():
        status = "Stalemate!"
    screen.blit(font.render("Status:", True, TEXT), (WIDTH + 20, 250))
    screen.blit(font.render(status, True, TEXT), (WIDTH + 20, 270))
    # instruction
    screen.blit(font.render("Click: select -> move", True, (200,200,200)), (WIDTH + 20, 320))
    if ai_thinking:
        screen.blit(font.render("AI thinking...", True, (255,200,0)), (WIDTH + 20, 350))

def button_at_pos(pos):
    x, y = pos
    mapping = {
        "new": pygame.Rect(WIDTH+20, 12, 160, 28),
        "undo": pygame.Rect(WIDTH+20, 52, 80, 28),
        "toggle": pygame.Rect(WIDTH+20, 124, 120, 28),
        "depth_minus": pygame.Rect(WIDTH+20, 196, 80, 28),
        "depth_plus": pygame.Rect(WIDTH+120, 196, 80, 28),
    }
    for name, rect in mapping.items():
        if rect.collidepoint(x, y):
            return name
    return None

def attempt_move(from_sq, to_sq):
    # supports auto-queen on promotion
    mv = chess.Move(from_sq, to_sq)
    if mv in board.legal_moves:
        board.push(mv)
        move_history.append(mv)
        return True
    # try promotion to queen
    promo = chess.Move(from_sq, to_sq, promotion=chess.QUEEN)
    if promo in board.legal_moves:
        board.push(promo)
        move_history.append(promo)
        return True
    return False

def ai_worker(depth):
    global ai_thinking
    ai_thinking = True
    time.sleep(AI_THINK_DELAY)
    # AI plays Black in this simple setup (human is White)
    maximizing = False  # since from eval's POV, maximizing True == White
    _, best = minimax(board, depth, -10**9, 10**9, maximizing)
    if best is not None:
        board.push(best)
        move_history.append(best)
    ai_thinking = False

def start_ai_if_needed():
    # If AI enabled and it's black's turn and not thinking and game not over -> start AI
    if ai_enabled and (not board.turn) and (not ai_thinking) and (not board.is_game_over()):
        t = threading.Thread(target=ai_worker, args=(ai_depth,), daemon=True)
        t.start()

def reset_game():
    global board, selected, highlight_squares, move_history, ai_thinking
    board = chess.Board()
    selected = None
    highlight_squares = []
    move_history = []
    ai_thinking = False

def undo():
    global move_history
    if len(move_history) >= 1:
        board.pop()
        move_history.pop()
    # If AI is enabled, also attempt to undo opponent move (so undo full turn)
    if ai_enabled and len(move_history) >= 1:
        board.pop()
        move_history.pop()

def main():
    global selected, highlight_squares, ai_enabled, ai_depth
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
                break
            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                pos = pygame.mouse.get_pos()
                # Board click?
                if pos[0] < WIDTH:
                    sq = coord_to_square(pos)
                    if sq is None:
                        continue
                    piece = board.piece_at(sq)
                    if selected is None:
                        # select a piece (must be piece of side to move)
                        if piece is not None and piece.color == board.turn:
                            selected = sq
                            # highlight legal moves from selected
                            highlight_squares = [m.to_square for m in board.legal_moves if m.from_square == sq]
                    else:
                        # attempt move
                        if attempt_move(selected, sq):
                            selected = None
                            highlight_squares = []
                            # after human move, maybe AI should move
                            start_ai_if_needed()
                        else:
                            # if clicked own other piece, change selection
                            p2 = board.piece_at(sq)
                            if p2 is not None and p2.color == board.turn:
                                selected = sq
                                highlight_squares = [m.to_square for m in board.legal_moves if m.from_square == sq]
                            else:
                                selected = None
                                highlight_squares = []
                else:
                    # panel click
                    btn = button_at_pos(pos)
                    if btn == "new":
                        reset_game()
                    elif btn == "undo":
                        undo()
                    elif btn == "toggle":
                        ai_enabled = not ai_enabled
                        # If toggled on and it's AI's turn, start AI
                        start_ai_if_needed()
                    elif btn == "depth_minus":
                        if ai_depth > 1:
                            ai_depth -= 1
                    elif btn == "depth_plus":
                        if ai_depth < 4:
                            ai_depth += 1

        # if AI is enabled and it's black to move, let it start (if not already started)
        start_ai_if_needed()

        # drawing
        screen.fill((0,0,0))
        draw_board()
        draw_panel()
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()

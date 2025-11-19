import os.path
import sys
import chess
import chess.polyglot
from tables import *

values = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 20000
}

transposition_table = {}
checkmate_value = 9999


def get_score_diff(board: chess.Board) -> int:
    wp = len(board.pieces(chess.PAWN, chess.WHITE)) * values[chess.PAWN]
    bp = len(board.pieces(chess.PAWN, chess.BLACK)) * values[chess.PAWN]
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE)) * values[chess.KNIGHT]
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK)) * values[chess.KNIGHT]
    wb = len(board.pieces(chess.BISHOP, chess.WHITE)) * values[chess.BISHOP]
    bb = len(board.pieces(chess.BISHOP, chess.BLACK)) * values[chess.BISHOP]
    wr = len(board.pieces(chess.ROOK, chess.WHITE)) * values[chess.ROOK]
    br = len(board.pieces(chess.ROOK, chess.BLACK)) * values[chess.ROOK]
    wq = len(board.pieces(chess.QUEEN, chess.WHITE)) * values[chess.QUEEN]
    bq = len(board.pieces(chess.QUEEN, chess.BLACK)) * values[chess.QUEEN]

    return (wp - bp) + (wn - bn) + (wb - bb) + (wr - br) + (wq - bq)


def is_endgame(board: chess.Board) -> bool:
    material = 0
    for piece_type, val in values.items():
        if piece_type != chess.KING:
            material += val * (
                    len(board.pieces(piece_type, chess.WHITE)) +
                    len(board.pieces(piece_type, chess.BLACK))
            )
    return material <= 1400


def get_king_safety(board: chess.Board, color: chess.Color) -> int:
    defense_score = 0
    king_square = board.king(color)
    if king_square is None:
        return 0

    king_file = chess.square_file(king_square)
    king_rank = chess.square_rank(king_square)

    for i in range(-1, 2):
        for j in range(-1, 2):
            nf = king_file + i
            nr = king_rank + j

            if nf < 0 or nf > 7 or nr < 0 or nr > 7:
                continue

            def_square = chess.square(nf, nr)
            if def_square == king_square:
                continue

            defender = board.piece_at(def_square)
            if defender is None or defender.color != color:
                continue

            if defender.piece_type == chess.PAWN:
                if color == chess.WHITE:
                    if nr == king_rank + 1:
                        defense_score += 5
                    else:
                        defense_score += 1
                elif color == chess.BLACK:
                    if nr == king_rank - 1:
                        defense_score += 5
                    else:
                        defense_score += 1
            elif defender.piece_type in [chess.KNIGHT, chess.BISHOP]:
                defense_score += 3
            else:
                defense_score += 1

    king_attackers = board.attackers(not color, king_square)
    defense_score -= len(king_attackers) * 20

    if not is_endgame(board):
        nb_empty_right = 0
        nb_empty_left = 0
        right_file = king_file + 1
        left_file = king_file - 1

        if 0 <= right_file <= 7:
            for rank in range(0, 8):
                square = chess.square(right_file, rank)
                piece = board.piece_at(square)
                if piece is None:
                    nb_empty_right += 1

        if 0 <= left_file <= 7:
            for rank in range(0, 8):
                square = chess.square(left_file, rank)
                piece = board.piece_at(square)
                if piece is None:
                    nb_empty_left += 1

        if nb_empty_right == 8:
            defense_score -= 10

        if nb_empty_left == 8:
            defense_score -= 10

    return defense_score

def rook_open_bonus(board: chess.Board, color: chess.Color) -> int:
    rooks = board.pieces(chess.ROOK, color)
    score = 0

    for rook in rooks:
        free_square = 0
        file = chess.square_file(rook)
        rank = chess.square_rank(rook)

        for r in range(0, 8):
            square = chess.square(file, r)
            piece = board.piece_at(square)
            if piece is None:
                free_square += 1
            elif piece.piece_type == chess.PAWN:
                free_square += 0.5

        if free_square >= 7:
            score += 20
        elif free_square >= 6.5:
            score += 10

        if (color == chess.WHITE and rank == 6) or (color == chess.BLACK and rank == 1):
            score += 40

    return score


def connected_rooks(board: chess.Board, color: chess.Color) -> int:
    rooks = list(board.pieces(chess.ROOK, color))
    if len(rooks) < 2:
        return 0

    score = 0

    for i in range(len(rooks)):
        for j in range(i + 1, len(rooks)):
            f1 = chess.square_file(rooks[i])
            r1 = chess.square_rank(rooks[i])
            f2 = chess.square_file(rooks[j])
            r2 = chess.square_rank(rooks[j])

            # Même rangée
            if r1 == r2 and f1 != f2:
                min_file = min(f1, f2)
                max_file = max(f1, f2)

                connected = True
                for file in range(min_file + 1, max_file):
                    sq = chess.square(file, r1)
                    if board.piece_at(sq) is not None:
                        connected = False
                        break

                if connected:
                    score += 20

            # Même colonne
            elif f1 == f2 and r1 != r2:
                min_rank = min(r1, r2)
                max_rank = max(r1, r2)

                connected = True
                for rank in range(min_rank + 1, max_rank):
                    sq = chess.square(f1, rank)
                    if board.piece_at(sq) is not None:
                        connected = False
                        break

                if connected:
                    score += 15

    return score


def passed_pawns_bonus(board: chess.Board, color: chess.Color) -> int:
    pawns = board.pieces(chess.PAWN, color)
    opp_pawns = board.pieces(chess.PAWN, not color)
    opp_pawns_files = [chess.square_file(p) for p in opp_pawns]

    score = 0

    for pawn in pawns:
        file = chess.square_file(pawn)
        if len(list({file, file - 1, file + 1} & set(opp_pawns_files))) == 0:
            rank = chess.square_rank(pawn)
            if color == chess.WHITE:
                if rank == 6:
                    score += 70
                else:
                    score += 30 + (rank ** 2)
            else:
                if rank == 1:
                    score += 70
                else:
                    score += 30 + ((7 - rank) ** 2)

    return score


def game_phase(board: chess.Board) -> int:
    phase_weights = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK: 2,
        chess.QUEEN: 4,
    }
    phase = 0
    max_phase = 24

    for piece_type, weight in phase_weights.items():
        phase += weight * (
                len(board.pieces(piece_type, chess.WHITE)) +
                len(board.pieces(piece_type, chess.BLACK))
        )

    phase = max_phase - phase
    phase = (phase * 256) // max_phase
    return phase


def king_tapered_eval(board: chess.Board) -> int:
    phase = game_phase(board)

    white_sq = board.king(chess.WHITE)
    black_sq = board.king(chess.BLACK)

    if white_sq is None or black_sq is None:
        return 0

    w_open = kingstable_opening[white_sq]
    b_open = kingstable_opening[chess.square_mirror(black_sq)]

    w_end = kingstable_endgame[white_sq]
    b_end = kingstable_endgame[chess.square_mirror(black_sq)]

    w_eval = (w_open * (256 - phase) + w_end * phase) // 256
    b_eval = (b_open * (256 - phase) + b_end * phase) // 256

    return round(w_eval - b_eval)


def pawn_structure_penalty(board: chess.Board, color: chess.Color) -> int:
    pawns = board.pieces(chess.PAWN, color)
    files = [chess.square_file(p) for p in pawns]
    penalty = 0

    # Pions doublés
    for f in range(8):
        count = files.count(f)
        if count > 1:
            penalty += 20 * (count - 1)

    # Pions isolés
    for p in pawns:
        f = chess.square_file(p)
        has_left_neighbor = any(chess.square_file(other) == f - 1 for other in pawns if other != p)
        has_right_neighbor = any(chess.square_file(other) == f + 1 for other in pawns if other != p)

        if not has_left_neighbor and not has_right_neighbor:
            penalty += 15

    return penalty


def mobility(board: chess.Board) -> int:
    score = 0

    # White mobility
    for move in board.legal_moves:
        piece = board.piece_type_at(move.from_square)
        score += {chess.PAWN: 0.5, chess.KNIGHT: 1,
                  chess.BISHOP: 1, chess.ROOK: 1.5,
                  chess.QUEEN: 2}.get(piece, 0)

    # Black mobility
    board.push(chess.Move.null())
    for move in board.legal_moves:
        piece = board.piece_type_at(move.from_square)
        score -= {chess.PAWN: 0.5, chess.KNIGHT: 1,
                  chess.BISHOP: 1, chess.ROOK: 1.5,
                  chess.QUEEN: 2}.get(piece, 0)
    board.pop()

    return score


def center_control(board: chess.Board, color: chess.Color) -> int:
    score = 0
    phase = game_phase(board)

    for center_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
        if board.is_attacked_by(color, center_square):
            score += 15 * (256 - phase) // 256

    return score

def central_pawn_bonus(board):
    phase = game_phase(board)
    if phase < 128:
        bonus = 0
        for sq in [chess.E4, chess.D4]:
            if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE):
                bonus += 30
        for sq in [chess.E5, chess.D5]:
            if board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK):
                bonus -= 30
        return bonus
    return 0

def evaluate(board: chess.Board) -> int:
    if board.is_checkmate():
        return -checkmate_value

    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    material = (values[chess.PAWN] * (wp - bp) +
                values[chess.KNIGHT] * (wn - bn) +
                values[chess.BISHOP] * (wb - bb) +
                values[chess.ROOK] * (wr - br) +
                values[chess.QUEEN] * (wq - bq))

    pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawnsq += sum([-pawntable[chess.square_mirror(i)]
                   for i in board.pieces(chess.PAWN, chess.BLACK)])

    knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knightsq += sum([-knightstable[chess.square_mirror(i)]
                     for i in board.pieces(chess.KNIGHT, chess.BLACK)])

    bishopsq = sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishopsq += sum([-bishopstable[chess.square_mirror(i)]
                     for i in board.pieces(chess.BISHOP, chess.BLACK)])

    rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rooksq += sum([-rookstable[chess.square_mirror(i)]
                   for i in board.pieces(chess.ROOK, chess.BLACK)])

    queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queensq += sum([-queenstable[chess.square_mirror(i)]
                    for i in board.pieces(chess.QUEEN, chess.BLACK)])

    kingsq = king_tapered_eval(board)

    eval = material + pawnsq + knightsq + bishopsq + rooksq + queensq + kingsq

    eval += mobility(board)

    eval += get_king_safety(board, chess.WHITE) * 2
    eval -= get_king_safety(board, chess.BLACK) * 2

    # bishop pairs
    if wb >= 2:
        eval += 30
    if bb >= 2:
        eval -= 30

    eval += rook_open_bonus(board, chess.WHITE)
    eval -= rook_open_bonus(board, chess.BLACK)

    eval += connected_rooks(board, chess.WHITE)
    eval -= connected_rooks(board, chess.BLACK)

    eval += passed_pawns_bonus(board, chess.WHITE)
    eval -= passed_pawns_bonus(board, chess.BLACK)

    eval -= pawn_structure_penalty(board, chess.WHITE)
    eval += pawn_structure_penalty(board, chess.BLACK)

    eval += center_control(board, chess.WHITE)
    eval -= center_control(board, chess.BLACK)

    eval += central_pawn_bonus(board)

    if board.turn == chess.BLACK:
        eval = -eval

    # print(f'evaluaing {board.fen()}')
    # print(board)
    # print(eval)
    return eval


def ordered_moves(board: chess.Board):
    moves = list(board.legal_moves)

    def move_score(m: chess.Move):
        score = 0

        # Captures (MVV-LVA)
        if board.is_capture(m):
            attacker = board.piece_at(m.from_square)
            attacker_type = attacker.piece_type if attacker else 0

            if board.is_en_passant(m):
                captured_square = chess.square(
                    chess.square_file(m.to_square),
                    chess.square_rank(m.from_square)
                )
                victim = board.piece_at(captured_square)
            else:
                victim = board.piece_at(m.to_square)

            victim_type = victim.piece_type if victim else 0

            game_score = get_score_diff(board)
            base_bonus = 1000 if (game_score >= 0 and board.turn) or (game_score <= 0 and not board.turn) else 200

            score = base_bonus + (10 * victim_type + (6 - attacker_type))
            return score

        # promotions
        if m.promotion:
            return 900 + m.promotion

        # checks
        board.push(m)
        is_check = board.is_check()
        board.pop()
        if is_check:
            return 500

        # same piece move penalty
        if game_phase(board) < 10 and len(board.move_stack) >= 2:
            last_piece_moved = board.piece_at(board.move_stack[-2].to_square)
            if board.piece_at(m.from_square) == last_piece_moved:
                return -100

        return score

    return sorted(moves, key=move_score, reverse=True)


def quiescence(board: chess.Board, alpha: int, beta: int, depth: int = 0) -> int:
    if depth >= 5:
        return evaluate(board)

    stand_pat = evaluate(board)

    if stand_pat >= beta:
        return beta

    if stand_pat > alpha:
        alpha = stand_pat
    for move in ordered_moves(board):
        if not board.is_capture(move):
            continue

        board.push(move)
        score = -quiescence(board, -beta, -alpha, depth + 1)
        board.pop()

        if score >= beta:
            return beta
        if score > alpha:
            alpha = score

    return alpha


def negamax(board: chess.Board, depth: int, alpha: int, beta: int) -> int:

    board_hash = chess.polyglot.zobrist_hash(board)
    if board_hash in transposition_table:
        stored_depth, stored_eval = transposition_table[board_hash]
        if stored_depth >= depth:
            return stored_eval

    if depth == 0:
        eval_result = quiescence(board, alpha, beta)
        transposition_table[board_hash] = (0, eval_result)
        return eval_result

    if board.is_game_over():
        if board.is_checkmate():
            eval_result = -checkmate_value
        else:
            eval_result = 0
        transposition_table[board_hash] = (depth, eval_result)
        return eval_result

    max_eval = -checkmate_value

    for move in ordered_moves(board):
        board.push(move)
        eval = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()

        if eval > max_eval:
            max_eval = eval

        alpha = max(alpha, eval)
        if alpha >= beta:
            break

    transposition_table[board_hash] = (depth, max_eval)

    return max_eval


def find_best_move(board: chess.Board, depth: int):

    best_move = None
    best_eval = -checkmate_value - 1

    print(f"# Recherche à profondeur {depth}", file=sys.stderr)

    for move in ordered_moves(board):
        board.push(move)
        eval = -negamax(board, depth - 1, -checkmate_value, checkmate_value)
        board.pop()

        print(f"# {move}: {eval}", file=sys.stderr)

        if eval > best_eval:
            best_eval = eval
            best_move = move

    return best_move, best_eval


# ========================================
# UCI Protocol Implementation
# ========================================

class UCIEngine:
    def __init__(self):
        self.board = chess.Board()
        self.search_depth = 4

        self.book_reader = chess.polyglot.open_reader(os.path.curdir + "/engines/Perfect2021.bin")
        self.use_book = True

    def uci(self):
        print("id name NulEngine")
        print("id author E")
        print("option name Depth type spin default 4 min 1 max 10")
        print("uciok")
        sys.stdout.flush()

    def isready(self):
        print("readyok")
        sys.stdout.flush()

    def setoption(self, name, value):
        if name.lower() == "depth":
            try:
                self.search_depth = int(value)
            except ValueError:
                pass

    def ucinewgame(self):
        self.board = chess.Board()
        transposition_table.clear()

    def position(self, tokens):
        if len(tokens) < 2:
            return

        if tokens[1] == "startpos":
            self.board = chess.Board()
            moves_start = 2
            if len(tokens) > 2 and tokens[2] == "moves":
                moves_start = 3
        elif tokens[1] == "fen":
            fen_parts = []
            i = 2
            while i < len(tokens) and tokens[i] != "moves":
                fen_parts.append(tokens[i])
                i += 1
            fen = " ".join(fen_parts)
            try:
                self.board = chess.Board(fen)
            except ValueError:
                self.board = chess.Board()
            moves_start = i + 1 if i < len(tokens) and tokens[i] == "moves" else len(tokens)
        else:
            return

        for i in range(moves_start, len(tokens)):
            try:
                move = chess.Move.from_uci(tokens[i])
                if move in self.board.legal_moves:
                    self.board.push(move)
            except (ValueError, AssertionError):
                pass

    def go(self, tokens):
        if self.use_book:
            entry = self.book_reader.weighted_choice(self.board)

            print(f"info string Book move found: {entry.move.uci()}")
            print(f"bestmove {entry.move.uci()}")
            sys.stdout.flush()
            return
        if len(transposition_table) > 100000:
            transposition_table.clear()

        try:
            best_move, best_eval = find_best_move(self.board, self.search_depth)

            if best_move:
                print(f"info depth {self.search_depth} score cp {best_eval}")
                print(f"bestmove {best_move.uci()}")
            else:
                print("bestmove 0000")
        except Exception as e:
            print(f"info string Error: {e}", file=sys.stderr)
            print("bestmove 0000")

        sys.stdout.flush()

    def quit(self):
        sys.exit(0)

    def run(self):
        while True:
            try:
                line = input().strip()
                if not line:
                    continue

                tokens = line.split()
                command = tokens[0].lower()

                if command == "uci":
                    self.uci()
                elif command == "isready":
                    self.isready()
                elif command == "setoption":
                    if len(tokens) >= 5 and tokens[1].lower() == "name" and tokens[3].lower() == "value":
                        name = tokens[2]
                        value = tokens[4]
                        self.setoption(name, value)
                elif command == "ucinewgame":
                    self.ucinewgame()
                elif command == "position":
                    self.position(tokens)
                elif command == "go":
                    self.go(tokens)
                elif command == "quit":
                    self.quit()

            except EOFError:
                break
            except Exception as e:
                print(f"info string Error: {e}", file=sys.stderr)
                sys.stderr.flush()


if __name__ == "__main__":
    engine = UCIEngine()
    engine.run()
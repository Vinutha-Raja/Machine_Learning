################################################################################
# Part 3:
#
# Pente is a two-player game normally played on a 19x19 grid of intersections.
# For the purposes of this exercise we'll use a 9x9 board. This allows each grid
# space to be identified by a letter and number as shown in the diagram below.
#
#   a b c d e f g h i
# 1 . . . . . . . . .
# 2 . . . . . . W . .
# 3 . B . . . B . . .
# 4 . . W . B B . . .
# 5 . . . W . B . . .
# 6 . . B W W . . . .
# 7 . . . . . W . . .
# 8 . . . . . . . . .
# 9 . . . . . . . . .
#
# Players alternate placing stones of their symbol (B for Black or W for White)
# on any empty intersection, B plays first. The goal is to either:
#
# 1) Align five stones of the same symbol in any vertical, horizontal, or
#    diagonal direction
# - OR -
# 2) Make five captures, for a total of ten captured stones
#
# Stones are captured by flanking an adjacent pair of an opponent's stones
# directly on either side with your own stones. Captures consist of exactly two
# stones; flanking a single stone, or three or more stones, does not result in a
# capture.
#
# Given the board above, if B plays a stone at f6, it would result in the W
# stones at d6 and e6 being removed from the board. Captured stones are counted
# for each player, and first player to capture ten stones wins. The game board
# after B moves to f6 would be as follows:
#
#   a b c d e f g h i
# 1 . . . . . . . . .
# 2 . . . . . . W . .
# 3 . B . . . B . . .
# 4 . . W . B B . . .
# 5 . . . W . B . . .
# 6 . . B . . B . . .
# 7 . . . . . W . . .
# 8 . . . . . . . . .
# 9 . . . . . . . . .
#
# A stone may legally be played on any empty intersection, though even if it
# forms a pair between two enemy stones, it does not result in a capture.
# Captures occur only when a played stone flanks two stones of the opposing
# color. As an example, if W now plays at d6, followed by B playing at d7, the
# board would be as follows:
#
#   a b c d e f g h i
# 1 . . . . . . . . .
# 2 . . . . . . W . .
# 3 . B . . . B . . .
# 4 . . W . B B . . .
# 5 . . . W . B . . .
# 6 . . B W . B . . .
# 7 . . . B . W . . .
# 8 . . . . . . . . .
# 9 . . . . . . . . .
#
# If W now plays at e6, no capture occurs. The board instead is as follows:
#
#   a b c d e f g h i
# 1 . . . . . . . . .
# 2 . . . . . . W . .
# 3 . B . . . B . . .
# 4 . . W . B B . . .
# 5 . . . W . B . . .
# 6 . . B W W B . . .
# 7 . . . B . W . . .
# 8 . . . . . . . . .
# 9 . . . . . . . . .
#
# In the diagram above, a win for W occurs if W plays at g8. If it were B's
# turn, B could prevent the win by either playing at g8 (which would block W
# from playing there for the win) or playing at d4 (this would capture the
# stones at d5 and d6 and break the string of four W stones) or playing at f2,
# which would result in a win for B.
################################################################################

class Pente:
    BLACK = 'B'
    WHITE = 'W'
    EMPTY = '.'

    def __init__(self):
        self.size = 9
        self.spaces = [[Pente.EMPTY] * 9 for _ in range(self.size)]
        self.Black_capture_count = 0
        self.White_capture_count = 0

    def render(self):
        print(' ', *(chr(ord('a') + x) for x in range(self.size)))
        for y in range(self.size):
            print(y + 1, *(self.spaces[x][y] for x in range(self.size)))

    def getMoveInput(self):
        while True:
            try:
                row, col = input("Please enter move (e.g. a1): ")
                x = max(0, min(self.size - 1, ord(row) - ord('a')))
                y = max(0, min(self.size - 1, ord(col) - ord('1')))
            except ValueError as e:
                print(e)
            else:
                return x, y

    def valid_move(self, x, y):
        if self.spaces[x][y] == Pente.EMPTY:
            return True
        else:
            return False

    def check_for_captures1(self, x, y):
        current_piece = self.spaces[x][y]
        directions = [(0, -1), (0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]
        for direc in directions:
            temp_x = x
            temp_y = y
            x1 = temp_x + direc[0]
            y1 = temp_y + direc[1]
            x2 = temp_x + (2 * direc[0])
            y2 = temp_y + (2 * direc[1])
            x3 = temp_x + (3 * direc[0])
            y3 = temp_y + (3 * direc[1])
            print(x1, y1)
            print(x2, y2)
            print(x3, y3)
            if self.spaces[x1][y1] != current_piece and \
                    self.spaces[x2][y2] != current_piece and \
                    self.spaces[x3][y3] == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x1][y1] = Pente.EMPTY
                self.spaces[x2][y2] = Pente.EMPTY



    def check_for_captures(self, x, y):

        current_piece = self.spaces[x][y]
        # right
        if y < self.size - 3:
            if self.spaces[x][y + 1] != current_piece and self.spaces[x][y + 2] != current_piece and self.spaces[x][y + 3] \
                    == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x][y + 1] = Pente.EMPTY
                self.spaces[x][y + 2] = Pente.EMPTY
            # left
        if y > 2:
            if self.spaces[x][y - 1] != current_piece and self.spaces[x][y - 2] != current_piece and self.spaces[x][y - 3] \
                    == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x][y - 1] = Pente.EMPTY
                self.spaces[x][y - 2] = Pente.EMPTY
        # top
        if x > 2:
            if self.spaces[x - 1][y] != current_piece and self.spaces[x - 2][y] != current_piece and self.spaces[x - 3][y] \
                    == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x - 1][y] = Pente.EMPTY
                self.spaces[x - 2][y] = Pente.EMPTY
        # bottom
        if x < self.size - 3:
            if self.spaces[x + 1][y] != current_piece and self.spaces[x + 2][y] != current_piece and self.spaces[x + 3][y] \
                    == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x + 1][y] = Pente.EMPTY
                self.spaces[x + 2][y] = Pente.EMPTY

        # Diagonal right top
        if x > 2 and y < self.size - 3:
            if self.spaces[x - 1][y + 1] != current_piece and self.spaces[x - 2][y + 2] != current_piece and self.spaces[x - 3][y + 3] == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x - 1][y + 1] = Pente.EMPTY
                self.spaces[x - 2][y + 2] = Pente.EMPTY
        # Diagonal left top
        if x > 2 and y > 2:
            if self.spaces[x - 1][y - 1] != current_piece and self.spaces[x - 2][y - 2] != current_piece and self.spaces[x - 3][y - 3] == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x - 1][y - 1] = Pente.EMPTY
                self.spaces[x - 2][y - 2] = Pente.EMPTY
        # Diagonal bottom left:
        if x < self.size - 3 and y > 2:
            if self.spaces[x + 1][y - 1] != current_piece and self.spaces[x + 2][y - 2] != current_piece and self.spaces[x + 3][y - 3] == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x + 1][y - 1] = Pente.EMPTY
                self.spaces[x + 2][y - 2] = Pente.EMPTY
        # Diagonal bottom right:
        if x < self.size - 3 and y < self.size - 3:
            if self.spaces[x + 1][y + 1] != current_piece and self.spaces[x + 2][y + 2] != current_piece and self.spaces[x + 3][y + 3] == current_piece:
                if current_piece == Pente.BLACK:
                    self.Black_capture_count += 2

                else:
                    self.White_capture_count += 2
                self.spaces[x + 1][y + 1] = Pente.EMPTY
                self.spaces[x + 2][y + 2] = Pente.EMPTY

    def check_win_condition_based_on_capture_count(self):
        if self.Black_capture_count >= 10:
            print("Black wins")
            exit(0)
        if self.White_capture_count >= 10:
            print("White wins")
            exit(0)

    def count_cont_pieces(self, direction, x, y, current_piece):
        count = 0
        first_dir = direction[0]
        sec_dir = direction[1]
        temp_x = x
        temp_y = y
        while (temp_x < self.size - 1 or temp_x > 0) and (temp_y < self.size - 1 or temp_y > 0):
            temp_x = temp_x + first_dir[0]
            temp_y = temp_y + first_dir[1]
            if self.spaces[temp_x][temp_y] == current_piece:
                count += 1
            else:
                break

        temp_x = x
        temp_y = y
        while (temp_x < self.size or temp_x >= 0) and (temp_y < self.size or temp_y >= 0):
            temp_x = temp_x + sec_dir[0]
            temp_y = temp_y + sec_dir[1]
            if self.spaces[temp_x][temp_y] == current_piece:
                count += 1
            else:
                break
        return count

    # check for 5 continuous piece
    def check_win_condition(self, x, y, current_piece):
        directions = [[(0, -1), (0, 1)],  # left right
                      [(-1, 0), (1, 0)],  # bottom top
                      [(-1, -1), (1, 1)],  # bottom diag left right
                      [(-1, 1), (1, -1)]]
        for direction in directions:
            count = self.count_cont_pieces(direction, x, y, current_piece)
            if count >= 4:
                print("{} piece wins!!".format(current_piece))
                exit(0)

    def check_if_board_is_full(self):
        for r in range(self.size):
            for c in range(self.size):
                if self.spaces[r][c] == Pente.EMPTY:
                    return False
        return True

    # TODO: Implement this function.
    # A fully-working solution must do the following:
    #   1. Alternate turns between BLACK and WHITE
    #   2. Only allow valid moves
    #   3. Properly identify and remove captured stones
    #   4. Determine win condition of 5 or more captures (10 or more captured stones)
    #   5. Determine win condition of 5 or more stones in a row
    #   6. If there are no open locations to move, and neither player has satisfied either
    #      win condition, end the game as a draw
    # Extra Credit:
    #   - Create an AI to play randomly
    #   - Create an AI that plays with some type of strategy
    def play(self):
        current_turn = Pente.BLACK
        while True:
            self.render()
            x, y = self.getMoveInput()
            if not self.valid_move(x, y):
                print("Not a valid move")
                continue

            self.spaces[x][y] = current_turn
            self.check_for_captures1(x, y)
            self.check_win_condition_based_on_capture_count()
            self.check_win_condition(x, y, current_turn)
            if self.check_if_board_is_full():
                print("DRAW!!")
                exit(0)

            current_turn = Pente.WHITE if current_turn == Pente.BLACK else Pente.BLACK

# Run the game
if __name__ == '__main__':
    print('Part 3:')
    pente = Pente()
    pente.play()
    print()

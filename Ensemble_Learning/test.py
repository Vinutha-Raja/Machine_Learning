import functools
from collections import Counter

def return_word_from_last_letter(text):
    words = text.split(' ')
    ans = ""
    # print(words)
    for word in words:
        # print(word)
        ans += word[len(word) - 1]
    return ans

print(return_word_from_last_letter('correct horse battery staple'))


def part1():

    class Person:
        def __init__(self, name, age):
            self.name = name
            self.age = age

        def __str__(self):
            return "Name: " + self.name + ", Age: " + str(self.age)

    def comparePersons(p1, p2):
        # TODO: Implement this function.
        # Return a negative number if p1 comes before p2.
        # Return a positive number if p1 comes after p2.
        # Return zero if p1 and p2 are equivalent.
        if p1.age < p2.age:
            return -1
        elif p1.age > p2.age:
            return 1
        else:
            if p1.name < p2.name:
                return -1
            else:
                return 1
        return 0



    # Example of using the comparison function.
    people = [
        Person("Matt", 50),
        Person("Lulu", 5),
        Person("Laura", 49),
        Person("Abby", 50),
        Person("Abb", 50),
        Person("Abb", 50),
        Person("Chris", 1),
        Person("Jen", 35),
        Person("Flavia", 12),
        Person("Alicia", 21),
        Person("Greg", 78),
        Person("Boris", 9),
    ]

    print("People sorted from 'least' to 'greatest':\n")
    for p in sorted(people, key=functools.cmp_to_key(comparePersons)):
        print('   ' + str(p))

part1()


def part2():
    def getNumberOfMatchingSockPairs(sockColors):
        # TODO your implementation goes here. This should return the total number of pairs.
        count = Counter(sockColors)
        # print(count)
        values = count.values()
        c = 0
        # print(values)
        for val in values:
            # if val % 2 == 0:
            # print(val, val// 2)
            c += (val // 2)

        return c

    # Examples of using the pair count function.
    tests = [
        ([], 0),
        (["red"], 0),
        (["red", "red"], 1),
        (["red", "blue"], 0),
        (["red", "red", "red"], 1),
        (["red", "blue", "red", "green", "green", "red"], 2),
        (
            [
                "red",
                "blue",
                "purple",
                "red",
                "green",
                "green",
                "purple",
                "red",
                "yellow",
                "red",
                "red",
                "yellow",
                "red",
                "purple"
            ],
            6
        ),
    ]

    for sockColors, expectedCount in tests:
        count = getNumberOfMatchingSockPairs(sockColors)
        print(
            f'    Got {count}, expected {expectedCount} for sockColors = {sockColors}')


part2()


def part3():
    from enum import Enum

    class Piece(Enum):
        """
        The possible contents of a square on the board.
        """
        X = 'X'
        O = 'O'
        EMPTY = ' '  # Represents an empty square on the board.

    class Square:
        """
        A single square on the board.

        Contains the piece on the square and the timer.

        A negative timer value indicates that the timer is disabled.
        """

        def __init__(self, piece=Piece.EMPTY, timer=-1, visited=0):
            self.piece = piece
            self.timer = timer
            self.visited = visited

        def isTimerEnabled(self):
            return self.timer >= 0

        def __repr__(self):
            return f'Square(piece={self.piece}, timer={self.timer})'

    class Coordinate:
        """
        A 2-D position coordinate.
        """

        def __init__(self, row, column):
            self.row = row
            self.column = column

        def __repr__(self):
            return f'Coordinate(row={self.row},column={self.column})'

    class Game:
        def __init__(self):
            self.size = 9
            self.board = [
                [Square() for _ in range(self.size)]
                for _ in range(self.size)
            ]

        def render(self):
            """
            Prints the current board state to the console.
            """
            print('        a    b    c    d    e    f    g    h    i')
            print('        o    o    o    o    o    o    o    o    o')
            print('     ----------------------------------------------')
            for index, row in enumerate(self.board):
                label = str(index + 1)
                contents = ' | '.join(
                    p.piece.value + (str(p.timer) if p.isTimerEnabled() else ' ') for p in row)
                print(f'{label}  x | {contents} | x')
                print('     ----------------------------------------------')
            print('        o    o    o    o    o    o    o    o    o')

        @staticmethod
        def get_user_move():
            """
            Gets the user move from stdin.

            Continues prompting the user for a move until the user enters a
            move that is within the boundaries of the board. Does not do any
            other move validity checking.
            """
            COLUMN_RANGE = 'abcdefghi'
            ROW_RANGE = '123456789'
            COLUMN_RANGE_MESSAGE = 'Please enter a column in the range "a" to "i".'
            ROW_RANGE_MESSAGE = 'Please enter a row in the range "1" to "9".'
            while True:
                raw_user_input = input('Enter Move (e.g. "a2"):\n')
                cs = ''.join(raw_user_input.strip().split()).lower()
                is_valid_format = (
                    len(cs) == 2 and cs[0].isalpha() and cs[1].isdigit())
                if not is_valid_format:
                    print('Invalid format. Please enter the move in the format "a2"')
                    continue
                if cs[0] not in COLUMN_RANGE:
                    print(f'Invalid column "{cs[0]}". {COLUMN_RANGE_MESSAGE}')
                    continue
                if cs[1] not in ROW_RANGE:
                    print(f'Invalid row "{cs[1]}". {ROW_RANGE_MESSAGE}"')
                    continue
                row = int(cs[1]) - 1
                column = ord(cs[0]) - ord('a')
                return Coordinate(row=row, column=column)

        def check_if_empty_square(self, move):
            print(self.board[move.row][move.column])
            if self.board[move.row][move.column].piece == Piece.EMPTY:
                return True
            else:
                return False

        def get_neighbor(self, row, col):
            neighbors = []
            current_val = self.board[row][col].piece
            if row > 0:
                if self.board[row - 1][col].piece == current_val:
                    neighbors.append((row - 1, col))
            if row < self.size - 1:
                if self.board[row + 1][col].piece == current_val:
                    neighbors.append((row + 1, col))
            if col > 0:
                if self.board[row][col - 1].piece == current_val:
                    neighbors.append((row, col - 1))
            if col < self.size - 1:
                if self.board[row][col + 1].piece == current_val:
                    neighbors.append((row, col + 1))
            return neighbors

        def no_neighbor(self, row, col):
            current_val = self.board[row][col].piece
            print("current_val:", current_val)
            if row > 0:
                if self.board[row - 1][col].piece == current_val:
                    return False
            if row < self.size - 1:
                if self.board[row + 1][col].piece == current_val:
                    return False
            if col > 0:
                if self.board[row][col - 1].piece == current_val:
                    return False
            if col < self.size - 1:
                if self.board[row][col + 1].piece == current_val:
                    return False
            return True

        def explode_bomb(self, row, col):
            self.board[row][col].piece = Piece.EMPTY
            # top
            if row > 0:
                self.board[row - 1][col].piece = Piece.EMPTY
                self.board[row - 1][col].timer = -1
            # bottom
            if row < self.size - 1:
                self.board[row + 1][col].piece = Piece.EMPTY
                self.board[row + 1][col].timer = -1
            # left
            if col > 0:
                self.board[row][col - 1].piece = Piece.EMPTY
                self.board[row][col - 1].timer = -1
            # right
            if col < self.size - 1:
                self.board[row][col + 1].piece = Piece.EMPTY
                self.board[row][col + 1].timer = -1
            self.board[row][col].timer = -1

        def reduce_bomb_timer(self, cur_row, cur_col):
            for row in range(len(self.board)):
                for col in range(len(self.board[0])):
                    if row != cur_row or col !=cur_col:
                        if self.board[row][col].isTimerEnabled():
                            if self.board[row][col].timer != 0:
                                self.board[row][col].timer -= 1
                            if self.board[row][col].timer == 0:
                                self.explode_bomb(row, col)

        def restart_timer(self):
            for row in range(len(self.board)):
                for col in range(len(self.board[0])):
                    if self.no_neighbor(row, col):
                        self.board[row][col].timer = 4

        def recursively_check_x(self, r, col):
            if self.board[r][col].visited == 1:
                return
            self.board[r][col].visited = 1
            if col == self.size - 1:
                print("X wins")
                exit(0)

            neighbors = self.get_neighbor(r, col)
            for n in neighbors:
                r = n[0]
                col = n[1]
                self.recursively_check_x(r, col)

        def recursively_check_o(self, r, col):
            if self.board[r][col].visited == 1:
                return
            self.board[r][col].visited = 1
            if r == self.size - 1:
                print("O wins")
                exit(0)

            neighbors = self.get_neighbor(r, col)
            for n in neighbors:
                r = n[0]
                col = n[1]
                self.recursively_check_o(r, col)

        def check_for_bridge(self):
            # check if x wins
            for row in range(len(self.board)):
                for col in range(len(self.board[0])):
                    self.board[row][col].visited = 0

            for row in range(self.size):
                r = row
                col = 0
                if self.board[row][col].piece == Piece.X:
                    self.recursively_check_x(r, col)

            for row in range(len(self.board)):
                for col in range(len(self.board[0])):
                    self.board[row][col].visited = 0

            # check if o wins
            for col in range(self.size):
                r = 0
                c = col
                if self.board[r][c].piece == Piece.O:
                    self.recursively_check_o(r, col)

        def check_if_board_full(self):
            for row in range(len(self.board)):
                for col in range(len(self.board[0])):
                    if self.board[row][col].piece == Piece.EMPTY:
                        print("Board is empty")
                        return False
            return True

        def disable_neighbor_timer(self, row, col):
            neighbors = self.get_neighbor(row, col)
            for n in neighbors:
                self.board[n[0]][n[1]].timer = -1

        # TODO: Implement this function.
        # A fully-working solution must do the following:
        #   1. Only allow play on an empty square.
        #   2. Alternate turns between X and O.
        #   3. Handle bomb logic correctly.
        #   4. Check for completed bridges or full board to find the winner.
        def play(self):
            current_turn = Piece.X
            while True:
                self.render()
                move = self.get_user_move()
                print(move.row, type(move.column))
                valid_move = self.check_if_empty_square(move)
                if not valid_move:
                    print("not valid")
                    continue

                self.board[move.row][move.column] = Square(piece=current_turn)
                if self.no_neighbor(move.row, move.column):
                    self.board[move.row][move.column].timer = 4
                else:
                    self.board[move.row][move.column].timer = -1
                    self.disable_neighbor_timer(move.row, move.column)
                self.reduce_bomb_timer(move.row, move.column)
                # self.restart_timer()
                self.check_for_bridge()
                if self.check_if_board_full():
                    print("{} player wins", current_turn)
                    exit(0)

                # switch current turn at the end
                current_turn = Piece.O if current_turn == Piece.X else Piece.X

    game = Game()
    game.play()


part3()
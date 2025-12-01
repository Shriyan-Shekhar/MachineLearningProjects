from abc import ABC
from sampleProgram.sample import BaseGame
from chessLib.move import KnightMove
from chessLib.position import Position
import random
# Documentation is added in PEP257-compliant way instead of JavaDoc style as this code is in Python.

class ComplexGame(BaseGame):
    """A complex chess-like game that implements the BaseGame interface.

    This class manages the setup of a board with multiple pieces (Knight, Bishop, Queen)
    and simulates gameplay by executing random valid moves.

    Author:
        Shriyan Shekhar
    """


    def play(self, moves: int): 
        """Execute the main gameplay loop.

        Implements the abstract method "play()" from BaseGame.
        Randomly selects pieces and performs valid moves for a specified
        number of turns. After each move, the current state of the board
        and all piece positions are printed.

        Args:
            moves (int): The number of moves to play.

        Returns:
            None
        """
        for i in range(moves):
            piece = random.choice(self.board.pieces) 
            valid = piece.valid_moves(self.board)
            while not valid:
                piece = random.choice(self.board.pieces)
                valid = piece.valid_moves(self.board)
            new_pos = random.choice(valid) 
            piece.move(new_pos)
            print(f"{i+1}: Moved {piece.name} to {new_pos.to_string()}")
            print("Current board positions:")
            for p in self.board.pieces:
                print(f"    {p.name}: {p.position.to_string()}")



    def setup(self, coordinates=None):
        """Initialize the game board and place all pieces.

        Implements the abstract method `setup()` from BaseGame.
        Creates an 8x8 board and places a Knight, Bishop, and Queen at
        either default or user-defined starting coordinates. Also defines
        the internal helper classes for board logic and piece movement.

        Args:
            coordinates (dict, optional): A dictionary mapping piece names (Knight,
            Bishop, Queen) to their starting `Position` objects. Defaults to preset
            coordinates if None.

        Returns:
            None
        """
        class Board:
            """Represents the game board and handles piece management."""
            def __init__(self, size=8):
                """Initialize a new board of given size."""
                self.size = size
                self.pieces = []

            def add_piece(self, piece):
                """Add a piece to the board at its position."""
                if self.is_occupied(piece.position):
                    raise ValueError("Position already occupied!")
                self.pieces.append(piece)

            def is_occupied(self, pos):
                """Check if a position on the board is occupied by any piece."""
                return any(p.position == pos for p in self.pieces)

            def is_within_bounds(self, x, y):
                """Check if coordinates are within board boundaries."""
                return 0 < x <= self.size and 0 < y <= self.size

            def get_validity_moves(self, start, directions):
                """Get valid moves in given directions from a start position."""
                moves = []
                for dx, dy in directions:
                    newX, newY = start.x + dx, start.y + dy
                    while self.is_within_bounds(newX, newY):
                        pos = Position(newX, newY)
                        if not self.is_occupied(pos):
                            moves.append(pos)
                        newX += dx
                        newY += dy
                return moves
            
        class Pieces(ABC):
            """Abstract base class for all game pieces."""
            def __init__(self, name: str, position: Position):
                """Initialize a piece with a name and position."""
                self.name = name
                self.position = position
            def move(self, new_position: Position):
                """Move the piece to a new position."""
                self.position = new_position
        
        class Knight(Pieces):
            """Represents a Knight piece with its movement logic."""
            def __init__(self, position: Position):
                """Initialize a Knight piece at a given position."""
                super().__init__("Knight", position)
                self.move_strategy = KnightMove()
            def valid_moves(self, board):
                """Get valid moves for the Knight, excluding occupied squares."""
                moves = self.move_strategy.valid_moves(self.position)
                moves = [m for m in moves if not board.is_occupied(m)]
                return moves

        class Bishop(Pieces):
            """Represents a Bishop piece with its movement logic."""
            def __init__(self, position):
                """Initialize a Bishop piece at a given position."""
                super().__init__("Bishop", position)

            def valid_moves(self, board):
                """Get valid moves for the Bishop."""
                directions = [(1,1),(1,-1),(-1,1),(-1,-1)]
                return board.get_validity_moves(self.position, directions)

        class Queen(Pieces):
            """Represents a Queen piece with its movement logic."""
            def __init__(self, position):
                """Initialize a Queen piece at a given position."""
                super().__init__("Queen", position)

            def valid_moves(self, board):
                """Get valid moves for the Queen."""
                directions = [(1,1),(1,-1),(-1,1),(-1,-1),(1,0),(-1,0),(0,1),(0,-1)]
                return board.get_validity_moves(self.position, directions)


        def initialize_board(self, coordinates=None):
            """Initialize the board and add all pieces to it.
            Args:
                coordinates (dict, optional): Custom piece positions of
                Knight, Bishop, Queen (in this order). Defaults to preset
                coordinates if None.
            Returns:
                None
            """
            self.board = Board()

            if coordinates == None:
                coordinates = {
                    'Knight': Position(1,1),
                    'Bishop': Position(2,3),
                    'Queen': Position(3,2)
                }

            self.pieces = [
                Knight(coordinates['Knight']),
                Bishop(coordinates['Bishop']),
                Queen(coordinates['Queen'])
            ]

            for p in self.pieces:
                self.board.add_piece(p)

            self.Knight = Knight
            self.Bishop = Bishop
            self.Queen = Queen
            self.Piece = Pieces
        
        def print_initial_board(self):
            """Print the initial positions of all pieces on the board."""
            print("Initial board positions:")
            for p in self.board.pieces:
                print(f"    {p.name}: {p.position.to_string()}")
        
        initialize_board(self, coordinates)
        print_initial_board(self)

import unittest
import os
import sys

# Documentation is added in PEP257-compliant way instead of JavaDoc style as this code is in Python.
# To run this file, cd to the programTests directory and execute: python -m unittest answertest.py or python answertest.py

current_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from sampleProgram.answer import ComplexGame  
from chessLib.position import Position

class TestComplexGame(unittest.TestCase):
    """Test suite (Unit Tests) for the ComplexGame class.

    Tests cover board setup and piece initialization behavior.

    Author:
        Shriyan Shekhar
    """
    def setUp(self):
        """Set up a ComplexGame instance before each test. Does not initialize the board."""
        self.game = ComplexGame()  

    def test_default_initial_positions(self):
        """Check that default positions are correctly initialized"""
        self.game.setup() 
        positions = {p.name: p.position for p in self.game.pieces}

        self.assertEqual(positions['Knight'].x, 1)
        self.assertEqual(positions['Knight'].y, 1)
        self.assertEqual(positions['Bishop'].x, 2)
        self.assertEqual(positions['Bishop'].y, 3)
        self.assertEqual(positions['Queen'].x, 3)
        self.assertEqual(positions['Queen'].y, 2)

    def test_custom_initial_positions(self):
        """Check that custom positions can be set via initialize_board"""
        custom_positions = {
            'Knight': Position(2,2),
            'Bishop': Position(7,7),
            'Queen': Position(4,4)
        }
        self.game.setup(coordinates=custom_positions)
        positions = {p.name: p.position for p in self.game.pieces}

        self.assertEqual(positions['Knight'].x, 2)
        self.assertEqual(positions['Knight'].y, 2)
        self.assertEqual(positions['Bishop'].x, 7)
        self.assertEqual(positions['Bishop'].y, 7)
        self.assertEqual(positions['Queen'].x, 4)
        self.assertEqual(positions['Queen'].y, 4)

    def test_no_overlap_on_initialization(self):
        """Ensure that no two pieces occupy the same square"""
        self.game.setup()
        positions = [p.position for p in self.game.pieces]
        unique_positions = set((p.x, p.y) for p in positions)
        self.assertEqual(len(positions), len(unique_positions),
                         "Two pieces occupy the same position!")

    def test_valid_moves_within_board(self):
        """Ensure valid moves for all pieces are within board bounds"""
        self.game.setup()
        for piece in self.game.pieces:
            moves = piece.valid_moves(self.game.board)
            for m in moves:
                self.assertTrue(0 < m.x <= 8 and 0 < m.y <= 8,
                                f"{piece.name} has move out of bounds: {m.x},{m.y}")

    def test_knight_cannot_move_to_occupied(self):
        """Ensure Knight does not move to occupied squares"""
        self.game.setup()
        knight = next(p for p in self.game.pieces if p.name == 'Knight')
        moves = knight.valid_moves(self.game.board)
        for m in moves:
            self.assertFalse(any(p.position == m for p in self.game.pieces if p != knight),
                             f"Knight can move to occupied square: {m.x},{m.y}")

    def test_new_1000_moves_play(self):
        """Run 1000 moves and ensure Knight does not move from initial position if blocked"""
        self.game.setup()
        for i in range (1000):
            self.game.play(1) 
            positions = {p.name: p.position for p in self.game.pieces}
            if positions['Knight'].x != 1 or positions['Knight'].y != 1:
                self.assertFalse("Knight should not have moved!")
            self.game.setup() 
    
    def test_bishop_1000_moves_play(self):
        """Run 1000 moves and ensure Bishop does not go to Knight position"""
        custom_positions = {
            'Knight': Position(2,2),
            'Bishop': Position(1,1),
            'Queen': Position(4,4)
        }
        self.game.setup(coordinates=custom_positions)
        for i in range (1000):
            self.game.play(1) 
            positions = {p.name: p.position for p in self.game.pieces}
            if positions['Bishop'].x == 2 and positions['Bishop'].y == 2 or positions['Bishop'].x == 4 and positions['Bishop'].y == 4:
                self.assertFalse("Bishop should not have moved!")
            self.game.setup(coordinates = custom_positions)

    def test_piece_count_after_20000_moves(self):
        """Ensure that after 20000 moves, there are still exactly 3 pieces"""
        custom_positions = {
            'Knight': Position(2,2),
            'Bishop': Position(1,1),
            'Queen': Position(4,4)
        }

        self.game.setup(coordinates=custom_positions)

        for _ in range(10000):
            self.game.play(1)  

        self.game.play (10000)
        self.assertEqual(len(self.game.board.pieces), 3, "There should still be 3 pieces on the board.")


    def test_bishop_moves(self):
        """Ensure Bishop does not move to occupied squares by checking possible moves"""
        self.game.setup()
        custom_positions = {
            'Knight': Position(2,2),
            'Bishop': Position(1,1),
            'Queen': Position(4,4)
        }
        

        self.game.setup(coordinates=custom_positions)
        bishop = next(p for p in self.game.pieces if p.name == 'Bishop')
        moves = bishop.valid_moves(self.game.board)
        self.assertEqual(len(moves), 5, "Bishop should have 5 valid moves.")

    def test_queen_moves(self):
        """Ensure Queen does not move to occupied squares by checking possible moves"""
        self.game.setup()
        custom_positions = {
            'Knight': Position(2,2),
            'Bishop': Position(3,3),
            'Queen': Position(1,1)
        }
        

        self.game.setup(coordinates=custom_positions)
        queen = next(p for p in self.game.pieces if p.name == 'Queen')
        moves = queen.valid_moves(self.game.board)
        self.assertEqual(len(moves), 19, "Queen should have 19 valid moves.")

if __name__ == "__main__":
    unittest.main()

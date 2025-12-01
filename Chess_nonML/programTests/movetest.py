import math
import unittest
import os
import sys

current_dir = os.path.dirname(__file__)

project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from chessLib.move import KnightMove
from chessLib.position import Position


class KnightTests(unittest.TestCase):
    def test_knight_move_from_inside_board(self):
        pos = Position(3, 3)
        knight = KnightMove()
        moves = knight.valid_moves(pos)
        self.assertIsNotNone(moves)
        self.assertEqual(8, moves.__len__())

        for move in moves:
            v = math.fabs(move.x - pos.x)
            if v == 1:
                self.assertEqual(2, math.fabs(move.y - pos.y))
            elif v == 2:
                self.assertEqual(1, math.fabs(move.y - pos.y))
            else:
                self.fail()

    def test_knight_move_from_corner(self):
        pos = Position(1, 1)
        knight = KnightMove()
        moves = knight.valid_moves(pos)
        self.assertIsNotNone(moves)
        self.assertEqual(2, moves.__len__())

        possibles = [Position(2, 3), Position(3, 2)]
        for move in possibles:
            self.assertTrue(moves.__contains__(move))

    def test_position(self):
        pos = Position(1, 1)
        self.assertEqual(1, pos.x)
        self.assertEqual(1, pos.y)
        pos2 = Position(1, 1)
        self.assertEqual(pos, pos2)

if __name__ == "__main__":
    unittest.main()
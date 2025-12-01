from sampleProgram.sample import SimpleGame
from sampleProgram.answer import ComplexGame
# To run this file, cd to the Questions directory and execute: python program.py
# Restructured file organization as program.py should be at the root level.

if __name__ == '__main__':
    game = SimpleGame()
    game = ComplexGame()
    game.setup()
    game.play(15)

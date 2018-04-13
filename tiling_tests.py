import unittest
from tiling import *


class TestTiling(unittest.TestCase):
    def test_tilesAsMatrix(self):
        tileCounter = TileCounter(10, 10, 1, 1)
        bbs = [
            [0.05,0.05, 0.15,0.15],
            [0.25,0.35, 0.25,0.45]
        ]
        tiles = tileCounter.getTilesAsMatrix(bbs)
        self.assertEqual(1, tiles[0,0])


    def test_tileAsTileList_width(self):
        tileCounter = TileCounter(10, 10, 1, 1)
        bbs = [
            [0.05,0.15, 0.05,0.15],
            [0.25,0.35, 0.25,0.45]
        ]
        tiles = tileCounter.getTiles(bbs)
        self.assertEqual(10, tiles[0].getWidth(100))


    def test_tileAsTileList(self):
        tileCounter = TileCounter(10, 10, 1, 1)
        bbs = [
            [0.15,0.15, 0.16,0.16],
        ]
        tiles = tileCounter.getTiles(bbs)
        self.assertEqual(10, tiles[0].getX1(100))
        self.assertEqual(20, tiles[0].getX2(100))
        self.assertEqual(10, tiles[0].getY1(100))
        self.assertEqual(20, tiles[0].getY2(100))


    def test_tileAsTileList(self):
        tileCounter = TileCounter(10, 10, 1, 1)
        bbs = [
            [0.15,0.15, 0.56,0.16],
        ]
        tiles = tileCounter.getTiles(bbs)
        self.assertEqual(10, tiles[0].getX1(100))
        self.assertEqual(20, tiles[0].getX2(100))
        self.assertEqual(10, tiles[0].getY1(100))
        self.assertEqual(20, tiles[0].getY2(100))

        self.assertEqual(50, tiles[1].getX1(100))
        self.assertEqual(20, tiles[1].getY2(100))


    def test_tileAsTileList_scaled(self):
        tileCounter = TileCounter(10, 10, 10, 10)
        bbs = [
            [1.5,1.5, 5.6,1.6],
        ]
        tiles = tileCounter.getTiles(bbs)

        self.assertEqual(2, len(tiles))

        self.assertEqual(10, tiles[0].getX1(100))
        self.assertEqual(20, tiles[0].getX2(100))
        self.assertEqual(10, tiles[0].getY1(100))
        self.assertEqual(20, tiles[0].getY2(100))

        self.assertEqual(50, tiles[1].getX1(100))
        self.assertEqual(20, tiles[1].getY2(100))


    def test_tileAsTileList_scaled_multipleLabels(self):
        tileCounter = TileCounter(10, 10, 10, 10)
        bbs = [
            [1.5,2.5, 5.6,2.6],
            [2.5,3.5, 5.6,2.6]
        ]
        tiles = tileCounter.getTiles(bbs)

        self.assertEqual(5, len(tiles))

        self.assertEqual(10, tiles[0].getX1(100))
        self.assertEqual(20, tiles[0].getX2(100))
        self.assertEqual(20, tiles[0].getY1(100))
        self.assertEqual(30, tiles[0].getY2(100))


if __name__ == '__main__':
    unittest.main()

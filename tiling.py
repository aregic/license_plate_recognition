import numpy as np
import math


# TODO consider case where label coordinates are in [0, width-1] x [0, height-1]

class TileCounter:
    def __init__(self, size_x = 8, size_y = 8, pic_width = 1.0, pic_height = 1.0):
        self.TILE_X = size_x
        self.TILE_Y = size_y
        self.pic_width = pic_width
        self.pic_height = pic_height


    def getTilesAsMatrix(self, scaled_labels : list) -> list:
        """
            Expected format of scaled_labels:
                [x1, y1, x2, y2], where x2 and y2 are coordinates too (so not height and width)
        """
        tiles = np.zeros([self.TILE_X, self.TILE_Y], dtype=np.int)
        for label in scaled_labels:
            x1 = label[0] / self.pic_width
            x2 = label[2] / self.pic_width
            y1 = label[1] / self.pic_height
            y2 = label[3] / self.pic_height

            # check all [x_i, y_j] where i,j \isin {1,2}
            self.findTileForCoord(x1, y1, tiles)
            self.findTileForCoord(x1, y2, tiles)
            self.findTileForCoord(x2, y1, tiles)
            self.findTileForCoord(x2, y2, tiles)

        return tiles


    def findTileForCoord(self, x : float, y : float, tiles : np.ndarray):
        """
            Finds out which tile contains the (x,y) point and returns the indices of that tile,
            starting from 0, i.e. return (i,j) \isin [0, TILE_X-1] x [0, TILE_Y-1]

            (x,y) must be in [0,1] x [0,1]

            Border points belong to the left / upper tile.

            Output: tiles \isin M_{NxN}
        """
        tile_x = math.floor( ( x * self.TILE_X ) )
        tile_y = math.floor( ( y * self.TILE_Y ) )
        tiles[tile_x, tile_y] += 1


    def getTiles(self, scaled_labels : list) -> list:
        """
            returns a list of Tile objects which have at least 1 edge inside. Tiles are effectively rectangles.
        """
        tiles = self.getTilesAsMatrix(scaled_labels)
        res = []
        for i in range(0, self.TILE_X):
            for j in range(0, self.TILE_Y):
                if tiles[i,j] > 0:
                    res.append(Tile(i/self.TILE_X, j/self.TILE_Y, (i+1)/self.TILE_X, (j+1)/self.TILE_Y))

        return res


class Tile:
    """
        This class is basically to convert between the "all coordinates" representation of a tile (= rectangle)
        and the "left-upper point + width, height" representation and to deal with the coordinates being in
        the [0,1] interval.
    """
    def __init__(self, x1,y1, x2,y2, height_representation = False):
        """
            If height_representation is True x2 will be considered width, y2 will be height
        """
        # TODO maybe find a better solution to be able to switch between height repr and coordinate repr
        self.x1 = x1
        self.y1 = y1

        if not height_representation:
            self.x2 = x2
            self.y2 = y2
        else:
            self.x2 = x1 + x2
            self.y2 = y1 + y2


    def getX1(self, pic_width : float = 1.0):
        return self.x1 * pic_width

    def getY1(self, pic_height : float = 1.0):
        return self.y1 * pic_height

    def getX2(self, pic_width : float = 1.0):
        return self.x2 * pic_width

    def getY2(self, pic_height : float = 1.0):
        return self.y2 * pic_height

    def getWidth(self, pic_width : float = 1.0):
        return (self.x2 - self.x1) * pic_width

    def getHeight(self, pic_height : float = 1.0):
        return (self.y2 - self.y1) * pic_height

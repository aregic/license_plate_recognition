import numpy as np
from typing import List


class LicensePlate(object):
    def __init__(self, label : np.ndarray):
        self.label = label
        self.x1, self.y1, self.x2, self.y2 = label

    def getHeightWidthRepresentation(self):
        """
        :return: the license plate bounding box as [x, y, width, height]
        """
        return [self.x1, self.y1, self.x2 - self.x1, self.y2 - self.y1]

    def getBoundingBoxCoordinates(self):
        return [self.x1, self.y1, self.x2, self.y2]


class LicensePlateList(object):
    def __init__(self, labels: List[np.ndarray]):
        self.labels = _readLicensePlate(labels)

    def getHeightWidthRepresentation(self):
        return [l.getHeightWidthRepresentation() for l in self.labels]

    def getBoundingBoxCoordinates(self):
        return [l.getBoundingBoxCoordinates() for l in self.labels]


def _readLicensePlate(labels : List[np.ndarray]):
    return [LicensePlate(label) for label in _convert_to_bounding_boxes(labels)]


def _convert_to_bounding_boxes(labels: List[list]) -> list:
    if labels is None:
        return None
    res = []
    for label in labels:
        for i in range(0,8,2):
            x1 = min
    for label in labels:
        label = np.array(label).reshape(4,2)
        x1 = min(label[0][0], label[1][0], label[2][0], label[3][0])
        x2 = max(label[0][0], label[1][0], label[2][0], label[3][0])
        y1 = min(label[0][1], label[1][1], label[2][1], label[3][1])
        y2 = max(label[0][1], label[1][1], label[2][1], label[3][1])
        res.append([x1, y1, x2, y2])
    return res
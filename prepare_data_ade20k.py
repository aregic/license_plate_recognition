import os
import os.path
import argparse
import sys
import scipy.ndimage
import numpy as np
# I'm awesome when it comes to naming things, ain't I?
from progress_bar import progress_bar
from shutil import copy2


class Example:
    def __init__(self, label_loc : dir):
        self.label_loc = label_loc
        self.basename = getBaseName(label_loc)
        self.picloc = self.basename + '.jpg'
        self.segloc = self.basename + '_seg.png'
        self.parts = self.findParts()


    # probably a bit lame and slow, but simple
    def findParts(self):
        res = []
        part_counter = 1
        parts_loc = self.basename + "_parts_" + str(part_counter) + '.png'
        while(os.path.isfile(parts_loc)):
            res.append(parts_loc)
            part_counter += 1
            parts_loc = self.basename + "_parts_" + str(part_counter) + '.png'
        return res


    def copyTo(self, output_dir : dir):
        files_to_copy = [self.label_loc, self.picloc, self.segloc]
        if self.parts:
            files_to_copy.extend(self.parts)
        copyFiles(files_to_copy, output_dir)


def copyFiles(file_list : list, output_dir : dir):
    for file_loc in file_list:
        copy2(file_loc, output_dir)


def copySamplesWithLabel(input_dir : dir, output_dir : dir, label_type : str):
    samples_with_label = collectSamplesWithLabel(input_dir, label_type)
    for file_loc in samples_with_label:
        example = Example(file_loc)
        example.copyTo(output_dir)


def copyLabelWithImage(label_loc : dir, output_dir : dir):
    basename = label_loc[:-8]
    picloc = basename + ".jpg"
    segloc = basename + "_seg.png"


def getBaseName(label_loc : dir) -> str:
    return label_loc[:-8]


def collectSamplesWithLabel(input_dir : dir, label_type : str):
    samples_with_label = []
    label_files = collectFilesFromDir(input_dir, 'txt')
    for file_loc in label_files:
        with open(file_loc, 'r') as f:
            if label_type in f.read():
                samples_with_label.append(file_loc)
    return samples_with_label


def collectFilesFromDir(path : dir, extension : str):
    res = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        fullnames = map(lambda x : os.path.join(dirpath, x), filenames)
        filtered_files = list(filter(lambda x : x.endswith('.%s' % extension), fullnames))
        if len(filtered_files) > 0:
            res.extend(filtered_files)
    return res


def readColorCode(label_loc : dir, label_type : str):
    with open(label_loc, 'r') as f:
        lines = f.read().split('\n')
        lines = map(lambda x : x.split(' # '), lines)
        color_codes = filter(lambda x : label_type in x, lines)
        return list(color_codes)


def isInColorCode(image : np.ndarray, x : int, y : int, mask_depth : int, color_mask: list):
    """
        Only one color_code should be passed to this function.
    """
    for i in range(mask_depth):
        if image[x,y,i] != int(color_mask[i]):
            return False
    return True


class SegmentBoundingBox:
    def __init__(self, x : int, y : int):
        self.min_x = x
        self.max_x = x
        self.min_y = y
        self.max_y = y

    def newPoint(self, x : int, y : int):
        self.min_x = min(self.min_x, x)
        self.max_x = max(self.max_x, x)
        self.min_y = min(self.min_y, y)
        self.max_y = max(self.max_y, y)

    def getAsList(self):
        return [self.min_x, self.min_y, self.max_x, self.max_y]


def findBoundingBox(example : Example, label_type : str, color_mask : list):
    color_codes = readColorCode(example.label_loc, label_type)
    segment_boxes = {}
    for part_pic in example.parts:
        im = scipy.ndimage.imread(part_pic)
        size_x, size_y, color_depth = np.shape(im)
        for x in range(size_x):
            for y in range(size_y):
                if isInColorCode(im, x, y, 2, color_mask):
                    category_id = im[x,y,2]
                    if category_id not in segment_boxes.keys():
                        segment_boxes[category_id] = SegmentBoundingBox(y,x)
                    else:
                        segment_boxes[category_id].newPoint(y,x)
    return segment_boxes


def createLabelFiles(path : dir):
    samples_with_label = collectSamplesWithLabel(path, 'license plate')
    number_of_samples = len(samples_with_label)
    i = 0
    for file_loc in samples_with_label:
        progress_bar(i, number_of_samples, 'Converting labels')
        example = Example(file_loc)
        bb_label_loc = example.basename + ".txt"
        if not os.path.isfile(bb_label_loc):
            license_plate_mask = [50, 159]
            license_plates = findBoundingBox(example, 'license plate', license_plate_mask)
            license_plates_as_list = map(lambda x : x[1].getAsList(), license_plates.items())
            with open(bb_label_loc, 'w') as f:
                for license_plate_coords in license_plates_as_list:
                    line = str(license_plate_coords)[1:-1]    # get rid of leading '[' and trailing ']'
                    f.write(line + '\n')
        i += 1
    progress_bar(number_of_samples, number_of_samples, 'Converting label')


if __name__=='__main__':
    description = ("Collects all the examples from input_dir to the output_dir. Input dataset is " +
        "expected to be ADE20K dataset, which you can find at http://groups.csail.mit.edu/vision/datasets/ADE20K/")
    argparser = argparse.ArgumentParser(description=description)
    parser.add_argument('input_dir', help='Location of the ADE20K dataset')
    parser.add_argument('output_dir', help='Matching examples will be put under this folder, without preserving the folder structure')
    parser.add_argument('label_type', help='If the description of the image contains this word, it will be put into the output_dir')

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(-1)

    args = parser.parse_args()
    collectSamplesWithLabel(args.input_dir, args.output_dir, args.label_type)

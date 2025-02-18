# this file borrows heavily from the tutorial at https://www.geeksforgeeks.org/find-the-number-of-islands-using-dfs/

import numpy as np
import os  # TODO temp

from mejiro.utils import util


def remove_single_pixels(masked_array):
    x_shape, y_shape = masked_array.shape

    for x in range(0, x_shape):
        for y in range(0, y_shape):
            if masked_array.data[x, y] != 0:
                if x == 0 and y == 0:  # bottom left corner
                    if masked_array.data[x + 1, y] == 0 and masked_array.data[x, y + 1] == 0:
                        masked_array.mask[x, y] = True
                elif x == 0 and y == y_shape - 1:  # top left corner
                    if masked_array.data[x + 1, y] == 0 and masked_array.data[x, y - 1] == 0:
                        masked_array.mask[x, y] = True
                elif x == x_shape and y == 0:  # bottom right corner
                    if masked_array.data[x - 1, y] == 0 and masked_array.data[x, y + 1] == 0:
                        masked_array.mask[x, y] = True
                elif x == x_shape - 1 and y == y_shape - 1:  # top right corner
                    if masked_array.data[x - 1, y] == 0 and masked_array.data[x, y - 1] == 0:
                        masked_array.mask[x, y] = True
                elif x == 0:  # left edge (not corners)
                    if masked_array.data[x + 1, y] == 0 and masked_array.data[x, y + 1] == 0 and masked_array.data[
                        x, y - 1] == 0:
                        masked_array.mask[x, y] = True
                elif x == x_shape - 1:  # right edge (not corners)
                    if masked_array.data[x - 1, y] == 0 and masked_array.data[x, y - 1] == 0 and masked_array.data[
                        x, y + 1] == 0:
                        masked_array.mask[x, y] = True
                elif y == 0:  # bottom edge (not corners)
                    if masked_array.data[x - 1, y] == 0 and masked_array.data[x + 1, y] == 0 and masked_array.data[
                        x, y + 1] == 0:
                        masked_array.mask[x, y] = True
                elif y == y_shape - 1:  # top edge (not corners)
                    if masked_array.data[x - 1, y] == 0 and masked_array.data[x + 1, y] == 0 and masked_array.data[
                        x, y - 1] == 0:
                        masked_array.mask[x, y] = True
                elif masked_array.data[x - 1, y] == 0 and masked_array.data[x + 1, y] == 0 and masked_array.data[
                    x, y - 1] == 0 and masked_array.data[x, y + 1] == 0:
                    masked_array.mask[x, y] = True

    return masked_array


def annular_mask(dimx, dimy, center, r_in, r_out):
    Y, X = np.ogrid[:dimx, :dimy]
    distance_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    return (r_in <= distance_from_center) & \
        (distance_from_center <= r_out)


def get_regions(masked_array, debug_dir=None):
    try:
        masked_array = masked_array.filled(0)  # fill masked values with 0
        formatted_array = masked_array.tolist()  # format as list of lists

        graph = Graph(*masked_array.shape, formatted_array)
        coord_list = graph.get_coord_list()

        indices_list = []
        for region in coord_list:
            x_coords = region[1]
            y_coords = region[0]
            original_indices = list(zip(x_coords, y_coords))
            # print(f'    non-truncated indices are {original_indices}')

            new_region = []
            # identify any new coordinates: these are the coordinates of the new region
            for coordinates in original_indices:
                if not already_present(coordinates, indices_list):
                    new_region.append(coordinates)

            indices_list.append(new_region)

        return indices_list
    except Exception as e:
        print(f'Error in get_regions: {e}')

        if debug_dir is not None:
            util.pickle(os.path.join(debug_dir, 'max_recursion_limit', f'masked_array_{id(masked_array)}.pkl'),
                        masked_array)
        return None


def already_present(coordinates_to_check, indices_list):
    for region in indices_list:
        if coordinates_to_check in region:
            return True
    return False


class Graph:

    def __init__(self, row, col, g):
        self.ROW = row
        self.COL = col
        self.graph = g

    def is_safe(self, i, j, visited):
        return (i >= 0 and i < self.ROW and
                j >= 0 and j < self.COL and
                not visited[i][j] and self.graph[i][j])

    def DFS(self, i, j, visited):
        rowNbr = [-1, 1, 0, 0]
        colNbr = [0, 0, -1, 1]

        # mark this cell as visited
        visited[i][j] = True

        # recur for all connected neighbors
        for k in range(4):
            if self.is_safe(i + rowNbr[k], j + colNbr[k], visited):
                self.DFS(i + rowNbr[k], j + colNbr[k], visited)

        return np.where(visited)

    def get_coord_list(self):
        coord_list = []
        visited = [[False for j in range(self.COL)] for i in range(self.ROW)]

        for i in range(self.ROW):
            for j in range(self.COL):
                # if a cell with value 1 is not visited yet, then new island found
                if visited[i][j] == False and self.graph[i][j] != 0:
                    # visit all cells in this island and increment island count
                    coords = self.DFS(i, j, visited)
                    coord_list.append(coords)

        return coord_list

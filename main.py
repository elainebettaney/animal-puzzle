import pandas as pd
import numpy as np
from itertools import cycle
import copy

import logging
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def load_data():
    df = pd.read_csv('bug_puzzle.csv')
    #df['bug_end'] = df['bug'].str.cat(df['head'], sep='_')

    tiles = {}
    for id in df.tile_id.unique():
        tiles[id] = Tile(id, df[df.tile_id == id])

    return df, tiles

class Tile(object):
    """
    Representation of a tile
    """
    def __init__(self, id, df):

        self.id = id

        self.bug_list = zip(df['bug'].values, df['head'].values)

        self.bug_cycle = cycle(self.bug_list)

        self.bug_set = set(self.bug_list)

        #self.bug_ends = set(df['bug_end'].values)

    def __str__(self):
        return str(self.id)

    #def contains(self, bug, head):

        #bug_end = '_'.join([bug, head])

        #if bug_end in self.bug_ends:
        #    return True
        #else:
        #    return False

    def match(self, condition):
        # Returns an anticlockwise list of bugs starting with condition if condition is met
        # Otherwise returns an empty list

        if condition == []:
            # If there is no condition then this tile is a match!
            return self.bug_list

        bug1 = condition[0]
        if bug1 in self.bug_set:
            # Get an ordered list of bugs starting with the first one in the condition
            bug_index = self.bug_list.index(bug1)
            bug_list = self.bug_list[bug_index:] + self.bug_list[:bug_index]

            nbugs = len(condition)
            if bug_list[:nbugs] == condition:
                return bug_list

        return []


class Path(object):
    """
    Defines the order in which the nodes should be visited.
    If we define this first then we know which edges we need to match too.
    I'm being lazy by hard coding this.  Should be able to build this programmatically.
    """
    def __init__(self):
        self.node_order = [(0,0), (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1), (1,1), (2,0), (2,-1), (2,-2), (1,-2), (0,-2), (-1,-1), (-2,0), (-2,1), (-2,2), (-1,2), (0,2)]

        self.fixed_edges = {(0,0):[],
             (1,0):[(0,0)],
             (1,-1):[(0,0), (1,0)],
             (0,-1):[(0,0), (1,-1)],
             (-1,0):[(0,0), (0,-1)],
             (-1,1):[(0,0), (-1,0)],
             (0,1):[(1,0), (0,0), (-1,1)],
             (1,1):[(1,0), (0,1)],
             (2,0):[(1,0), (1,1)],
             (2,-1):[(1,-1), (1,0), (2,0)],
             (2,-2):[(1,-1), (2,-1)],
             (1,-2):[(0,-1), (1,-1), (2,-2)],
             (0,-2):[(0,-1), (1,-2)],
             (-1,-1):[(-1,0), (0,-1), (0,-2)],
             (-2,0):[(-1,0), (-1,-1)],
             (-2,1):[(-1,1), (-1,0), (-2,0)],
             (-2,2):[(-1,1), (-2, 1)],
             (-1,2):[(0,1), (-1,1), (-2,2)],
             (0,2):[(1,1), (0,1), (-1,2)]}

    def get_node(self, index):
        if index > len(self.node_order) - 1:
            return None
        else:
            return self.node_order[index]

    def get_fixed_edges(self, node):
        fixed_neighbours = self.fixed_edges.get(node, None)
        fixed_edges = [(node, n) for n in fixed_neighbours]
        return fixed_edges

class HexGrid(object):
    """
    Axial hex grid
    """
    def __init__(self, size=2):

        self.size = size

        # Initialise the grid with no tiles placed
        # (i, j) : tile_id
        self.nodes = {}
        for i in range(-size, size + 1):
            for j in range(-size, size + 1):
                if i + j <= size:
                    self.nodes[(i, j)] = None

        # Keep track of the edges as well
        # NOTE: Tracks both a -> b and b -> a
        # ((x1, y1), (x2, y2)) : (bug, head)
        self.edges = {}
        for tile_x in self.nodes.keys():
            for tile_y in self.nodes.keys():
                if self._is_neighbour(tile_x, tile_y):
                    self.edges[(tile_x, tile_y)] = None

    def __str__(self):

        s = super(HexGrid, self).__str__() + '\n'
        for node in self.nodes:
            tile = self.nodes[node]
            if tile:
                s += '{}: {}\n'.format(node, str(tile.id))
                edges = [k for k in self.edges if k[0] == node]
                for e in edges:
                    bug = self.edges[e]
                    if bug:
                        s += '\t {} {}\n'.format(str(e), str(bug))

        return s

    def __deepcopy__(self, memo):
        # I have no idea what I'm doing here

        cls = self.__class__
        result = cls.__new__(cls)

        result.size = self.size
        result.nodes = copy.copy(self.nodes)
        result.edges = copy.copy(self.edges)

        return result

    def print_path(self, path):
        s = ''
        prev_node = path.node_order[0]
        for node in path.node_order:
            if self.nodes[node]:
                if node <> prev_node:
                    s = s + ' -{}- '.format(self.edges[(node, prev_node)][0])

                s = s + str(self.nodes[node].id)
                prev_node = node

        return s

    def get_nodes(self):
        return self.nodes

    def get_edges(self):
        return self.edges

    def get_edge_val(self, edge):
        return self.edges[edge]

    def _is_neighbour(self, x, y):
        # Given 2 co-ordinates, are they neighbours?
        if abs(x[0] - y[0]) <= 1 and abs(x[1] - y[1]) <= 1:
            return True
        else:
            return False

    def get_node_edges(self, node):
        # For a given node, return the edges connected to it

        edges = {}
        for k in self.edges.keys():
            if k[0] == node:
                edges[k] = self.edges[k]

        return edges

    def place_tile(self, node, tile, dEdges):

        # Make sure this node is empty
        assert(self.nodes[node] == None)

        # Place tile
        self.nodes[node] = tile

        # Track new edges
        for edge, bug in dEdges.items():
            edge_rev = (edge[1], edge[0])
            bug_rev = (bug[0], int(not bug[1]))

            self.assign_edge(edge, bug)
            self.assign_edge(edge_rev, bug_rev)

        logger.debug('Grid: {}'.format(str(self)))

    def assign_edge(self, edge, bug):
        #logger.debug('Assigning edge {} {}'.format(str(edge), str(bug)))
        if edge in self.edges:
            if self.edges[edge]:
                # If this edge is already assigned a bug then check it's the same one
                assert (self.edges[edge] == bug)
            else:
                self.edges[edge] = bug

    def get_ordered_neighbours(self, node, start_bug):
        """
        Given a central node return the list of nodes surrounding the central node in anti-clockwise order starting with start.
        NOTE: This returns surrounding nodes whether or not they exist in the grid.
        """
        axial_directions = [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]

        if not start_bug:
            # Can return nodes in any order
            surrounding_nodes = [(node[0] + dir[0], node[1] + dir[1]) for dir in axial_directions]

        else:
            start_node = [(k, v) for k, v in self.edges.items() if k[1] == node and v == (start_bug[0], int(not start_bug[1]))][0][0][0]
            start_direction = (start_node[0] - node[0], start_node[1] - node[1])
            start_index = axial_directions.index(start_direction)

            ordered_directions = axial_directions[start_index:] + axial_directions[:start_index]

            surrounding_nodes = [(node[0] + dir[0], node[1] + dir[1]) for dir in ordered_directions]

        return surrounding_nodes


def place_tile(Grid, path, i, max_i, tile_ids, dTiles):
    logger.debug('i={}'.format(i))

    if i <= max_i:
        node = path.get_node(i)
        logger.debug('Filling node {}'.format(node))

        fixed_edges = path.get_fixed_edges(node)
        logger.debug('Fixed edges: {}'.format(str(fixed_edges)))

        # condition is a list of consecutive edge conditions in anti-clockwise order
        condition = [Grid.get_edge_val(e) for e in fixed_edges]
        logger.debug('Condition: {}'.format(str(condition)))

        # For each tile in tiles
        # Does it meet the condition?
        for tile_id in tile_ids:

            tile = dTiles[tile_id]
            edge_bugs = tile.match(condition)
            logger.debug('\ti={}, Tile {}: {}'.format(i, tile.id, str(edge_bugs)))

            if edge_bugs:
                # Put the tile in the grid
                new_grid = copy.deepcopy(Grid)

                if condition:
                    neighbours = new_grid.get_ordered_neighbours(node, condition[0])
                else:
                    neighbours = new_grid.get_ordered_neighbours(node, None)
                edges = [(node, n) for n in neighbours]
                dEdges = {k:v for k, v in zip(edges, edge_bugs)}

                #print new_grid
                new_grid.place_tile(node, tile, dEdges)
                logger.info(new_grid.print_path(path))

                # Try to place the next tile
                tile_ids_new = [id for id in tile_ids if id <> tile_id]
                success, new_grid = place_tile(new_grid, path, i + 1, max_i, tile_ids_new, dTiles)

                if success:
                    return True, new_grid

    logger.debug('i={}: Failed to find tile'.format(i))
    return False, None

def run():

    tile_df, dTiles = load_data()

    Grid = HexGrid(2)
    path = Path()

    ntiles = len(dTiles)

    i = 0
    success = False
    while not success:
        if i <= ntiles-1:
            success, new_grid = place_tile(Grid, path, i, ntiles-1, dTiles.keys(), dTiles)
            i += 1
        else:
            break

    if success:
        print new_grid
    else:
        print 'Failed to find a solution'


if __name__ == '__main__':
    run()
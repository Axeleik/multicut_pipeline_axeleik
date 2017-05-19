import numpy as np
import vigra
import cPickle as pickle
from copy import deepcopy
from skimage.morphology import skeletonize_3d
from Queue import LifoQueue
from time import time
import h5py
import nifty_with_cplex as nifty




def check_box(volume,point,is_queued_map,is_visited_map,stage=1):
    """checks the Box around the point for points which are 1,
    but were not already put in the queue and returns them in a list"""
    list_not_visited=[]
    list_not_queued = []
    list_are_near = []
    list_is_node = []



    for x in xrange(-1, 2):

        # Edgecase for x
        if point[0] + x < 0 or point[0] + x > volume.shape[0] - 1:
            continue

        for y in xrange(-1, 2):

            # Edgecase for y
            if point[1] + y < 0 or point[1] + y > volume.shape[1] - 1:
                continue

            for z in xrange(-1, 2):

                # Edgecase for z
                if point[2] + z < 0 or point[2] + z > volume.shape[2] - 1:
                    continue

                # Dont look at the middle point
                if x == 0 and y == 0 and z == 0:
                    continue

                if volume[point[0] + x, point[1] + y, point[2] + z] > 0:

                    list_are_near.extend([[point[0] + x, point[1] + y, point[2] + z]])


                    if is_queued_map[point[0] + x, point[1] + y, point[2] + z]==0:
                        list_not_queued.extend([[point[0] + x, point[1] + y, point[2] + z]])


                    #leftover, maybe i`ll need it sometime
                    if is_visited_map[point[0] + x, point[1] + y, point[2] + z] == 0:
                        list_not_visited.extend([[point[0] + x, point[1] + y, point[2] + z]])

    is_visited_map[point[0],point[1],point[2]]=1
    return list_not_queued,list_not_visited,is_visited_map,list_are_near





def init(volume):
    """searches for the first node to start with"""

    point = np.array((np.where(volume)[:][0][0], np.where(volume)[:][1][0], np.where(volume)[:][2][0]))

    is_visited_map = np.zeros(volume.shape, dtype=int)
    is_visited_map[point[0], point[1], point[2]]=1

    is_queued_map =np.zeros(volume.shape, dtype=int)
    is_queued_map[point[0], point[1], point[2]]=1

    not_queued,_,_,_ = check_box(volume,point,is_queued_map,np.zeros(volume.shape, dtype=int))

    if len(not_queued)==2:
        while True:
            point = np.array(not_queued[0])
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            not_queued,_,_,_ = check_box(volume, point, is_queued_map, np.zeros(volume.shape, dtype=int))

            if len(not_queued)!=1:
                break


    return point







def stage_one(img):


    print "initializing first stage..."

    #initializing
    volume = deepcopy(img)
    is_visited_map = np.zeros(volume.shape, dtype=int)
    is_queued_map = np.zeros(volume.shape, dtype=int)
    is_node_map = np.zeros(volume.shape, dtype=int)
    is_term_map  = np.zeros(volume.shape, dtype=int)
    is_branch_map  = np.zeros(volume.shape, dtype=int)
    is_standart_map = np.zeros(volume.shape, dtype=int)
    nodes = {}
    edges = []
    last_node = 1
    current_node = 1
    queue = LifoQueue()
    point=init(volume)
    leftover_list= []
    branch_point_list=[]
    node_list = []
    length=0
    edge_list=[]
    edge_list.extend([[point[0], point[1], point[2]]])

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued,not_visited,is_visited_map,are_near=check_box(volume, point, is_queued_map, is_visited_map)
    nodes[current_node]=point


    for i in xrange(0,len(not_queued)):
        queue.put(np.array([not_queued[i],current_node,length,edge_list]))
        is_queued_map[not_queued[i][0], not_queued[i][1], not_queued[i][2]] = 1

    assert(len(not_queued)>0)

    if len(not_queued)==1:
        is_term_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node

    else:
        is_branch_map[point[0], point[1], point[2]] = last_node
        is_node_map[point[0], point[1], point[2]] = last_node

    print "initialized first stage"
    print "-----------------------"
    print "starting first stage..."

    while queue.qsize():

        #pull item from queue
        point,current_node,length,edge_list=queue.get()

        not_queued,not_visited,is_visited_map,are_near = check_box(volume, point, is_queued_map, is_visited_map)



        #standart point
        if len(not_queued)==1:
            edge_list.extend([[point[0], point[1], point[2]]])
            length = length + np.linalg.norm([point[0] - not_queued[0][0], point[1] - not_queued[0][1], (point[2] - not_queued[0][2]) * 10])
            queue.put(np.array([not_queued[0],current_node,length,edge_list]))
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            branch_point_list.extend([[point[0], point[1], point[2]]])
            is_standart_map[point[0], point[1], point[2]] = 1



        #terminating point
        elif len(not_queued)==0:
            last_node=last_node+1
            nodes[last_node] = point
            edge_list.extend([[point[0], point[1], point[2]]])
            node_list.extend([[point[0], point[1], point[2]]])
            edges.extend([[[current_node, last_node],length,edge_list]])
            is_term_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node
            print "found terminating point"



        # branch point
        elif len(not_queued)>1:
            edge_list.extend([[point[0], point[1], point[2]]])
            last_node = last_node + 1
            nodes[last_node ] = point    #build node
            edges.extend([[[current_node, last_node],length,edge_list]]) #build edge
            node_list.extend([[point[0], point[1], point[2]]])
            edge_list = []
            edge_list.extend([[point[0], point[1], point[2]]])
            #putting node branches in the queue
            for x in not_queued:
                length = np.linalg.norm([point[0] - x[0], point[1] - x[1], (point[2] - x[2]) * 10])
                queue.put(np.array([x, last_node,length,edge_list]))
                is_queued_map[x[0], x[1], x[2]] = 1

            is_branch_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node

            print "found node point "


        else:
            leftover_list.extend([[[point[0], point[1], point[2]],not_queued,not_visited,len(are_near)]])



    assert(len(leftover_list)==0), " there are unclassified leftovers !"


    print "------------------------------"
    print "queue is empty"
    print "------------------------------"
    assert((len(np.where(volume)[0]) - len(np.where(is_branch_map)[0]) - len(np.where(is_term_map)[0]) - len(np.where(is_standart_map)[0]))==0), "too few points were looked at/some were looked at twice !"
    print "all points were looked at"
    print "------------------------------"
    print "phase one finished succesfully"


    return  is_node_map,is_term_map,is_branch_map,nodes,edges




def stage_two(is_node_map, is_term_map, edges):

    print "------------------------"
    print "------------------------"
    print "starting second stage..."

    list_term = np.array(np.where(is_term_map)).transpose()

    for point in list_term:
        print "term point: ", point


        _,_,_,list_near_nodes = check_box(is_node_map, point, np.zeros(is_node_map.shape, dtype=int), np.zeros(is_node_map.shape, dtype=int),2 )

        for i in list_near_nodes:
            edge_list = []
            edge_list.extend([[point[0], point[1], point[2]]])
            edge_list.extend([[i[0], i[1], i[2]]])
            edges.extend([[[is_term_map[point[0],point[1],point[2]], is_node_map[i[0],i[1],i[2]]],np.linalg.norm([point[0] - i[0], point[1] - i[1], (point[2] - i[2]) * 10]),edge_list]]) #build edge

    print "------------------------"
    print "second stage finished"
    print "------------------------"
    return edges


#forming list of terminal numbers
def form_term_list(is_term_map):

    term_where = np.array(np.where(is_term_map)).transpose()
    term_list=[]
    for point in term_where:
        term_list.extend([is_term_map[point[0],point[1],point[2]]])
    term_list = np.array([term for term in term_list])

    return term_list


def skeleton_to_graph(img):

    """main function"""
    time_before_stage_one_1=time()
    is_node_map, is_term_map, is_branch_map, nodes, edges = stage_one(img)
    edges1=deepcopy(edges)

    time_between_stage_1_and_stage_2 = time()

    edges = stage_two(is_node_map, is_term_map, edges)
    edges2 = deepcopy(edges)
    time_after_stage_2=time()

    print len(edges1), " new edges from stage one "
    print len(edges2)-len(edges1) ," new edges from stage two "
    print  "--> Total of ",len(edges2)," edges"

    term_list = form_term_list(is_term_map)
    term_list -= 1
    return nodes,np.array(edges),term_list,time_between_stage_1_and_stage_2-time_before_stage_one_1,time_after_stage_2-time_between_stage_1_and_stage_2


#for debugging purposes
def show(volume,is_queued_map,is_visited_map,is_node_map,point,mode="vo",z=2):

    if mode=="vo":
        print volume[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]

    if mode=="no":
        print is_node_map[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]

    if mode=="qu":
        print is_queued_map[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]

    if mode=="vi":
        print is_visited_map[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]

    if mode=="vovi":
        print is_visited_map[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]==volume[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]











# make the rows of array unique
# see http://stackoverflow.com/questions/16970982/find-unique-rows-in-numpy-array
def get_unique_rows(array, return_index=False):
    array_view = np.ascontiguousarray(array).view(np.dtype((np.void, array.dtype.itemsize * array.shape[1])))
    _, idx = np.unique(array_view, return_index=True)
    unique_rows = array[idx]
    if return_index:
        return unique_rows, idx
    else:
        return unique_rows





def graph_and_edge_weights(nodes,edges_and_lens):


    edges = []
    edge_lens = []
    edges.extend(edges_and_lens[:, 0])
    edges = np.array(edges,dtype="uint32")
    edge_lens.extend(edges_and_lens[:, 1])
    edge_lens = np.array([edge for edge in edge_lens])
    edges = np.sort(edges, axis=1)
    # remove duplicates from edges and edge-lens
    edges, unique_idx = get_unique_rows(edges, return_index=True)
    edge_lens = edge_lens[unique_idx]
    edges_and_lens = edges_and_lens[unique_idx]

    assert len(edges) == len(edge_lens)
    assert edges.shape[1] == 2
    node_list = np.array(nodes.keys())
    edges = np.array(edges, dtype='uint32')
    edges = np.sort(edges, axis=1)
    edges -= 1
    n_nodes = edges.max() + 1
    assert len(node_list) == n_nodes
    assert len(node_list) == n_nodes
    g = nifty.graph.UndirectedGraph(n_nodes)
    g.insertEdges(edges)
    assert g.numberOfEdges == len(edge_lens), '%i, %i' % (g.numberOfEdges, len(edge_lens))
    return g, edge_lens,edges_and_lens





# check that we have the correct number of connected components
def check_connected_components(g):
    cc = nifty.graph.components(g)
    cc.build()

    components = cc.componentLabels()

    #print components
    n_ccs = len(np.unique(components))
    assert n_ccs == 1, str(n_ccs)
    print "Only 1 connected componets -> Passed Test"











def edge_paths_and_counts_for_nodes(g, weights, node_list):
    """
    Returns the path of edges for all pairs of nodes in node list as
    well as the number of times each edge is included in a shortest path.
    @params:
    g         : nifty.graph.UndirectedGraph, the underlying graph
    weights   : list[float], the edge weights for shortest paths
    node_list : list[int],   the list of nodes that will be considered for the shortest paths
    @returns:
    edge_paths: dict[tuple(int,int),list[int]] : Dictionary with list of edge-ids corresponding to the
                shortest path for each pair in node_list
    edge_counts: np.array[int] : Array with number of times each edge was visited in a shortest path
    """
    edge_paths = {}
    edge_counts = np.zeros(g.numberOfEdges, dtype = 'uint32')
    path_finder = nifty.graph.ShortestPathDijkstra(g)
    for ii, u in enumerate(node_list):
        print ii ," of ", len(node_list)
        # target all nodes in node list that we have not visited as source
        # already (for these the path is already present)
        target_nodes = node_list[ii+1:]
        shortest_paths = path_finder.runSingleSourceMultiTarget(weights.tolist(), u, target_nodes)
        assert len(shortest_paths) == len(target_nodes)
        ## associate each shortest paths with the correct uv pair
        ## and return it on the edge level
        #
        # I have implemented some nifty functions to do the conversion
        # paths of nodes to list of edges in c++ directly
        # this speeds up things, but is not in the conda package yet
        #
        #paths_found, this_paths = g.edgesFromNodePaths(shortest_paths)
        #assert len(this_paths) == len(target_nodes)
        #assert np.all(paths_found)
        #for ii, edge_ids in enumerate(this_paths):
        #    edge_counts[edge_ids] += 1
        #    v = target_nodes[ii]
        #    edge_paths[(u,v)] = edge_ids
        # pure python impl
        for jj, sp in enumerate(shortest_paths):
            edge_list = []
            # get the correct target node
            v = target_nodes[jj]
            # walk along the paths and find the edges
            last_node = sp[0]
            assert last_node == v, "%i, %i" % (last_node, v)
            for node in sp[1:]:
                edge_id = g.findEdge(last_node, node)
                edge_list.append(edge_id)
                edge_counts[edge_id] += 1
                last_node = node
            # make sur that the last node we found is the target node
            assert last_node == u, "%i, %i" % (last_node, u)
            edge_paths[(u,v)] = edge_list
    return edge_paths, edge_counts





if __name__ == "__main__":


    time_before_volume=time()

    #print "loading volume.."

    img = np.load("/mnt/localdata03/amatskev/neuraldata/test/skel_img.npz")['arr_0']

    #print "volume loaded"


    #time_after_volume = time()


    nodes, edges,term_list,time_stage_one,time_stage_two = skeleton_to_graph(img)

    #time_loading_volume = time_after_volume - time_before_volume


    #GRAPH STUFF

    #graph1_time=time()
    g,edge_lens,edges = graph_and_edge_weights(nodes,edges)
    #graph2_time = time()
    check_connected_components(g)
    #graph3_time=time()
    edge_paths, edge_counts = edge_paths_and_counts_for_nodes(g, edge_lens,term_list[:30])
    #graph4_time=time()
    edge_list = g.uvIds()

    edge_paths_julian={}

    for pair in edge_paths.keys():
        edge_paths_julian[pair]=[]

        for idx in edge_paths[pair]:
            edge_paths_julian[pair].extend(edges[idx][2])
            print "hi"






    # print "----------------------------------------------------"
    # print " loading volume took ", time_loading_volume," seconds"
    # # print " masking volume took ", time_masking_volume, " seconds"
    # # print " skeletonizing took  ", time_skeletonizing_volume, " seconds" #
    # print " stage one took      ", time_stage_one, " seconds"
    # print " stage two took      ", time_stage_two, " seconds"
    # # print " all of them took    ", time_loading_volume + time_masking_volume + time_skeletonizing_volume + time_stage_one + time_stage_two, " seconds"
    # print "  extract_edges_and_lengths took  ", graph2_time-graph1_time, " seconds"
    # print "  check_connected_components took  ", graph3_time - graph2_time, " seconds"
    # print "  shortest_paths_in_skeleton took  ", graph4_time-graph3_time, " seconds"
    # print " all of them took    ", time_loading_volume + time_stage_one + time_stage_two + (graph2_time-graph1_time) + (graph3_time-graph2_time) + (graph4_time-graph3_time) , " seconds in debugging mode"



"""
 loading volume took  0.702250957489  seconds
 masking volume took  0.647576093674  seconds
 skeletonizing took   73.1700987816  seconds
 stage one took       17.0688860416  seconds
 stage two took       1.80663585663  seconds
 all of them took     93.395447731  seconds
"""



























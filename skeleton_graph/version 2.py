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

                    if stage==1:
                        list_are_near.extend([[point[0] + x, point[1] + y, point[2] + z]])

                    elif stage==2:
                        list_are_near.extend([volume[point[0] + x, point[1] + y, point[2] + z]])

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

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued,not_visited,is_visited_map,are_near=check_box(volume, point, is_queued_map, is_visited_map)
    nodes[current_node]=point


    for i in xrange(0,len(not_queued)):
        queue.put(np.array([not_queued[i],current_node]))
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
        point,current_node=queue.get()

        not_queued,not_visited,is_visited_map,are_near = check_box(volume, point, is_queued_map, is_visited_map)



        #standart point
        if len(not_queued)==1:
            queue.put(np.array([not_queued[0],current_node]))
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            branch_point_list.extend([[point[0], point[1], point[2]]])
            is_standart_map[point[0], point[1], point[2]] = 1



        #terminating point
        elif len(not_queued)==0:
            last_node=last_node+1
            nodes[last_node] = point
            node_list.extend([[point[0], point[1], point[2]]])
            edges.extend([[current_node, last_node]])
            is_term_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node
            print "found terminating point"



        # branch point
        elif len(not_queued)>1:
            last_node = last_node + 1
            nodes[last_node ] = point    #build node
            edges.extend([[current_node, last_node]]) #build edge
            node_list.extend([[point[0], point[1], point[2]]])
            #putting node branches in the queue
            for x in not_queued:
                queue.put(np.array([x, last_node]))
                is_queued_map[x[0], x[1], x[2]] = 1

            is_branch_map[point[0], point[1], point[2]] = last_node
            is_node_map[point[0], point[1], point[2]] = last_node

            print "found node point "


        else:
            leftover_list.extend([[[point[0], point[1], point[2]],not_queued,not_visited,len(are_near)]])



    assert(len(leftover_list)==0), " there are unclassified leftovers !"


    print "------------------------------"
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

        for i in xrange(0,len(list_near_nodes)):
            edges.extend([[is_term_map[point[0],point[1],point[2]], list_near_nodes[i]]]) #build edge

    print "------------------------"
    print "second stage finished"
    print "------------------------"
    return edges






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



    return nodes,edges,time_between_stage_1_and_stage_2-time_before_stage_one_1,time_after_stage_2-time_between_stage_1_and_stage_2


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





if __name__ == "__main__":


    time_before_volume=time()

    print "loading volume.."

    # # pull volume from file
    # print "pulling Volume.."
    # print "-------------------"
    # f = h5py.File("/mnt/localdata03/amatskev/neuraldata/skelexp/result.h5", mode='r')
    # Volume = np.array(f["z/1/test"], dtype=np.int32)
    # print "volume pulled"
    # print "-------------------"

    with open('/mnt/localdata03/amatskev/neuraldata/test/first_try_skel_img.pkl', mode='r') as f:
        img = pickle.load(f)

    print "volume loaded"

    time_after_volume = time()

    # # mask
    # Volume[Volume != 10] = 0
    # Volume[Volume == 10] = 1
    # print "volume masked"
    # print "-------------------"
    #
    # time_after_masking = time()
    #
    # # skeletonize
    # img = skeletonize_3d(Volume)
    # print "volume skeletonized"
    # print "-------------------"
    #
    # time_after_skeletonizing = time()
    #


    nodes, edges,time_stage_one,time_stage_two = skeleton_to_graph(img)

    time_loading_volume = time_after_volume - time_before_volume
    # time_masking_volume = time_after_masking - time_after_volume
    # time_skeletonizing_volume = time_after_skeletonizing - time_after_masking

    print "----------------------------------------------------"
    print " loading volume took ", time_loading_volume," seconds"
    # print " masking volume took ", time_masking_volume, " seconds"
    # print " skeletonizing took  ", time_skeletonizing_volume, " seconds" #
    print " stage one took      ", time_stage_one, " seconds"
    print " stage two took      ", time_stage_two, " seconds"
    # print " all of them took    ", time_loading_volume + time_masking_volume + time_skeletonizing_volume + time_stage_one + time_stage_two, " seconds"
    print " all of them took    ", time_loading_volume + time_stage_one + time_stage_two, " seconds"



    # 1. construct graph
    graph = nifty.graph.UndirectedGraph(len(nodes))
    graph.insertEdges(np.array(edges,dtype="uint32"))

    pass


"""
 loading volume took  0.702250957489  seconds
 masking volume took  0.647576093674  seconds
 skeletonizing took   73.1700987816  seconds
 stage one took       17.0688860416  seconds
 stage two took       1.80663585663  seconds
 all of them took     93.395447731  seconds
"""



























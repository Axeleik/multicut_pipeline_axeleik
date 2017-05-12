import numpy as np
import vigra
import cPickle as pickle
from copy import deepcopy
from skimage.morphology import skeletonize_3d
from Queue import LifoQueue




def check_box(volume,point,is_queued_map,is_visited_map):
    """checks the Box around the point for points which are 1,
    but were not already put in the queue and returns them in a list"""
    list_not_visited=[]
    list_not_queued = []
    list_are_near = []

    if point[0]==126 and point[1]==303 and point[2]==7:
        print"blabla"
        pass


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

                # TODO case if loop, all are queued but not visited
                if volume[point[0] + x, point[1] + y, point[2] + z] == 1:

                    list_are_near.extend([[point[0] + x, point[1] + y, point[2] + z]])

                    if is_queued_map[point[0] + x, point[1] + y, point[2] + z]==0:
                        list_not_queued.extend([[point[0] + x, point[1] + y, point[2] + z]])
                    if is_visited_map[point[0] + x, point[1] + y, point[2] + z]==0:
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






def Term_or_loop():
    pass



def grow():
    pass





def skeleton_to_graph(img,skel):

    """main function"""

    print "initializing..."

    #initializing
    volume = deepcopy(img)
    is_visited_map = np.zeros(volume.shape, dtype=int)
    is_queued_map = np.zeros(volume.shape, dtype=int)
    nodes = {}
    edges = []
    last_node = 1
    queue = LifoQueue()
    point=init(volume)
    ignore = []
    leftover_list= []
    special_case_list=[]
    branch_point_list=[]

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued,not_visited,is_visited_map,are_near=check_box(volume, point, is_queued_map, is_visited_map)
    nodes[last_node]=point


    for i in xrange(0,len(not_queued)):
        queue.put(np.array([not_queued[i],last_node]))
        is_queued_map[not_queued[i][0], not_queued[i][1], not_queued[i][2]] = 1

    print "initialized"
    print "-----------"
    print "starting..."

    while queue.qsize():

        point,last_node=queue.get()

        not_queued,not_visited,is_visited_map,are_near = check_box(volume, point, is_queued_map, is_visited_map)

        #standart branch point
        if len(not_queued)==1:
            queue.put(np.array([not_queued[0],last_node]))
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            branch_point_list.extend([[point[0], point[1], point[2]]])


        #terminating point
        elif len(not_queued)==0 and len(not_visited)==0 and len(are_near)==1:
            nodes[last_node+1] = point
            edges.extend([[last_node, last_node+1]])
            print "found terminating point"


        #node point
        elif len(not_queued)>1:

            nodes[last_node + 1] = point    #build node
            edges.extend([[last_node, last_node + 1]]) #build edge

            #putting node branches in the queue
            for x in not_queued:
                queue.put(np.array([x, last_node+1]))
                is_queued_map[x[0], x[1], x[2]] = 1

            print "found node point "

        #special case 1
        elif len(not_queued) == 0 and len(not_visited) == 0 and len(are_near) > 1:
            print "found special case point"
            special_case_list.extend([[point[0], point[1], point[2]]])
            continue


        # TODO is this the right looping condition ?
        else:
            leftover_list.extend([[point[0], point[1], point[2]]])


    print "not noted points :", len(np.where(volume)[0]) - len(branch_point_list) - len(special_case_list) - len(test_list) - len(nodes)

    return nodes,edges































if __name__ == "__main__":

    print "loading volume.."

    with open('/mnt/localdata03/amatskev/neuraldata/test/first_try_skel_img.pkl', mode='r') as f:
        img = pickle.load(f)

    with open('/mnt/localdata03/amatskev/neuraldata/test/first_try_skel.pkl', mode='r') as f:
        skel = pickle.load(f)

    print "volume loaded"
    skel=skel.transpose()
    nodes, edges = skeleton_to_graph(img,skel)

    print
































""" dump:

Volume = deepcopy(img)
    is_visited = np.zeros(Volume.shape, dtype=int)
    is_queued = np.zeros(Volume.shape, dtype=int)
    nodes = {}
    edges = []
    last_node=-1
    queue = LifoQueue()


    #Copy of img and producing map
    Volume=deepcopy(img)
    Map = np.zeros(Volume.shape, dtype=int)
    Queue_Map = np.zeros(Volume.shape, dtype=int)
    #Lists and Queue
    Queue = LifoQueue() #Last in first out
    Nodes={}
    Edges = np.array([[0,0]]) # dump array
    Node_number = 0
    test=[]
    Anomaly_number=0
    Queue_points=[]


    Point = np.array((np.where(Volume)[:][0][0], np.where(Volume)[:][1][0], np.where(Volume)[:][2][0]))
    print "Starting... "
    while True:

        Contacts=0
        Queue_timer=0

        #Small Box
        for x in xrange(-1, 2):

            #Edgecase for x
            if Point[0]+x<0 or Point[0]+x>Volume.shape[0] - 1:
                continue

            for y in xrange(-1, 2):

                # Edgecase for y
                if Point[1]+y<0 or Point[1]+y>Volume.shape[1] - 1:
                    continue

                for z in xrange(-1, 2):

                    # Edgecase for z
                    if Point[2]+z<0 or Point[2]+z>Volume.shape[2] - 1:
                        continue

                    #Dont put the middle point in the queue
                    if x==0 and y==0 and z==0:
                        continue


                    if Volume[Point[0]+x,Point[1]+y,Point[2]+z]==1:

                        #only if the point was not looked at already
                        if Queue_Map[Point[0]+x,Point[1]+y,Point[2]+z]==0:
                            Queue.put(np.array([Point[0] + x, Point[1] + y, Point[2] + z]))
                            Queue_Map[Point[0]+x,Point[1]+y,Point[2]+z]=-1
                            Queue_timer=Queue_timer+1
                            Queue_points.append([Point[0] + x, Point[1] + y, Point[2] + z])

                        if Queue_Map[Point[0]+x,Point[1]+y,Point[2]+z]>0:
                            if Queue_Map[Point[0]+x,Point[1]+y,Point[2]+z]!=Queue_Map[Point[0],Point[1],Point[2]]:
                                Edges = np.delete(Edges, np.where(Edges[:, 0] == Queue_Map[Point[0]+x,Point[1]+y,Point[2]+z])[0][0], axis=0)
                                Edges[np.where(Edges[:, 1] == 0)[0][len(np.where(Edges[:, 1] == 0)[0]) - 1]][1] = Queue_Map[Point[0]+x,Point[1]+y,Point[2]+z]
                        Contacts=Contacts+1


        assert (Contacts != 0), "Not connected point in skeleton"

        #Writing contact points in Map
        Map[Point[0],Point[1],Point[2]]=Contacts

        #test
        if Contacts>=3 and Queue_timer==1:
            Anomaly_number=Anomaly_number+1
            print "FOUND ANOMALY NUMBER ", Anomaly_number
            test.append(Node_number)


        #Only node if terminating or branch point
        if Contacts!=2 and Queue_timer!=1: # TODO Contacts - queue number or so
            Node_number = Node_number + 1
            print "Node: ", Node_number
            Nodes[Node_number] = np.array([Point[0], Point[1], Point[2]])

            #write the edge
            Edges[np.where(Edges[:, 1] == 0)[0][len(np.where(Edges[:, 1] == 0)[0]) - 1]][1] = Node_number

            # future edges
            if Contacts!=1:
                for i in xrange(0, Queue_timer):
                    Edges = np.append(Edges, np.array([[Node_number, 0]]), axis=0)
                    Queue_Map[Queue_points[i][0],Queue_points[i][1],Queue_points[i][2]]=Node_number

        if Queue.qsize()!=0:
            Point=Queue.get()

        else:
            break

    Edges=np.delete(Edges, 0, axis=0)



    return Map,Nodes,Edges



"""
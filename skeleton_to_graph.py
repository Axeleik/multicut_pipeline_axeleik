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

    if point[0]==1227 and point[1]==735 and point[2]==27:
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

                    if point[0] == 126 and point[1] == 302 and point[2] == 7:
                        print"blabla"
                        pass

                    if point[0] == 126 and point[1] == 303 and point[2] == 7:
                        print"blabla"
                        pass

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
    is_node_map = np.zeros(volume.shape, dtype=int)
    nodes = {}
    edges = []
    last_node = 1
    current_node = 1
    queue = LifoQueue()
    point=init(volume)
    looping_list_vector = []
    looping_list_number = []
    leftover_list= []
    special_case_list=[]
    branch_point_list=[]
    special_branch_point_list=[]
    node_list = []

    is_queued_map[point[0], point[1], point[2]] = 1
    not_queued,not_visited,is_visited_map,are_near=check_box(volume, point, is_queued_map, is_visited_map)
    nodes[current_node]=point


    for i in xrange(0,len(not_queued)):
        queue.put(np.array([not_queued[i],current_node]))
        is_queued_map[not_queued[i][0], not_queued[i][1], not_queued[i][2]] = 1

    print "initialized"
    print "-----------"
    print "starting..."

    while queue.qsize():



        point,current_node=queue.get()


        i = 0
        #looping condition from other side
        if len(looping_list_vector)!=0:

            loop_index=0
            for idx,val in enumerate(looping_list_vector):
                if all(val == [point[0],point[1],point[2]]):
                    branch_point_list.extend([[point[0], point[1], point[2]]])
                    edges.extend([[current_node, looping_list_number[idx]]])
                    i=i+1 # build edge
                    is_visited_map[point[0], point[1], point[2]] = 1
                    loop_index=idx

        assert(i<2)

        if i == 1:
            continue



        not_queued,not_visited,is_visited_map,are_near = check_box(volume, point, is_queued_map, is_visited_map)

        #standart branch point
        if len(not_queued)==1:
            queue.put(np.array([not_queued[0],current_node]))
            is_queued_map[not_queued[0][0], not_queued[0][1], not_queued[0][2]] = 1
            branch_point_list.extend([[point[0], point[1], point[2]]])


        #terminating point
        elif len(not_queued)==0 and len(not_visited)==0 and len(are_near)==1:
            last_node=last_node+1
            nodes[last_node] = point
            node_list.extend([[point[0], point[1], point[2]]])
            edges.extend([[current_node, last_node]])
            print "found terminating point"


        #node point
        elif len(not_queued)>1:
            last_node = last_node + 1
            nodes[last_node ] = point    #build node
            edges.extend([[current_node, last_node]]) #build edge
            node_list.extend([[point[0], point[1], point[2]]])
            #putting node branches in the queue
            for x in not_queued:
                queue.put(np.array([x, last_node]))
                is_queued_map[x[0], x[1], x[2]] = 1

            is_node_map[point[0], point[1], point[2]] = 1

            print "found node point "

        #special case 1
        elif len(not_queued) == 0 and len(not_visited) == 0 and len(are_near) > 1:
            print "found special case point"
            special_case_list.extend([[point[0], point[1], point[2]]])
            continue




        # TODO is this the right looping condition ?
        elif len(not_queued) == 0 and len(not_visited) == 1 and len(are_near) == 2:

            #workaround for searching for vector in list
            looping_list_vector.extend(np.array([[not_visited[0][0] , not_visited[0][1] , not_visited[0][2]]]))
            looping_list_number.extend(np.array([current_node]))
            branch_point_list.extend([[point[0], point[1], point[2]]])
            print "found loop "

            pass

        #special branch point
        elif len(not_queued)==0 and len(not_visited)==0 and len(are_near)>1:
            special_branch_point_list.extend([[point[0], point[1], point[2]]])


        else:
            leftover_list.extend([[point[0], point[1], point[2]]])




    print "not noted points :", len(np.where(volume)[0]) - len(branch_point_list) - len(special_case_list) - len(leftover_list) - len(node_list) - len(special_branch_point_list)






    allofthem=np.concatenate((np.array(branch_point_list),np.array(special_case_list),np.array(leftover_list),np.array(node_list),np.array(special_branch_point_list)))
    nr1=np.ascontiguousarray(allofthem).view(np.dtype((np.void, allofthem.dtype.itemsize * allofthem.shape[1])))
    assert(len(skel)>len(allofthem))
    nr2=np.ascontiguousarray(skel).view(np.dtype((np.void, skel.dtype.itemsize * skel.shape[1])))

    a=np.setdiff1d(nr1,nr2)

    new=a.view(allofthem.dtype)
    new=new.reshape(new.shape[0]/3,3)







    delbranch=[]
    z=len(branch_point_list)
    #test for not fetched
    for idx1, val1 in enumerate(branch_point_list):

        if idx1==100:

            pass
            pass

        print idx1, " of ", z-1, " elements"
        for idx2,val2 in enumerate(skel):



            if all(val1==val2):
                delbranch.extend([val2])
                continue

    delnodes = []
    z = len(branch_point_list)
    # test for not fetched
    for idx1, val1 in enumerate(nodes):

        print idx1, " of ", z - 1, " elements"
        for idx2, val2 in enumerate(skel):

            if all(val1 == val2):
                delnodes.extend([val2])
                continue

    delspecial = []
    z = len(branch_point_list)
    # test for not fetched
    for idx1, val1 in enumerate(special_case_list):

        print idx1, " of ", z - 1, " elements"
        for idx2, val2 in enumerate(skel):

            if all(val1 == val2):
                delspecial.extend([val2])
                continue

    delleft = []
    z = len(leftover_list)
    # test for not fetched
    for idx1, val1 in enumerate(leftover_list):

        print idx1, " of ", z - 1, " elements"
        for idx2, val2 in enumerate(skel):

            if all(val1 == val2):
                delleft.extend([val2])
                continue








    return nodes,edges






def show(volume,point,mode="v",z=2):

    if mode=="vo":
        print volume[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]

    if mode=="no":
        print volume[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]

    if mode=="qu":
        print volume[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]

    if mode=="vi":
        print volume[point[0]-z:point[0]+z+1, point[1]-z:point[1]+z+1, point[2]-z:point[2]+z+1]


























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
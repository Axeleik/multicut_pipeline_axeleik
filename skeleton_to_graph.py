import numpy as np
import vigra
import cPickle as pickle
from copy import deepcopy
from skimage.morphology import skeletonize_3d
from Queue import LifoQueue






def skeleton_to_graph(img):

    #Copy of img and producing map
    Volume=deepcopy(img)
    Map = np.zeros(Volume.shape, dtype=int)

    #Lists and Queue
    Queue = LifoQueue() #Last in first out
    Nodes={}
    Edges = np.array([[0,0]]) # dump array
    Node_number = 0
    test=[]

    Point = np.array((np.where(Volume)[:][0][0], np.where(Volume)[:][1][0], np.where(Volume)[:][2][0]))
    print "Starting... "
    while True:

        Contacts=0
        Future_nodes=0


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
                        if Map[Point[0]+x,Point[1]+y,Point[2]+z]==0:
                            Queue.put(np.array([Point[0] + x, Point[1] + y, Point[2] + z]))
                            Future_nodes = Future_nodes + 1

                        Contacts=Contacts+1

        assert (Contacts != 0), "Not connected point in skeleton"

        #Writing contact points in Map
        Map[Point[0],Point[1],Point[2]]=Contacts

        if Contacts==2 and Future_nodes==0:
            test.append(Node_number)
            #Edges[np.where(Edges[:, 1] == 0)[0][len(np.where(Edges[:, 1] == 0)[0]) - 1]][1] = Node_number

        #Only node if terminating or branch point
        if Contacts!=2 or (Contacts==2 and Future_nodes==0): #als letztes geaendert

            Node_number = Node_number + 1
            print "Node: ", Node_number
            Nodes[Node_number] = np.array([Point[0], Point[1], Point[2]])

            #write the edge
            Edges[np.where(Edges[:, 1] == 0)[0][len(np.where(Edges[:, 1] == 0)[0]) - 1]][1] = Node_number

            # future edges
            if Contacts!=1:
                for i in xrange(0, Future_nodes):
                    Edges = np.append(Edges, np.array([[Node_number, 0]]), axis=0)


        if Queue.qsize()!=0:
            Point=Queue.get()

        else:
            break

    Edges=np.delete(Edges, 0, axis=0)


    """schauen ob es auch anhaefungen von pixeln gibt wie beim alten algorithmus, und nicht nur ecke an ecke oder seite an seite
    also 3 pixel nebeneinander, obwohl sie eine linie bilden--> der dritte """


    """ausserdem schau auf die tafel, wenn z.b. 3 neben einander sind aber in
    3d oder es einen ein pixel langen branch gibt hat er zwar
    dann einen terminating point, aber auch gleichzeitig zwei
    contacts und schreibt den terminating point nicht """
    return Map,Nodes,Edges


if __name__ == "__main__":

    with open('/mnt/localdata03/amatskev/neuraldata/test/first_try_skel_img.pkl', mode='r') as f:
        img = pickle.load(f)

    Map,Nodes,Edges=skeleton_to_graph(img)

    input("hi")
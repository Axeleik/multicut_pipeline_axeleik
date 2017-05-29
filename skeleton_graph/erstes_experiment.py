import numpy as np
import h5py
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.morphology import medial_axis, skeletonize, skeletonize_3d

"""

Benutze mit mc_pipeline_new !

"""


#
# def printname(name):
#     print (name)
#
#
# f= h5py.File("/media/axeleik/EA62ECB562EC8821/data/pipeline/erstellt/results/splB_z1/result.h5", mode='r')
# f.visit(printname)
# a = np.array(f["z/1/data"],dtype=np.int32)
#
# print (a.shape)
#
# params = skeletopyze.Parameters()
# #params.min_segment_length= 300
# print("Skeletonizing")
#
# """
# print("Skeleton contains nodes:")
# for n in b.nodes():
#     print( str(n) + ": " + "(%d, %d, %d), diameter %f"%(b.locations(n).x(), b.locations(n).y(), b.locations(n).z(), b.diameters(n)))
#
# print("Skeleton contains edges:")
# for e in b.edges():
#     print (e.u, e.v)
#
# edges = [e for e in b.edges()]
# nodes = [e for e in b.nodes()]
#
# path=np.array(([0,0,0]))
# """
# skeletons=[]
# for label in np.unique(a):
#     if label==0:
#         continue
#     unique=deepcopy(a)
#     unique[a != label] = 0
#     unique[a == label] = 1
#     b = skeletopyze.get_skeleton_graph(unique, params)
#     """
#     for e in b.edges():
#
#         for o in b.edges():
#             if e.u == o.u and e.v == o.v:
#                 continue
#             if e.u == o.u or e.v == o.v:
#                 print ("FOUND! \n \n" )
#                 print (e.u, e.v)
#                 print (o.u, o.v)
#                 print ("\n\n")
#     """
#
#
#     skeletons.append(b)
#
#
#
# paths=[]
#
# for b in skeletons:
#     array = np.array([[0,0,0]])
#
#     for n in b.nodes():
#         array = np.append(array, [[b.locations(n).x(), b.locations(n).y(), b.locations(n).z()]], axis=0)
#
#     paths.append(array[1:])
#
# for data in paths:
#     data = data.transpose()
#     fig = plt.figure()
#     ax = Axes3D(fig)
#     ax.plot(data[0], data[1], data[2], label='original_true', lw=2, c='Dodgerblue')  # gezackt
#     plt.show()
#
#
# input("hi")
#
# """
# for n in edges:
#     path= np.append(path, [[4, 2, 3]], axis=0)
# """
#params = skeletopyze.Parameters()
a = np.zeros((200,250,200), dtype=np.int32)
b = np.zeros((200,250,200), dtype=np.int32)



a[53:79,50:185,20:40] = 1
a[121:147,50:185,20:40] = 1
a[19:190,176:185,20:40] = 1  #unterschied von 20:47 auf 48, linker arm faellt weg
a[25:175,25:50,20:40] = 1
a[80:120,25:176,20:40] = 1

a[54:79,80:100,1:100] = 1
a[53:79,50:185,80:110] = 1
a[121:147,50:185,80:110] = 1
a[1:199,176:185,80:110] = 1  #unterschied von 20:47 auf 48, linker arm faellt weg
a[25:175,25:50,80:110] = 1
a[80:120,25:176,80:110] = 1

# b[25:175,25:50,20:40] = 1
# b[80:120,25:176,20:40] = 1



# Volume=np.array([[54,51,20],[54,51,40],
#                  [79,51,20],[79,51,40],
#                  [121,51,20],[121,51,40],
#                  [156,51,20],[146,51,40],
#                  [25,176,20],[25,176,40],
#                  [175,176,20],[175,176,40],
#                  [25, 195, 20], [25, 195, 40],
#                  [175, 195, 20], [175, 195, 40],
#                  [54, 176, 20], [54, 176, 40],
#                  [79,176,20],[79,176,40],
#                  [121,176,20],[121,176,40],
#                  [146,176,20],[146,176,40],
#                  [100, 176, 20], [100, 176, 40],
#                  [100, 195, 20], [100, 195, 40]])

# #a=1-a
# c = skeletopyze.get_skeleton_graph(a, params)
#
# # testarray=np.array([[0,0,0]])
# # for n in c.nodes():
# #     testarray = np.append(testarray, [[c.locations(n).x(), c.locations(n).y(), c.locations(n).z()]], axis=0)
# testarray = np.array([[c.locations(n).x(), c.locations(n).y(), c.locations(n).z()] for n in c.nodes()])
#
#
# plt.imshow(a[:,:,30])
# plt.show()

#np.all(c, axis=1)

plt.imshow(a[:,:,29])
plt.show()

skel_img_a = skeletonize_3d(a)
skel_img_b = skeletonize_3d(b)
# mask = np.ones(skel_img.shape, dtype=np.bool)
# mask[95:105,45:176,27:31] = 0
# skel_img[mask] = 0
plt.imshow((skel_img_a + skel_img_a)[:,:,29])
plt.show()

skel_a=np.array([[e1,e2,e3] for e1,e2,e3 in zip(np.where(skel_img_a)[0],np.where(skel_img_a)[1],np.where(skel_img_a)[2])])

# skel_a_1_1=np.array([[e1,e2,e3] for e1,e2,e3 in skel_a if e3==29 and e1<100])
# skel_a_1_2=np.array([[e1,e2,e3] for e1,e2,e3 in skel_a if e3==29 and e1>100])
# skel_a_2=np.array([[e1,e2,e3] for e1,e2,e3 in skel_a if e3==30])
# skel_a_3=np.array([[e1,e2,e3] for e1,e2,e3 in skel_a if e3!=30 and e3!=29])

skel_b= np.array([[e1,e2,e3] for e1,e2,e3 in zip(np.where(skel_img_b)[0],np.where(skel_img_b)[1],np.where(skel_img_b)[2])])

# skel_a_list=[]
# skel_b_list=[]

# for i in skel_a:
#     skel_a_list.append(np.array([i]))

# if skel_a.shape[0]!=0:
#     for i in np.unique(skel_a[:,2]):
#         skel_a_list.append(np.array([[e1,e2,e3] for e1,e2,e3 in skel_a if e3==i]))

# if skel_b.shape[0]!=0:
#     for i in np.unique(skel_b[:,2]):
#         skel_b_list.append(np.array([[e1,e2,e3] for e1,e2,e3 in skel_b if e3==i]))

# skel_b_1_1=np.array([[e1,e2,e3] for e1,e2,e3 in skel_b if e3==29])
# skel_b_2=np.array([[e1,e2,e3] for e1,e2,e3 in skel_b if e3==30])
# skel_b_3=np.array([[e1,e2,e3] for e1,e2,e3 in skel_b if e3!=30 and e3!=29])


# data_a_1_1 = skel_a_1_1.transpose()
# data_a_1_2 = skel_a_1_2.transpose()
# data_a_2 = skel_a_2.transpose()
# data_a_3 = skel_a_3.transpose()
data_ab_4 = np.array([[0,200],[0,250],[0,200]])

# for i,obj in enumerate(skel_a_list):
#     skel_a_list[i]=obj.transpose()

skel_a=skel_a.transpose()
skel_b=skel_b.transpose()

# for i,obj in enumerate(skel_b_list):
#     skel_b_list[i]=obj.transpose()

# data_b_1_1 = skel_b_1_1.transpose()
# data_b_2 = skel_b_2.transpose()
# data_b_3 = skel_b_3.transpose()



fig = plt.figure()
ax = Axes3D(fig)


ax.scatter(skel_a[0], skel_a[1], skel_a[2], label='original_true', lw=0.000001, c='Dodgerblue')
# ax.scatter(skel_b[0], skel_b[1], skel_b[2], label='original_true', lw=0.000001, c='Red')


# for i,obj in enumerate(skel_a_list):
#     ax.scatter(obj[0], obj[1], obj[2], label='original_true', lw=0.000001, c='Dodgerblue')

# Volume=Volume.transpose()
# ax.scatter(Volume[0], Volume[1], Volume[2], label='original_true', lw=3,c="Yellow")

# ax.scatter(obj[0], obj[1], obj[2], label='original_true', lw=5, c='Yellow')

# for i,obj in enumerate(skel_b_list):
#     ax.plot(obj[0], obj[1], obj[2], label='original_true', lw=5, c='Red')


# if data_a_1_1.shape[0]!=0:
#     ax.plot(data_a_1_1[0],data_a_1_1[1],data_a_1_1[2], label='original_true', lw=5, c='Dodgerblue')
#
# if data_a_1_2.shape[0]!=0:
#     ax.plot(data_a_1_2[0],data_a_1_2[1],data_a_1_2[2], label='original_true', lw=5, c='Dodgerblue')
#
# if data_a_2.shape[0]!=0:
#     ax.plot(data_a_2[0],data_a_2[1],data_a_2[2], label='original_true', lw=5, c='Red')

if data_ab_4.shape[0]!=0:
    ax.plot(data_ab_4[0],data_ab_4[1],data_ab_4[2], label='original_true', lw=0, c='Yellow')

# if data_b_1_1.shape[0]!=0:
#     ax.plot(data_b_1_1[0],data_b_1_1[1],data_b_1_1[2], label='original_true', lw=5, c='Dodgerblue')
#
# if data_b_2.shape[0]!=0:
#     ax.plot(data_b_2[0],data_b_2[1],data_b_2[2], label='original_true', lw=5, c='Red')
#
# if data_a_3.shape[0]!=0:
#     ax.plot(data_a_3[0],data_a_3[1],data_a_3[2], label='original_true', lw=5, c='Yellow')
#
# if data_b_3.shape[0]!=0:
#     ax.plot(data_b_3[0],data_b_3[1],data_b_3[2], label='original_true', lw=5, c='Yellow')

plt.show()



# import numpy as np
import cv2
# import contour_pair_list


SCALE_FACTOR = 100.0
CONTOUR_POINT_COUNT = 16
ret_str = ""

coord_correction = []
for i in range(18):
    coord_correction.append([-1028, -774])

coord_correction[4][1] += -2
coord_correction[5][1] += 0
coord_correction[6][1] += 0
coord_correction[7][1] += 5
coord_correction[8][1] += 10
coord_correction[9][1] += 0

all_contours = []
all_centroids = []
for i in range(18):
    num = '0' + str(i+1)
    filename = 'images/Spiriferella-spiralia-' + num[-2:]
    print(filename)
    im = cv2.imread(filename + '.png')

    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    j = 0
    centroids_in_a_section = []
    contours_in_a_section = []
    x_correction = coord_correction[i][0]
    y_correction = coord_correction[i][1]

    for c in contours:
        max_x = [0, 0]
        min_x = [9999, 9999]
        idx = 0
        max_idx = -1
        min_idx = -1
        num_point = CONTOUR_POINT_COUNT
        idx_list = []
        for pt in c:
            if pt[0][0] > max_x[0]:
                max_x = pt[0]
                max_idx = idx
            if pt[0][0] < min_x[0]:
                min_x = pt[0]
                min_idx = idx
            idx += 1
        idx_diff1 = int((max_idx - min_idx) / (num_point / 2))
        idx_diff2 = int((min_idx + (len(c) - max_idx)) / (num_point / 2))
        for k in range(int(num_point / 2)):
            idx_list.append(int(min_idx + idx_diff1 * k))
        for k in range(int(num_point / 2)):
            idx_list.append(int(max_idx + idx_diff2 * k) % len(c))
        # print(max_x, min_x, max_idx, min_idx, idx_list, len(c))
        simp_cont = []
        for k in range(num_point):
            simp_cont.append([c[idx_list[k]][0][0] + x_correction, -1 * (c[idx_list[k]][0][1] + y_correction), (i - 9) * 40.3])
        # print(simp_cont)

        contours_in_a_section.append(simp_cont)
        # for idx in range(len(simp_cont)):
        # cv2.putText(im,str(idx) ,(simp_cont[idx][0],simp_cont[idx][1]),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)

        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        centroids_in_a_section.append([cx + x_correction, -1 * (cy + y_correction), (i - 9) * 40.3])
        verts = []
        for v in simp_cont:
            verts.append("(" + ",".join([str(coord) for coord in v]) + ")")
        # contour_list.append(  "[" + ",\n".join( verts ) + "]" )
        j += 1

    all_centroids.append(centroids_in_a_section)
    all_contours.append(contours_in_a_section)

# convert contours into vertices
all_contours_str_list = []
for contours_in_a_section in all_contours:
    contour_str_list = []
    for contour in contours_in_a_section:
        v_str_list = []
        for v in contour:
            # print("v:",v)
            v_str = "(" + ",".join([str(x/SCALE_FACTOR) for x in v]) + ")"
            v_str_list.append(v_str)
        contour_str = "[" + ",".join(v_str_list) + "]"
        # print( "contour_str:", contour_str )
        contour_str_list.append(contour_str)
    contours_in_a_section_str = "[" + ",\n".join(contour_str_list) + "]"
    all_contours_str_list.append(contours_in_a_section_str)
ret_str += "contour_list=[" + ",\n".join(all_contours_str_list) + "]\n"
# ret_str += "centroids=[" + ",\n".join(centroids)+"]\n"

# convert centroids into vertices
all_centroids_str_list = []

for centroids_in_a_section in all_centroids:
    centroids_str_list = []
    centroids_in_a_section_str = ""
    for centroid in centroids_in_a_section:
        centroid_str = "(" + ",".join([str(x/SCALE_FACTOR) for x in centroid]) + ")"
        # print(centroid_str)
        centroids_str_list.append(centroid_str)
    centroids_in_a_section_str = "[" + ",\n".join(centroids_str_list) + "]"
    all_centroids_str_list.append(centroids_in_a_section_str)
ret_str += "centroid_list=[" + ",".join(all_centroids_str_list) + "]\n"


ret_str += '''
import bpy
import bmesh
import math
#import contour_pair_list

SCALE_FACTOR = 100.0
SPIRALIA_RADIUS = 500 / SCALE_FACTOR

mesh = bpy.data.meshes.new("mesh")  # add a new mesh
obj = bpy.data.objects.new("MyObject", mesh)  # add a new object using the mesh

scene = bpy.context.scene
scene.objects.link(obj)  # put the object into the scene (link)
scene.objects.active = obj  # set as the active object in the scene
obj.select = True  # select object

mesh = bpy.context.object.data
bm = bmesh.new()

contour_pair_list = [
[[1,0],[2,3]],
[[2,0],[3,3]],
[[2,1],[3,4]],
[[2,2],[3,6]],
[[2,2],[3,5]],
[[2,3],[3,8]],
[[2,3],[3,9]],
[[2,5],[3,10]],
[[2,5],[3,12]],
[[2,4],[3,13]],
[[2,6],[3,14]],
[[2,7],[3,15]],
[[3,0],[4,0]],
[[3,1],[4,1]],
[[3,2],[4,2]],
[[3,3],[4,5]],
[[3,3],[4,3]],
[[3,4],[4,7]],
[[3,6],[4,10]],
[[3,8],[4,11]],
[[3,10],[4,13]],
[[3,13],[4,16]],
[[3,14],[4,19]],
[[3,16],[4,22]],
[[3,17],[4,24]],
[[3,4],[4,6]],
[[3,5],[4,8]],
[[3,9],[4,12]],
[[3,12],[4,17]],
[[3,15],[4,23]],
[[3,20],[4,27]],
[[3,22],[4,29]],
[[4,0 ],[5,2 ]],
[[4,0 ],[5,0 ]],
[[4,1 ],[5,6 ]],
[[4,1 ],[5,1 ]],
[[4,2 ],[5,8 ]],
[[4,2 ],[5,4 ]],
[[4,5 ],[5,10]],
[[4,7 ],[5,14]],
[[4,10],[5,18]],
[[4,11],[5,20]],
[[4,13],[5,22]],
[[4,16],[5,25]],
[[4,19],[5,27]],
[[4,22],[5,30]],
[[4,24],[5,33]],
[[4,26],[5,34]],
[[4,3 ],[5,7 ]],
[[4,6 ],[5,11]],
[[4,8 ],[5,17]],
[[4,12],[5,21]],
[[4,17],[5,29]],
[[4,23],[5,32]],
[[4,27],[5,35]],
[[4,29],[5,37]],
[[4,30],[5,38]],
[[5,2 ],[6,3 ]],
[[5,6 ],[6,6 ]],
[[5,8 ],[6,11]],
[[5,10],[6,14]],
[[5,14],[6,17]],
[[5,18],[6,20]],
[[5,20],[6,23]],
[[5,22],[6,24]],
[[5,25],[6,28]],
[[5,27],[6,29]],
[[5,30],[6,33]],
[[5,33],[6,36]],
[[5,34],[6,37]],
[[5,0 ],[6,0 ]],
[[5,1 ],[6,1 ]],
[[5,4 ],[6,5 ]],
[[5,7 ],[6,10]],
[[5,11],[6,15]],
[[5,17],[6,22]],
[[5,21],[6,26]],
[[5,29],[6,32]],
[[5,32],[6,35]],
[[5,35],[6,38]],
[[5,37],[6,40]],
[[5,38],[6,41]],
[[6,0 ],[7,1 ]],
[[6,1 ],[7,4 ]],
[[6,5 ],[7,11]],
[[6,10],[7,17]],
[[6,15],[7,23]],
[[6,22],[7,27]],
[[6,26],[7,31]],
[[6,32],[7,36]],
[[6,35],[7,39]],
[[6,38],[7,42]],
[[6,40],[7,43]],
[[6,41],[7,44]],
[[6,42],[7,45]],
[[6,3 ],[7,6 ]],
[[6,6 ],[7,10]],
[[6,11],[7,13]],
[[6,14],[7,16]],
[[6,17],[7,20]],
[[6,20],[7,24]],
[[6,23],[7,25]],
[[6,24],[7,29]],
[[6,28],[7,30]],
[[6,29],[7,34]],
[[6,33],[7,35]],
[[6,36],[7,38]],
[[6,37],[7,40]],
[[7,1 ],[8,1]],
[[7,4 ],[8,5]],
[[7,11],[8,11]],
[[7,17],[8,17]],
[[7,23],[8,22]],
[[7,27],[8,26]],
[[7,31],[8,29]],
[[7,36],[8,34]],
[[7,39],[8,38]],
[[7,42],[8,42]],
[[7,43],[8,43]],
[[7,44],[8,44]],
[[7,45],[8,45]],
[[7,6 ],[8,6]],
[[7,10],[8,9]],
[[7,13],[8,13]],
[[7,16],[8,16]],
[[7,20],[8,19]],
[[7,24],[8,23]],
[[7,25],[8,25]],
[[7,29],[8,27]],
[[7,30],[8,31]],
[[7,34],[8,33]],
[[7,35],[8,35]],
[[7,38],[8,39]],
[[7,40],[8,40]],
[[8,6 ],[9,5 ]],
[[8,9 ],[9,9 ]],
[[8,13],[9,13]],
[[8,16],[9,15]],
[[8,19],[9,19]],
[[8,23],[9,22]],
[[8,25],[9,25]],
[[8,27],[9,28]],
[[8,31],[9,29]],
[[8,33],[9,33]],
[[8,35],[9,36]],
[[8,39],[9,37]],
[[8,40],[9,40]],
[[8,1 ],[9,1 ]],
[[8,5 ],[9,6 ]],
[[8,11],[9,12]],
[[8,17],[9,18]],
[[8,22],[9,23]],
[[8,26],[9,26]],
[[8,29],[9,31]],
[[8,34],[9,35]],
[[8,38],[9,39]],
[[8,42],[9,41]],
[[8,43],[9,42]],
[[8,44],[9,43]],
[[8,45],[9,44]],
[[9,5 ],[10,6 ]],
[[9,9 ],[10,8 ]],
[[9,13],[10,12]],
[[9,15],[10,15]],
[[9,19],[10,18]],
[[9,22],[10,22]],
[[9,25],[10,24]],
[[9,28],[10,27]],
[[9,29],[10,30]],
[[9,33],[10,32]],
[[9,36],[10,35]],
[[9,37],[10,38]],
[[9,40],[10,41]],
[[9,1 ],[10,1 ]],
[[9,6 ],[10,3 ]],
[[9,12],[10,10]],
[[9,18],[10,16]],
[[9,23],[10,21]],
[[9,26],[10,25]],
[[9,31],[10,29]],
[[9,35],[10,33]],
[[9,39],[10,37]],
[[9,41],[10,40]],
[[9,42],[10,42]],
[[9,43],[10,43]],
[[9,44],[10,44]],
[[10,1 ],[11,1 ]],
[[10,3 ],[11,5 ]],
[[10,10],[11,13]],
[[10,16],[11,19]],
[[10,21],[11,23]],
[[10,25],[11,27]],
[[10,29],[11,32]],
[[10,33],[11,35]],
[[10,37],[11,39]],
[[10,40],[11,42]],
[[10,42],[11,43]],
[[10,43],[11,44]],
[[10,6 ],[11,3 ]],
[[10,8 ],[11,6 ]],
[[10,12],[11,9 ]],
[[10,15],[11,12]],
[[10,18],[11,15]],
[[10,22],[11,18]],
[[10,24],[11,22]],
[[10,27],[11,26]],
[[10,30],[11,28]],
[[10,32],[11,31]],
[[10,35],[11,33]],
[[10,38],[11,36]],
[[10,41],[11,40]],
[[11,1 ],[12,1 ]],
[[11,5 ],[12,5 ]],
[[11,13],[12,12]],
[[11,19],[12,19]],
[[11,23],[12,23]],
[[11,27],[12,26]],
[[11,32],[12,30]],
[[11,35],[12,35]],
[[11,39],[12,39]],
[[11,42],[12,41]],
[[11,43],[12,43]],
[[11,44],[12,44]],
[[11,3 ],[12,3 ]],
[[11,6 ],[12,6 ]],
[[11,9 ],[12,9 ]],
[[11,12],[12,11]],
[[11,15],[12,15]],
[[11,18],[12,18]],
[[11,22],[12,22]],
[[11,26],[12,25]],
[[11,28],[12,28]],
[[11,31],[12,31]],
[[11,33],[12,34]],
[[11,36],[12,37]],
[[11,40],[12,40]],
[[11,41],[12,42]],
[[12,3 ],[13,4]],
[[12,6 ],[13,7]],
[[12,9 ],[13,10]],
[[12,11],[13,13]],
[[12,15],[13,17]],
[[12,18],[13,21]],
[[12,22],[13,24]],
[[12,25],[13,28]],
[[12,28],[13,30]],
[[12,31],[13,34]],
[[12,34],[13,38]],
[[12,37],[13,39]],
[[12,40],[13,40]],
[[12,1 ],[13,1 ]],
[[12,5 ],[13,6 ]],
[[12,12],[13,11]],
[[12,19],[13,16]],
[[12,23],[13,20]],
[[12,26],[13,25]],
[[12,30],[13,29]],
[[12,35],[13,33]],
[[12,39],[13,36]],
[[12,41],[13,38]],
[[12,43],[13,39]],
[[12,44],[13,40]],
[[13,1 ],[14,4 ]],
[[13,6 ],[14,6 ]],
[[13,11],[14,9 ]],
[[13,16],[14,12]],
[[13,20],[14,15]],
[[13,25],[14,19]],
[[13,29],[14,21]],
[[13,33],[14,24]],
[[13,36],[14,27]],
[[13,38],[14,28]],
[[13,39],[14,29]],
[[13,7 ],[14,4 ]],
[[13,10],[14,6 ]],
[[13,13],[14,9 ]],
[[13,17],[14,12]],
[[13,21],[14,15]],
[[13,24],[14,19]],
[[13,28],[14,21]],
[[13,30],[14,24]],
[[13,34],[14,27]],
[[3,7],[4,9]],
[[3,11],[4,15]],
[[3,18],[4,14]],
[[3,18],[4,20]],
[[3,19],[4,18]],
[[3,21],[4,25]],
[[4,4],[5,12]],
[[4,9],[5,13]],
[[4,9],[5,16]],
[[4,15],[5,15]],
[[4,15],[5,24]],
[[4,14],[5,19]],
[[4,20],[5,28]],
[[4,18],[5,23]],
[[4,25],[5,31]],
[[4,21],[5,26]],
[[4,28],[5,36]],
[[5,3],[6,4]],
[[5,5],[6,8]],
[[5,9],[6,7]],
[[5,9],[6,12]],
[[5,12],[6,9]],
[[5,12],[6,16]],
[[5,13],[6,13]],
[[5,16],[6,19]],
[[5,15],[6,18]],
[[5,24],[6,27]],
[[5,19],[6,21]],
[[5,28],[6,31]],
[[5,23],[6,25]],
[[5,31],[6,34]],
[[5,26],[6,30]],
[[5,36],[6,39]],
[[6,2],[7,0]],
[[6,2],[7,3]],
[[6,4],[7,2]],
[[6,4],[7,8]],
[[6,8],[7,5]],
[[6,8],[7,12]],
[[6,7],[7,7]],
[[6,12],[7,15]],
[[6,9],[7,9]],
[[6,16],[7,18]],
[[6,13],[7,14]],
[[6,19],[7,22]],
[[6,18],[7,19]],
[[6,27],[7,28]],
[[6,21],[7,21]],
[[6,31],[7,33]],
[[6,25],[7,26]],
[[6,34],[7,37]],
[[6,30],[7,32]],
[[6,39],[7,41]],
[[7, 0],[8, 0]],
[[7, 3],[8, 4]],
[[7, 2],[8, 2]],
[[7, 8],[8, 8]],
[[7, 5],[8, 3]],
[[7,12],[8,12]],
[[7, 7],[8,7 ]],
[[7,15],[8,14]],
[[7, 9],[8,10]],
[[7,18],[8,18]],
[[7,14],[8,15]],
[[7,22],[8,21]],
[[7,19],[8,20]],
[[7,28],[8,28]],
[[7,21],[8,24]],
[[7,33],[8,32]],
[[7,26],[8,30]],
[[7,37],[8,37]],
[[7,32],[8,36]],
[[7,41],[8,41]],
[[8, 0],[9, 0]],
[[8, 4],[9, 4]],
[[8, 2],[9, 2]],
[[8, 8],[9, 8]],
[[8, 3],[9, 3]],
[[8,12],[9,11]],
[[8,7 ],[9, 7]],
[[8,14],[9,14]],
[[8,10],[9,10]],
[[8,18],[9,17]],
[[8,15],[9,16]],
[[8,21],[9,21]],
[[8,20],[9,20]],
[[8,28],[9,27]],
[[8,24],[9,24]],
[[8,32],[9,30]],
[[8,30],[9,32]],
[[8,37],[9,34]],
[[8,41],[9,38]],
[[9, 0],[10, 0]],
[[9, 2],[10, 2]],
[[9, 3],[10, 5]],
[[9, 7],[10, 9]],
[[9,10],[10,13]],
[[9,16],[10,19]],
[[9,20],[10,23]],
[[9,24],[10,28]],
[[9,32],[10,36]],
[[9, 4],[10, 4]],
[[9, 8],[10, 7]],
[[9,11],[10,11]],
[[9,14],[10,14]],
[[9,17],[10,17]],
[[9,21],[10,20]],
[[9,27],[10,26]],
[[9,30],[10,31]],
[[9,34],[10,34]],
[[9,38],[10,39]],
[[10, 0],[11, 0]],
[[10, 2],[11, 2]],
[[10, 5],[11, 7]],
[[10, 9],[11,10]],
[[10,13],[11,16]],
[[10,19],[11,21]],
[[10,23],[11,24]],
[[10,28],[11,30]],
[[10,36],[11,37]],
[[10, 4],[11, 4]],
[[10, 7],[11, 8]],
[[10,11],[11,11]],
[[10,14],[11,14]],
[[10,17],[11,17]],
[[10,20],[11,20]],
[[10,26],[11,25]],
[[10,31],[11,29]],
[[10,34],[11,34]],
[[10,39],[11,38]],
[[11, 0],[12, 0]],
[[11, 2],[12, 2]],
[[11, 7],[12, 8]],
[[11,10],[12,13]],
[[11,16],[12,16]],
[[11,21],[12,21]],
[[11,24],[12,27]],
[[11,30],[12,32]],
[[11,37],[12,38]],
[[11, 4],[12, 4]],
[[11, 8],[12, 7]],
[[11,11],[12,10]],
[[11,14],[12,14]],
[[11,17],[12,17]],
[[11,20],[12,20]],
[[11,25],[12,24]],
[[11,29],[12,29]],
[[11,34],[12,33]],
[[11,38],[12,36]],
[[12, 0],[13, 0]],
[[12, 2],[13, 2]],
[[12, 8],[13, 9]],
[[12,13],[13,14]],
[[12,16],[13,18]],
[[12,21],[13,22]],
[[12,27],[13,27]],
[[12,32],[13,32]],
[[12,38],[13,37]],
[[12, 4],[13, 3]],
[[12, 7],[13, 5]],
[[12,10],[13, 8]],
[[12,14],[13,12]],
[[12,17],[13,15]],
[[12,20],[13,19]],
[[12,24],[13,23]],
[[12,29],[13,26]],
[[12,33],[13,31]],
[[12,36],[13,35]],
[[13, 0],[14, 0]],
[[13, 2],[14, 3]],
[[13, 9],[14, 7]],
[[13,14],[14,10]],
[[13,18],[14,13]],
[[13,22],[14,17]],
[[13,27],[14,20]],
[[13,32],[14,23]],
[[13,37],[14,26]],
[[13, 3],[14, 1]],
[[13, 5],[14, 2]],
[[13, 8],[14, 5]],
[[13,12],[14, 8]],
[[13,15],[14,11]],
[[13,19],[14,14]],
[[13,23],[14,16]],
[[13,26],[14,18]],
[[13,31],[14,22]],
[[13,35],[14,25]],
[[14, 0],[15, 0]],
[[14, 3],[15, 2]],
[[14, 7],[15, 5]],
[[14,10],[15, 6]],
[[14,13],[15, 8]],
[[14,17],[15,11]],
[[14,20],[15,13]],
[[14,23],[15,15]],
[[14,26],[15,16]],
[[14, 2],[15, 1]],
[[14, 5],[15, 3]],
[[14, 8],[15, 4]],
[[14,11],[15, 7]],
[[14,14],[15, 9]],
[[14,16],[15,10]],
[[14,18],[15,12]],
[[14,22],[15,14]],
[[14,25],[15,16]],
[[15, 0],[16, 0]],
[[15, 2],[16, 3]],
[[15, 5],[16, 5]],
[[15, 6],[16, 7]],
[[15, 8],[16, 9]],
[[15,11],[16,11]],
[[15,13],[16,12]],
[[15,15],[16,13]],
[[15, 1],[16, 1]],
[[15, 3],[16, 2]],
[[15, 4],[16, 4]],
[[15, 7],[16, 6]],
[[15, 9],[16, 8]],
[[15,10],[16,10]],
[[15,12],[16,12]],
[[15,14],[16,13]],
[[16, 0],[17,0]],
[[16, 3],[17,2]],
[[16, 5],[17,5]],
[[16, 7],[17,6]],
[[16, 9],[17,7]],
[[16,11],[17,8]],
[[16, 1],[17,1]],
[[16, 2],[17,3]],
[[16, 4],[17,4]],
[[16, 6],[17,6]],
[[16, 8],[17,7]],
[[16,10],[17,8]],
[[17,0],[18,0]],
[[17,2],[18,2]],
[[17,5],[18,3]],
[[17,6],[18,4]],
[[17,1],[18,1]],
[[17,3],[18,2]],
[[17,4],[18,3]]
]

#centroid links
#centoid_vertex_list = []
#for cs in centroids:
#    centroids_in_a_layer = []
#    for v in cs:
#        centroids_in_a_layer.append(bm.verts.new(v))
#    centoid_vertex_list.append(centroids_in_a_layer)


# contour 
all_contour_model = []
section_index = 0
for contours_in_a_section in contour_list:
    contour_model_in_a_section = []

    contour_index = 0
    for contour in contours_in_a_section:
        contour_model = { 'index':[section_index+1,contour_index],'v':[], 'e':[], 'bottom':True, 
            'top':True, 'above':[], 'below':[], 'coords':[], 'branching_processed': False, 'centroid':[], 
            'contour_length':-1, 'fused_top':False, 'fused_bottom': False }
        contour_model['contour_length'] = len( contour )
        for v in contour:
            contour_model['v'].append( bm.verts.new(v) )
            contour_model['coords'].append( v )
        for i in range( len( contour_model['v'] ) ):
            contour_model['e'].append( bm.edges.new( ( contour_model['v'][i], 
                contour_model['v'][(i+1)%len(contour_model['v'])] ) ) )
        contour_model['centroid'] = centroid_list[section_index][contour_index]
        contour_model_in_a_section.append( contour_model )
        contour_index += 1
    all_contour_model.append( contour_model_in_a_section )
    section_index += 1

# contour model connectivity processing
for contour_pair in contour_pair_list:
    contouridx1 = contour_pair[0]
    contouridx2 = contour_pair[1]
    
    contour1 = all_contour_model[contouridx1[0]-1][contouridx1[1]]
    contour2 = all_contour_model[contouridx2[0]-1][contouridx2[1]]
    contour1['above'].append( contour2 )
    contour2['below'].append( contour1 )
    # print( contour1['index'], contour2['index'] )

# check if fused
for contours_in_a_section in all_contour_model:
    for contour in contours_in_a_section:
        if len( contour['below'] ) == 2: 
            contour['fused_top'] = True
            # print( "fused when checked from below 1", contour['index'], [ c['index'] for c in contour['below'] ] )
        if len( contour['below'] ) == 1 and contour['below'][0]['fused_top'] == True: 
            contour['fused_top'] = True
            # print( "fused when checked from below 2", contour['index'], [ c['index'] for c in contour['below'] ] )
        if len( contour['above'] ) == 2:
            contour['fused_bottom'] = True
            # print( "fused when checked from above 1", contour['index'], [ c['index'] for c in contour['above'] ] )
            temp = contour
            while len( temp['below'] ) == 1:
                temp = temp['below'][0]
                temp['fused_bottom'] = True
                # print( "fused when checked from above 2", temp['index'], [ c['index'] for c in contour['above'] ] )
        
        #print( contour1['index'], contour2['index'] )

def get_distance( coord1, coord2 = [0,0,0]):
    x_diff = coord1[0] - coord2[0]
    y_diff = coord1[1] - coord2[1]
    z_diff = coord1[2] - coord2[2]
    sum_diff_squared = x_diff*x_diff + y_diff*y_diff + z_diff*z_diff
    distance = math.sqrt(sum_diff_squared)  
    return distance
    
# close top and bottom
for contour_model_in_a_section in all_contour_model:
    for contour_model in contour_model_in_a_section:
        if len( contour_model['below'] ) == 0:
            if contour_model['fused_bottom'] == False:
                face1 = bm.faces.new( tuple( contour_model['v'] ) )
                print( "bottom not fused", contour_model['index'], [ x['index'] for x in contour_model['above'] ] )
            else:
                # print( "fused bottom", contour_model['index'], [ x['index'] for x in contour_model['above'] ] )
                above_centroid = [0,0,0]
                above_left = [0,0,0]
                above_right = [0,0,0]
                #print( contour_model['index'] )
                left_idx = 0
                right_idx = int(contour_model['contour_length']/2)
                #print( left_idx, right_idx )
    
                if len( contour_model['above'] ) == 2:
                    above1 = contour_model['above'][0]
                    above2 = contour_model['above'][1]
                    if above1['coords'][0][0] > above2['coords'][0][0] :
                        temp = above2
                        above2 = above1
                        above1 = temp
                    above_centroid = list( above1['centroid'] ).copy()
                    for i in range(3):
                        above_centroid[i] = ( above1['centroid'][i] + above2['centroid'][i] ) / 2
                    above_left = above1['coords'][0]
                    above_right = above2['coords'][int(above2['contour_length']/2)]
                else:
                    above_centroid = contour_model['above'][0]['centroid']
                #print( "top cover", contour_model['index'] )
                #print( "current centroid, below centroid", contour_model['centroid'], below_centroid )
                
                curr_left = contour_model['coords'][0]
                curr_centroid = contour_model['centroid']
                curr_right = contour_model['coords'][int(contour_model['contour_length']/2)]
                
                left_dist = get_distance( curr_left, curr_centroid )
                right_dist = get_distance( curr_right, curr_centroid )
                mean_dist = ( left_dist + right_dist ) / 2
                #print( "left, right, mean dist", left_dist, right_dist, mean_dist )
                sin_theta = mean_dist / SPIRALIA_RADIUS
                theta = math.asin( sin_theta )
                z_diff = SPIRALIA_RADIUS - SPIRALIA_RADIUS * math.cos( theta )
                #print( "sin theta, theta, z_diff", sin_theta, theta, z_diff )
                
                delta_centroid = []
                for k in range(3):
                    delta_centroid.append( ( contour_model['centroid'][k] - above_centroid[k] ) * (z_diff / (40.3/SCALE_FACTOR)) )
                #delta_centroid = [ contour_model['centroid'][x] - above_centroid[x] for x in range(3) ]
                #delta_centroid[2] = z_diff
                
                #delta_left = [ contour_model['coords'][0][x] - below_left[x] for x in range(3) ]
                #delta_right = [ contour_model['coords'][int(contour_model['contour_length']/2)][x] - below_right[x] for x in range(3) ]
                
                new_centroid = [ contour_model['centroid'][x] + delta_centroid[x] for x in range(3) ] 
                #new_left = [ contour_model['coords'][0][x] + delta_left[x] for x in range(3) ] 
                #new_right = [ contour_model['coords'][int(contour_model['contour_length']/2)][x] + delta_right[x] for x in range(3) ]
    
                #print( below_centroid, below_left, below_right )
                #print( contour_model['centroid'], contour_model['coords'][0], contour_model['coords'][int(contour_model['contour_length']/2)] )
                #print( delta_centroid, delta_left, delta_right )
                #print( new_centroid, new_left, new_right )
    
                v_centroid = bm.verts.new(new_centroid)
                #v_left = bm.verts.new(new_left) 
                #v_right =  bm.verts.new(new_right) 
                #contour_model['e'].append( bm.edges.new( ( v_left, v_centroid ) ) )
                #contour_model['e'].append( bm.edges.new( ( v_right, v_centroid ) ) )
                for i in range( contour_model['contour_length'] ):
                    face1 = bm.faces.new( ( contour_model['v'][i], contour_model['v'][(i+1)%contour_model['contour_length']], v_centroid ) )

        elif len( contour_model['above'] ) == 0:
            if contour_model['fused_top'] == False:
                face1 = bm.faces.new( tuple( reversed( contour_model['v'] ) ) )
                print( "top not fused", contour_model['index'], [ x['index'] for x in contour_model['below'] ] )
            else:
                #print( "fused", contour_model['index'], [ x['index'] for x in contour_model['below'] ] )
                below_centroid = [0,0,0]
                below_left = [0,0,0]
                below_right = [0,0,0]
                #print( contour_model['index'] )
                left_idx = 0
                right_idx = int(contour_model['contour_length']/2)
                #print( left_idx, right_idx )
    
                if len( contour_model['below'] ) == 2:
                    below1 = contour_model['below'][0]
                    below2 = contour_model['below'][1]
                    if below1['coords'][0][0] > below2['coords'][0][0] :
                        temp = below2
                        below2 = below1
                        below1 = temp
                    below_centroid = list( below1['centroid'] ).copy()
                    for i in range(3):
                        below_centroid[i] = ( below1['centroid'][i] + below2['centroid'][i] ) / 2
                    below_left = below1['coords'][0]
                    below_right = below2['coords'][int(below2['contour_length']/2)]
                else:
                    below_centroid = contour_model['below'][0]['centroid']
                #print( "top cover", contour_model['index'] )
                #print( "current centroid, below centroid", contour_model['centroid'], below_centroid )
                
                curr_left = contour_model['coords'][0]
                curr_centroid = contour_model['centroid']
                curr_right = contour_model['coords'][int(contour_model['contour_length']/2)]
                
                left_dist = get_distance( curr_left, curr_centroid )
                right_dist = get_distance( curr_right, curr_centroid )
                mean_dist = ( left_dist + right_dist ) / 2
                # print( "left, right, mean dist", left_dist, right_dist, mean_dist )
                
                sin_theta = mean_dist / SPIRALIA_RADIUS
                theta = math.asin( sin_theta )
                z_diff = SPIRALIA_RADIUS - SPIRALIA_RADIUS * math.cos( theta )
                #print( "sin theta, theta, z_diff", sin_theta, theta, z_diff )
                
                delta_centroid = []
                for k in range(3):
                    delta_centroid.append( ( contour_model['centroid'][k] - below_centroid[k] ) * (z_diff / (40.3/SCALE_FACTOR)) )
                #delta_centroid = [ contour_model['centroid'][x] - below_centroid[x] for x in range(3) ]
                #delta_centroid[2] = z_diff
                
                #delta_left = [ contour_model['coords'][0][x] - below_left[x] for x in range(3) ]
                #delta_right = [ contour_model['coords'][int(contour_model['contour_length']/2)][x] - below_right[x] for x in range(3) ]
                
                new_centroid = [ contour_model['centroid'][x] + delta_centroid[x] for x in range(3) ] 
                #new_left = [ contour_model['coords'][0][x] + delta_left[x] for x in range(3) ] 
                #new_right = [ contour_model['coords'][int(contour_model['contour_length']/2)][x] + delta_right[x] for x in range(3) ]
    
                #print( below_centroid, below_left, below_right )
                #print( contour_model['centroid'], contour_model['coords'][0], contour_model['coords'][int(contour_model['contour_length']/2)] )
                #print( delta_centroid, delta_left, delta_right )
                #print( new_centroid, new_left, new_right )
    
                v_centroid = bm.verts.new(new_centroid)
                #v_left = bm.verts.new(new_left) 
                #v_right =  bm.verts.new(new_right) 
                #contour_model['e'].append( bm.edges.new( ( v_left, v_centroid ) ) )
                #contour_model['e'].append( bm.edges.new( ( v_right, v_centroid ) ) )
                for i in range( contour_model['contour_length'] ):
                    face1 = bm.faces.new( ( contour_model['v'][i], v_centroid, contour_model['v'][(i+1)%contour_model['contour_length']] ) )
                        

# create faces
for contour_pair in contour_pair_list:
    contouridx1 = contour_pair[0]
    contouridx2 = contour_pair[1]
    
    contour1 = all_contour_model[contouridx1[0]-1][contouridx1[1]]
    contour2 = all_contour_model[contouridx2[0]-1][contouridx2[1]]

    if len( contour1['above'] ) == 1 and len( contour2['below'] ) == 1:
        for i in range( len( contour1['v'] ) ):
            idx1 = i 
            idx2 = ( idx1 + 1 ) % len( contour1['v'] )
                
            #face1 = bm.faces.new( ( contour1['v'][idx1], contour1['v'][idx2], contour2['v'][idx1] ) )
            #face2 = bm.faces.new( ( contour1['v'][idx2], contour2['v'][idx2], contour2['v'][idx1] ) )
            face1 = bm.faces.new( ( contour1['v'][idx2], contour1['v'][idx1], contour2['v'][idx1], contour2['v'][idx2] ) )
    elif len( contour1['above'] ) == 2 and len( contour2['below'] ) == 1:
        if contour1['branching_processed'] == False:
            above1 = contour1['above'][0]
            above2 = contour1['above'][1]
            if above1['coords'][0][0] > above2['coords'][0][0] :
                temp = above2
                above2 = above1
                above1 = temp
            #print( contour1['index'], above1['index'] )
            quarter_length = int( len( contour1['v'] ) / 4 )
            for i in range( quarter_length ):
                idx1 = i 
                idx2 = idx1+1
                idx3 = idx1*2
                idx4 = idx1*2+1
                idx5 = idx1*2+2
                face1 = bm.faces.new( ( contour1['v'][idx2], contour1['v'][idx1], above1['v'][idx3], above1['v'][idx4] ) )
                face1 = bm.faces.new( ( contour1['v'][idx2], above1['v'][idx4], above1['v'][idx5] ) )
            for i in range( quarter_length) :
                idx1 = i +quarter_length
                idx2 = idx1+1
                idx3 = i*2
                idx4 = i*2+1
                idx5 = i*2+2
                face1 = bm.faces.new( ( contour1['v'][idx2], contour1['v'][idx1], above2['v'][idx3], above2['v'][idx4] ) )
                face1 = bm.faces.new( ( contour1['v'][idx2], above2['v'][idx4], above2['v'][idx5] ) )
            for i in range( quarter_length) :
                idx1 = i +quarter_length*2
                idx2 = idx1+1
                idx3 = ( quarter_length*2 +(i*2) ) % len( contour1['v'] )
                idx4 = ( quarter_length*2 +(i*2)+1 )% len( contour1['v'] )
                idx5 = ( quarter_length*2 +(i*2)+2 ) % len( contour1['v'] ) 
                face1 = bm.faces.new( ( contour1['v'][idx2], contour1['v'][idx1], above2['v'][idx3], above2['v'][idx4] ) )
                face1 = bm.faces.new( ( contour1['v'][idx2], above2['v'][idx4], above2['v'][idx5] ) )
            for i in range( quarter_length) :
                idx1 = i +quarter_length*3
                idx2 = ( idx1+1 ) % len( contour1['v'] )
                idx3 = ( quarter_length*2 +(i*2) ) % len( contour1['v'] )
                idx4 = ( quarter_length*2 +(i*2)+1 )% len( contour1['v'] )
                idx5 = ( quarter_length*2 +(i*2)+2 ) % len( contour1['v'] ) 
                face1 = bm.faces.new( ( contour1['v'][idx2], contour1['v'][idx1], above1['v'][idx3], above1['v'][idx4] ) )
                face1 = bm.faces.new( ( contour1['v'][idx2], above1['v'][idx4], above1['v'][idx5] ) )
            idx1 = quarter_length
            idx2 = quarter_length * 3
            idx3 = quarter_length * 2
            idx4 = 0
            #print( idx1, idx2, idx3, idx4 )
            face1 = bm.faces.new( ( contour1['v'][idx2], contour1['v'][idx1], above1['v'][idx3] ) )
            face1 = bm.faces.new( ( contour1['v'][idx1], contour1['v'][idx2], above2['v'][0] ) )
        contour1['branching_processed'] = True
    elif len( contour1['above'] ) == 1 and len( contour2['below'] ) == 2:
        if contour2['branching_processed'] == False:
            below1 = contour2['below'][0]
            below2 = contour2['below'][1]
            if below1['coords'][0][0] > below2['coords'][0][0] :
                temp = below2
                below2 = below1
                below1 = temp
            #print( contour2['index'], below1['index'], below2['index'] )
            quarter_length = int( len( contour1['v'] ) / 4 )
            for i in range( quarter_length ):
                idx1 = i 
                idx2 = idx1+1
                idx3 = idx1*2
                idx4 = idx1*2+1
                idx5 = idx1*2+2
                #print( "1", idx1, idx2, idx3, idx4 )
                face1 = bm.faces.new( ( contour2['v'][idx1], contour2['v'][idx2], below1['v'][idx4], below1['v'][idx3] ) )
                face1 = bm.faces.new( ( contour2['v'][idx2], below1['v'][idx5], below1['v'][idx4] ) )
            for i in range( quarter_length) :
                idx1 = i +quarter_length
                idx2 = idx1+1
                idx3 = i*2
                idx4 = i*2+1
                idx5 = i*2+2
                #print( "2", idx1, idx2, idx3, idx4, idx5 )
                face1 = bm.faces.new( ( contour2['v'][idx1], contour2['v'][idx2], below2['v'][idx4], below2['v'][idx3] ) )
                face1 = bm.faces.new( ( contour2['v'][idx2], below2['v'][idx5], below2['v'][idx4] ) )
            for i in range( quarter_length) :
                idx1 = i +quarter_length*2
                idx2 = idx1+1
                idx3 = ( quarter_length*2 +(i*2) ) % len( contour1['v'] )
                idx4 = ( quarter_length*2 +(i*2)+1 )% len( contour1['v'] )
                idx5 = ( quarter_length*2 +(i*2)+2 ) % len( contour1['v'] ) 
                #print( "3", idx1, idx2, idx3, idx4, idx5 )
                face1 = bm.faces.new( ( contour2['v'][idx1], contour2['v'][idx2], below2['v'][idx4], below2['v'][idx3] ) )
                face1 = bm.faces.new( ( contour2['v'][idx2], below2['v'][idx5], below2['v'][idx4] ) )
            for i in range( quarter_length) :
                idx1 = i +quarter_length*3
                idx2 = ( idx1+1 ) % len( contour1['v'] )
                idx3 = ( quarter_length*2 +(i*2) ) % len( contour1['v'] )
                idx4 = ( quarter_length*2 +(i*2)+1 )% len( contour1['v'] )
                idx5 = ( quarter_length*2 +(i*2)+2 ) % len( contour1['v'] ) 
                #print( "4", idx1, idx2, idx3, idx4, idx5 )
                face1 = bm.faces.new( ( contour2['v'][idx1], contour2['v'][idx2], below1['v'][idx4], below1['v'][idx3] ) )
                face1 = bm.faces.new( ( contour2['v'][idx2], below1['v'][idx5], below1['v'][idx4] ) )
            idx1 = quarter_length
            idx2 = quarter_length * 3
            idx3 = quarter_length * 2
            idx4 = 0
            #print( idx1, idx2, idx3, idx4 )
            face1 = bm.faces.new( ( contour2['v'][idx1], contour2['v'][idx2], below1['v'][idx3] ) )
            face1 = bm.faces.new( ( contour2['v'][idx2], contour2['v'][idx1], below2['v'][0] ) )
        contour2['branching_processed'] = True

CONTOUR_LENGTH = 106 / SCALE_FACTOR
CONTOUR_WIDTH = 16 / SCALE_FACTOR
CONTOUR_POINT_COUNT = 16

toppair_list = [ [ [18,0],[18,1] ], [ [12,42], [10,44] ] ]
# create imaginary spiral based on cross-section
def cross(a, b):
    c = [a[1]*b[2] - a[2]*b[1],
         a[2]*b[0] - a[0]*b[2],
         a[0]*b[1] - a[1]*b[0]]

    return c
    
for toppair in toppair_list:
    contour1 = all_contour_model[toppair[0][0]-1][toppair[0][1]]
    contour2 = all_contour_model[toppair[1][0]-1][toppair[1][1]]
    #print("toppair", contour1['index'], contour2['index'])
    num_point = contour1['contour_length']
    
    below_centroid = [0,0,0]
    below1 = contour1['below'][0]
    below2 = contour2['below'][0]
    for i in range(3):
        below_centroid[i] = ( below1['centroid'][i] + below2['centroid'][i] ) / 2
    #print("below:", below1['index'], below2['index'])
    
    curr_centroid = [0,0,0]
    for i in range(3):
        curr_centroid[i] = ( contour1['centroid'][i] + contour2['centroid'][i] ) / 2
        
    left_to_right_vector = [0,0,0]
    for i in range(3):
        left_to_right_vector[i] = below2['centroid'][i] - below1['centroid'][i]
    
    left_dist = get_distance( contour1['centroid'], curr_centroid )
    right_dist = get_distance( contour2['centroid'], curr_centroid )
    mean_dist = ( left_dist + right_dist ) / 2
    #print( "left, right, mean dist", left_dist, right_dist, mean_dist )
    
    sin_theta = mean_dist / SPIRALIA_RADIUS
    theta = math.asin( sin_theta )
    z_diff = SPIRALIA_RADIUS - SPIRALIA_RADIUS * math.cos( theta )
    #print( "sin theta, theta, z_diff", sin_theta, theta, z_diff )
    
    delta_centroid = []
    for k in range(3):
        delta_centroid.append( curr_centroid[k] - below_centroid[k] )
    
    delta_length = get_distance( delta_centroid )
    #print( "delta", delta_centroid, delta_length )
    unit_delta = [0,0,0]
    for k in range(3):
        unit_delta[k] = delta_centroid[k] / delta_length
    
    new_centroid = [ curr_centroid[x] + unit_delta[x] * z_diff for x in range(3) ]
    outer = [  curr_centroid[x] + unit_delta[x] * delta_length + unit_delta[x] * ( CONTOUR_LENGTH / 4 ) for x in range(3) ] 
    inner = [  curr_centroid[x] + unit_delta[x] * delta_length - unit_delta[x] * ( CONTOUR_LENGTH / 4 ) for x in range(3) ] 
    
    v_centroid = bm.verts.new(new_centroid)
    #v_outer = bm.verts.new(outer)
    #v_inner = bm.verts.new(inner)
    #bm.edges.new( ( v_outer, v_centroid ) )
    #bm.edges.new( ( v_inner, v_centroid ) )
    
    perpendicular_vector = cross( left_to_right_vector, unit_delta )
    perpendicular_length = get_distance( perpendicular_vector )
    unit_perpendicular = [0,0,0]
    for k in range(3):
        unit_perpendicular[k] = perpendicular_vector[k] / perpendicular_length
    
    v_list = []
    for i in range( CONTOUR_POINT_COUNT ):
        rotation_in_radian = ( math.pi * 2 / CONTOUR_POINT_COUNT )
        radius_displacement= math.cos( rotation_in_radian * i ) * (CONTOUR_LENGTH / 4)
        perpendicular_displacement = math.sin( rotation_in_radian * i ) * (CONTOUR_WIDTH / 4)
        new_vert = [0,0,0]
        for j in range(3):
            new_vert[j] = new_centroid[j] + unit_delta[j] * radius_displacement + unit_perpendicular[j] * perpendicular_displacement 

        v_list.append( bm.verts.new(new_vert) )
    for i in range( len( v_list ) ):
        bm.edges.new( ( v_list[i], v_list[(i+1)%CONTOUR_POINT_COUNT] ) )
        
    for i in range( contour1['contour_length'] ):
        face1 = bm.faces.new( ( contour1['v'][(i+1)%CONTOUR_POINT_COUNT], contour1['v'][i], v_list[i], v_list[(i+1)%CONTOUR_POINT_COUNT] ) )
    for i in range( contour2['contour_length'] ):
        face1 = bm.faces.new( ( contour2['v'][(i+1)%CONTOUR_POINT_COUNT], contour2['v'][i], v_list[(int(CONTOUR_POINT_COUNT/2)+CONTOUR_POINT_COUNT-i)%CONTOUR_POINT_COUNT], v_list[(int(CONTOUR_POINT_COUNT/2)+CONTOUR_POINT_COUNT-i-1)%CONTOUR_POINT_COUNT] ) )

bottompair_list = [ [[2,4],[2,7]], [[2,6],[3,20]],[[3,16],[3,22]],[[3,17],[4,30]],[[4,26],[6,42]],[[3,19],[3,21]],[[4,21],[4,28]] ]

for bottompair in bottompair_list:
    contour1 = all_contour_model[bottompair[0][0]-1][bottompair[0][1]]
    contour2 = all_contour_model[bottompair[1][0]-1][bottompair[1][1]]
    print("bottompair", contour1['index'], contour2['index'])
    num_point = contour1['contour_length']
    
    above_centroid = [0,0,0]
    above1 = contour1['above'][0]
    above2 = contour2['above'][0]
    for i in range(3):
        above_centroid[i] = ( above1['centroid'][i] + above2['centroid'][i] ) / 2
    #print("above:", above1['index'], above2['index'])
    
    curr_centroid = [0,0,0]
    for i in range(3):
        curr_centroid[i] = ( contour1['centroid'][i] + contour2['centroid'][i] ) / 2
        
    left_to_right_vector = [0,0,0]
    for i in range(3):
        left_to_right_vector[i] = above2['centroid'][i] - above1['centroid'][i]
    
    left_dist = get_distance( contour1['centroid'], curr_centroid )
    right_dist = get_distance( contour2['centroid'], curr_centroid )
    mean_dist = ( left_dist + right_dist ) / 2
    print( "radius, left, right, mean dist", SPIRALIA_RADIUS, left_dist, right_dist, mean_dist )
    
    sin_theta = mean_dist / SPIRALIA_RADIUS
    theta = math.asin( sin_theta )
    z_diff = SPIRALIA_RADIUS - SPIRALIA_RADIUS * math.cos( theta )
    print( "RADIUS, RADIUS * cos(theta)", SPIRALIA_RADIUS,SPIRALIA_RADIUS * math.cos( theta )) 
    print( "sin theta, theta, theta in degree, cos(theta), z_diff", sin_theta, theta, math.degrees( theta ), math.cos(theta), z_diff )
    
    delta_centroid = []
    for k in range(3):
        delta_centroid.append( curr_centroid[k] - above_centroid[k] )
    
    delta_length = get_distance( delta_centroid )
    print( "delta", delta_centroid, delta_length )
    unit_delta = [0,0,0]
    for k in range(3):
        unit_delta[k] = delta_centroid[k] / delta_length
    
    new_centroid = [ curr_centroid[x] + unit_delta[x] * z_diff for x in range(3) ]
    outer = [  curr_centroid[x] + unit_delta[x] * delta_length + unit_delta[x] * ( CONTOUR_LENGTH / 4 ) for x in range(3) ] 
    inner = [  curr_centroid[x] + unit_delta[x] * delta_length - unit_delta[x] * ( CONTOUR_LENGTH / 4 ) for x in range(3) ] 
    
    v_centroid = bm.verts.new(new_centroid)
    #v_outer = bm.verts.new(outer)
    #v_inner = bm.verts.new(inner)
    #bm.edges.new( ( v_outer, v_centroid ) )
    #bm.edges.new( ( v_inner, v_centroid ) )
    
    perpendicular_vector = cross( left_to_right_vector, unit_delta )
    perpendicular_length = get_distance( perpendicular_vector )
    unit_perpendicular = [0,0,0]
    for k in range(3):
        unit_perpendicular[k] = perpendicular_vector[k] / perpendicular_length
    
    v_list = []
    for i in range( CONTOUR_POINT_COUNT ):
        rotation_in_radian = ( math.pi * 2 / CONTOUR_POINT_COUNT )
        radius_displacement= math.cos( rotation_in_radian * i ) * (CONTOUR_LENGTH / 4)
        perpendicular_displacement = math.sin( rotation_in_radian * i ) * (CONTOUR_WIDTH / 4)
        new_vert = [0,0,0]
        for j in range(3):
            new_vert[j] = new_centroid[j] + unit_delta[j] * radius_displacement + unit_perpendicular[j] * perpendicular_displacement 

        v_list.append( bm.verts.new(new_vert) )
    for i in range( len( v_list ) ):
        bm.edges.new( ( v_list[i], v_list[(i+1)%CONTOUR_POINT_COUNT] ) )
        
    for i in range( contour1['contour_length'] ):
        face1 = bm.faces.new( ( contour1['v'][i], contour1['v'][(i+1)%CONTOUR_POINT_COUNT], v_list[(CONTOUR_POINT_COUNT-i-1)%CONTOUR_POINT_COUNT], v_list[(+CONTOUR_POINT_COUNT-i)%CONTOUR_POINT_COUNT] ) )
    for i in range( contour2['contour_length'] ):
        face1 = bm.faces.new( ( contour2['v'][i], contour2['v'][(i+1)%CONTOUR_POINT_COUNT], v_list[(int(CONTOUR_POINT_COUNT/2)+i+1)%CONTOUR_POINT_COUNT], v_list[(int(CONTOUR_POINT_COUNT/2)+i)%CONTOUR_POINT_COUNT] ) )


# make the bmesh the object's mesh
bm.to_mesh(mesh)  

bm.free()  # always do this when finished
'''

file = open("Spiriferella_spiralia_simplified_contour_.py", "w")
file.write(ret_str)
file.close()

#print( contour_pair_list.contour_pair_list )
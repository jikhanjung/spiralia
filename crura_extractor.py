import numpy as np
import cv2
#import contour_pair_list

ret_str = ""

coord_correction = []
for i in range(18):
    coord_correction.append( [-1028,-774])

coord_correction[4][1] += -2
coord_correction[5][1] += 0
coord_correction[6][1] += 0
coord_correction[7][1] += 5
coord_correction[8][1] += 10
coord_correction[9][1] += 0

num_point = 8
all_contours = []
all_contours = []
all_centroids = []
SCALE_FACTOR = 100

for i in range(7):
    num = '0' + str(i+1)
    filename = 'images/Spiriferella-crura-' + num[-2:]
    print(filename)
    im = cv2.imread( filename + '.png')

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
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
        num_point = 16
        idx_list = []
        for pt in c:
            if pt[0][0] > max_x[0]:
                max_x = pt[0]
                max_idx = idx
            if pt[0][0] < min_x[0]:
                min_x = pt[0]
                min_idx = idx
            idx += 1
        idx_diff1 = int( ( max_idx - min_idx ) / ( num_point / 2))
        idx_diff2 = int( ( min_idx + ( len(c) -max_idx ) ) / (num_point /2))
        for k in range(int(num_point /2)):
            idx_list.append( int( min_idx + idx_diff1 * k )  )
        for k in range(int(num_point /2)):
            idx_list.append(int(max_idx + idx_diff2 * k )%len(c))
        print( max_x, min_x, max_idx, min_idx, idx_list, len(c))
        simp_cont = []
        for k in range(num_point):
            simp_cont.append( [ c[idx_list[k]][0][0]+x_correction, c[idx_list[k]][0][1]+y_correction, (i-9)*40.3 ])
        print( simp_cont )

        contours_in_a_section.append( simp_cont )
        cv2.drawContours(im, [c], -1, (0, 255, 0), 3)

        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(im,str(j) ,(cx,cy),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)

        centroids_in_a_section.append( [ cx+x_correction, -1 * (cy+y_correction ), (i-9)*40.3 ] )
        verts = []
        for v in simp_cont:
            verts.append( "(" + ",".join( [ str( coord ) for coord in v ] ) + ")" )
        #contour_list.append(  "[" + ",\n".join( verts ) + "]" )
        cv2.imwrite('images/Spiriferella-crura-labeled-' + num[-2:] + '.png', im)

        j += 1

    all_centroids.append( centroids_in_a_section )
    all_contours.append( contours_in_a_section )

all_contours_str_list = []
for contours_in_a_section in all_contours:
    contour_str_list = []
    for contour in contours_in_a_section:
        v_str_list = []
        for v in contour:
            #print("v:",v)
            v_str = "(" + ",".join( [str(x/SCALE_FACTOR) for x in v]) + ")"
            v_str_list.append( v_str )
        contour_str = "[" + ",".join( v_str_list ) + "]"
        #print( "contour_str:", contour_str )
        contour_str_list.append( contour_str )
    contours_in_a_section_str = "[" + ",\n".join( contour_str_list ) + "]"
    all_contours_str_list.append( contours_in_a_section_str )
ret_str += "contour_list=["
ret_str += ",\n".join( all_contours_str_list )
ret_str += "]\n"

all_centroids_str_list = []

for centroids_in_a_section in all_centroids:
    centroids_str_list = []
    centroids_in_a_section_str = ""
    for centroid in centroids_in_a_section:
        centroid_str = "(" + ",".join( [ str(x/SCALE_FACTOR) for x in centroid ] )+")"
        print( centroid_str )
        centroids_str_list.append( centroid_str )
    centroids_in_a_section_str = "[" + ",\n".join( centroids_str_list ) + "]"
    all_centroids_str_list.append( centroids_in_a_section_str  )
ret_str += "centroid_list=[" + ",".join(all_centroids_str_list) + "]\n"


ret_str += """
import bpy
import bmesh
import time 

mesh = bpy.data.meshes.new("mesh")  # add a new mesh
obj = bpy.data.objects.new("MyObject", mesh)  # add a new object using the mesh

scene = bpy.context.scene
scene.objects.link(obj)  # put the object into the scene (link)
scene.objects.active = obj  # set as the active object in the scene
obj.select = True  # select object

mesh = bpy.context.object.data
bm =  bmesh.new()

# contour 
all_contour_model = []
section_index = 0
for contours_in_a_section in contour_list:
    contour_model_in_a_section = []

    contour_index = 0
    for contour in contours_in_a_section:
        contour_model = { 'index':[section_index+1,contour_index],'v':[], 'e':[], 'bottom':True, 'top':True, 'above':[], 'below':[], 'coords':[], 'branching_processed': False, 'centroid':[], 'contour_length':-1 }
        contour_model['contour_length'] = len( contour )
        for v in contour:
            contour_model['v'].append( bm.verts.new(v) )
            contour_model['coords'].append( v )
        for i in range( len( contour_model['v'] ) ):
            contour_model['e'].append( bm.edges.new( ( contour_model['v'][i],contour_model['v'][(i+1)%len(contour_model['v'])] ) ) )
        contour_model['centroid'] = centroid_list[section_index][contour_index]
        contour_model_in_a_section.append( contour_model )
        contour_index += 1
    all_contour_model.append( contour_model_in_a_section )
    section_index += 1


# contour model connectivity processing
for contour_model_in_a_section in all_contour_model:
    contour1 = contour_model_in_a_section[0].copy()
    contour2 = contour_model_in_a_section[1].copy()

    if contour1['coords'][0] > contour2['coords'][0]:
        contour_model_in_a_section[0] = contour2
        contour_model_in_a_section[1] = contour1

# close top and bottom
face1 = bm.faces.new( tuple( all_contour_model[0][0]['v'] ) )
face1 = bm.faces.new( tuple( all_contour_model[0][1]['v'] ) )
face1 = bm.faces.new( tuple( reversed( all_contour_model[-1][0]['v'] ) ) )
face1 = bm.faces.new( tuple( reversed( all_contour_model[-1][1]['v'] ) ) )

# create faces
for i in range( len( all_contour_model ) - 1 ): 
  
    contours_in_below_section = all_contour_model[i]
    contours_in_above_section = all_contour_model[i+1]

    for j in range(2):
        contour1 = contours_in_below_section[j]
        contour2 = contours_in_above_section[j]
        print( contour1['coords'] )
        print( contour2['coords'] )
        for k in range( len( contour1['v'] ) ):
            idx1 = k 
            idx2 = ( idx1 + 1 ) % len( contour1['v'] )
            print( i, j, idx1, idx2 )
            face1 = bm.faces.new( ( contour1['v'][idx2], contour1['v'][idx1], contour2['v'][idx1], contour2['v'][idx2] ) )
        # make the bmesh the object's mesh
        bm.to_mesh(mesh)  
        #time.sleep(2)


bm.free()  # always do this when finished
"""

file = open("Spiriferella_crura_contour_.py", "w")
file.write(ret_str)
file.close()

#print( all_contours )


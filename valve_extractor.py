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
num_point = 100

end_point_list = [[[[ 40, 949],[1959, 977]],[[109,1056],[1956, 974]]],
[[[ 28, 948],[1942,1015]],[[ 73,1063],[1951,1019]]],
[[[ 19,1005],[1936,1023]],[[ 67,1082],[1937,1021]]],
[[[ 19, 995],[1923,1048]],[[ 83,1078],[1924,1046]]],
[[[  8,1003],[1928, 997]],[[ 77,1069],[1927, 995]]],
[[[110,1095],[1935, 930]],[[129,1103],[1904, 945]]],
[[[ 67, 991],[1926, 822]],[[ 67, 988],[1934, 828]]],
[[[ 43,1005],[1924, 904]],[[ 45,1001],[1925, 902]]],
[[[ 32, 957],[1927, 890]],[[ 32, 952],[1930, 889]]],
[[[ 36, 966],[1863, 938]],[[ 41, 965],[1863, 936]]],
[[[ 71,1016],[1923, 918]],[[ 59, 988],[1924, 916]]],
[[[ 65, 929],[1832, 964]],[[ 99, 916],[1830, 960]]],
[[[ 64, 870],[1846, 918]],[[116, 945],[1850, 915]]],
[[[105, 861],[1854, 877]],[[102, 856],[1855, 875]]],
[[[121, 836],[1843, 780]],[[122, 827],[1847, 775]]],
[[[140, 826],[1779, 774]],[[140, 817],[1784, 771]]],
[[[199, 842],[1772, 760]],[[194, 839],[1776, 753]]],
[[[187, 817],[1754, 763]],[[187, 812],[1756, 760]]]]

all_contours = []
for i in range(18):
    num = '0' + str(i+1)
    filename = 'images/Spiriferella-valves-' + num[-2:]
    print(filename)
    im = cv2.imread( filename + '.png')

    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    j = 0
    contours_in_a_section = []
    x_correction = coord_correction[i][0]
    y_correction = coord_correction[i][1]

    for c in contours:
        print("contour",j,"len:",len(c))
        new_contour = []
        interval = len(c) / num_point
        k = 0
        idx1 = -1
        idx2 = -1
        for pt in c:
            if pt[0][0] == end_point_list[i][j][0][0] and pt[0][1] == end_point_list[i][j][0][1]:
                idx1 = k
                #print( "idx1:", pt[0])
            elif pt[0][0] == end_point_list[i][j][1][0] and pt[0][1] == end_point_list[i][j][1][1]:
                #print( "idx2:", pt[0])
                idx2 = k
            k+=1
        #print(i,j,idx1,idx2)
        if c[idx1][0][0] > c[idx2][0][0]:
            left_idx = idx2
            right_idx = idx1
        else:
            left_idx = idx1
            right_idx = idx2
        #print( c[min_idx], c[max_idx])
        idx_list = []
        if left_idx < right_idx:
            idx_diff1 = int( ( right_idx - left_idx ) / ( num_point / 2))
            idx_diff2 = int( ( left_idx + ( len(c) - right_idx ) ) / (num_point /2))
        else:
            idx_diff1 = int((left_idx - right_idx) / (num_point / 2))
            idx_diff2 = int((right_idx + (len(c) - left_idx)) / (num_point / 2))

        for k in range(int(num_point /2)):
            idx_list.append( int( left_idx + idx_diff1 * k ) %len(c) )
        for k in range(int(num_point /2)):
            idx_list.append(int(right_idx + idx_diff2 * k )%len(c))

        #print( left_idx, idx_diff1, right_idx, idx_diff2,  idx_list )

        cv2.drawContours(im, [c], -1, (0, 255, 0), 3)
        for idx in idx_list:
            cx = c[idx][0][0]
            cy = c[idx][0][1]
            x = ( cx + x_correction )
            y = ( cy + y_correction )
            z = ( (i-9)*40.3 )
            new_contour.append( [ x, y, z ] )
            cv2.putText(im, str(idx), (cx, cy), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)
        contours_in_a_section.append( new_contour )
        M = cv2.moments(c)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        cv2.putText(im,str(j),(cx,cy),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,255)
        j+=1

    all_contours.append( contours_in_a_section )
    cv2.imwrite('images/Spiriferella-valves-labeled-' + num[-2:] +'.png', im)
for cis in all_contours:
    print( "no of contours:", len(cis), "length each", [ len(x) for x in cis ] )

all_contours_str_list = []
for contours_in_a_section in all_contours:
    contour_str_list = []
    for contour in contours_in_a_section:
        v_str_list = []
        for v in contour:
            #print("v:",v)
            v_str = "(" + ",".join( [str(x/1000.0) for x in v]) + ")"
            v_str_list.append( v_str )
        contour_str = "[" + ",".join( v_str_list ) + "]"
        #print( "contour_str:", contour_str )
        contour_str_list.append( contour_str )
    contours_in_a_section_str = "[" + ",\n".join( contour_str_list ) + "]"
    all_contours_str_list.append( contours_in_a_section_str )
ret_str += "contour_list=["
ret_str += ",\n".join( all_contours_str_list )
ret_str += "]"

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
        contour_model = { 'index':[section_index+1,contour_index],'coords':[],'v':[], 'e':[] }
        #print( "contour model", section_index, contour_index, contour_model )
        for v in contour:
            contour_model['v'].append( bm.verts.new(v) )
            contour_model['coords'].append( v )
        #for i in range( len( contour_model['v'] ) ):
        #    contour_model['e'].append( bm.edges.new( ( contour_model['v'][i],contour_model['v'][(i+1)%len(contour_model['v'])] ) ) )
        contour_model_in_a_section.append( contour_model )
        contour_index += 1
    all_contour_model.append( contour_model_in_a_section )
    section_index += 1





# close top and bottom
for contour_model_in_a_section in all_contour_model:
    face1 = bm.faces.new( tuple( contour_model_in_a_section[0]['v'] ) )
    face1 = bm.faces.new( tuple( reversed( contour_model_in_a_section[1]['v'] ) ) )

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
            face1 = bm.faces.new( ( contour1['v'][idx1], contour1['v'][idx2], contour2['v'][idx2], contour2['v'][idx1] ) )
        # make the bmesh the object's mesh
        bm.to_mesh(mesh)  
        #time.sleep(2)


bm.free()  # always do this when finished
"""

file = open("Spiriferella_valve_contour_.py", "w")
file.write(ret_str)
file.close()

#print( all_contours )


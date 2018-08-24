import math
import bpy
import bmesh

NUMBER_OF_WHORL = 10
RADIUS = 300
TRANSLATION = [0, 0, 20]
CONTOUR_LENGTH = 106
CONTOUR_WIDTH = 16
NUMBER_OF_SECTION = 48
CENTER = [350, 0]
displacement_list = [ [ 7, 0 ], ]
CONTOUR_POINT_COUNT = 16
SCALE_FACTOR = 1000
CLOCKWISE = 1
COUNTER_CLOCKWISE = -1

def get_spiral_contours( center, number_of_whorl, number_of_section, contour_width, contour_length, contour_point_count, scale_factor, radius, translation, start_angle, whorl_direction):
    contour_list = []
    centroid_list = []
    for i in range( number_of_whorl):
        for j in range( number_of_section ):
            angle_in_degree = ( 360 / number_of_section) * j * whorl_direction + start_angle
            print( "degree", j, whorl_direction, start_angle, angle_in_degree )
            angle_in_radian = math.radians( angle_in_degree )
            x = math.cos( angle_in_radian ) * radius + translation[0] * ( j/number_of_section) + translation[0] * i
            y = math.sin( angle_in_radian ) * radius + translation[1] * ( j/number_of_section) + translation[1] * i
            z = translation[2] * (j / number_of_section) + translation[2] * i
            centroid_list.append([(center[0] + x) / scale_factor, (center[1] + y) / scale_factor, z / scale_factor])
            contour = []
            for k in range( contour_length ):
                rotation_in_radian = ( math.pi * 2 / contour_point_count )
                radius_displacement= math.cos( rotation_in_radian * k ) * (contour_length / 2)
                z_displacement = math.sin( rotation_in_radian * k ) * (contour_width / 2)
                x = math.cos(angle_in_radian) * ( radius + radius_displacement ) + translation[0] * ( j/number_of_section) + translation[0] * i
                y = math.sin(angle_in_radian) * ( radius + radius_displacement ) + translation[1] * ( j/number_of_section) + translation[1] * i
                z = translation[2] * (j / number_of_section)  + translation[2] * i + z_displacement
                contour.append([(center[0] + x) / scale_factor, (center[1] + y) / scale_factor, z / scale_factor])
            if whorl_direction == 1:
                contour_list.append( contour )
            else:
                contour_list.append( reversed(contour ) )
            #print( x, y, z, contour )
            #print( x, y, z )
    return contour_list

all_list=[]
all_list.append( get_spiral_contours( [350,0], NUMBER_OF_WHORL, NUMBER_OF_SECTION, CONTOUR_WIDTH, CONTOUR_LENGTH, CONTOUR_POINT_COUNT, SCALE_FACTOR, RADIUS, [15, 0, 20 ], -135, CLOCKWISE ) )
all_list.append( get_spiral_contours( [-350,0], NUMBER_OF_WHORL, NUMBER_OF_SECTION, CONTOUR_WIDTH, CONTOUR_LENGTH, CONTOUR_POINT_COUNT, SCALE_FACTOR, RADIUS, [-15, 0, 20 ], -45, COUNTER_CLOCKWISE ) )

mesh = bpy.data.meshes.new("mesh")  # add a new mesh
obj = bpy.data.objects.new("MyObject", mesh)  # add a new object using the mesh

scene = bpy.context.scene
scene.objects.link(obj)  # put the object into the scene (link)
scene.objects.active = obj  # set as the active object in the scene
obj.select = True  # select object

mesh = bpy.context.object.data
bm = bmesh.new()

vert_list = []
#for centroid in centroid_list:
#    vert_list.append( bm.verts.new( centroid ) )

#for idx in range( len( vert_list )-1):
#    bm.edges.new((vert_list[idx], vert_list[idx+1]))
for contour_list in all_list:
    all_vert = []
    for contour in contour_list:
        vert_in_a_contour = []
        for v in contour:
            vert_in_a_contour.append(bm.verts.new(v))
        #for i in range( len( contour ) ):
        #    bm.edges.new((vert_in_a_contour[i], vert_in_a_contour[i + 1]))
        all_vert.append( vert_in_a_contour )

    for i in range( len( all_vert ) - 1):
        for j in range( len( all_vert[i]) ):
            idx1 = j
            idx2 = ( j + 1 ) % CONTOUR_POINT_COUNT
            face1 = bm.faces.new((all_vert[i][idx1], all_vert[i][idx2], all_vert[i+1][idx2], all_vert[i+1][idx1] ) )


bm.to_mesh(mesh)
obj.location = ( 0, -0.2, -0.1)
obj.rotation_euler = ( math.radians( -60), 0, 0 )
bm.free()  # always do this when finished

import math
import bpy
import bmesh

WHORL_NUMBER = 8
RADIUS = 300
TRANSLATION = [0,0,20]
LENGTH = 106
WIDTH = 16
SECTION = 48
CENTER = [350,0]
centroid_list = []
displacement_list = [ [ 7, 0 ], ]
contour_length = 16
SCALE_FACTOR = 1000

contour_list = []
for i in range(WHORL_NUMBER):
    for j in range( SECTION  ):
        angle_in_radian = math.radians( ( 360 / SECTION ) * j )
        x = math.cos( angle_in_radian ) * RADIUS + TRANSLATION[0] * ( angle_in_radian / ( 2 * math.pi ) ) + TRANSLATION[0] * i
        y = math.sin( angle_in_radian ) * RADIUS + TRANSLATION[1] * ( angle_in_radian / ( 2 * math.pi ) ) + TRANSLATION[1] * i
        z = TRANSLATION[2] * ( angle_in_radian / ( 2 * math.pi ) ) + TRANSLATION[2] * i
        centroid_list.append( [ x/SCALE_FACTOR, y/SCALE_FACTOR, z/SCALE_FACTOR ] )
        contour = []
        for k in range( contour_length ):
            rotation_in_radian = ( math.pi * 2 / contour_length )
            radius_displacement= math.cos( rotation_in_radian * k ) * ( LENGTH / 2 )
            z_displacement = math.sin( rotation_in_radian * k ) * ( WIDTH / 2 )
            x = math.cos(angle_in_radian) * ( RADIUS + radius_displacement ) + TRANSLATION[0] * (angle_in_radian / (2 * math.pi)) + TRANSLATION[0] * i
            y = math.sin(angle_in_radian) * ( RADIUS + radius_displacement ) + TRANSLATION[1] * (angle_in_radian / (2 * math.pi)) + TRANSLATION[1] * i
            z = TRANSLATION[2] * ( angle_in_radian / ( 2 * math.pi ) ) + TRANSLATION[2] * i + z_displacement
            contour.append( [ x/SCALE_FACTOR, y/SCALE_FACTOR, z/SCALE_FACTOR] )
        contour_list.append( contour )
        #print( x, y, z, contour )
        #print( x, y, z )


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
        idx2 = ( j + 1 ) % contour_length
        face1 = bm.faces.new((all_vert[i][idx1], all_vert[i][idx2], all_vert[i+1][idx2], all_vert[i+1][idx1] ) )


bm.to_mesh(mesh)

bm.free()  # always do this when finished

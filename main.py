import bpy
import math
import bmesh

# okay, so to refresh, because I lost everything I was doing on the blender
#thing :(
# okay: the pick is 0.1
#the teeth I had at 0.15
z_ = [0,0,1] #unit vector in z direction

pick_height = 0.04
teeth_width = 0.05
teeth_gap = 0.01
teeth_length = 3
teeth_thickness = 0.03
no_teeth = 18
comb_height = teeth_width*no_teeth + teeth_gap*(no_teeth-1)
pick_radius_modifier = 1.8 #to vary the width of the pick relative to the tooth_width
pick_radius = teeth_width/(2*pick_radius_modifier)

gear_width = 0.4
gear_gap = 0.2
top_gap = 0.1

cylinder_height_modifier = gear_width + gear_gap + top_gap
cylinder_height = comb_height + cylinder_height_modifier

pick_gap = 0.08




notes = [["1a&1b&1c&2f", 1], ["1c",1], ["2c", 0.5], ["1g&2c", 0.5],["1a&2c&0c", 0.5], ["2c",0.5], ["1c", 0.5], ["1g&0e", 2],["1a&1b&2c", 1], ["1c",1], ["0c", 0.5], ["0g&0e", 2],["1a&1b&2c", 1], ["2c",1], ["2c", 0.5], ["1g&0e", 2],["1a&1b&0c", 1], ["0c",1], ["1c", 0.5], ["1g&2e", 2],["1a&2b&2c", 1], ["0c",1], ["0c", 0.5], ["0g&0e", 2],["1a&1b&1c&2f", 1], ["1c",1], ["2c", 0.5], ["1g&1e", 2],["1a&1b&0c", 1], ["2c",1], ["1c", 0.5], ["1g&0e", 2],["1a&1b&2c", 1], ["1c",1], ["0c", 0.5], ["0g&0e", 2],["1a&1b&2c", 1], ["2c",1], ["2c", 0.5], ["1g&0e", 2],["1a&1b&0c", 1], ["0c",1], ["1c", 0.5], ["1g&2e", 2],["1a&2b&2c", 1], ["0c",1], ["0c", 0.5], ["0g&0e", 2]]

print(len(notes), "No_notes")

key = {} # json of the notes, to improve this: make this json be generated from musicXML, and make that be generated from PDFs


#maths functions

def get_vector_angle(v):
    return math.atan(v[1]/v[0])

#can't get the vector library to work in blender, so this is just a quick rotation about the z axis function    
def Rz(angle, v1): 
    #takes an array with 3 entries for v1. angle is in radians
    Rz = [[math.cos(angle), -math.sin(angle), 0],[math.sin(angle), math.cos(angle), 0], [0,0,1]]
    v2 = [(Rz[0][0]*v1[0] + Rz[0][1]*v1[1] + Rz[0][2]*v1[2]), (Rz[1][0]*v1[0] + Rz[1][1]*v1[1] + Rz[1][2]*v1[2]), (Rz[2][0]*v1[0] + Rz[2][1]*v1[1] + Rz[2][2]*v1[2])]
    return v2

#get orthogonal vector on x,y

def add(v1, v2):
    assert(len(v1) == len(v2))
    new_v = []
    for i in range(len(v1)):
        temp = v1[i] + v2[i]
        new_v.append(temp)
    return new_v

def x_by_scalar(scalar, v):
    new_v = []
    for i in v:
        new_v.append(i*scalar)
    return new_v


def magnitude(v):
    length = 0
    
    for i in v:
        length += i**2
    return math.sqrt(length)

def normalise(v):
    length = magnitude(v)
    return x_by_scalar(1/length, v)

#note: returns a unit vector with z component = 0
def get_orthogonal(vector):
    return normalise([-vector[1], vector[0], 0])

def dot_product(vector1, vector2):
    dot_product = 0
    if not len(vector1) == len(vector2):
        raise("Dot product requires that both vectors are the same size")
    for i in range(len(vector1)):
        dot_product += vector1[i]*vector2[i]
    return dot_product

#get vector component of vector1 along vector2
def proj(vector1, vector2):
    dot_product = dot_product(vector1, vector2)
    mag = magnitude(vector2)**2
    scalar_component = dot_product/mag
    proj = x_by_scalar(scalar_component, vector2)
    return proj

def angle_between(vector1, vector2):
    #get the angle between two vectors
    d_product = dot_product(vector1, vector2)
    v1_magnitude = magnitude(vector1)
    v2_magnitude = magnitude(vector2)
    theta = math.acos(d_product/(v1_magnitude * v2_magnitude))
    
    return theta
#get angle between x-axis and vector
def angle(vector):
    return math.atan(vector[1]/vector[0])

#reflect a vector about a given vector -- NOTE only reflects about the Z axis
def reflect(vector, pre_image):
    angle_between_vectors = angle_between(vector, pre_image)
    #angle between the vector and the x-axis
    angle_vector = angle(vector)
    

    #angle between the pre_image and the x-axis
    angle_pre_image = angle(pre_image)

    reflection_angle = angle_vector - angle_pre_image

    reflected_vector = Rz(2*reflection_angle, pre_image)
    return reflected_vector

def get_smallest_note(notes):
    smallest = notes[0][1]
    for i in notes:
        if i[1] < smallest:
            smallest = i[1]
    return smallest


def populate_key(key):
    global no_teeth
    global teeth_notes
    for c in range(no_teeth):    
                    x = 97 + (c+2)%7 #c + 2 mod 7 so that it cycles between c and a on the ascii table
                    fl = math.floor((c+2)/7)
                    newKey = str(fl) + chr(x)
                    key[newKey] = c


populate_key(key)



#get the number of beats in the notes array
def get_no_beats(notes):
    total_beats = 0
    for i in notes:
        total_beats+= i[1]

    return total_beats

no_beats = get_no_beats(notes)

#no_beats = 80
cylinder_circumference_modifier = 0.5 #To add some space between the final note and the first note
#cylinder_circumference = (pick_gap+(pick_radius))*no_beats + cylinder_circumference_modifier
shortest_note = get_smallest_note(notes)
cylinder_circumference = (pick_gap/shortest_note)*no_beats + len(notes)*(2*pick_radius)  + cylinder_circumference_modifier#(remember that no_beats is (n1 + n2 + n3... + nn)
cylinder_radius = cylinder_circumference/(2*math.pi)

pick_gap_angle = (pick_gap+(pick_radius))/(cylinder_radius)
cylinder_vertices = 100

def generate_main_cylinder():
    
    bpy.ops.mesh.primitive_cylinder_add(radius=cylinder_radius,vertices = cylinder_vertices, depth=cylinder_height, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    
    bpy.ops.object.move_to_collection(collection_index=0, is_new=True, new_collection_name='mainCylinder')
    bpy.context.active_object.name = 'main_cylinder'
    bpy.context.object.location[2] = cylinder_height/2
    
    

def make_pick(angle, pos, note):
    #make the pick
    bpy.ops.mesh.primitive_cylinder_add(radius=pick_radius, depth=pick_height, enter_editmode=False, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    bpy.context.object.name = 'pick'
    #rotate the pick pi/2 relative to y
    bpy.context.object.rotation_euler[1] = 1.5708
    #rotate the pick angle radians relative to z
    bpy.context.object.rotation_euler[2] = angle
    # move the pick to the location
    bpy.context.object.location[0] = pos[0]
    bpy.context.object.location[1] = pos[1]
    bpy.context.object.location[2] = pos[2] + pick_radius
    col = bpy.data.collections.get("mainCylinder")
    col.objects.link(bpy.context.active_object)
    
#cylinder is not a perfect circle, so if I position things at cylinder_radius
#away from center I end up with floating objects. This function calculates the 
#largest gap
def calc_pick_height_modifier():
    h = cylinder_radius*math.cos(math.pi/cylinder_vertices)
    return cylinder_radius - h
def generate_picks(notes):
    #loop through notes
    global pick_gap_angle
    global pick_height
    cumulative_angle = 0
    maximum_gap = calc_pick_height_modifier()
    pick_height += maximum_gap
    pos_vector = [cylinder_radius - maximum_gap + pick_height/2, 0, 0]
    shortest_note = get_smallest_note(notes)
    for c,i in enumerate(notes):
        chord = i[0].split("&")
        if c == 0:
            angle = 0
        else:
            angle = ((2*pick_radius) + (pick_gap*(i[1]/shortest_note)))/cylinder_radius
        cumulative_angle += angle
        assert(cumulative_angle < math.pi*2)
        pos_vector = Rz(angle, pos_vector)
        
        pick_rotation_angle = get_vector_angle(pos_vector)
        for x in chord:
            pos_vector[2] = key[x]*(teeth_width + teeth_gap) + cylinder_height_modifier - top_gap
            
            make_pick(pick_rotation_angle, pos_vector, chord)#pick_height_modifier)
#    col = bpy.data.collections.get("mainCylinder")
 #   bpy.ops.object.select_all(action='SELECT')
  #  bpy.ops.object.join()
    

def join_objects(collection):
     col = bpy.data.collections.get(collection)
     bpy.ops.object.select_all(action='SELECT')
     sel_objs = bpy.context.selected_objects
     objs_in_col = [obj for obj in col.objects if obj in sel_objs]
     bpy.ops.object.join()


#gears functions and classes

#find the desired module of the gear, using 32 teeth total as the ideal:

def render(v1,v2):
    vertices = [v1,v2]
    edges = [[0,1]]
    
    name = 'test'
    mesh = bpy.data.meshes.new(name)
    faces = []
    obj = bpy.data.objects.new(name, mesh)

    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(vertices,edges, faces)


class Gear():
    def __init__(self,**kwargs):
        no_points = 10
        self.__dict__.update(**kwargs)
        self.root_radius = self.get_root_radius()
        self.pitch = self.get_pitch()
        self.z_component = [0,0,self.diameters['face_width']]
        
        self.tooth_pitch = self.pitch/2
        self.pitch_angle = self.pitch/self.get_pitch_radius()
        self.pitch_angle_root = self.pitch/self.root_radius
        self.tooth_pitch_angle = self.pitch_angle/2
        
        self.profile_angle = self.tooth_pitch_angle/2
        self.involute_range = self.calculate_involute_range(no_points)
        involute_angle = self.involute_range['angle']
        
        self.tooth_pitch_angle_root = self.pitch_angle_root/2
        
        involute_curve_profile = self.generate_involute_profile([self.get_tip_radius(),0,0], no_points, self.involute_range)        
        self.tip_angle = involute_curve_profile['tip_range']
        self.pitch_angle_root = self.pitch/self.root_radius
        
        involute_curve_vertices = involute_curve_profile['vertices']
        
        tip_start_vector = involute_curve_profile['tip_start_vector']
        tip_vertices = self.generate_tip_vertices(self.tip_angle, no_points, tip_start_vector)
        self.vertices = {'involute_curve' : involute_curve_vertices, 'tip' : tip_vertices}    
        #1) calculate root diameter, dedendum and addendum
    def get_pitch(self):
        pitch = (self.diameters['pitch']*math.pi)/self.diameters['no_teeth']
        print(pitch, "LLL")
        return pitch    
        #return self.diameters['module']
        
    
    def get_circles(self):
        return self.diameters
    def get_root_radius(self):
        return self.diameters['root']/2
    def get_pitch_radius(self):
        return self.diameters['pitch']/2
    def get_tip_radius(self):
        return self.diameters['tip']/2
    def get_involute_curve_vertices(self):
        return self.involute_curve_vertices
    def get_tip_vertices(self):
        return self.tip_vertices        
    def get_vertices(self, part, position):
        vertices = []
        
        for i in range(len(self.vertices[part])):
            temp_verts = []    
            for c in self.vertices[part][i]:
                vert = Rz(position, c)
                temp_verts.append(vert)
            vertices.append(temp_verts)
            
        return vertices
    
    def calculate_tip_angle(self, involute_sector_angle, no_points):
        tip_angle = (self.pitch_angle/2 - involute_sector_angle*2)
        #tip_angle = (self.pitch_angle/2 - (involute_sector_angle*2))
        return tip_angle
    
    def generate_tip_vertices(self, tip_angle, no_points, start_vector):
        no_points = round(no_points)
        inter_angle = tip_angle/(no_points)
        top_tip_vertex = add(start_vector, self.z_component)
        vertices = [[start_vector], [top_tip_vertex]]
        for i in range(no_points):
            start_vector = Rz(inter_angle, start_vector)
            
            vertices[0].append(start_vector)
            top_tip_vertex = add(start_vector, self.z_component)
            
            vertices[1].append(top_tip_vertex)
        
        return vertices
    #calculates the angle and minimum iterant used in deriving the points for the involute curve
    
    def calculate_involute_sector(self, no_points, angle, min_it):
        
        
        #rotate pos_vector to the start of min_it*angle
        arc_length = angle*self.root_radius
        start_vector = [self.root_radius,0,0]
        end_vector = Rz(angle*min_it, start_vector)
        end_vector_orth = get_orthogonal(end_vector)
        end_vector_orth = x_by_scalar(-min_it*arc_length, end_vector_orth)
        point = add(end_vector, end_vector_orth)
        involute_curve_sector = angle_between(start_vector, point)
        
        
        return involute_curve_sector
        
        
    def calculate_involute_range(self, no_points):
        #no_points refers to the number of points along the involute curve to be used in the spline
        #the involute curve is generated from the root circle!
            
        tip_radius = self.get_tip_radius()
        pitch_radius = self.get_pitch_radius()
        
        root_radius_vector = [0,self.root_radius, 0]
        root_radius_vector_orth = get_orthogonal(root_radius_vector)
        
        
        #calculate the scalar required to lengthen a unit vector tangent to the root circle to the tip circle
        tip_scale_factor = math.sqrt((tip_radius**2 - root_radius_vector[0]**2 - root_radius_vector[1]**2)/(root_radius_vector_orth[0]**2 + root_radius_vector_orth[1]**2))
        #calculate the scalar required to lengthen a unit vector tangent to the root circle to the pitch circle
        pitch_scale_factor = math.sqrt((pitch_radius**2 - root_radius_vector[0]**2 - root_radius_vector[1]**2)/(root_radius_vector_orth[0]**2 + root_radius_vector_orth[1]**2))
        
        #tip_scale_factor is made up of no_points * length of segment
        theta = tip_scale_factor/(no_points*self.root_radius)
        #using theta to calculate the minimum iterant for rendering the addendum of the involute curve:
        min_it = math.floor(pitch_scale_factor/(theta*self.root_radius))
        involute_sector = self.calculate_involute_sector(no_points, theta, min_it)
        return {'angle' : theta, 'min_it' : min_it, 'involute_sector_angle' : involute_sector}
    
    #account for error coming from the rendering of the circle being an approximation of a circle - not uniform radius
    def calc_maximum_gap(self, root_radius, cylinder_vertices):
        h = root_radius*math.cos(math.pi/cylinder_vertices)

        return root_radius - h
    
    #generate the vertices of the involute curve profile of the gear tooth
    
    def generate_involute_profile(self, mid_vector, no_points, involute_range):
        
        #z vector
        radius_modifier = self.root_radius - self.calc_maximum_gap(self.root_radius, cylinder_vertices) #account for error in positioning teeth
        
        #create a vector that is in line with mid_vector, with magnitude 1
        start_vector = normalise(mid_vector)    
        #lengthen start_vector to length radius_modifier
        start_vector = x_by_scalar(radius_modifier, start_vector)
        #rotate start_vector so it points to where the involute curve starts
        start_vector = Rz(-(self.tooth_pitch_angle_root/2 + self.involute_range['involute_sector_angle']), start_vector)
        
        angle = involute_range['angle']
        #min_it = involute_range['min_it']
        
        min_it = 0 #setting this to zero so the cruve gets rendered from the root circle, since I am too pressed for time to model a fillet
        
        #rotate pos_vector to the start of min_it*angle
        pos_vector = Rz(angle*min_it, start_vector)
        arc_length = angle*self.root_radius
        vertices = [[],[],[],[],[],[]]
        
        for i in range(min_it, no_points + 1):
            pos_vector_orth = get_orthogonal(pos_vector)
            pos_vector_orth = x_by_scalar(-i*arc_length, pos_vector_orth)
            point = add(pos_vector, pos_vector_orth)
            z_vertex = add(self.z_component, point)
            vertices[0].append(point)
            vertices[1].append(z_vertex)    
            vertices[4].append(point)
            
            reflected_point = reflect(mid_vector, point)
            vertices[5].append(reflected_point)
            z_vertex = add(self.z_component, reflected_point)
            
            vertices[3].append(reflected_point)
            vertices[2].append(z_vertex)
            
            if (i < no_points):
                pos_vector = Rz(angle, pos_vector)
        #vertices.append([0,0,0])
        #vertices.append([0,self.get_tip_radius(),0])
        tip_range = angle_between(point, reflected_point)
        #calculate angle between start and end of profile
        involute_curve_range = angle_between(start_vector, point)
        return  {'vertices' : vertices, 'tip_start_vector' : point, 'tip_range' : tip_range, 'involute_curve_range' : involute_curve_range}
    
    
    def generate_involute_vertices(self, no_points, involute_range):
        #start with mid_vector of [0,tip_radius,0]
        #generate involute_profile           
        pass     

#takes the relevant circle diameters and calculates vertices for all points
#of the tooth (fillet, clearance, face, tip, + extruding 2d profile to gear_width 

#steps are: get involute curve vectors.Use these vectors to map out
#parameters. interpolate the curve described by the points
#use this formula to calculate the intersection of the curve with the tip_circle
#the intercection + all below it form the vertices for the face

#add some clearance
#look into blender curves to make the fillet curve tangent to the root_circle
#tip: follow line of tip_circle. blender issue.

#mirror these vectors across the y_axis, then give every vertex a twin at cylinder_width in the z_direction.

#Then connect the faces using something like rendering_info

#add a "gear with hole" boolean check when in the gear class. That way you can actually
#just call a gear class and plug in your diameters, then you can use a gear_renderer class to just make a gear

class Gear_Tooth():
    def __init__(self, modulus, pitch_diameter, **kwargs):
        self.__dict__.update(kwargs)
    
    
    
class Renderer():
    
    def render_gear_teeth(self, gear):
        pitch = gear.diameters['module']*math.pi
        angle = pitch/(gear.get_pitch_radius())
        for i in range(gear.diameters['no_teeth']):
            self.render_tooth(gear, angle*i, i)
            
        
    def render_tooth(self, gear, position, tooth_no):
        self.render_part(gear, 'involute_curve', position, tooth_no)
        self.render_part(gear, 'tip', position, tooth_no)
        
    def render_construction_circles(self,gear):
        bpy.ops.curve.primitive_bezier_circle_add(radius=gear.get_root_radius(), enter_editmode=True, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.curve.subdivide(number_cuts=100)
        bpy.ops.curve.primitive_bezier_circle_add(radius=gear.get_pitch_radius(), enter_editmode=True, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.curve.subdivide(number_cuts=100)
        bpy.ops.curve.primitive_bezier_circle_add(radius=gear.get_tip_radius(), enter_editmode=True, align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
        bpy.ops.curve.subdivide(number_cuts=100)
        
    def render_teeth(self,gear):
        pitch = gear.diameters['module']*math.pi
        angle = pitch/(gear.get_pitch_radius())
        
        cylinder_h = gear.get_tip_radius() - gear.get_root_radius()
        cylinder_rad = pitch/2
        pos_vector = [gear.get_root_radius() + cylinder_h/2,0,0]
        
        for i in range(gear.diameters['no_teeth']):
            bpy.ops.mesh.primitive_cylinder_add(radius=cylinder_rad, depth=cylinder_h, enter_editmode=False, align='WORLD', location=(pos_vector[0], pos_vector[1], 0), scale=(1, 1, 1))
            bpy.context.object.rotation_euler[1] = 1.5708
            bpy.context.object.rotation_euler[2] = angle*i
            pos_vector = Rz(angle, pos_vector)
    def render_involute_vertices(self, gear):
        vertices = gear.get_involute_curve_vertices()
        edges = []
        for i in range(len(vertices)-1):
            edges.append([i, i+1])
        
        name = 'tooth'
        mesh = bpy.data.meshes.new(name)
        faces = []
        obj = bpy.data.objects.new(name, mesh)

        col = bpy.data.collections.get("Collection")
        col.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        mesh.from_pydata(vertices,edges, faces)
    
    def render_side(self, vert1, vert2, part, tooth_no):
        vertices = []
        faces = []
        edges = []
        if not (len(vert1) == len(vert2)):
            raise("to render a face using render_side(), both sets of vertices must be the same size")
                
        for i in range(len(vert1)-1):
            
            vertices.append(vert1[i])
                    
            vertices.append(vert1[i+1])
            vertices.append(vert2[i+1])
            vertices.append(vert2[i])
        
            i_mod = 4*i
            faces.append([i_mod, i_mod+1, i_mod+2, i_mod+3])
       
      
        name = 'tooth' + part + str(tooth_no)
        mesh = bpy.data.meshes.new(name)
        
        obj = bpy.data.objects.new(name, mesh)

        col = bpy.data.collections.get("Collection")
        col.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        mesh.from_pydata(vertices,edges, faces)
        
    
    def render_part(self, gear, part, position, tooth_no):
        vertices = gear.get_vertices(part, position)
        
        for i in range(len(vertices)-1):
            self.render_side(vertices[i], vertices[i+1], part, tooth_no)
        
        #for c in range(len(vertices)):
         #   edges = []
 

#            for i in range(len(vertices[c])-1):
 #               edges.append([i, i+1])
  #          
   #             name = 'tooth' + part + str(tooth_no)
    #            mesh = bpy.data.meshes.new(name)
     #           faces = []
      #          obj = bpy.data.objects.new(name, mesh)
#
 #               col = bpy.data.collections.get("Collection")
  #              col.objects.link(obj)
   #             bpy.context.view_layer.objects.active = obj
    #            mesh.from_pydata(vertices[c],edges, faces)
                    
class Gear_System():
    def __init__(self, angle, cylinder_radius, face_width, gear_ratio_ar, no_teeth, **kwargs):
        #fcalculate modulus
        self.__dict__.update(kwargs)
        self.pressure_angle = angle
        self.gear_ratio = gear_ratio_ar[0]/gear_ratio_ar[1]
        self.gear_ratio_ar = gear_ratio_ar
        self.face_width = face_width
       # if (self.gear_ratio*no_teeth + no_teeth < 32):
        #    raise Exception("number of teeth and gear ratio supplied will result in undercut")
        
        
        
        gear1_pitch_diameter = self.calculate_pitch_diameter(cylinder_radius)
        
        self.module = self.calculate_module(gear1_pitch_diameter, no_teeth)  
        
        gear2_pitch_diameter = self.gear_ratio*gear1_pitch_diameter
        
        
        gear1_tip_diameter = self.calculate_tip_diameter(gear1_pitch_diameter)
        gear2_tip_diameter = self.calculate_tip_diameter(gear2_pitch_diameter)
        
        gear1_root_diameter = 2*cylinder_radius
        gear2_root_diameter = self.calculate_root_diameter(gear2_pitch_diameter)
       
        gear1_no_teeth = no_teeth
        gear2_no_teeth = int(self.gear_ratio*no_teeth)
        
        gear1 = {'pitch' : gear1_pitch_diameter, 'tip' : gear1_tip_diameter, 'root' : gear1_root_diameter, 'module' : self.module, 'no_teeth' : no_teeth, 'face_width' : face_width}
        gear2 = {'pitch' : gear2_pitch_diameter, 'tip' : gear2_tip_diameter, 'root' : gear2_root_diameter, 'module' : self.module, 'no_teeth' : no_teeth, 'face_width' : face_width}
        
        
        #todo: don't put this all in diameters
        self.gear1 = Gear(diameters = gear1)
        self.gear2 = Gear(diameters = gear2)
        
        self.gear_list = [self.gear1, self.gear2]        
      
    def getGears(self):
        return self.gear_list 
        
    def gear_info(self, request):
        if request == 'all':
            return self._dict__
        else:
            return self.__dict__[request]
    
    def calculate_module(self, pitch_diameter, no_teeth):
        return pitch_diameter/no_teeth
        
        
    def calculate_pitch(self):
        return (self.pitch_diameter*math.pi)/self.no_teeth
        
    def calculate_pitch_diameter(self, radius):
        return 2*radius/math.cos(self.pressure_angle*(math.pi/180))
        #return self.module*no_teeth
    
     
   
    def calculate_tip_diameter(self, pitch_diameter):
        return pitch_diameter + 2*self.module
    def calculate_root_diameter(self, pitch_diameter):
        return pitch_diameter*math.cos(self.pressure_angle*(math.pi/180))



    
def generate_completed_cylinder(notes):
    generate_main_cylinder()
    generate_picks(notes)
    join_objects("mainCylinder")



#comb functions

def add_tooth(pos_vector, tooth_thickness,tooth_width, tooth_length, note):
    # make mesh
    
    #get related vectors
    orth_vec = x_by_scalar(tooth_thickness, get_orthogonal(pos_vector)) 
    tooth_width_vector = x_by_scalar(tooth_width, z_)
    tooth_length_vector = x_by_scalar(tooth_length, normalise([pos_vector[0], pos_vector[1], 0]))
    
    
    #vertex 1 and 2
    
    vertices = [add(pos_vector, x_by_scalar(1/2,orth_vec)), add(pos_vector, x_by_scalar(1/2,x_by_scalar(-1, orth_vec)))]
    #vertex 3
    vertices.append(add(vertices[0], tooth_width_vector))
    #vertex 4
    vertices.append(add(vertices[1], tooth_width_vector))
    #vertex 5
    vertices.append(add(vertices[0], tooth_length_vector))
    #vertex 6
    vertices.append(add(vertices[1], tooth_length_vector))
    #vertex 7
    vertices.append(add(vertices[2], tooth_length_vector))
    #vertex 8
    vertices.append(add(vertices[3], tooth_length_vector))
    
    edges = [[0,1], [0,2], [1,3], [3,2], [0,4],[1,5],[3,7],[2,6], [4,5],[4,6],[5,7], [6,7]]
    
    #faces: front and back
    faces = [[0,1,3,2], [4,5,7,6]]
    #side`
    faces.append([0,4,6,2])
    #side2
    faces.append([1,5,7,3])
    #top
    faces.append([2,3,7,6])
    #botto
    faces.append([0,1,5,4])
    
    
    name = "tooth_" + str(note)
    mesh = bpy.data.meshes.new(name)
   
    obj = bpy.data.objects.new(name, mesh)

    col = bpy.data.collections.get("Collection")
    col.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    mesh.from_pydata(vertices,edges, faces)



def generate_teeth(no_teeth, teeth_width, teeth_length, teeth_thickness, teeth_gap):
    #pos_vector, tooth_thickness,tooth_width, tooth_length, note):
    lowest_pick_z = cylinder_height_modifier - top_gap - (teeth_width - pick_radius*2)/2
    tooth_rot = (pick_radius + teeth_thickness)/cylinder_radius
    pos_vector = Rz(-tooth_rot, [cylinder_radius + pick_height/2,0,0])
    pos_vector[2] = lowest_pick_z
    for i in range(no_teeth):
        add_tooth(pos_vector, teeth_thickness, teeth_width, teeth_length, 1)
        pos_vector[2] += teeth_width + teeth_gap

#x = Gear_System(20,cylinder_radius,gear_width, [1,2], 28)
x = Gear_System(20,cylinder_radius,gear_width, [1,2], 50)
y = Renderer()
#y.render_construction_circles(x.gear2)
y.render_gear_teeth(x.gear1)
#y.render_gear_teeth(x.gear1)
#y.render_involute_vertices(x.gear2)
#y.render_part(x.gear2,'involute_curve',math.pi,1)
#y.render_part(x.gear1, 'tip',0,1)
generate_completed_cylinder(notes)
#generate_teeth(no_teeth, teeth_width, teeth_length, teeth_thickness, teeth_gap)


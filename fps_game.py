import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from ctypes import c_void_p
import math
import random
import platform

WIDTH, HEIGHT = 800, 600
light_on = True
current_item = "torch"  # 'torch' or 'sword'

lastX, lastY = WIDTH / 2, HEIGHT / 2
first_mouse = True
yaw, pitch = -90.0, 0.0
camera_pos = pyrr.Vector3([0.0, 1.0, 3.0])
camera_front = pyrr.Vector3([0.0, 0.0, -1.0])
camera_up = pyrr.Vector3([0.0, 1.0, 0.0])
delta_time = 0.0
last_frame = 0.0

velocity_y = 0.0
gravity = -9.8
is_jumping = False
ground_level = 1.0 # Camera Y position on ground
objects = []

is_swinging = False
swing_angle = 0
swing_direction = 1

def mouse_button_callback(window, button, action, mods):
    global is_swinging
    if button == glfw.MOUSE_BUTTON_LEFT and action == glfw.PRESS:
        if current_item == "sword" and not is_swinging:
            is_swinging = True

def key_callback(window, key, scancode, action, mods):
    global light_on, current_item
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
    if key == glfw.KEY_1 and action == glfw.PRESS:
        current_item = "torch"
        light_on = True
        print("Currently holding: Torch")
    if key == glfw.KEY_2 and action == glfw.PRESS:
        current_item = "sword"
        light_on = False
        print("Currently holding: Sword")
    if key == glfw.KEY_F and action == glfw.PRESS:
        # Toggle light only if holding torch
        if current_item == "torch":
            light_on = not light_on
            print(f"Torch light {'on' if light_on else 'off'}")

def mouse_callback(window, xpos, ypos):
    global lastX, lastY, yaw, pitch, camera_front, first_mouse
    if first_mouse:
        lastX = xpos
        lastY = ypos
        first_mouse = False
    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos
    sensitivity = 0.1
    xoffset *= sensitivity
    yoffset *= sensitivity
    yaw += xoffset
    pitch += yoffset
    pitch = max(-89.0, min(89.0, pitch))
    front = pyrr.Vector3([
        math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)),
        math.sin(math.radians(pitch)),
        math.sin(math.radians(yaw)) * math.cos(math.radians(pitch))
    ])
    camera_front[:] = pyrr.vector.normalise(front)

def process_input(window):
    global camera_pos, velocity_y, is_jumping, delta_time

    camera_speed = 2.5 * delta_time
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        camera_pos += camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        camera_pos -= camera_speed * camera_front
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        camera_pos -= pyrr.vector.normalise(pyrr.vector3.cross(camera_front, camera_up)) * camera_speed
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        camera_pos += pyrr.vector.normalise(pyrr.vector3.cross(camera_front, camera_up)) * camera_speed

    # Jumping
    if glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS and not is_jumping:
        velocity_y = 5.0  # initial jump velocity
        is_jumping = True

    # Apply gravity if jumping
    if is_jumping:
        velocity_y += gravity * delta_time
        camera_pos.y += velocity_y * delta_time
        if camera_pos.y <= ground_level:
            camera_pos.y = ground_level
            is_jumping = False
            velocity_y = 0.0
    else:
        # Fix Y position on ground (no flying)
        camera_pos.y = ground_level

def create_shader():
    import platform
    if platform.system() == "Darwin":
        glsl_version = "#version 410 core\n"
    else:
        glsl_version = "#version 330 core\n"
    with open("vertex_shader.glsl") as f:
        vertex_src = f.read()
    with open("fragment_shader.glsl") as f:
        fragment_src = f.read()
    # Replace the first line (should start with #version) with glsl_version
    vertex_lines = vertex_src.splitlines()
    fragment_lines = fragment_src.splitlines()
    if vertex_lines and vertex_lines[0].lstrip().startswith("#version"):
        vertex_lines[0] = glsl_version.rstrip('\n')
    else:
        vertex_lines.insert(0, glsl_version.rstrip('\n'))
    if fragment_lines and fragment_lines[0].lstrip().startswith("#version"):
        fragment_lines[0] = glsl_version.rstrip('\n')
    else:
        fragment_lines.insert(0, glsl_version.rstrip('\n'))
    vertex_src = "\n".join(vertex_lines)
    fragment_src = "\n".join(fragment_lines)
    return compileProgram(
        compileShader(vertex_src, GL_VERTEX_SHADER),
        compileShader(fragment_src, GL_FRAGMENT_SHADER)
    )


def create_box(size=1.0):
    s = size / 2
    vertices = [
        -s,-s,-s, 0,0,-1,  s,-s,-s, 0,0,-1,  s, s,-s, 0,0,-1,
         s, s,-s, 0,0,-1, -s, s,-s, 0,0,-1, -s,-s,-s, 0,0,-1,
        -s,-s, s, 0,0, 1,  s,-s, s, 0,0, 1,  s, s, s, 0,0, 1,
         s, s, s, 0,0, 1, -s, s, s, 0,0, 1, -s,-s, s, 0,0, 1,
        -s, s, s, 0, 1, 0, -s, s,-s, 0, 1, 0,  s, s,-s, 0, 1, 0,
         s, s,-s, 0, 1, 0,  s, s, s, 0, 1, 0, -s, s, s, 0, 1, 0,
        -s,-s, s, 0,-1, 0, -s,-s,-s, 0,-1, 0,  s,-s,-s, 0,-1, 0,
         s,-s,-s, 0,-1, 0,  s,-s, s, 0,-1, 0, -s,-s, s, 0,-1, 0,
        -s,-s, s,-1,0,0, -s, s, s,-1,0,0, -s, s,-s,-1,0,0,
        -s, s,-s,-1,0,0, -s,-s,-s,-1,0,0, -s,-s, s,-1,0,0,
         s,-s, s, 1,0,0,  s, s, s, 1,0,0,  s, s,-s, 1,0,0,
         s, s,-s, 1,0,0,  s,-s,-s, 1,0,0,  s,-s, s, 1,0,0
    ]
    return np.array(vertices, dtype=np.float32)


def create_cylinder(radius=0.05, height=0.4, segments=16):
    vertices = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        next_theta = 2.0 * math.pi * (i + 1) / segments

        # Bottom circle triangle
        vertices.extend([
            0, 0, 0, 0, -1, 0,
            radius * math.cos(next_theta), 0, radius * math.sin(next_theta), 0, -1, 0,
            radius * math.cos(theta), 0, radius * math.sin(theta), 0, -1, 0
        ])

        # Top circle triangle
        vertices.extend([
            0, height, 0, 0, 1, 0,
            radius * math.cos(theta), height, radius * math.sin(theta), 0, 1, 0,
            radius * math.cos(next_theta), height, radius * math.sin(next_theta), 0, 1, 0
        ])

        # Side rectangle as two triangles
        # First triangle
        normal_x0 = math.cos(theta)
        normal_z0 = math.sin(theta)
        normal_x1 = math.cos(next_theta)
        normal_z1 = math.sin(next_theta)

        vertices.extend([
            radius * math.cos(theta), 0, radius * math.sin(theta), normal_x0, 0, normal_z0,
            radius * math.cos(next_theta), 0, radius * math.sin(next_theta), normal_x1, 0, normal_z1,
            radius * math.cos(next_theta), height, radius * math.sin(next_theta), normal_x1, 0, normal_z1
        ])

        # Second triangle
        vertices.extend([
            radius * math.cos(theta), 0, radius * math.sin(theta), normal_x0, 0, normal_z0,
            radius * math.cos(next_theta), height, radius * math.sin(next_theta), normal_x1, 0, normal_z1,
            radius * math.cos(theta), height, radius * math.sin(theta), normal_x0, 0, normal_z0
        ])

    return np.array(vertices, dtype=np.float32)


def create_sword_blade(length=0.6, width=0.05, thickness=0.02):
    l = length
    w = width / 2
    t = thickness / 2

    vertices = [
        # Front face
        -w, 0, t,  0, 0, 1,
         w, 0, t,  0, 0, 1,
         w, l, t,  0, 0, 1,
         w, l, t,  0, 0, 1,
        -w, l, t,  0, 0, 1,
        -w, 0, t,  0, 0, 1,

        # Back face
        -w, 0, -t, 0, 0, -1,
         w, 0, -t, 0, 0, -1,
         w, l, -t, 0, 0, -1,
         w, l, -t, 0, 0, -1,
        -w, l, -t, 0, 0, -1,
        -w, 0, -t, 0, 0, -1,

        # Left face
        -w, 0, -t, -1, 0, 0,
        -w, 0, t,  -1, 0, 0,
        -w, l, t,  -1, 0, 0,
        -w, l, t,  -1, 0, 0,
        -w, l, -t, -1, 0, 0,
        -w, 0, -t, -1, 0, 0,

        # Right face
         w, 0, -t, 1, 0, 0,
         w, 0, t,  1, 0, 0,
         w, l, t,  1, 0, 0,
         w, l, t,  1, 0, 0,
         w, l, -t, 1, 0, 0,
         w, 0, -t, 1, 0, 0,

        # Top face
        -w, l, -t, 0, 1, 0,
         w, l, -t, 0, 1, 0,
         w, l, t,  0, 1, 0,
         w, l, t,  0, 1, 0,
        -w, l, t,  0, 1, 0,
        -w, l, -t, 0, 1, 0,

        # Bottom face
        -w, 0, -t, 0, -1, 0,
         w, 0, -t, 0, -1, 0,
         w, 0, t,  0, -1, 0,
         w, 0, t,  0, -1, 0,
        -w, 0, t,  0, -1, 0,
        -w, 0, -t, 0, -1, 0,
    ]
    return np.array(vertices, dtype=np.float32)


def create_vao(vertices):
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)
    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * 4, c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * 4, c_void_p(12))
    glEnableVertexAttribArray(1)
    return vao, len(vertices) // 6


def add_object(vertices, position, color):
    vao, count = create_vao(vertices)
    objects.append((vao, count, pyrr.matrix44.create_from_translation(position), color))


def draw_inventory_bar(shader):
    global current_item
    # Disable depth test and enable blending for UI
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    # Orthographic projection for 2D UI: left=0, right=WIDTH, bottom=0, top=HEIGHT
    ortho = pyrr.matrix44.create_orthogonal_projection(0, WIDTH, 0, HEIGHT, -1, 1)
    glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, ortho)
    glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, pyrr.matrix44.create_identity())
    
    slot_size = 64
    margin = 10
    total_width = slot_size * 2 + margin
    start_x = WIDTH // 2 - total_width // 2
    y = 20

    def draw_quad(x, y, w, h, color):
        glUniform3f(glGetUniformLocation(shader, "objectColor"), *color)
        model = pyrr.matrix44.create_from_translation(pyrr.Vector3([x, y, 0]))
        model = pyrr.matrix44.multiply(model, pyrr.matrix44.create_from_scale(pyrr.Vector3([w, h, 1])))
        glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
        quad_vertices = np.array([
            0, 0, 0, 0, 0, 1,
            1, 0, 0, 0, 0, 1,
            1, 1, 0, 0, 0, 1,
            1, 1, 0, 0, 0, 1,
            0, 1, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1,
        ], dtype=np.float32)
        vao, count = create_vao(quad_vertices)
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, count)
        glDeleteVertexArrays(1, [vao])

    # Background dark gray slots
    draw_quad(start_x, y, slot_size, slot_size, (0.1, 0.1, 0.1))
    draw_quad(start_x + slot_size + margin, y, slot_size, slot_size, (0.1, 0.1, 0.1))

    # Torch slot - yellow highlight if selected, dull otherwise
    torch_color = (1.0, 1.0, 0.0) if current_item == "torch" else (0.6, 0.6, 0.0)
    draw_quad(start_x + slot_size + margin, y, slot_size, slot_size, torch_color)

    # Sword slot - blue highlight if selected, dull otherwise
    sword_color = (0.0, 0.7, 1.0) if current_item == "sword" else (0.0, 0.3, 0.5)
    draw_quad(start_x, y, slot_size, slot_size, sword_color)

    # Restore depth test and disable blending
    glEnable(GL_DEPTH_TEST)
    glDisable(GL_BLEND)


def main():
    global delta_time, last_frame, velocity_y, is_jumping, is_swinging, swing_angle, swing_direction

    if not glfw.init():
        raise Exception("glfw cannot be initialized!")
    # Set window hints for macOS (Darwin)
    if platform.system() == "Darwin":
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
    window = glfw.create_window(WIDTH, HEIGHT, "FPS Torchlight Demo", None, None)
    glfw.make_context_current(window)
    # macOS Core Profile requires a VAO to be bound before using shaders
    default_vao = glGenVertexArrays(1)
    glBindVertexArray(default_vao)
    glfw.set_cursor_pos_callback(window, mouse_callback)
    glfw.set_key_callback(window, key_callback)
    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_input_mode(window, glfw.CURSOR, glfw.CURSOR_DISABLED)

    shader = create_shader()
    glEnable(GL_DEPTH_TEST)

    # Add 3 objects (triangle and cubes)
    triangle = np.array([
        0.0, 0.5, 0.0, 0, 0, 1,
       -0.5,-0.5, 0.0, 0, 0, 1,
        0.5,-0.5, 0.0, 0, 0, 1
    ], dtype=np.float32)
    add_object(triangle, pyrr.Vector3([2, 0.5, -4]), (1.0, 0.3, 0.3))

    cube = create_box(1.0)
    add_object(cube, pyrr.Vector3([-2, 0.5, -3]), (0.2, 0.8, 0.4))
    add_object(cube, pyrr.Vector3([0, 0.5, -5]), (0.2, 0.3, 0.8))

    # Small cubes floating above those
    for base_pos in [(2, 0.5, -4), (-2, 0.5, -3), (0, 0.5, -5)]:
        offset = pyrr.Vector3([base_pos[0], base_pos[1] + 1.0, base_pos[2]])
        small_cube = create_box(0.3)
        add_object(small_cube, offset, (random.random(), random.random(), random.random()))

    # Floor and walls
    floor = create_box(10.0)
    add_object(floor, pyrr.Vector3([0, -5, 0]), (0.4, 0.4, 0.4))
    wall1 = create_box(10.0)
    add_object(wall1, pyrr.Vector3([0, 0, -10]), (0.3, 0.3, 0.5))
    wall2 = create_box(10.0)
    add_object(wall2, pyrr.Vector3([-10, 0, 0]), (0.5, 0.3, 0.3))
    wall3 = create_box(10.0)
    add_object(wall3, pyrr.Vector3([10, 0, 0]), (0.3, 0.5, 0.3))

    # Create held item models
    torch_handle = create_cylinder()
    torch_handle_vao, torch_handle_count = create_vao(torch_handle)

    sword_blade = create_sword_blade()
    sword_blade_vao, sword_blade_count = create_vao(sword_blade)

    while not glfw.window_should_close(window):
        current_frame = glfw.get_time()
        delta_time = current_frame - last_frame
        last_frame = current_frame

        process_input(window)
        glClearColor(0.05, 0.05, 0.05, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glUseProgram(shader)

        # View and projection matrices
        view = pyrr.matrix44.create_look_at(camera_pos, camera_pos + camera_front, camera_up)
        projection = pyrr.matrix44.create_perspective_projection(45, WIDTH / HEIGHT, 0.1, 100.0)

        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniformMatrix4fv(glGetUniformLocation(shader, "projection"), 1, GL_FALSE, projection)
        glUniform3f(glGetUniformLocation(shader, "lightPos"), *(camera_pos + camera_front * 2.0))
        glUniform3f(glGetUniformLocation(shader, "viewPos"), *camera_pos)
        glUniform3f(glGetUniformLocation(shader, "lightColor"), 1.0, 1.0, 1.0)

        # Light ON only if holding torch and light_on is True
        glUniform1i(glGetUniformLocation(shader, "lightOn"), 1 if current_item == "torch" and light_on else 0)

        # Draw world objects
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        for vao, count, model, color in objects:
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, model)
            glUniform3f(glGetUniformLocation(shader, "objectColor"), *color)
            glBindVertexArray(vao)
            glDrawArrays(GL_TRIANGLES, 0, count)

        # Clear depth buffer so held item is always drawn on top
        glClear(GL_DEPTH_BUFFER_BIT)

        # To draw the held item, we remove the translation from the view matrix.
        # This makes the item stay in a fixed position relative to the camera.
        view_no_translation = view.copy()
        view_no_translation[3][:3] = [0, 0, 0]
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view_no_translation)

        # Also disable lighting for the held item for a consistent look
        glUniform1i(glGetUniformLocation(shader, "lightOn"), 0)

        if is_swinging:
            swing_speed = 300
            swing_angle += swing_speed * delta_time * swing_direction
            if swing_angle > 45:
                swing_direction = -1
            elif swing_angle < 0:
                is_swinging = False
                swing_angle = 0
                swing_direction = 1

        if current_item == "torch":
            # Torch model
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.4, -0.25, -0.6]))
            rotation = pyrr.matrix44.create_from_x_rotation(math.radians(25))
            scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([0.1, 0.4, 0.1]))
            item_model = pyrr.matrix44.multiply(pyrr.matrix44.multiply(translation, rotation), scale)
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, item_model)
            glUniform3f(glGetUniformLocation(shader, "objectColor"), 0.8, 0.7, 0.1)
            glBindVertexArray(torch_handle_vao)
            glDrawArrays(GL_TRIANGLES, 0, torch_handle_count)

        elif current_item == "sword":
            # Sword model
            translation = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.5, -0.1, -0.8]))
            rotation = pyrr.matrix44.create_from_x_rotation(math.radians(15))
            rotation_z = pyrr.matrix44.create_from_z_rotation(math.radians(-10))
            combined_rotation = pyrr.matrix44.multiply(rotation, rotation_z)
            swing_rotation = pyrr.matrix44.create_from_y_rotation(math.radians(swing_angle))
            combined_rotation = pyrr.matrix44.multiply(swing_rotation, combined_rotation)
            scale = pyrr.matrix44.create_from_scale(pyrr.Vector3([0.05, 0.6, 0.05]))
            item_model = pyrr.matrix44.multiply(pyrr.matrix44.multiply(translation, combined_rotation), scale)
            glUniformMatrix4fv(glGetUniformLocation(shader, "model"), 1, GL_FALSE, item_model)
            glUniform3f(glGetUniformLocation(shader, "objectColor"), 0.3, 0.9, 1.0)
            glBindVertexArray(sword_blade_vao)
            glDrawArrays(GL_TRIANGLES, 0, sword_blade_count)

        # Restore view matrix and lighting for the next frame's world render
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
        glUniform1i(glGetUniformLocation(shader, "lightOn"), 1 if current_item == "torch" and light_on else 0)

        # Restore view matrix for UI
        glUniformMatrix4fv(glGetUniformLocation(shader, "view"), 1, GL_FALSE, view)
 
        # Draw inventory UI bar
        draw_inventory_bar(shader)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()

if __name__ == "__main__":
    main()

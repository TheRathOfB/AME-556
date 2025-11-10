import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import matplotlib.pyplot as plt
import imageio

xml_path = 'AME 556 HW 3 Prob 3.xml' #xml file (assumes this is in the same folder as this file)
fps = 60
hold_sec = 0.5
simend = 2 + hold_sec #simulation time
print_camera_config = 0 #set to 1 to print camera config
                        #this is useful for initializing view of the model)

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

def init_controller(model,data):
    #initialize the controller here. This function is called once, in the beginning
    pass

def controller(model, data):
    #put the controller here. This function is called inside the simulation.
    pass

def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(
        window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(
        window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height,
                      dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 *
                      yoffset, scene, cam)

#get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)                # MuJoCo data
cam = mj.MjvCamera()                        # Abstract camera
opt = mj.MjvOption()                        # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# Example on how to set camera configuration
# cam.azimuth = 90
# cam.elevation = -45
# cam.distance = 2
# cam.lookat = np.array([0.0, 0.0, 0])
cam.azimuth = -88.80000000000001 ; cam.elevation = -20.599999999999966 ; cam.distance = 2.587489162283811
cam.lookat = np.array([ 0.0 , 0.0 , 0.0 ])

#initialize the controller
init_controller(model,data)
data.qpos[0] = 0.1
data.qpos[2] = 0.15
data.qpos[3] = np.cos(0.5 * (np.pi/2))
data.qpos[5] = np.sin(0.5 * (np.pi/2))
data.qpos[7] = 0
data.qpos[8] = 0
data.qpos[9] = 0
data.qpos[10] = 0
data.qpos[11] = 0.1
data.qvel[0] = 0
data.qvel[3] = 0
data.qvel[5] = 0
data.qvel[7] = 0
data.qvel[8] = 0
data.qvel[9] = 0
data.qvel[10] = 0

q_ref = [data.qpos[7], data.qpos[8], data.qpos[9], data.qpos[10]]

#set the controller

def controller(model, data):
    q  = data.qpos[7:11]
    K = np.array([-1.0000, 29.1556, -1.8826, 4.8442])
    tau = K*(q_ref - q)
    data.ctrl[:] = np.clip(tau, -100.0, 100.0)

mj.set_mjcb_control(controller)
mj.mj_forward(model, data)

N = 500
qpos_hist = []
time_hist = []

# -------------------- Video Writer --------------------
video_path = r"C:\Users\brand\Documents\Pulsar\AME 556\AME 556 HW 3 Prob 3 Sim.mp4"
writer = imageio.get_writer(video_path, fps=fps, macro_block_size=1)

def render_and_write_frame():
    """Render current state, append to video. Returns True if successful frame."""
    fbw, fbh = glfw.get_framebuffer_size(window)
    if fbw == 0 or fbh == 0:
        glfw.poll_events()
        return False
    viewport = mj.MjrRect(0, 0, fbw, fbh)
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    # Swap/poll first to keep GLFW happy across platforms
    glfw.swap_buffers(window)
    glfw.poll_events()
    # Read back pixels and append
    rgb = np.empty((fbh, fbw, 3), dtype=np.uint8)
    mj.mjr_readPixels(rgb, None, viewport, context)
    writer.append_data(np.flipud(rgb))
    return True

# -------------------- 1) Pre-roll: show initial pose --------------------
n_hold_frames = int(round(hold_sec * fps))
for _ in range(n_hold_frames):
    if glfw.window_should_close(window):
        break
    render_and_write_frame()

# -------------------- 2) Simulate & record --------------------
while not glfw.window_should_close(window):
    time_prev = data.time
    # advance physics to maintain fps pacing
    while (data.time - time_prev) < (1.0 / fps):
        mj.mj_step(model, data)

    qpos_hist.append(data.qpos.copy())
    time_hist.append(data.time)

    if data.time >= simend:
        # final render at simend
        render_and_write_frame()
        break

    render_and_write_frame()

# -------------------- Shutdown (order matters) --------------------
writer.close()          # flush/encode video
if print_camera_config == 1:
    print('cam.azimuth =', cam.azimuth, ';',
          'cam.elevation =', cam.elevation, ';',
          'cam.distance =', cam.distance)
    print('cam.lookat = np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

glfw.terminate()
print("Saved video to:", video_path)

qpos_hist = np.array(qpos_hist)    # shape (N_steps, nq)
time_hist = np.array(time_hist)

x_hist = qpos_hist[:, 0] # : extracts every element from a specific column

w = qpos_hist[:, 3]
y = qpos_hist[:, 5]
theta = 2 * np.arctan2(y, w)

plt.plot(time_hist, x_hist)
plt.xlabel('time (sec)')
plt.ylabel('x (m)')
plt.title(r"x(t) of the COM position over time")
plt.show()

plt.plot(time_hist, theta)
plt.xlabel('time (sec)')
plt.ylabel(r"$\theta$ (rad)")
plt.title(r"$\theta(t)$ of the COM orientation over time")
plt.show()

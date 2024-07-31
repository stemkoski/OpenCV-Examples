import tkinter as tk
from tkinter import filedialog
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np

# References:
# 1. PyOpenGL documentation: http://pyopengl.sourceforge.net/documentation/index.html
# 2. OpenGL Programming Guide: https://www.glprogramming.com/red/
# 3. Tkinter documentation: https://docs.python.org/3/library/tkinter.html

# Initialize global variables
window = None
vertices = []
faces = []

def load_obj(filename):
    """
    Load an OBJ file and extract vertices and faces.
    """
    global vertices, faces
    vertices = []
    faces = []

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):
                parts = line.split()
                face = [int(i.split('/')[0]) - 1 for i in parts[1:]]
                faces.append(face)

def draw_wireframe():
    """
    Draw the OBJ model as a wireframe.
    """
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(0.0, 0.0, -5)

    glBegin(GL_LINES)
    for face in faces:
        for i in range(len(face)):
            glVertex3fv(vertices[face[i]])
            glVertex3fv(vertices[face[(i + 1) % len(face)]])
    glEnd()
    glutSwapBuffers()

def on_resize(width, height):
    """
    Handle window resizing.
    """
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / float(height), 1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def main():
    """
    Initialize the OpenGL and create the main Tkinter window.
    """
    global window
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
    window = tk.Tk()
    window.title('OBJ Wireframe Viewer')
    window.geometry('800x600')

    frame = tk.Frame(window, width=800, height=600)
    frame.pack(expand=True, fill=tk.BOTH)
    frame.bind("<Configure>", lambda event: on_resize(event.width, event.height))

    # Initialize OpenGL context
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"OBJ Wireframe Viewer")
    glutDisplayFunc(draw_wireframe)
    glutIdleFunc(draw_wireframe)

    # Load OBJ button
    button = tk.Button(window, text="Load OBJ", command=load_obj_file)
    button.pack()

    window.mainloop()

def load_obj_file():
    """
    Open a file dialog to load an OBJ file.
    """
    filename = filedialog.askopenfilename(filetypes=[("OBJ files", "*.obj")])
    if filename:
        load_obj(filename)
        glutPostRedisplay()

if __name__ == "__main__":
    main()
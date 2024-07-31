import numpy as np

# Define the golden ratio
phi = (1 + np.sqrt(5)) / 2

# Define the vertices of the icosahedron
vertices = [
    [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
    [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
    [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
]

# Define the faces of the icosahedron (each face is a triangle)
faces = [
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
]

# Write the OBJ file
with open('icosahedron.obj', 'w') as file:
    for vertex in vertices:
        file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
    for face in faces:
        file.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

print("OBJ file for icosahedron created as 'icosahedron.obj'")
import sys
from pathlib import Path
from plyfile import PlyData, PlyElement
import numpy as np

path_in = sys.argv[1] # 'path/to/mesh_semantic.ply'

print("Reading input...")
file_in = PlyData.read(path_in)
vertices_in = file_in.elements[0]
faces_in = file_in.elements[1]

print("Filtering data...")
# 0/7-floor, [3,6,7,9,11,12,13,48]:chair, 5-table, 14-pot, 56-pot-leaf, 18-tabletop
objects = {}
for f in faces_in:
     object_id = f[1]
     if  object_id not in [3, 6, 7, 9, 11, 12, 13, 48, 14, 56, 5, 18]:
         continue
     if not object_id in objects:
         objects[object_id] = []
     objects[object_id].append((f[0],))

print("Writing data...")
all_faces = [np.array(faces, dtype=[('vertex_indices', 'O')]) for faces in objects.values()]
all_faces = np.concatenate(all_faces, axis=0)
all_faces = PlyElement.describe(all_faces, 'face')
path_out = Path(path_in).parent / f"mesh_selected.ply"
PlyData([vertices_in, all_faces]).write(path_out)

# for object_id, faces in objects.items():
#     path_out = Path(path_in).parent / f"mesh_{object_id}.ply"
#     print(path_out)
#     faces_out = PlyElement.describe(np.array(faces, dtype=[('vertex_indices', 'O')]), 'face')
#     PlyData([vertices_in, faces_out]).write(path_out)

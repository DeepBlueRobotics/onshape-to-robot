import numpy as np
import shutil
import math
import subprocess
import stl
import os
from stl import mesh
from colorama import Fore, Back, Style


def load_mesh(stl_file):
    return mesh.Mesh.from_file(stl_file)


def save_mesh(mesh, stl_file):
    mesh.save(stl_file, mode=stl.Mode.BINARY)


def combine_meshes(meshData):

    return mesh.Mesh(np.concatenate(meshData))


def apply_matrix(mesh, matrix):
    rotation = matrix[0:3, 0:3]
    translation = matrix[0:3, 3:4].T.tolist()

    def transform(points):
        return (rotation*np.matrix(points).T).T + translation*len(points)

    mesh.v0 = transform(mesh.v0)
    mesh.v1 = transform(mesh.v1)
    mesh.v2 = transform(mesh.v2)
    mesh.normals = transform(mesh.normals)


# Script taken from doing the needed operation
# (Filters > Remeshing, Simplification and Reconstruction >
# Quadric Edge Collapse Decimation, with parameters:
# 0.9 percentage reduction (10%), 0.3 Quality threshold (70%)
# Target number of faces is ignored with those parameters
# conserving face normals, planar simplification and
# post-simplimfication cleaning)
# And going to Filter > Show current filter script
filter_script_mlx = """<!DOCTYPE FilterScript>
<FilterScript>
 <filter name="Simplification: Quadric Edge Collapse Decimation">
  <Param type="RichFloat" value="%reduction%" name="TargetPerc"/>
  <Param type="RichFloat" value="0.3" name="QualityThr"/>
  <Param type="RichBool" value="false" name="PreserveBoundary"/>
  <Param type="RichFloat" value="1" name="BoundaryWeight"/>
  <Param type="RichBool" value="false" name="PreserveNormal"/>
  <Param type="RichBool" value="false" name="PreserveTopology"/>
  <Param type="RichBool" value="false" name="OptimalPlacement"/>
  <Param type="RichBool" value="true" name="PlanarQuadric"/>
  <Param type="RichBool" value="false" name="QualityWeight"/>
  <Param type="RichBool" value="true" name="AutoClean"/>
  <Param type="RichBool" value="false" name="Selected"/>
 </filter>
</FilterScript>
"""


def create_tmp_filter_file(filename='filter_file_tmp.mlx', reduction=0.9):
    with open('/tmp/' + filename, 'w', encoding="utf-8") as stream:
        stream.write(filter_script_mlx.replace('%reduction%', str(reduction)))
    return '/tmp/' + filename


def reduce_faces(in_file, out_file, reduction=0.5):
    import pymeshlab
    filter_script_path = create_tmp_filter_file(reduction=reduction)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(in_file)
    ms.load_filter_script(filter_script_path)
    ms.apply_filter_script()
    ms.save_current_mesh(out_file)

def simplify_stl(stl_file, max_size=3):
    size_M = os.path.getsize(stl_file)/(1024*1024)

    if size_M > max_size:
        print(Fore.BLUE + '+ '+os.path.basename(stl_file) +
              (' is %.2f M, running mesh simplification' % size_M))
        shutil.copyfile(stl_file, '/tmp/simplify.stl')
        reduce_faces('/tmp/simplify.stl', stl_file, max_size / size_M)

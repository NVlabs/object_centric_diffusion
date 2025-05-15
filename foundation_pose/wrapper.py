import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np
import trimesh
from foundation_pose.estimater import ScorePredictor, PoseRefinePredictor, FoundationPose
from foundation_pose.Utils import draw_posed_3d_box, draw_xyz_axis, trimesh_add_pure_colored_texture


class FoundationPoseWrapper:
    def __init__(self, mesh_dir, debug_dir=None) -> None:
        # load object mesh
        self.debug_dir = "./debug" #debug_dir
        self.mesh_dir = mesh_dir
        self.mesh = None

        self.grasp_obj_name = None
        self.cur_grasp_obj_name = None
    
    def update_grasp_obj_name(self, obj_name):
        self.grasp_obj_name = obj_name

    def load_mesh(self):
        assert self.grasp_obj_name is not None
        mesh_path = os.path.join(self.mesh_dir, self.grasp_obj_name + ".obj")
        print(mesh_path)

        mesh = trimesh.load(mesh_path)
        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.load(mesh_path, force='mesh', skip_materials=False)

        # solve the material issue (wrongly recognize vertex color as material image)
        if mesh.visual.material is not None:
            if mesh.visual.material.image is None:  # no texture
                mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True) # use vertex color
        # mesh.show()

        if "light_bulb_in" in mesh_path:
            mesh = trimesh.load(mesh_path, force='mesh', skip_materials=True) # use vertex color

        mesh.vertices = mesh.vertices - np.mean(mesh.vertices, axis=0)

        self.mesh = mesh
        self.cur_grasp_obj_name = self.grasp_obj_name

    def create_estimator(self, debug_level=-1):
        # load mesh if mesh have not been loaded or grasp_obj_name changed
        if (self.mesh is None) or not (self.cur_grasp_obj_name == self.grasp_obj_name):
            self.load_mesh()

        debug_level = 0 if (self.debug_dir is None) or (debug_level < 0) else debug_level

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        return FoundationPose(
            model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, 
            scorer=scorer, refiner=refiner, 
            debug_dir=self.debug_dir, debug=debug_level,
        )
    

class FoundationPoseWrapperReal:
    def __init__(self) -> None:
        # load object mesh
        self.debug_dir = "./debug" #debug_dir
        self.mesh = None

    def create_estimator(self, debug_level=-1):
        assert self.mesh is not None

        debug_level = 0 if (self.debug_dir is None) or (debug_level < 0) else debug_level

        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        return FoundationPose(
            model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh, 
            scorer=scorer, refiner=refiner, 
            debug_dir=self.debug_dir, debug=debug_level,
        )
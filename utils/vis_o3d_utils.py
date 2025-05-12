# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os, shutil, copy
from pathlib import Path
from functools import partial

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as Rt

def draw_export_current_image(geometry=None,
         title="Open3D",
         width=1024,
         height=768,
         actions=None,
         lookat=None,
         eye=None,
         up=None,
         field_of_view=60.0,
         intrinsic_matrix=None,
         extrinsic_matrix=None,
         bg_color=(1.0, 1.0, 1.0, 1.0),
         bg_image=None,
         ibl=None,
         ibl_intensity=None,
         show_skybox=None,
         show_ui=None,
         raw_mode=False,
         point_size=None,
         line_width=None,
         animation_time_step=1.0,
         animation_duration=None,
         rpc_interface=False,
         on_init=None,
         on_animation_frame=None,
         on_animation_tick=None,
         non_blocking_and_return_uid=False):
    """Draw 3D geometry types and 3D models. This is a high level interface to
    :class:`open3d.visualization.O3DVisualizer`.

    The initial view may be specified either as a combination of (lookat, eye,
    up, and field of view) or (intrinsic matrix, extrinsic matrix) pair. A
    simple pinhole camera model is used.

    Args:
        geometry (List[Geometry] or List[Dict]): The 3D data to be displayed can be provided in different types:
            - A list of any Open3D geometry types (``PointCloud``, ``TriangleMesh``, ``LineSet`` or ``TriangleMeshModel``).
            - A list of dictionaries with geometry data and additional metadata. The following keys are used:
                - **name** (str): Geometry name.
                - **geometry** (Geometry): Open3D geometry to be drawn.
                - **material** (:class:`open3d.visualization.rendering.MaterialRecord`): PBR material for the geometry.
                - **group** (str): Assign the geometry to a group. Groups are shown in the settings panel and users can take take joint actions on a group as a whole.
                - **time** (float): If geometry elements are assigned times, a time bar is displayed and the elements can be animated.
                - **is_visible** (bool): Show this geometry?
        title (str): Window title.
        width (int): Viewport width.
        height (int): Viewport height.
        actions (List[(str, Callable)]): A list of pairs of action names and the
            corresponding functions to execute. These actions are presented as
            buttons in the settings panel. Each callable receives the window
            (``O3DVisualizer``) as an argument.
        lookat (array of shape (3,)): Camera principal axis direction.
        eye (array of shape (3,)): Camera location.
        up (array of shape (3,)): Camera up direction.
        field_of_view (float): Camera horizontal field of view (degrees).
        intrinsic_matrix (array of shape (3,3)): Camera intrinsic matrix.
        extrinsic_matrix (array of shape (4,4)): Camera extrinsic matrix (world
            to camera transformation).
        bg_color (array of shape (4,)): Background color float with range [0,1],
            default white.
        bg_image (open3d.geometry.Image): Background image.
        ibl (open3d.geometry.Image): Environment map for image based lighting
            (IBL).
        ibl_intensity (float): IBL intensity.
        show_skybox (bool): Show skybox as scene background (default False).
        show_ui (bool): Show settings user interface (default False). This can
            be toggled from the Actions menu.
        raw_mode (bool): Use raw mode for simpler rendering of the basic
            geometry (Default false).
        point_size (int): 3D point size (default 3).
        line_width (int): 3D line width (default 1).
        animation_time_step (float): Duration in seconds for each animation
            frame.
        animation_duration (float): Total animation duration in seconds.
        rpc_interface (bool or str): Start an RPC interface at this local
            address and listen for drawing requests. If rpc_interface is True, the
            default address "tcp://localhost:51454" is used. The requests can be
            made with :class:`open3d.visualization.ExternalVisualizer`.
        on_init (Callable): Extra initialization procedure for the underlying
            GUI window. The procedure receives a single argument of type
            :class:`open3d.visualization.O3DVisualizer`.
        on_animation_frame (Callable): Callback for each animation frame update
            with signature::

                Callback(O3DVisualizer, double time) -> None

        on_animation_tick (Callable): Callback for each animation time step with
            signature::

                Callback(O3DVisualizer, double tick_duration, double time) -> TickResult

            If the callback returns ``TickResult.REDRAW``, the scene is redrawn.
            It should return ``TickResult.NOCHANGE`` if redraw is not required.
        non_blocking_and_return_uid (bool): Do not block waiting for the user
            to close the window. Instead return the window ID. This is useful
            for embedding the visualizer and is used in the WebRTC interface and
            Tensorboard plugin.

    Example:
        See `examples/visualization/draw.py` for examples of advanced usage. The ``actions()``
        example from that file is shown below::

            import open3d as o3d
            import open3d.visualization as vis

            SOURCE_NAME = "Source"
            RESULT_NAME = "Result (Poisson reconstruction)"
            TRUTH_NAME = "Ground truth"

            bunny = o3d.data.BunnyMesh()
            bunny_mesh = o3d.io.read_triangle_mesh(bunny.path)
            bunny_mesh.compute_vertex_normals()

            bunny_mesh.paint_uniform_color((1, 0.75, 0))
            bunny_mesh.compute_vertex_normals()
            cloud = o3d.geometry.PointCloud()
            cloud.points = bunny_mesh.vertices
            cloud.normals = bunny_mesh.vertex_normals

            def make_mesh(o3dvis):
                mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    cloud)
                mesh.paint_uniform_color((1, 1, 1))
                mesh.compute_vertex_normals()
                o3dvis.add_geometry({"name": RESULT_NAME, "geometry": mesh})
                o3dvis.show_geometry(SOURCE_NAME, False)

            def toggle_result(o3dvis):
                truth_vis = o3dvis.get_geometry(TRUTH_NAME).is_visible
                o3dvis.show_geometry(TRUTH_NAME, not truth_vis)
                o3dvis.show_geometry(RESULT_NAME, truth_vis)

            vis.draw([{
                "name": SOURCE_NAME,
                "geometry": cloud
            }, {
                "name": TRUTH_NAME,
                "geometry": bunny_mesh,
                "is_visible": False
            }],
                 actions=[("Create Mesh", make_mesh),
                          ("Toggle truth/result", toggle_result)])
    """
    o3d.visualization.gui.Application.instance.initialize()
    w = o3d.visualization.O3DVisualizer(title, width, height)
    w.set_background(bg_color, bg_image)

    if actions is not None:
        for a in actions:
            w.add_action(a[0], a[1])

    if point_size is not None:
        w.point_size = point_size

    if line_width is not None:
        w.line_width = line_width

    def add(g, n):
        if isinstance(g, dict):
            w.add_geometry(g)
        else:
            w.add_geometry("Object " + str(n), g)

    n = 1
    if isinstance(geometry, list):
        for g in geometry:
            add(g, n)
            n += 1
    elif geometry is not None:
        add(geometry, n)

    w.reset_camera_to_default()  # make sure far/near get setup nicely
    if lookat is not None and eye is not None and up is not None:
        w.setup_camera(field_of_view, lookat, eye, up)
    elif intrinsic_matrix is not None and extrinsic_matrix is not None:
        w.setup_camera(intrinsic_matrix, extrinsic_matrix, width, height)

    w.animation_time_step = animation_time_step
    if animation_duration is not None:
        w.animation_duration = animation_duration

    if show_ui is not None:
        w.show_settings = show_ui

    if ibl is not None:
        w.set_ibl(ibl)

    if ibl_intensity is not None:
        w.set_ibl_intensity(ibl_intensity)

    if show_skybox is not None:
        w.show_skybox(show_skybox)

    if rpc_interface:
        if not isinstance(rpc_interface, str):
            rpc_interface = "tcp://127.0.0.1:51454"
        w.start_rpc_interface(address=rpc_interface, timeout=10000)

        def stop_rpc():
            w.stop_rpc_interface()
            return True

        w.set_on_close(stop_rpc)

    if raw_mode:
        w.enable_raw_mode(True)

    if on_init is not None:
        on_init(w)
    if on_animation_frame is not None:
        w.set_on_animation_frame(on_animation_frame)
    if on_animation_tick is not None:
        w.set_on_animation_tick(on_animation_tick)
    
    # w.scene.scene.render_to_image()
    
    o3d.visualization.gui.Application.instance.add_window(w)
    # img_o3d = o3d.visualization.gui.Application.render_to_image(w.scene, 1920, 1080)
    # img = np.array(img_o3d)
    
    # import cv2
    # cv2.imshow("foot_model", img)
    # cv2.waitKey()

    if non_blocking_and_return_uid:
        return w.uid
    else:
        o3d.visualization.gui.Application.instance.run()

    # w.export_current_image('test.png')
    # w.scene.scene.render_to_image()

class O3DVisualizer:
    def __init__(self) -> None:
        self.geo_list = []
        self.mat_list = []

        # visualizer setting
        self.cam_param = None
        self.vis = None

        # check if initialized
        self.ready = False

    def load_cam_param_from_file(self, filename=None):
        default_path = "./config/camera/cur_camera.json"
        filename = filename if filename is not None else default_path
        print(f"Load camera parameter from {filename}.")
        assert os.path.exists(filename), f"The path {filename} does not exist."
        self.cam_param = o3d.io.read_pinhole_camera_parameters(filename)

    def add_geometry(self, geometry: o3d.geometry.Geometry, mat: o3d.visualization.rendering.MaterialRecord = None):
        self.geo_list.append(geometry)
        self.mat_list.append(mat)
    
    def add_mesh_from_file(self, filename):
        print(f"Load mesh from {filename}.")
        assert os.path.exists(filename), f"The path {filename} does not exist."
        mesh = o3d.io.read_triangle_mesh(filename, enable_post_processing=True)
        self.add_geometry(mesh)
    
    def _check_color(self, rgb):
        if np.max(rgb) > 1:
            assert np.max(rgb) <= 255
            rgb = rgb.astype(np.float16) / 255.0
        assert np.max(rgb) <= 1
        return rgb
    
    def add_pc_from_numpy(self, xyz, color=None):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        if color is not None:
            assert len(xyz) == len(color)
            color = self._check_color(color)    # convert to (0, 1)
            pcd.colors = o3d.utility.Vector3dVector(color)
        self.add_geometry(pcd)
    
    def add_pc_from_dummy(self, n_point=1024):
        xyz = np.random.rand(n_point, 3)
        color = np.random.rand(n_point, 3)
        self.add_pc_from_numpy(xyz, color=color)
    
    def add_pc_from_file(self, filename):
        print(f"Load pointcloud from {filename}.")
        assert os.path.exists(filename), f"The path {filename} does not exist."
        pc = o3d.io.read_point_cloud(filename, remove_nan_points=True, remove_infinite_points=True)
        # pc.colors = o3d.utility.Vector3dVector(np.asarray(pc.colors) * 255.)
        # print(np.asarray(pc.colors))
        # print(pc.has_colors())
        self.add_geometry(pc)
    
    def add_pc_from_rgbd(self, rgbd: o3d.geometry.RGBDImage, intrinsic, extrinsic=None):
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, extrinsic=extrinsic)
        self.add_geometry(pcd)
    
    def add_pc_from_images(self, rgb, depth, intrinsic, extrinsic=None):        # reconstruct point cloud
        rgb_o3d = o3d.geometry.Image(np.ascontiguousarray(rgb))
        depth_o3d = o3d.geometry.Image(depth)
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            np.asarray(rgb_o3d).shape[1], np.asarray(rgb_o3d).shape[2], intrinsic_matrix=intrinsic,
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, convert_rgb_to_intensity=False)
        self.add_pc_from_rgbd(rgbd, intrinsic, extrinsic=extrinsic)

    def _pose_to_tf(self, pose):
        n_channel = np.array(pose).shape[-1]
        pos = pose[:3]
        if n_channel == 6:
            rot = Rt.from_euler('xyz', pose[3:]).as_matrix()[:3, :3]
        elif n_channel == 7:
            rot = Rt.from_quat(pose[3:]).as_matrix()[:3, :3] # assume xyzw
        else:
            raise NotImplementedError
        tf = np.eye(4)
        tf[:3, :3] = rot
        tf[:3, -1] = pos
        return tf


    def add_pose_from_traj(self, traj, pos_only=False, paint_color=None):
        n_channel = np.array(traj).shape[-1]
        assert n_channel in [3, 6, 7], f"Current channel number {n_channel} is unsupported."
        pos_only = pos_only or n_channel <= 3   # no orientation
        if pos_only:
            assert paint_color is None, f"paint color is not supported when pos_only is true"
        
        # get color map
        from matplotlib import cm
        color_map = cm.get_cmap("gist_rainbow")   

        # add coordinate frame mesh
        if not pos_only:
            for timestamp, pose in enumerate(traj):
                tf = self._pose_to_tf(pose)
                # get mesh
                size = 0.04 # 0.05
                pose_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
                if paint_color is not None:
                    pose_mesh.paint_uniform_color(paint_color)
                else:
                    # colored by timestamp
                    n_pose = len(traj)
                    normalized_timestamps = timestamp / n_pose
                    pose_mesh.paint_uniform_color(color_map(normalized_timestamps)[:3])
                pose_mesh.transform(tf)
                self.add_geometry(pose_mesh)

        # add point cloud (colored by timestamp)
        else:
            points = [pose[:3] for pose in traj]
            n_pose = len(traj)         
            normalized_timestamps = [ts / n_pose for ts in range(n_pose)]
            colors = [color_map(ts)[:3] for ts in normalized_timestamps] # range is (0, 1)
            self.add_pc_from_numpy(points, color=colors)
    
    def add_mesh_from_traj(self, traj, frame_idx, obj_mesh, obj_mat):
        n_channel = np.array(traj).shape[-1]
        pose = traj[frame_idx]
        # pose2 = traj[frame_idx+1]
        # pose[:3] = (pose[:3] + pose2[:3]) / 2.
        mesh = copy.deepcopy(obj_mesh) # create new

        tf = self._pose_to_tf(pose)
        mesh.transform(tf)
        if obj_mat is None:
            self.add_geometry(mesh)
        else:
            self.add_geometry(mesh, mat=obj_mat)



    def init(self):
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()
        self.ready = True
    
    def run(self):
        assert self.ready, "must run init() first"
        # add geometry
        if len(self.geo_list) > 0:
            [self.vis.add_geometry(g) for g in self.geo_list]
        # update camera
        if self.cam_param is not None:
            ctr = self.vis.get_view_control()
            ctr.convert_from_pinhole_camera_parameters(self.cam_param, True)
        # run
        # print("Press h for helper function.") # for more details: https://github.com/isl-org/Open3D/tree/main/cpp/open3d/visualization/visualizer/VisualizerCallback.cpp#L102
        self.vis.run()
        # close
        self.vis.destroy_window()
        self.ready = False

    def run_test_cam(self):
        def _save_camera_pose(vis, save_dir="./config/camera/"):
            # get current cam parameter
            save_path = os.path.join(save_dir, "cur_camera.json")
            ctr = self.vis.get_view_control()
            params = ctr.convert_to_pinhole_camera_parameters()
            cur_extrinsic = params.extrinsic
            cur_intrinsic = params.intrinsic.intrinsic_matrix
            print(f"Current extrinsic: {cur_extrinsic}")
            print(f"Save to {save_path}.")
            o3d.io.write_pinhole_camera_parameters(save_path, params)
        # add key callback to save camera parameter
        self.vis.register_key_callback(ord("S"), partial(_save_camera_pose))
        self.run()
    
    def draw(self):
        geo_mat_list = []
        for idx, (geo, mat) in enumerate(zip(self.geo_list, self.mat_list)):
            cur_dict = {
                "name": f"geo_{idx}",
                "geometry": geo,
                "material": mat,
            }
            geo_mat_list.append(cur_dict)
        # high level interface to open3d.visualization.O3DVisualizer
        o3d.visualization.draw(
            geo_mat_list, 
            bg_color=(0.1, 0.1, 0.1, 1.0), show_skybox=False,
            intrinsic_matrix=None if self.cam_param is None else self.cam_param.intrinsic.intrinsic_matrix,
            extrinsic_matrix=None if self.cam_param is None else self.cam_param.extrinsic,
            point_size=15,
        )


if __name__ == "__main__":
    # vis = O3DVisualizer()
    # vis.add_pc_from_dummy()
    # vis.init()
    # vis.run_test_cam()

    # vis.load_cam_param_from_file()
    # vis.init()
    # vis.run()

    vis = O3DVisualizer()
    vis.add_pc_from_dummy()
    vis.draw()
# RoboJuDo-Plus G1 assets

Copied from `RoboJuDo-Plus/assets/robots/g1` at commit
`328837233f08159a5c2439797e7386b65dc32f10`
([chifongip/RoboJuDo-Plus](https://github.com/chifongip/RoboJuDo-Plus),
upstream copyright HansZ8).
The XML and STL files are copied unchanged. The source repository's license
is included in `LICENSE` (CC BY 4.0).

Training uses `g1_29dof_rev_1_0.xml`; the 23- and 12-DoF variants are retained
as supplied by the source. The 29-DoF robot has 26 collision meshes, four
shoulder cylinders, and four spheres of radius 0.005 m under each foot.

`src/tasks/supermimic/config/robot_cfg.py` adapts the model at load time:
it removes the standalone floor and lights, replaces torque motors with the
existing mjlab training PD actuators, and names the originally unnamed geoms.
All original robot collision shapes are retained. Training disables robot
self-collisions while retaining full-body ground contacts. The scene supplies
the ground and applies foot friction randomization to all eight foot spheres.

#!/usr/bin/env python3
import argparse
import pickle
import numpy as np
from pydrake.all import (
    DiagramBuilder,
    Parser,
    AddMultibodyPlantSceneGraph,
    RigidTransform,
    ConnectMeshcatVisualizer,
    FindResourceOrThrow
)

NQ = 7
Q_NOMINAL = np.array([0.0, 0.1, 0, -1.2, 0, 1.6, 0])
np.random.seed(0)


def gen_random_q(lower, upper):
    up = upper - Q_NOMINAL
    lp = Q_NOMINAL - lower
    scale = np.array([min(u, l) for u,l in zip(up, lp)])/4
    print(scale)
    rand_q = np.random.normal(Q_NOMINAL, scale = scale)
    rand_q[0] = np.random.uniform(lower[0], upper[0])
    rand_q = np.clip(rand_q, lower, upper)
    return rand_q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Use --url to specify the zmq_url for a meshcat server\nuse --problem to specify a .yaml problem file"
    )
    parser.add_argument("-u", "--url", nargs="?", default=None)
    parser.add_argument("-n", "--num", nargs="?", default = 1000)
    args = parser.parse_args()
    zmq_url, num = args.url, args.num

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step = 1e-4)
    parser = Parser(plant, scene_graph)
    panda = parser.AddModelFromFile(FindResourceOrThrow("drake/manipulation/models/franka_description/urdf/panda_arm_hand.urdf"))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("panda_link0", panda), RigidTransform())
    plant.Finalize()


    if zmq_url is not None:
        meshcat = ConnectMeshcatVisualizer(
            builder,
            scene_graph,
            zmq_url=zmq_url,
        )
        meshcat.load()
    else:
        print("No meshcat server provided, running without GUI")

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    diagram.Publish(diagram_context)

    plant_context = plant.GetMyContextFromRoot(diagram_context)

    # get joint limits

    inds = plant.GetJointIndices(panda)[0:NQ]
    joint_lower_limits = []
    joint_upper_limits = []
    for i in inds:
        joint = plant.get_joint(i)
        joint_lower_limits.append(joint.position_lower_limits()[0])
        joint_upper_limits.append(joint.position_upper_limits()[0])


    for i in range(num):
        rand_q = gen_random_q(joint_lower_limits, joint_upper_limits) #np.random.uniform(joint_lower_limits, joint_upper_limits)
        rand_q = np.concatenate((rand_q, np.zeros(2)))
        plant.SetPositions(plant_context, panda, rand_q)
        diagram.Publish(diagram_context)
        input()





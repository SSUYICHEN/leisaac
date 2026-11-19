import torch


def dynamic_reset_gripper_effort_limit_sim(env, teleop_device):
    need_to_set = []
    if "bi-so101leader" in teleop_device:
        need_to_set = [env.scene.articulations["left_arm"], env.scene.articulations["right_arm"]]
    elif "so101leader" in teleop_device or teleop_device in ["keyboard", "gamepad"]:
        need_to_set = [env.scene["robot"]]
    for arm in need_to_set:
        write_gripper_effort_limit_sim(env, arm)
    return


def write_gripper_effort_limit_sim(env, env_arm):
    # NOTE: Isaac Lab 3.x data properties return ProxyArray; use .torch to get a torch.Tensor view.
    gripper_pos = env_arm.data.body_link_pos_w.torch[:, -1]  # [num_envs, 3]

    object_positions = []
    object_masses = []
    object_names = []

    for name, obj in env.scene._rigid_objects.items():
        pos = obj.data.body_link_pos_w.torch[:, 0]  # [num_envs, 3]
        object_positions.append(pos)
        object_masses.append(obj.data.default_mass.torch)
        object_names.append(name)

    if not object_positions:
        return

    object_positions = torch.stack(object_positions)  # [num_objects, num_envs, 3]
    object_masses = torch.stack(object_masses)  # [num_objects, num_envs, 1]

    distances = torch.sqrt(torch.sum((object_positions - gripper_pos.unsqueeze(0)) ** 2, dim=2))

    min_distances, min_indices = torch.min(distances, dim=0)  # [num_envs]

    target_masses = object_masses[min_indices.cpu(), 0, 0]  # [num_envs]

    joint_effort_limits = env_arm.data.joint_effort_limits.torch  # [num_envs, num_joints]
    target_effort_limits = (target_masses / 0.15).to(joint_effort_limits.device)

    current_effort_limit_sim = joint_effort_limits[:, -1]  # [num_envs]
    need_update = torch.abs(target_effort_limits - current_effort_limit_sim) > 0.1

    if torch.any(need_update):
        new_limits = current_effort_limit_sim.clone()
        new_limits[need_update] = target_effort_limits[need_update]

        # Isaac Lab 3.x: write_joint_effort_limit_to_sim is deprecated; the *_index variant expects
        # limits of shape (len(env_ids), len(joint_ids)) = (num_envs, 1) for the gripper joint (index 5).
        env_arm.write_joint_effort_limit_to_sim_index(limits=new_limits.unsqueeze(-1), joint_ids=[5])


def get_task_type(task: str, task_type: str | None = None) -> str:
    """
    Make sure the task type is in the supported teleop devices.
    """
    if task_type is not None:
        return task_type
    if "BiArm" in task:
        return "bi-so101leader"
    elif "LeKiwi" in task:
        return "lekiwi-leader"
    else:
        return "so101leader"


def delete_attribute(obj, attr_name):
    if hasattr(obj, attr_name):
        delattr(obj, attr_name)

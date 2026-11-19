def apply_default_sim_settings(env_cfg) -> None:
    """Apply leisaac's default PhysX / RTX settings to an environment config (Isaac Lab 3.x API).

    Isaac Lab 3.0 replaced ``SimulationCfg.physx`` by ``SimulationCfg.physics`` and removed ``SimulationCfg.render``;
    the RTX settings are now carried by each camera's ``renderer_cfg``.
    """
    from isaaclab_physx.physics import PhysxCfg
    from isaaclab_tasks.utils.presets import set_isaac_rtx_global_settings

    env_cfg.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01, friction_correlation_distance=0.00625)
    for value in vars(env_cfg.scene).values():
        renderer_cfg = getattr(value, "renderer_cfg", None)
        if renderer_cfg is not None:
            set_isaac_rtx_global_settings(renderer_cfg, enable_translucency=True)

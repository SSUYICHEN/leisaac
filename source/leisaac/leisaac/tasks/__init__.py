# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

##
# Register Gym environments.
##

import pkgutil

from isaaclab_tasks.utils import import_packages

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils", ".mdp"]


# Import each task sub-package separately so that a task which fails to import (for example because an
# optional Isaac Sim extension is unavailable) does not prevent the remaining tasks from being registered.
def _import_all_tasks():
    for module_info in pkgutil.iter_modules(__path__):
        if not module_info.ispkg or module_info.name in _BLACKLIST_PKGS:
            continue
        try:
            import_packages(f"{__name__}.{module_info.name}", _BLACKLIST_PKGS)
        except Exception as e:  # noqa: BLE001
            print(f"[leisaac] WARNING: failed to import task package '{module_info.name}': {e!r}")


_import_all_tasks()

# Copyright (c) 2019, NVIDIA Corporation. All rights reserved
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
# Cheng-Bin Jin Re-implementation.

"""Helpers for managing the run/training loop."""

class RunContext(object):
    """Helper class for managing the run/training loop.

    The context will hide the implementation details of a basic run/training loop.
    It will set things up properly, tell if run should be stopped, and then cleans up.
    User should call update periodically and use should_stop to determine if run should be stopped.

    Args:
        submit_config: The SubmitConfig that is used for the current run.
        config_module: (deprecated) The whole config module that is used for the current run.
        """
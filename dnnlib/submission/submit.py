# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code Lincense-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
# Cheng-Bin Jin re-implementation

"""Submit a function to be run either locally or in a computing cluster."""

import copy
import re
from enum import Enum

from .. import util
from . import internal

class SubmitTarget(Enum):
    """The target where the function should be run.

    LOCAL: Run it locally
    """
    LOCAL = 1


class PlatformExtras:
    """A mixed bag of values used by dnnlib heuristics.

    Attributes:

        data_reader_buffer_size: Used by DataReader to size internal shared memory buffers.
        data_reader_process_count: Number of worker processes to spawn (zero for single thread operation)
    """
    def __init__(self):
        self.data_reader_buffer_size = 1 << 30  # 1GB
        self.data_reader_process_count = 0      # single threaded default




class SubmitConfig(util.EasyDict):
    """Strongly typed config dict needed to submit runs.

    Attributes:
        run_dir_root: Path to the run dir root. Can be optionally templated with tags. Needs to always be run through get_path_from_template.
        run_desc: Description of the run. Will be used in the run dir and task name.
        run_dir_ignore: List of file patterns used to ignore files when copying files to the run dir.
        run_dir_extra_files: List of (abs_path, rel_path) tuples of file paths. rel_path root will be the src directory inside the run dir.
        submit_target: Submit target enum value. Used to select where the run is actually launched.
        num_gpus: Number of GPUs used/requested for the run.
        print_info: Wheter to print debug information when submitting.
        local.do_not_copy_source_files: Do not copy source files from the working directory to the run dir.
        run_id: Automatically populated value during submit.
        run_name: Automatically populated value during submit.
        run_dir: Automatically populated value during submit.
        run_func_name: Automatically populated value during submit.
        run_func_kwargs: Automatically populated value during submit.
        user_name: Automatically populated value during submit. Can be set by the user which will then override the automatic value.
        task_name: Automatically populated value during submit.
        host_name: Automatically populated value during submit.
        platform_extras: Automatically populated values during submit. Used by various dnnlib libraries such as the DataReader class.
    """
    def __init__(self):
        super().__init__()

        # run (set these)
        self.run_dir_root = ""  # should always be passed through get_path_from_template
        self.run_desc = ""
        self.run_dir_ignore = ["__pycache__", "*.pyproj", "*.sln", "*.suo", ".cache", ".idea", ".vs", ".vscode", "_cudacache"]
        self.run_dir_extra_files = []

        # submit (set these)
        self.submit_target = SubmitTarget.LOCAL
        self.num_gpus = 1
        self.print_info = False
        self.nvprof = False
        self.local = internal.local.TargetOptions()
        self.datasets = []

        # (automatically populated)
        self.run_id = None
        self.run_name = None
        self.run_dir = None
        self.run_func_name = None
        self.run_func_kwargs = None
        self.user_name = None
        self.task_name = None
        self.host_name = None
        self.platform_extras = PlatformExtras()


def submit_run(submit_config: SubmitConfig, run_func_name: str, **run_func_kwargs) -> None:
    """Create a run dir, gather files related to the run, copy files to the run dir, and launch the run in appropriate place."""
    submit_config = copy.deepcopy(submit_config)

    submit_target = submit_config.submit_target
    farm = None
    if submit_target == SubmitTarget.LOCAL:
        farm = internal.local.Target()
    assert farm is not None  # unknown target

    # Disallow submitting jobs with zero num_gpus.
    if (submit_config.num_gpus is None) or (submit_config.num_gpus == 0):
        raise RuntimeError("submit_config.num_gpus must be set to a non-zero value")

    if submit_config.user_name is None:
        submit_config.user_name = get_user_name()

    submit_config.run_func_name = run_func_name
    submit_config.run_func_kwargs = run_func_kwargs

    #---------------------------------------------------------------------------------------
    # Prepare submission by populating the run dir
    #---------------------------------------------------------------------------------------
    host_run_dir = _create_run_dir_local(submit_config)

    submit_config.task_name = "{0}-{1:05d}-{2}".format(submit_config.user_name, submit_config.run_id, submit_config.run_desc)
    docker_valid_name_regex = "^[a-zA-Z0-9][a-zA-Z0-9_.-]+$"
    if not re.match(docker_valid_name_regex, submit_config.task_name):
        raise RuntimeError("Invalid task name. Probable reason: unacceptable characters in your submit_config.run_desc. "
                           "Task name must be accepted by the following regex: " + docker_valid_name_regex + ", got " +
                           submit_config.task_name)

    # Farm specific preparations for a submit
    farm.finalize_submit_config(submit_config, host_run_dir)
    _populate_run_dir(submit_config, host_run_dir)
    return farm.submit(submit_config, host_run_dir)
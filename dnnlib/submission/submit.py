# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code Lincense-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html
# Cheng-Bin Jin re-implementation

"""Submit a function to be run either locally or in a computing cluster."""

class SubmitConfig():
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
        Platform_extra: Automatically populated values during submit. Used by various dnnlib libraries such as the DataReader class.
    """
    def __init__(self):


import os
from shutil import copyfile


def create_dirs(dirs, json_filename):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return exit_code: 0:success -1:failed
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)

        # copy the current json file for future reference
        dst = os.path.abspath(os.path.join(dirs[0], '..')) + "/configs_file.json"
        if not os.path.exists(dst):
            copyfile(json_filename, dst)

        return 0
    except Exception as err:
        print("Creating directories error: {0}".format(err))
        exit(-1)

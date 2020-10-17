"""
Module utils.iotools
---------------------

Created on 1 Jul 2013

utils.iotools.py holds helper functions for input/output issues. This module is not intended to
be executed.

Feel free to use and extend this module.

.. moduleauthor:: vimaier

"""
import json
import shutil
from pathlib import Path
from typing import List, Union

from generic_parser.entry_datatypes import get_instance_faker_meta
from generic_parser.entrypoint_parser import save_options_to_config

from omc3.definitions import formats
from omc3.utils import logging_tools

LOG = logging_tools.get_logger(__name__)


def delete_content_of_dir(path_to_dir):
    """
    Deletes all folders, files and symbolic links in given directory.
    :param string path_to_dir:
    """
    directory = Path(path_to_dir)
    if not directory.is_dir():
        return
    for item in directory.iterdir():
        delete_item(item)


def delete_item(path_to_item) -> None:
    """ Deletes the item given by path_to_item. It distinguishes between a file, a directory and a
    symbolic link.
    """
    item = Path(path_to_item)
    try:
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)
    except FileNotFoundError:
        LOG.error(f"No item to delete at '{item}'")
    except OSError:
        LOG.error(f"Could not delete item '{item}'")


def copy_content_of_dir(src_dir, dst_dir) -> None:
    """ Copies all files and directories from src_dir to dst_dir. """
    source = Path(src_dir)
    destination = Path(dst_dir)

    if not source.is_dir():
        return

    create_dirs(destination)

    for element in source.iterdir():
        copy_item(element, destination)


def create_dirs(directory) -> None:
    """ Creates all dirs to directory if not exists. """
    directory = Path(directory)
    if not directory.exists():
        directory.mkdir()
        LOG.debug(f"Created directory at: '{directory}'")


def copy_item(src_item, dest) -> None:
    """ Copies a file or a directory to dest. dest may be a directory.
    If src_item is a directory then all containing files and dirs will be copied into dest. """
    source = Path(src_item)
    destination = Path(dest)
    try:
        if source.is_file():
            shutil.copy2(src_item, dest)
        elif source.is_dir():
            copy_content_of_dir(src_item, dest)
        else:
            raise IOError
    except IOError:
        LOG.error("Could not copy item because of IOError. Item: '{}'".format(src_item))


def delete_files_except_gitignore(path_to_directory: Union[Path, str]) -> bool:
    """
    Deletes all files in the given path_to_directory except of the file with the name '.gitignore'

    Args:
        path_to_directory (Union[Path, str]): Path to the directory to wipe.

    Returns:
        True if the directory exists and the files are deleted otherwise False
    """
    directory = Path(path_to_directory)
    if not directory.is_dir():
        return False

    for element in directory.iterdir():
        if element.is_file() and element.name != ".gitignore":
            element.unlink()
    return True


def exists_directory(path_to_dir) -> bool:
    return Path(path_to_dir).is_dir()


def not_exists_directory(path_to_dir) -> bool:
    return not exists_directory(path_to_dir)


def no_dirs_exist(*dirs) -> bool:
    return not dirs_exist(*dirs)


def dirs_exist(*dirs) -> bool:
    for directory in dirs:
        if not Path(directory).is_dir():
            return False
    return True


def get_all_filenames_in_dir_and_subdirs(path_to_dir) -> List[Path]:
    """ Looks for files(not dirs) in dir and subdirs and returns them as a list.  """
    return [element for element in Path(path_to_dir).rglob("*") if element.is_file()]


def get_all_filenames_in_dir(path_to_dir) -> List[Path]:
    """ Looks for files in dir(not subdir) and returns them as a list """
    return [element for element in Path(path_to_dir).iterdir() if element.is_file()]


def get_all_dir_names_in_dir(path_to_dir) -> List[Path]:
    """ Looks for directories in dir and returns them as a list """
    return [element for element in Path(path_to_dir).iterdir() if element.is_dir()]


def is_not_empty_dir(directory) -> bool:
    return len(Path(directory).iterdir() != 0)


def read_all_lines_in_textfile(path_to_textfile) -> str:
    textfile = Path(path_to_textfile)
    if not textfile.exists() or not textfile.is_file():
        LOG.error(f"No file at location '{textfile}'")
        return ""
    return textfile.read_text()


def append_string_to_textfile(path_to_textfile, str_to_append) -> None:
    """ If file does not exist, a new file will be created. """
    with Path(path_to_textfile).open("a") as file_to_append:
        file_to_append.write(str_to_append)


def write_string_into_new_file(path_to_textfile, str_to_insert) -> None:
    """ An existing file will be truncated. """
    Path(path_to_textfile).write_text(str_to_insert)


def replace_keywords_in_textfile(
    path_to_textfile: Union[Path, str],
    dict_for_replacing: dict,
    new_output_path: Union[Path, str] = None,
) -> None:
    """
    Replaces all keywords in a textfile with the corresponding values in the dictionary.
    E.g.: A textfile with the content "%(This)s will be replaced!" and the dict {"This": "xyz"}
    results to the change "xyz will be replaced!" in the textfile.

    Args:
        path_to_textfile (Union[Path, str]): Path to the file containing text in which to replace
            elements.
        dict_for_replacing (dict): dictionary of elements to replace and the values to replace
            them with.
        new_output_path: Optional path to another file in which to write the result. If not
            provided, then the source file's content will be replaced.
    """
    destination_file = path_to_textfile if new_output_path is None else new_output_path
    all_lines = read_all_lines_in_textfile(path_to_textfile)
    lines_with_replaced_keys = all_lines % dict_for_replacing
    write_string_into_new_file(destination_file, lines_with_replaced_keys)


def json_dumps_readable(json_outfile, object_to_dump) -> None:
    """ This is how you write a beautiful json file
    
    Args:
        json_outfile: File to write
        object_to_dump: object to dump to json format
    """
    object_to_dump = json.dumps(object_to_dump)\
        .replace(", ", ",\n    ")\
        .replace("[", "[\n    ")\
        .replace("],\n    ", "],\n\n")\
        .replace("{", "{\n")\
        .replace("}", "\n}")
    with Path(json_outfile).open("w") as json_file:
        json_file.write(object_to_dump)


class PathOrStr(metaclass=get_instance_faker_meta(Path, str)):
    """ A class that behaves like a Path when possible, otherwise like a string."""
    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip("\'\"")  # behavior like dict-parser, IMPORTANT FOR EVERY STRING-FAKER
        return Path(value)


def convert_paths_in_dict_to_strings(dict_: dict) -> dict:
    """ Converts all Paths in the dict to strings,
     including those in iterables. """
    dict_ = dict_.copy()
    for key, value in dict_.items():
        if isinstance(value, Path):
            dict_[key] = str(value)
        else:
            try:
                list_ = list(value)
            except TypeError:
                pass
            else:
                has_changed = False
                for idx, item in enumerate(list_):
                    if isinstance(item, Path):
                        list_[idx] = str(item)
                        has_changed = True
                if has_changed:
                    dict_[key] = list_
    return dict_


def remove_none_dict_entries(dict_: dict) -> dict:
    """ Removes None entries from dict.
    This can be used as a workaround to
    https://github.com/pylhc/generic_parser/issues/26.
    """
    return {key: value for key, value in dict_.items() if value is not None}


def save_config(output_dir: Path, opt: dict, script: str):
    """  Quick wrapper for save_options_to_config.

    Args:
        output_dir (Path): Path to the output directory (does not need to exist)
        opt (dict): opt-structure to be saved
        script (str): path/name of the invoking script (becomes name of the .ini)
                      usually ``__file__``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    opt = remove_none_dict_entries(opt)  # temporary fix (see docstring)
    opt = convert_paths_in_dict_to_strings(opt)
    save_options_to_config(output_dir / formats.get_config_filename(script),
                           dict(sorted(opt.items()))
                           )

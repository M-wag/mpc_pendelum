import os
import sys

def add_parent_dir_to_sys_path():
    # Path to the directory of the current script
    current_script_path = os.path.dirname(os.path.abspath(__file__))

    # Path to the parent of the script's directory
    parent_dir = os.path.abspath(os.path.join(current_script_path, '..'))

    # Add the parent directory to sys.path if it's not already there
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)

# Call the function to add the parent directory to sys.path
add_parent_dir_to_sys_path()


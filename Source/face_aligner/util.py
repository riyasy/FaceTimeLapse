import os
import shutil


def clear_directory(directory):
    """Removes all files and subdirectories in the specified directory."""
    if os.path.exists(directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error clearing {item_path}: {e}")
        print(f"Cleared directory: {directory}")
    else:
        os.makedirs(directory)
        print(f"Created directory: {directory}")

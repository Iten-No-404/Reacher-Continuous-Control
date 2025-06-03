import os

def create_next_id_folder(base_path: str) -> str:
    """
    Creates a new directory with the next available integer ID inside the given base_path.

    For example, if base_path contains directories "1", "2", "test", it will
    create a new directory "3" and return its full path. Non-integer folder names
    are ignored.

    Args:
        base_path (str): The path to the parent directory where the new folder
                         should be created.

    Returns:
        str: The full path to the newly created directory.

    Raises:
        OSError: If there's an issue creating the directory.
    """
    # Ensure the base path exists
    try:
        os.makedirs(base_path, exist_ok=True)
    except OSError as e:
        print(f"Error ensuring base path '{base_path}' exists: {e}")
        raise

    existing_ids = []
    try:
        # List all items (files and directories) in the base_path
        for item_name in os.listdir(base_path):
            full_item_path = os.path.join(base_path, item_name)
            # Check if it's a directory and if its name can be converted to an integer
            if os.path.isdir(full_item_path):
                try:
                    existing_ids.append(int(item_name))
                except ValueError:
                    # Ignore items that are not valid integers (e.g., "test")
                    pass
    except FileNotFoundError:
        # This case should ideally not happen if os.makedirs(base_path, exist_ok=True) succeeds
        print(f"Warning: Base path '{base_path}' not found when listing contents. Proceeding with ID 0.")
    except OSError as e:
        print(f"Error listing contents of '{base_path}': {e}")
        raise

    # Determine the next available ID
    if existing_ids:
        next_id = max(existing_ids) + 1
    else:
        next_id = 0 # Start with 0 if no integer folders exist

    new_folder_name = str(next_id)
    new_folder_path = os.path.join(base_path, new_folder_name)

    # Create the new directory
    try:
        os.makedirs(new_folder_path)
        print(f"Successfully created directory: '{new_folder_path}'")
        return new_folder_path
    except OSError as e:
        print(f"Error creating new directory '{new_folder_path}': {e}")
        raise
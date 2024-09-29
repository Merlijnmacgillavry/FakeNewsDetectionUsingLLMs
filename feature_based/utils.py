import os


def create_file_with_directories(file_path, content=""):
    """
    Creates a file at the specified path, including any missing directories.

    Args:
        file_path (str): The path to the file to create.
        content (str, optional): The content to write to the file. Defaults to "".
    """

    # Get directory path from the file path
    directory_path = os.path.dirname(file_path)

    # Create directories if they don't exist
    try:
        os.makedirs(directory_path)
    except OSError as error:
        print(f"Error creating directories: {error}")
        pass

    # Open the file in write mode (creates it if it doesn't exist)
    try:
        with open(file_path, "w") as f:
            f.write(content)
        print(f"File created at: {file_path}")
    except OSError as error:
        print(f"Error creating file: {error}")

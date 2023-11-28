import os
import tarfile

# Define the directory where the files are located
source_directory = "/home/arpit/2g1n6qdydwa9u22shpxqzp0t8m/P34/rgb_frames"
# source_directory = "/home/arpit/test_projects/hoi-forecast/temp"
# Change this to your source directory

# List all the files in the source directory
files = [f for f in os.listdir(source_directory) if os.path.isfile(os.path.join(source_directory, f))]

# Create a folder for each file
for file in files:
    # Extract the file name without the extension
    filename, file_extension = os.path.splitext(file)
    
    # Create a folder with the same name as the file
    folder_path = os.path.join(source_directory, filename)
    os.makedirs(folder_path, exist_ok=True)
    
    # Move the file into the newly created folder
    source_file_path = os.path.join(source_directory, file)
    dest_file_path = os.path.join(folder_path, file)
    os.rename(source_file_path, dest_file_path)

print("Folders created for each file.")


# Walk through the source directory and its subdirectories
for root, dirs, files in os.walk(source_directory):
    print("root: ", root)
    print("dirs: ", dirs)
    print("files: ", files)
    dest_directory = root
    for file in files:

        source_file_path = os.path.join(root, file)
        with tarfile.open(source_file_path, "r") as tar:
            tar.extractall(path=dest_directory)

        os.remove(source_file_path)
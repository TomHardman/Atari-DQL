import os

def count_files(directory):
    """Counts files in directory"""
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        total_files += len(filenames)  # Count files in current directory
    return total_files


def delete_oldest_file(directory):
    """Delete the oldest file in the directory."""
    files = [(os.path.join(directory, f), os.path.getctime(os.path.join(directory, f))) for f in os.listdir(directory)]
    if files:
        oldest_file = min(files, key=lambda x: x[1])[0]
        print(f"Deleting oldest file: {oldest_file}")
        os.remove(oldest_file)
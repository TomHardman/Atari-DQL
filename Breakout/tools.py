import os

def count_files(directory):
    """Counts files in directory"""
    total_files = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        total_files += len(filenames)  # Count files in current directory
    return total_files


def delete_worst_model(directory):
    """Delete the oldest file in the directory."""
    files = [(os.path.join(directory, f), os.path.getctime(os.path.join(directory, f))) for f in os.listdir(directory)]
    if files:
        worst_model = min(files, key=lambda x: float(x[0][:-3].split('_')[-1]))[0]
        print(f"Deleting worst model: {worst_model}")
        os.remove(worst_model)
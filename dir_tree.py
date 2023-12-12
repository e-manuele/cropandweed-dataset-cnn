import os
def print_dir_tree(dirpath, indent=0):
    print(' ' * indent + dirpath)
    for dirname in os.listdir(dirpath):
        path = os.path.join(dirpath, dirname)
        if os.path.isdir(path):
            print_dir_tree(path, indent + 1)
        elif dirname.endswith('.csv') or dirname.endswith('.jpg') or dirname.endswith('.png'):
            continue  # Skip files with these extensions
        else:
            print(' ' * indent + dirname)

print_dir_tree('.')

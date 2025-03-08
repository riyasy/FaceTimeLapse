import os
from collections import defaultdict

def check_repeating_prefixes(directory):
    """
    Checks if the first 15 letters of filenames in a directory repeat and prints groups.
    
    Args:
        directory (str): Path to the directory to check.
    """
    if not os.path.isdir(directory):
        print(f"Error: '{directory}' is not a valid directory.")
        return

    # Dictionary to group files by their first 15 letters
    prefix_groups = defaultdict(list)

    # Iterate through files in the directory
    for filename in os.listdir(directory):
        # Use first 15 letters or full name if shorter
        prefix = filename[:15] if len(filename) >= 15 else filename
        prefix_groups[prefix].append(filename)

    # Print groups with more than one file
    found_repeats = False
    for prefix, files in prefix_groups.items():
        if len(files) > 1:  # Only print if there's a repeat
            found_repeats = True
            print(f"\nGroup with prefix '{prefix}':")
            for file in files:
                print(f"  - {file}")

    if not found_repeats:
        print(f"No filenames with repeating first 15 letters found in '{directory}'.")

# Example usage
if __name__ == "__main__":
    check_repeating_prefixes("/Users/riyasyoosuf/Desktop/Input/Phase2_FaceVideos_CleanedUp/FaceVideos_CFR_Aligned")
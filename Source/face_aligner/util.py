import datetime
import os
import shutil

from datetime import datetime  

def get_filtered_videos(video_dir, date_from="20200101_000000", date_to="20290101_000000"):
    """
    Returns a sorted list of video files from video_dir that:
    1. Follow the format YYYYMMDD_HHMMSS
    2. Fall between date_from and date_to (inclusive)
    
    Args:
        video_dir (str): Directory containing video files
        date_from (str): Start date in format YYYYMMDD_HHMMSS
        date_to (str): End date in format YYYYMMDD_HHMMSS
    
    Returns:
        list: Sorted list of filtered filenames
    """
    # Convert boundary dates to datetime objects for comparison
    try:
        dt_from = datetime.strptime(date_from, "%Y%m%d_%H%M%S")
        dt_to = datetime.strptime(date_to, "%Y%m%d_%H%M%S")
    except ValueError as e:
        raise ValueError("Invalid date format in date_from or date_to. Use YYYYMMDD_HHMMSS format") from e

    filtered_files = []
    
    # Get all files in directory
    for filename in os.listdir(video_dir):
        # Check if filename matches the pattern YYYYMMDD_HHMMSS
        if len(filename) < 15:  # Minimum length for YYYYMMDD_HHMMSS
            continue
            
        date_part = filename[:15]  # Extract YYYYMMDD_HHMMSS part
        
        try:
            # Try to parse the date from filename
            file_dt = datetime.strptime(date_part, "%Y%m%d_%H%M%S")
            
            # Check if file date falls within range
            if dt_from <= file_dt <= dt_to:
                filtered_files.append(filename)
                
        except ValueError:
            # Skip files that don't match the date format
            continue
    
    # Return sorted list
    return sorted(filtered_files)

def get_eye_to_screen_ratio(filename):
    """
    Returns an eye distance to screen ratio based on the date in the filename.
    
    Args:
        filename (str): Filename in format YYYYMMDD_HHMMSS...
    
    Returns:
        float: Eye distance to screen ratio
    """
    # Default ratio for invalid formats or other dates
    DEFAULT_RATIO = 0.075
    
    # Check if filename has minimum length and extract date part
    if len(filename) < 15:
        return DEFAULT_RATIO
        
    date_part = filename[:15]  # YYYYMMDD_HHMMSS
    
    try:
        # Parse the date from filename
        file_dt = datetime.strptime(date_part, "%Y%m%d_%H%M%S")
        year = file_dt.year
        month = file_dt.month
        
        # 2020 or 2021: ratio 0.1
        if year == 2020:
            return 0.1
        
        if year == 2021:
            return 0.095
                   

        if year == 2022:
            return 0.085
            
        # 2023 Jan-May: ratio 0.08
        if year == 2023 and 1 <= month <= 5:
            return 0.08
            
        # Any other date: ratio 0.08
        return DEFAULT_RATIO
        
    except ValueError:
        # Return default ratio if date parsing fails
        return DEFAULT_RATIO
    

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

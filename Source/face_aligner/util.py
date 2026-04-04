import os
import shutil
from datetime import datetime

# Eye-to-screen ratios keyed by year/period.
# This ratio controls how much of the output width the inter-eye distance spans.
# Larger value → face appears bigger in the output frame.
EYE_RATIO_BY_YEAR = {
    2020: 0.100,
    2021: 0.095,
    2022: 0.085,
}
EYE_RATIO_2023_JAN_MAY = 0.080   # First half of 2023
EYE_RATIO_DEFAULT      = 0.075   # All other dates

# Date-range filter defaults used by get_filtered_videos()
DEFAULT_DATE_FROM = "20200101_000000"
DEFAULT_DATE_TO   = "20290101_000000"


def get_filtered_videos(video_dir, date_from=DEFAULT_DATE_FROM, date_to=DEFAULT_DATE_TO):
    """
    Returns a sorted list of filenames in video_dir whose names begin with
    a timestamp matching 'YYYYMMDD_HHMMSS' and fall within [date_from, date_to].

    Files that don't follow the naming convention are silently skipped.
    """
    try:
        dt_from = datetime.strptime(date_from, "%Y%m%d_%H%M%S")
        dt_to   = datetime.strptime(date_to,   "%Y%m%d_%H%M%S")
    except ValueError as e:
        raise ValueError("date_from / date_to must be in YYYYMMDD_HHMMSS format") from e

    filtered = []
    for filename in os.listdir(video_dir):
        if len(filename) < 15:
            continue  # Too short to contain a valid timestamp

        try:
            # The first 15 characters encode YYYYMMDD_HHMMSS
            file_dt = datetime.strptime(filename[:15], "%Y%m%d_%H%M%S")
            if dt_from <= file_dt <= dt_to:
                filtered.append(filename)
        except ValueError:
            continue  # Filename doesn't start with a recognised timestamp

    return sorted(filtered)


def get_eye_to_screen_ratio(filename):
    """
    Returns the eye-distance-to-output-width ratio for a given filename.

    The ratio is chosen based on the date embedded in the filename
    (format: YYYYMMDD_HHMMSS…) because the subject's camera distance
    varied across recording sessions. Returns DEFAULT if parsing fails.
    """
    if len(filename) < 15:
        return EYE_RATIO_DEFAULT

    try:
        file_dt = datetime.strptime(filename[:15], "%Y%m%d_%H%M%S")
        year, month = file_dt.year, file_dt.month

        if year in EYE_RATIO_BY_YEAR:
            return EYE_RATIO_BY_YEAR[year]

        if year == 2023 and 1 <= month <= 5:
            return EYE_RATIO_2023_JAN_MAY

        return EYE_RATIO_DEFAULT

    except ValueError:
        return EYE_RATIO_DEFAULT  # Filename doesn't start with a valid timestamp


def clear_directory(directory):
    """
    Empties all contents of directory (files and sub-folders).
    Creates the directory if it doesn't already exist.
    """
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
        print(f"Cleared: {directory}")
    else:
        os.makedirs(directory)
        print(f"Created: {directory}")

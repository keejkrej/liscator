import os
import numpy as np
import numba as nb
import scipy
import h5py
import skimage as sk
import cv2
import pandas as pd
import re
import math


# import matplotlib.pyplot as plt

import pathlib

import scipy.ndimage as smg
from nd2reader import ND2Reader

STRUCT3 = np.ones((3,3), dtype=np.bool_)
STRUCT5 = np.ones((5,5), dtype=np.bool_)
STRUCT5[[0,0,-1,-1], [0,-1,0,-1]] = False

@nb.njit
def window_std(img: np.ndarray) -> float:
    """
    Calculate unnormed variance of 'img'
    Refer to https://en.wikipedia.org/wiki/Variance#Unbiased_sample_variance
    Refer to Pyama https://github.com/SoftmatterLMU-RaedlerGroup/pyama/tree/master

    Parameters:
    img (np.ndarray): Input image

    Returns:
    float: Unnormed variance of the image
    """
    return np.sum((img - np.mean(img))**2)

@nb.njit
def generic_filter(img: np.ndarray, fun: callable, size: int = 3, reflect: bool = False) -> np.ndarray:
    """
    Apply filter to image.

    Parameters:
    img (np.ndarray): The image to be filtered
    fun (callable): The filter function to be applied, must accept subimage of 'img' as only argument and return a scalar. "Fun" stands for function and callable should stand for function in Python
    size (int): The size (side length) of the kernel. Must be an odd integer
    reflect (bool): Switch for border mode: True for 'reflect', False for 'mirror'. Reflect and Mirror should be filling the borders of the img.

    Returns:
    np.ndarray: Filtered image as a np.float64 array with same shape as 'img'

    Raises:
    ValueError: If 'size' is not an odd integer
    """
    if size % 2 != 1:
        raise ValueError("'size' must be an odd integer")
    height, width = img.shape
    s2 = size // 2

    # Set up temporary image for correct border handling
    img_temp = np.empty((height+2*s2, width+2*s2), dtype=np.float64)
    img_temp[s2:-s2, s2:-s2] = img
    if reflect:
        img_temp[:s2, s2:-s2] = img[s2-1::-1, :]
        img_temp[-s2:, s2:-s2] = img[:-s2-1:-1, :]
        img_temp[:, :s2] = img_temp[:, 2*s2-1:s2-1:-1]
        img_temp[:, -s2:] = img_temp[:, -s2-1:-2*s2-1:-1]
    else:
        img_temp[:s2, s2:-s2] = img[s2:0:-1, :]
        img_temp[-s2:, s2:-s2] = img[-2:-s2-2:-1, :]
        img_temp[:, :s2] = img_temp[:, 2*s2:s2:-1]
        img_temp[:, -s2:] = img_temp[:, -s2-2:-2*s2-2:-1]

    # Create and populate result image
    filtered_img = np.empty_like(img, dtype=np.float64)
    for y in range(height):
        for x in range(width):
            filtered_img[y, x] = fun(img_temp[y:y+2*s2+1, x:x+2*s2+1])

    return filtered_img


def binarize_frame(img: np.ndarray, mask_size: int = 3) -> np.ndarray:
    """
    Coarse segmentation of phase-contrast image frame
    Refer to OpenCV tutorials for more information on binarization/thresholding techniques.

    Parameters:
    img (np.ndarray): The image to be binarized
    mask_size (int): The size of the mask to be used in the binarization process (mask refers to kernel size in image processing)

    Returns:
    np.ndarray: Binarized image of frame
    """
    # Get logarithmic standard deviation at each pixel
    std_log = generic_filter(img, window_std, size=mask_size)
    std_log[std_log>0] = (np.log(std_log[std_log>0]) - np.log(mask_size**2 - 1)) / 2

    # Get width of histogram modulus
    counts, edges = np.histogram(std_log, bins=200)
    bins = (edges[:-1] + edges[1:]) / 2
    hist_max = bins[np.argmax(counts)]
    sigma = np.std(std_log[std_log <= hist_max])

    # Apply histogram-based threshold
    img_bin = std_log >= hist_max + 3 * sigma

    # Remove noise
    img_bin = smg.binary_dilation(img_bin, structure=STRUCT3)
    img_bin = smg.binary_fill_holes(img_bin)
    img_bin &= smg.binary_opening(img_bin, iterations=2, structure=STRUCT5)
    img_bin = smg.binary_erosion(img_bin, border_value=1)

    return img_bin

def csv_output(out_dir: str, pos: list, mins: float, use_square_rois: bool = True) -> None:
    """
    Generate CSV output for tracked positions

    Parameters:
    out_dir (str): Output directory path
    pos (list): List of positions to process
    mins (float): Minutes per frame
    use_square_rois (bool): Whether to use square ROIs

    Returns:
    None
    """
    folders = get_tracked_folders(out_dir,pos)
    for folder in folders:
        csv_output_position(folder[0],folder[1],mins,use_square_rois)

def csv_output_position(pos: int, pos_path: pathlib.Path, mins: float, use_square_rois: bool) -> None:
    """
    Generate CSV output for a single position

    Parameters:
    pos (int): Position number
    pos_path (pathlib.Path): Path to position directory
    mins (float): Minutes per frame
    use_square_rois (bool): Whether to use square ROIs

    Returns:
    None
    """
    tracks_path = pos_path.joinpath('tracks.csv')
    tracks = pd.read_csv(tracks_path.absolute(),index_col=0)

    data_path = pos_path.joinpath('data.h5')

    with h5py.File(data_path.absolute(), "r") as data:
        frames = range(data.attrs['frame_min'],data.attrs['frame_max']+1)
        fl_channel_names = data.attrs['fl_channel_names']

    excel_path = pos_path.joinpath('output.xlsx')

    particles = [int(p) for p in tracks['particle'].unique()]
    particles.sort()

    print("Starting Data Export for position:",str(pos))

    with pd.ExcelWriter(excel_path.absolute()) as writer:
        if use_square_rois == True and 'square_area' in tracks:
            area = csv_get_table(particles,tracks,frames,mins,'square_area')
        else:
            area = csv_get_table(particles,tracks,frames,mins,'area')
        area.to_excel(writer, sheet_name='Area', index=False)

        for i in range(len(fl_channel_names)):
            col_name = 'brightness_' + str(i)
            if use_square_rois == True and 'square_' + col_name in tracks:
                brightness = csv_get_table(particles,tracks,frames,mins,'square_' + col_name)
            else:
                brightness = csv_get_table(particles,tracks,frames,mins,col_name)
            brightness.to_excel(writer, sheet_name=fl_channel_names[i], index=False)

            table_to_image(pos_path,particles,brightness,fl_channel_names[i])

    print('Done')

def table_to_image(pos_path: pathlib.Path, particles: list, table: pd.DataFrame, name: str) -> None:
    """
    Convert table data to image and save it.
    This is a post-processing step.
    These converts the table data to the fluorescent tracks image.

    Parameters:
    pos_path (pathlib.Path): Path to position directory
    particles (list): List of particle IDs
    table (pd.DataFrame): Data table
    name (str): Name for the output file
    """
    # Import matplotlib and set backend at the start of the function
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Create figure without displaying it
    fig = plt.figure()

    for p in particles:
        plt.plot(table['time'].values, table[str(p)].values, color='gray', alpha=0.5)

    plt.xlabel('Time (Frame)')
    plt.ylabel('Brightness (Pixelsum)')
    plt.title(name)
    plt.tight_layout()

    # Save figure and close it
    fig.savefig(pos_path.joinpath(name + '.png').absolute())
    plt.close(fig)


def csv_get_table(particles: list, tracks: pd.DataFrame, frames: list, mins: float, col: str) -> pd.DataFrame:
    """
    Post-processing step that converts from TrackPy to Pyama format.
    Extract data from tracks and create a table.

    Parameters:
    particles (list): List of particle IDs
    tracks (pd.DataFrame): Tracking data
    frames (list): List of frame numbers
    mins (float): Minutes per frame
    col (str): Column name to extract

    Returns:
    pd.DataFrame: Extracted data table
    """
    keys = []
    keys.append('time')
    for p in particles:
        keys.append(str(p))

    data = {}
    for key in keys:
        data[key] = []

    print('Fetching Data:', col)
    for f in frames:
        #print('Frame',f)
        data['time'].append(f * mins / 60)
        for p in particles:
            t = tracks[(tracks['particle'] == p) & (tracks['frame'] == f)]
            if len(t) > 0:
                data[str(p)].append(t.iloc[0][col])
            else:
                # change this behaviour if memory during tracking is used
                data[str(p)].append(0)

    return pd.DataFrame(data)


def square_roi(out_dir: str, pos: list, micron_size: float) -> None:
    """
    Post-processing step where the micron_size defines the length of the squares.
    Apply square ROI to tracked positions

    Parameters:
    out_dir (str): Output directory path
    pos (list): List of positions to process
    micron_size (float): Size of ROI in microns

    Returns:
    None
    """
    folders = get_tracked_folders(out_dir,pos)
    print(folders)
    for folder in folders:
        square_roi_position(folder[0],folder[1],micron_size)

def square_roi_position(pos: int, pos_path: pathlib.Path, micron_size: float) -> None:
    """
    Post-processing step where the micron_size defines the length of the squares.
    Apply square ROI to a single position.

    Parameters:
    pos (int): Position number
    pos_path (pathlib.Path): Path to position directory
    micron_size (float): Size of ROI in microns

    Returns:
    None
    """
    tracks_path = pos_path.joinpath('tracks.csv')
    tracks = pd.read_csv(tracks_path.absolute(),index_col=0)

    data_path = pos_path.joinpath('data.h5')
    data = h5py.File(data_path.absolute(), "r")

    size = math.ceil(micron_size / data.attrs['pixel_microns'])

    width,height = data.attrs['width'],data.attrs['height']

    print("Starting Square ROIs for position:",str(pos))

    for frame in sorted(tracks['frame'].unique()):
        frame_data_index = frame-data.attrs['frame_min']
        print("Frame",str(int(frame)))

        # t = tracks[(tracks['frame'] == frame) & (tracks['enabled'] == True)]
        true_values = [1, '1', '1.0', 1.0, True, 'True', 'true', 'TRUE']

        # Convert true_values to lowercase strings
        true_values_lower = [str(v).lower() for v in true_values]

        # Create the enabled condition:
        # 1. Convert the enabled column to string
        # 2. Convert all values to lowercase
        # 3. Check if they're in our true_values list
        enabled_condition = tracks['enabled'].astype(str).str.lower().isin(true_values_lower)

        # Apply both conditions to filter the dataframe
        t = tracks[
            (tracks['frame'] == frame) & (enabled_condition)]


        for index, record in t.iterrows():

            x = int((record['bbox_x1'] + record['bbox_x2']) // 2)
            y = int((record['bbox_y1'] + record['bbox_y2']) // 2)

            x1 = max(0,x - size)
            y1 = max(0,y - size)

            x2 = min(height-1,x + size)
            y2 = min(width-1,y +  size)

            tracks.loc[(tracks['frame'] == frame) & (tracks['particle'] == record['particle']), 'square_area'] = (x2-x1) * (y2-y1)
            for i in range(len(data.attrs['fl_channels'])):

                im_slice = data['fluorescence'][int(frame_data_index),i][x1:x2,y1:y2]
                tracks.loc[(tracks['frame'] == frame) & (tracks['particle'] == record['particle']), 'square_brightness_' + str(i)] = im_slice.sum()

    data.close()
    tracks.to_csv(tracks_path.absolute())
    print("Done")


# to be deprecated since the function above is the only one being used.
def square_roi_position_old(nd2_path: str, out_dir: str, pos: int, micron_size: float) -> None:
    """
    Old version of square ROI application to a single position

    Parameters:
    nd2_path (str): Path to ND2 file
    out_dir (str): Output directory path
    pos (int): Position number
    micron_size (float): Size of ROI in microns

    Returns:
    None
    """
    if not pathlib.Path(nd2_path).is_file():
        print("Invalid ND2 Path")
        return

    nd2 = ND2Reader(nd2_path)

    pos_dir = position_path(out_dir,pos)
    if pos_dir is None:
        print("Could not find directory")
        return

    tracks_path = pos_dir.joinpath('tracks.csv')
    if not tracks_path.is_file():
        print("Could not find features.csv")
        return
    tracks = pd.read_csv(tracks_path.absolute())

    # pixel_microns: the amount of microns per pixel
    size = math.ceil(micron_size / nd2.metadata['pixel_microns'])

    data_path = pos_dir.joinpath('data.h5')
    if not data_path.is_file():
        print("Could not find data.h5")
        return


    tracks['square_mass'] = 0
    data = h5py.File(data_path.absolute(), "r")

    width,height = nd2.metadata['width'],nd2.metadata['height']

    print("Starting Square ROIs for position " + str(pos))
    for frame in sorted(tracks['frame'].unique()):
        frame_data_index = frame-data.attrs['frame_min']
        print("Frame " + str(int(frame)))

        t = tracks[(tracks['frame'] == frame) & (tracks['enabled'] == True)]
        #t = tracks[(tracks['frame'] == frame)]
        fl_image = data['bg_corr'][int(frame_data_index)]

        for index, record in t.iterrows():
            x = int((record['bbox_x1'] + record['bbox_x2']) // 2)
            y = int((record['bbox_y1'] + record['bbox_y2']) // 2)

            x1 = max(0,x - size)
            y1 = max(0,y - size)

            x2 = min(height-1,x + size)
            y2 = min(width-1,y +  size)

            im_slice = fl_image[x1:x2,y1:y2]
            tracks.loc[(tracks['frame'] == frame) & (tracks['particle'] == record['particle']), 'square_mass'] = im_slice.sum()

    data.close()
    tracks.to_csv(tracks_path.absolute())
    print("Done")


def get_position_folders(out_dir: str) -> list:
    """
    Get a list of position folders from the output directory

    Parameters:
    out_dir (str): Output directory path

    Returns:
    list: List of tuples containing position number and path
    """
    folders =  []
    for path in pathlib.Path(out_dir).iterdir():
        if not path.is_dir():
            continue
        if not re.search('^XY0*\\d+$', path.name):
            continue

        number_str = path.name[2:].lstrip('0')
        pos = int(number_str) if number_str else 0
        folders.append((pos,path))
    return folders

def get_tracking_folders(out_dir: str, pos: list) -> list:
    """
    Get a list of tracking folders for specified positions

    Parameters:
    out_dir (str): Output directory path
    pos (list): List of position numbers

    Returns:
    list: List of tuples containing position number and path
    """
    pos = list(set(pos))
    pos_folders = get_position_folders(out_dir)

    if len(pos) > 0:
        pos_folders = [p for p in pos_folders if p[0] in pos]

    folders = []
    for folder in pos_folders:
        features_path = folder[1].joinpath('features.csv')
        if not features_path.is_file():
            print("Position " + str(folder[0]) + ":", "Could not find features.csv")
            continue

        data_path = folder[1].joinpath('data.h5')
        if not data_path.is_file():
            print("Position " + str(folder[0]) + ":", "Could not find data.h5")
            continue

        folders.append(folder)
    return folders

def get_tracked_folders(out_dir: str, pos: list) -> list:
    """
    Get a list of tracked folders for specified positions

    Parameters:
    out_dir (str): Output directory path
    pos (list): List of position numbers

    Returns:
    list: List of tuples containing position number and path
    """
    pos = list(set(pos))
    pos_folders = get_position_folders(out_dir)

    if len(pos) > 0:
        pos_folders = [p for p in pos_folders if p[0] in pos]

    folders = []
    for folder in pos_folders:
        features_path = folder[1].joinpath('features.csv')
        if not features_path.is_file():
            print("Position " + str(folder[0]) + ":", "Could not find features.csv")
            continue

        data_path = folder[1].joinpath('data.h5')
        if not data_path.is_file():
            print("Position " + str(folder[0]) + ":", "Could not find data.h5")
            continue

        tracks_path = folder[1].joinpath('tracks.csv')
        if not tracks_path.is_file():
            print("Position " + str(folder[0]) + ":", "Could not find tracks.csv")
            continue

        folders.append(folder)
    return folders

def tracking_pyama(out_dir: str, pos: list, expand: int = 0) -> None:
    """
    Perform Pyama tracking on specified positions and saves them into the output directory

    Parameters:
    out_dir (str): Output directory path
    pos (list): List of position numbers
    expand (int): Expansion factor for labels

    Returns:
    None
    """
    folders = get_tracking_folders(out_dir,pos)
    for folder in folders:
        track_position_pyama(folder[0],folder[1],expand)

def track_position_pyama(pos: int, pos_path: pathlib.Path, expand: int) -> None:
    """
    Perform Pyama tracking on a single position.
    data.h5 contains the segmentation and the background corrected fluorescence images.
    features.csv contains the features of the particles. Bounding boxes, integrated fluorescence.
    The track is being saved as tracks.csv file

    Parameters:
    pos (int): Position number
    pos_path (pathlib.Path): Path to position directory
    expand (int): Expansion factor for labels

    Returns:
    None
    """
    features_path = pos_path.joinpath('features.csv')
    features = pd.read_csv(features_path.absolute(),index_col=0)

    data_path = pos_path.joinpath('data.h5')
    data = h5py.File(data_path.absolute(), "r")
    data_labels = data['labels']

    min_track_length = data.attrs['frame_max']-data.attrs['frame_min']+1

    tracks = []

    frames = features['frame'].unique()
    frames.sort()

    print("Starting Pyama Tracking for position " + str(pos))
    for frame in frames:
        print("Frame " + str(frame))
        frame_data_index = frame-data.attrs['frame_min']
        frame_features = features[features['frame'] == frame]
        if len(tracks) == 0:
            for index, row in frame_features.iterrows():
                tracks.append([row])
        else:
            matched_labels = []

            frame_labels = data_labels[frame_data_index]
            prev_labels = data_labels[frame_data_index-1]

            # Add optional label expansion here
            if expand > 0:
                frame_labels = sk.segmentation.expand_labels(frame_labels,expand)
                prev_labels = sk.segmentation.expand_labels(prev_labels,expand)

            # Add frames left to check if len(track) + frames_left < min_frames
            remove_indices = []
            for i in range(len(tracks)):
                track = tracks[i]
                prev_row = track[len(track)-1]

                # no memory so ignore any lost
                if frame - prev_row['frame'] > 1:
                    remove_indices.append(i)

                    # dont add track to completed (we only want entire tracks)
                    continue

                #frame_labels = data_labels[frame_data_index]
                #prev_labels = data_labels[frame_data_index-1]

                # Add optional label expansion here
                #if expand > 0:
                    #frame_labels = sk.segmentation.expand_labels(frame_labels,expand)
                    #prev_labels = sk.segmentation.expand_labels(prev_labels,expand)

                label_slice = frame_labels[prev_labels == prev_row['label']]
                found_labels = sorted(np.unique(label_slice))

                if len(found_labels) > 0 and found_labels[0] == 0:
                    found_labels.pop(0)

                if len(found_labels) == 0:
                    # No match for this track
                    continue

                #print(found_labels)

                local_matches = []
                for label in found_labels:
                    row = frame_features[frame_features['label'] == label].iloc[0]

                    # already found parent
                    if row['label'] in matched_labels:
                        continue

                    local_matches.append({'s': row['area'], 'r': row})

                if len(local_matches) > 0:
                    local_matches = sorted(local_matches, key=lambda r: r['s'], reverse=True)
                    selected_match = local_matches[0]

                    track.append(selected_match['r'])
                    matched_labels.append(selected_match['r']['label'])

            # Remove tracks that can be ignored
            if len(remove_indices) > 0:
                remove_indices.reverse()
                for index in remove_indices:
                    tracks.pop(index)


            unmatched_rows = frame_features[~np.isin(frame_features['label'],matched_labels)]
            for index, row in unmatched_rows.iterrows():
                tracks.append([row])

    data.close()

    result_data = []

    particle_id = 0
    for track in tracks:
        if len(track) < min_track_length:
            continue



        for row in track:
            row['particle'] = particle_id
            row['enabled'] = True
            result_data.append(row)

        particle_id += 1

    tracks = pd.DataFrame(result_data)

    # Find large particles and disable
    large_particles = tracks[tracks['area'] > 10000]['particle'].unique()
    tracks.loc[np.isin(tracks['particle'], large_particles), 'enabled'] = False

    tracks_path = pos_path.joinpath('tracks.csv')
    tracks.to_csv(tracks_path.absolute())
    print("Done")


def position_path(out_dir: str, pos: int) -> pathlib.Path:
    """
    Get the path for a specific position

    Parameters:
    out_dir (str): Output directory path
    pos (int): Position number

    Returns:
    pathlib.Path: Path to the position directory
    """
    for path in pathlib.Path(out_dir).iterdir():
        if not path.is_dir():
            continue
        if not re.search('XY0*' + str(pos) + '$', path.name):
            continue
        return path
    return None


def pyama_segmentation(img: np.ndarray) -> np.ndarray:
    """
    Perform Pyama segmentation on an image

    Parameters:
    img (np.ndarray): Input image

    Returns:
    np.ndarray: Labeled segmentation of the image
    """
    binary_segmentation = binarize_frame(img)

    # remove small objects MIN_SIZE=1000
    sk.morphology.remove_small_objects(binary_segmentation,min_size=1000,out=binary_segmentation)

    # convert binary mask to labels (1,2,3,...)
    return sk.measure.label(binary_segmentation, connectivity=1)

def segment_positions(nd2_path: str, out_dir: str, pos: list, seg_channel: int, fl_channels: list, frame_min: int = None, frame_max: int = None, bg_corr: bool = True) -> None:
    """
    Segment positions from an ND2 file

    Parameters:
    nd2_path (str): Path to ND2 file
    out_dir (str): Output directory path
    pos (list): List of position numbers
    seg_channel (int): Segmentation channel index
    fl_channels (list): List of fluorescence channel indices
    frame_min (int): Minimum frame number
    frame_max (int): Maximum frame number
    bg_corr (bool): Whether to perform background correction

    Returns:
    None
    """
    if not pathlib.Path(nd2_path).is_file():
        print("Invalid ND2 Path")
        return

    fl_channels = list(set(fl_channels))
    pos = list(set(pos)) # remove duplicates

    nd2 = ND2Reader(nd2_path)

    if seg_channel < 0 or seg_channel > len(nd2.metadata['channels']) - 1:
        print("Invalid Segmentation Channel")
        return

    for c in fl_channels:
        if c < 0 or c > len(nd2.metadata['channels']) - 1:
            print("Invalid Fluorescence Channel")
            return

    positions = list(nd2.metadata['fields_of_view'])
    if len(pos) > 0:
        positions = [p for p in positions if p in pos]

    if len(positions) == 0:
        print("Invalid Positions")
        return

    fl_channel_names = [nd2.metadata['channels'][c] for c in fl_channels]

    try:
        # Check and calculate padding
        max_field = max(nd2.metadata['fields_of_view'])
        if max_field > 0:
            padding = int(np.ceil(np.log10(max_field)))
        else:
            # Save metadata to a text file
            with open("metadata_output.txt", "w") as file:
                file.write(str(nd2.metadata))

            print("Warning: fields_of_view contains zero or negative values.")
            padding = 0  # or any default you prefer
    except KeyError:
        print("Error: 'fields_of_view' key not found in metadata.")
        padding = 0  # or any default you prefer



    frames = list(nd2.metadata['frames'])

    if frame_min is not None:
        if frame_min not in frames:
            print('Invalid frame_min')
            return
    else:
        frame_min = frames[0]

    if frame_max is not None:
        if frame_max not in frames:
            print('Invalid frame_max')
            return
    else:
        frame_max = frames[-1]

    if frame_max < frame_min:
        print('frame_max must be greater or equal to frame_min')
        return

    frames = [f for f in frames if frame_min <= f <= frame_max]

    width, height, num_frames = nd2.metadata['width'], nd2.metadata['height'], len(frames)

    print('Segmentation Channel: ' + nd2.metadata['channels'][seg_channel])
    print('Fluorescence Channels: ' + ', '.join(fl_channel_names))

    for pos in positions:
        print(f"Segmenting position {pos}")
        pos_dir = pathlib.Path(out_dir).joinpath(f'XY{str(pos).zfill(padding)}')
        pos_dir.mkdir(parents=True, exist_ok=True)

        file_path = pos_dir.joinpath('data.h5')

        feature_keys = ['x', 'y'] + [f'brightness_{i}' for i in range(len(fl_channels))] + ['area', 'frame', 'label', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2']
        feature_data = {key: [] for key in feature_keys}

        with h5py.File(file_path.absolute(), "w") as file_handle:
            data_labels = file_handle.create_dataset('labels', (num_frames, height, width), dtype=np.uint16, chunks=(1, height, width))
            data_fl = file_handle.create_dataset('fluorescence', (num_frames, len(fl_channels), height, width), dtype=np.float64, chunks=(1, 1, height, width))

            file_handle.attrs['frame_min'] = frame_min
            file_handle.attrs['frame_max'] = frame_max
            file_handle.attrs['seg_channel'] = seg_channel
            file_handle.attrs['fl_channels'] = fl_channels
            file_handle.attrs['fl_channel_names'] = fl_channel_names
            file_handle.attrs['width'] = nd2.metadata['width']
            file_handle.attrs['height'] = nd2.metadata['height']
            file_handle.attrs['pixel_microns'] = nd2.metadata['pixel_microns']

            for index, frame in enumerate(frames):
                frame_image = nd2.get_frame_2D(t=frame, c=seg_channel, v=pos)
                binary_segmentation = binarize_frame(frame_image)

                sk.morphology.remove_small_objects(binary_segmentation, min_size=1000, out=binary_segmentation)
                label_segmentation = sk.measure.label(binary_segmentation, connectivity=1)

                frame_fl_images = []
                for c in fl_channels:
                    frame_fl_image = nd2.get_frame_2D(t=frame, c=c, v=pos)
                    if bg_corr:
                        frame_fl_image = background_correction(frame_fl_image, label_segmentation, 5, 5, 0.5)
                    frame_fl_images.append(frame_fl_image)

                props = sk.measure.regionprops(label_segmentation)
                print(f"Frame {frame}: {len(props)} features")

                for prop in props:
                    if prop.bbox[0] == 0 or prop.bbox[1] == 0 or prop.bbox[2] == height or prop.bbox[3] == width:
                        label_segmentation[label_segmentation == prop.label] = 0
                        continue

                    x, y = prop.centroid
                    feature_data['x'].append(x)
                    feature_data['y'].append(y)

                    for i, fl_image in enumerate(frame_fl_images):
                        feature_data[f'brightness_{i}'].append(fl_image[tuple(prop.coords.T)].sum())

                    feature_data['area'].append(prop.area)
                    feature_data['frame'].append(frame)
                    feature_data['label'].append(prop.label)
                    feature_data['bbox_x1'].append(prop.bbox[0])
                    feature_data['bbox_y1'].append(prop.bbox[1])
                    feature_data['bbox_x2'].append(prop.bbox[2] - 1)
                    feature_data['bbox_y2'].append(prop.bbox[3] - 1)

                data_labels[index, :, :] = label_segmentation
                for i, fl_image in enumerate(frame_fl_images):
                    data_fl[index, i, :, :] = fl_image

        features_path = pos_dir.joinpath('features.csv')
        features = pd.DataFrame(feature_data)
        features.to_csv(features_path.absolute())

    print("Done")

def background_spline(image, img_mask, countX, countY, overlap):
    """
    Creates a background model using a grid of sampling points and spline interpolation.

    Used for background correction of microscopy images by modeling systematic
    illumination variations. Part of the pipeline for processing fluorescence data.

    Parameters:
    image (np.ndarray): Input microscopy image
    img_mask (np.ndarray): Binary mask of regions to exclude (e.g. cells)
    countX (int): Number of grid points in X direction
    countY (int): Number of grid points in Y direction
    overlap (float): Overlap between grid windows (0-1)

    Returns:
    np.ndarray: Interpolated background map same size as input image
    """
    # Get image dimensions
    h,w = image.shape

    # Calculate size of sampling windows based on grid density and overlap
    sizeX = int(w/((countX - (countX-1)*overlap)*2))
    sizeY = int(h/((countY - (countY-1)*overlap)*2))

    # Create grid points for sampling background
    pointsX = np.linspace(sizeX,w-(sizeX),countX).astype(int)
    pointsY = np.linspace(sizeY,h-(sizeY),countY).astype(int)

    # Create masked array to ignore foreground objects
    masked_img = np.ma.masked_array(image, mask=img_mask)

    # Sample background at each grid point
    pos = []
    vals = []
    for ix in range(len(pointsX)):
        for iy in range(len(pointsY)):
            x = pointsX[ix]
            y = pointsY[iy]

            # Get sampling window boundaries
            x1,x2 = max(0,x-sizeX),min(w-1,x+sizeX)
            y1,y2 = max(0,y-sizeY),min(h-1,y+sizeY)

            # Extract window and calculate statistics
            sub_image = masked_img[y1:y2,x1:x2]
            vals.append([np.ma.mean(sub_image),np.ma.median(sub_image),np.ma.var(sub_image)])
            pos.append([x,y,ix,iy])

    # Convert to numpy arrays
    vals = np.array(vals)
    pos = np.array(pos)

    # Create support points for spline interpolation using median values
    fit_support = np.empty((countX, countY))
    for i in range(len(pos)):
        fit_support[pos[i,2],pos[i,3]] = vals[i,1]

    # Interpolate background using bicubic spline
    bg_spline = scipy.interpolate.RectBivariateSpline(x=pointsX, y=pointsY, z=fit_support)
    return bg_spline(x=range(w), y=range(h)).T

def background_correction(image,img_mask,countX,countY,overlap = 0.1):
    """

    """

    h,w = image.shape
    patch = background_spline(image,img_mask,countX,countY,overlap)
    bg_mean = patch.mean()

    bg_interp = patch.copy()

    A = np.divide(bg_interp, bg_mean)
    bg_interp = np.subtract(image, bg_interp)
    bg_interp = np.divide(bg_interp, np.median(A, axis=0, keepdims=True))

    return bg_interp


def moonraedler_dir():
    p = pathlib.Path('/project/ag-moonraedler')
    if p.is_dir():
        return p.absolute()

    p = pathlib.Path('//z-sv-dfsroot.ad.physik.uni-muenchen.de/dfsextern/project/ag-moonraedler')
    if p.is_dir():
        return p.absolute()

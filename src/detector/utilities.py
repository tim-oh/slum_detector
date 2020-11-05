import pickle
import numpy as np
from matplotlib import pyplot as plt
import imageio
import os


def scale_distance(distance, dist_thresh=0.5):
    """
    Scale pseudo-probability values of slums-world predictions to pixel classes.

    The 'distance_estimated' of slums-world is actually the probability of being slum (dist > .05 = slum).
    This function scales those values to the repo convention of marking slums as >63.
    Use of uint8 data type avoids scaling of values to max=255 by imageio.imwrite.

    :param distance_pkl: Output of slums-world in terms of estimated distances to slum boundary.
    :return: Scaled pixel values.
    """
    dist_raw = np.array(distance)
    factor = 63 / dist_thresh
    dist_scaled = dist_raw * factor
    dist_scaled = dist_scaled.astype('uint8')
    return dist_scaled


def load_slums_world_pkl(input_path, output_path, show=False):
    """
    Load pkl of slums_world prediction, convert values to correct scale and save as png, with optional visualisation.

    When show=True, the histograms shows the distribution of prediction values and the respective converted values.
    The first image plot shows the original estimated distances, the second image plot shows the resulting slum map.

    Usage:
    load_slums_world_pkl('PATH/TO/input.pkl', 'PATH/TO/output.png', show=True)

    :param input_path: Path to .pkl containing distance_estimated.
    :param output_path: File name path to save output to.
    :param show: Visualise original, converted values and resulting slum mask.
    """
    with open(input_path, 'rb') as f:
        distance_pkl = pickle.load(f, encoding="bytes")
    scaled_distance = scale_distance(distance_pkl)
    imageio.imwrite(output_path, scaled_distance)
    if show:
        plt.hist(distance_pkl)
        plt.show()
        plt.hist(scaled_distance)
        plt.show()
        plt.imshow(scaled_distance)
        plt.show()
        plt.imshow(scaled_distance > 63)
        plt.show()



def assemble_tiles(base_path, show_type, show_plot=False):
    '''
    Re-assemble image from validation or test tiles, in their original place; optional display.

    Usage example:
    composite_img = assemble_tiles('/path/to/tiled/png/base_dir', 'validate', show_plot=True)

    :param base_path: Path to directory under which png files were saved. Should have sub-directories 'meta_data',
    'training', 'validation', and 'test'.
    :param show_type: Either 'validate' or 'test'.
    :param show_plot: Boolean flag to display image. Default = False.
    :return: Image showing validation or test set tiles, saved in images/collage/collage.png of the respective images.
    '''
    if show_type == 'test':
        img_dir = 'testing/images'
    elif show_type == 'validate':
        img_dir = 'validation/images'
    else:
        raise ValueError(f'show_type: arg must be set to string test or validate but is {show_type!r} ')
    meta_data = np.load(os.path.join(base_path, 'meta_data/tile_register.npz'), allow_pickle=True)
    register = meta_data['register']
    coordinates = meta_data['coordinates']
    collage = np.zeros((coordinates[1, 0, -1] + 1, coordinates[1, 1, -1] + 1, 3), dtype='uint8')
    counter = 0
    for i in np.arange(len(register)):
        if register[i] == show_type:
            x_0 = coordinates[0, 0, i]
            x_1 = coordinates[1, 0, i]
            y_0 = coordinates[0, 1, i]
            y_1 = coordinates[1, 1, i]
            tile_i = imageio.imread(os.path.join(base_path, img_dir, 'image_' + str(counter) + '.png')).astype('uint8')
            collage[x_0:x_1+1, y_0:y_1+1, :] = tile_i
            counter += 1
    os.makedirs(os.path.join(base_path, img_dir, 'collage'), exist_ok=True)
    imageio.imwrite(os.path.join(base_path, img_dir, 'collage/collage.png'), collage)
    if show_plot:
        plt.imshow(collage)
        plt.show()
    return collage


def load_slums_world_npy(pred_in_path, true_in_path, pred_out_path, true_out_path):
    """
    Load npy of slums_world predictions generated for validation, convert values to correct scale and save as png.

    Usage:
    load_slums_world_pkl('PATH/TO/pred_in.npy', 'PATH/TO/true_in.npy', 'PATH/TO/pred_out.png', 'PATH/TO/true_out.png')

    :param pred_in_path: Path to .npy containing distance_estimated as 1D vector.
    :param true_in_path: Path to .npy containing true distances as 3D tiles.
    :param pred_out_path: Path to .png containing distance_estimated as stacked column.
    :param true_out_path: Path to .png containing true distances as stacked column.
    """
    true_npy = np.load(true_in_path)
    tile_size = true_npy.shape[1]
    pred_npy = np.load(pred_in_path)
    tiled_pred = np.reshape(pred_npy, (-1, tile_size, tile_size))
    reshaped_pred = np.reshape(tiled_pred, (-1, tile_size))
    reshaped_true = np.reshape(true_npy, (-1, tile_size)).astype('uint8')
    scaled_pred = scale_distance(reshaped_pred)
    imageio.imwrite(pred_out_path, scaled_pred)
    imageio.imwrite(true_out_path, reshaped_true)

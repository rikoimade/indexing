#!/usr/bin/env python
# coding: utf-8

# ## Descriptions
# this script try to construct the orientation mapping using HRTEM
# 1. Generate stacks of FFT from hrtem and save as 2D electron diffraction.
# 2. Generate phase library for indexing
# 3. Index FFT pattern, and construct the orientation mapping.


import hyperspy.api as hs
import matplotlib.pyplot as plt
import pyxem as pxm
import numpy as np
import pandas as pd
import glob
import os
# In[2]:


import datetime


import diffpy
from diffsims.generators.structure_library_generator import StructureLibraryGenerator
from diffsims.libraries.structure_library import StructureLibrary
from diffsims.generators.diffraction_generator import DiffractionGenerator
from diffsims.generators.library_generator import DiffractionLibraryGenerator, VectorLibraryGenerator

from pyxem.generators.indexation_generator import IndexationGenerator
from pyxem.generators.indexation_generator import VectorIndexationGenerator

from pyxem.utils.sim_utils import sim_as_signal
from pyxem.utils.indexation_utils import peaks_from_best_template
from pyxem.utils.plot import generate_marker_inputs_from_peaks



today = datetime.datetime.now().strftime("%Y%m%d")


# ### 1. Load the HRTEM image using hyperspy

## setup
extension = "dm*"  # to capture both dm3 and dm4
outputfolder = "processed"
clean_fft = False
flist = glob.glob("*."+extension)
tiles = 16
library_angular_resolution = 0.1
tem_beam_energy = 300

def create_diffraction_from_fft(image, number_of_tiles=16):
    image = hs.load(image)

    ## here we are going to split the images into
    number_of_tiles = number_of_tiles
    split_images = image.split(axis='x', number_of_parts=number_of_tiles)
    for i in range(len(split_images)):
        split_images[i] = split_images[i].split(axis='y', number_of_parts=number_of_tiles)

    ## getting all the callibration value to use for diffraction pattern tiles later
    size = split_images[0][0].axes_manager['x'].size
    scale = split_images[0][0].axes_manager['x'].scale
    units = split_images[0][0].axes_manager['x'].units
    scan_step = size * scale

    fft_list = []

    for j in range(number_of_tiles):
        col = []
        for i in range(number_of_tiles):
            im = split_images[i][j]
            fft = im.fft(shift=True, apodization=True).amplitude
            fft_list.append(fft.data)

    pattern_size = fft.axes_manager['x'].size
    calibration = fft.axes_manager['x'].scale
    units = fft.axes_manager['x'].units

    ## reshape and stack

    dp = pxm.ElectronDiffraction2D(np.asarray(fft_list).reshape(number_of_tiles, number_of_tiles,
                                                                pattern_size, pattern_size))
    dp.set_diffraction_calibration(calibration / 10)
    dp.set_scan_calibration(scan_step)


    return dp

def clean_diffraction_pattern(dp):
    ## clean up data
    dp.data.astype('float64')
    dp = dp.remove_background('gaussian_difference',
                              sigma_min=2, sigma_max=8)
    dp.data -= dp.data.min()
    dp.data *= 1 / dp.data.max()
    return dp
    # In[21]:


if __name__ == '__main__':
    dps = pd.DataFrame()
    for image in flist:
        
        try:
            print(image+" start")
            saveas = image.split('.')[0] + '_stacked_fft_gekko_' + str(number_of_tiles) + 'x' + str(
                number_of_tiles)+'.hspy'

            if os.path.isfile(saveas) == False:
                dp = create_diffraction_from_fft(image, number_of_tiles=tiles)
                if clean_fft: dp = clean_diffraction_pattern(dp)


            dp.save(saveas)
            dps.loc[image,'dp'] = saveas
        except Exception:
            continue



    # ## 2. Indexing with pattern matching

    # ### 2.1 Generate diffraction pattern library


    for image in flist:
        ## create crystall structure from CIF files

        library_name = image.split('.')[0] + "_template_" + str(angular_resolution) + "_deg.pickle"
        if os.path.isfile(library_name) == False:

            aug = diffpy.structure.loadStructure('augite.cif')
            pig = diffpy.structure.loadStructure('pigeonite.cif')


            structure_library_generator = StructureLibraryGenerator(
                [('Aug', aug, 'monoclinic'),
                 ('Pig', pig, 'monoclinic')])
            in_plane_rotation = [(0,), (0,)]
            angular_resolution = library_angular_resolution
            structure_library = structure_library_generator.get_orientations_from_stereographic_triangle(
                in_plane_rotation,  # In-plane rotations
                angular_resolution)  # Angular resolution of the library

            ### load fft and set as ElectronDiffraction2D
            im = pxm.load_hspy(dps.loc[image,'dp'])
            dp = pxm.ElectronDiffraction2D(im.data)
            dp.set_diffraction_calibration(im.axes_manager['kx'].scale)
            dp.set_scan_calibration(im.axes_manager['x'].scale)


            pattern_size = dp.axes_manager['x'].size
            calibration = dp.axes_manager['kx'].scale

            diffraction_calibration = calibration
            half_pattern_size = pattern_size // 2
            reciprocal_radius = diffraction_calibration*(half_pattern_size - 1)

            beam_energy = tem_beam_energy

            ediff = DiffractionGenerator(beam_energy, 0.025)
            diff_gen = DiffractionLibraryGenerator(ediff)
            template_library = diff_gen.get_diffraction_library(structure_library,
                                                                calibration=diffraction_calibration,
                                                                reciprocal_radius=reciprocal_radius,
                                                                half_shape=(half_pattern_size, half_pattern_size),
                                                                with_direct_beam=False)

            template_library.pickle_library(library_name)
        dps.loc[image,'template_library'] = library_name




        ## let's take a look at the generated pattern
        #pattern0 = sim_as_signal(template_library.get_library_entry(phase='Aug', angle=(0, 0, 0))["Sim"],
        #                         pattern_size, 0.03, reciprocal_radius)



        # ### 2.2 start indexing

        #
        indexer = IndexationGenerator(dp, template_library)
        match_results = indexer.correlate(n_largest=3, inplane_rotations=np.arange(0, 360, 1))


        '''
        match_results.plot_best_matching_results_on_signal(dp, template_library,
                                                           permanent_markers=False,
                                                           cmap='viridis')
        '''

        cryst_map = match_results.get_crystallographic_map()

        mtex_file = image.split('.')[0]+"_pattern_match_results_tile"+str(number_of_tiles)+"_"+today+".csv"
        cryst_map.save_mtex_map(mtex_file)
        dps.loc[image, 'mtex_file'] = mtex_file


        ori_map = cryst_map.get_orientation_map()

        ori_map_file = image.split('.')[0]+"_ori_map_tile"+str(number_of_tiles)+"_"+today+".hspy"
        ori_map.save(ori_map_file)
        dps.loc[image, 'ori_map_file'] = ori_map_file
        print(image + " finished")

# ## 3. Vector matching

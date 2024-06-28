# from mejiro.helpers import gs
# from mejiro.lenses.test import SampleStrongLens


# def test_get_image():
#     sample_lens = SampleStrongLens()

#     num_pix = 51
#     final_pixel_side = 45
#     side = 5.61
#     grid_oversample = 3
#     bands = ['F106', 'F129', 'F184']

#     # lens.add_subhalos(*pyhalo.unpickle_subhalos(os.path.join(pickle_dir, 'pyhalo', 'cdm_subhalos_tuple')))

#     arrays = []
#     for band in bands:
#         array = sample_lens.get_array(num_pix=num_pix * grid_oversample, side=side, band=band)
#         arrays.append(array)

#     # one band, one array (single band happy path)
#     gs_images, execution_time = gs.get_images(sample_lens, arrays[0], bands[0], input_size=num_pix,
#                                               output_size=final_pixel_side, grid_oversample=grid_oversample)
#     # TODO some kind of check

#     # 3 bands, 3 arrays (color happy path)
#     gs_images, execution_time = gs.get_images(sample_lens, arrays, bands, input_size=num_pix,
#                                               output_size=final_pixel_side, grid_oversample=grid_oversample)
#     # TODO some kind of check

#     # TODO one band, multiple arrays

#     # TODO two bands, any number of arrays

#     # TODO multiple bands, one array

#     # TODO any number of bands, two arrays

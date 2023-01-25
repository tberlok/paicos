

# def test_units():
#     import paicos as pa
#     import numpy as np

#     pa.use_units(True)

#     snap = pa.Snapshot(pa.root_dir + '/data', 247,
#                        snap_basename='reduced_snap', load_catalog=False)

#     for key in snap.info(0, False):
#         snap.load_data(0, key)

#     # Check the conversion to CGS against a few values
#     for key in ['Density', 'Coordinates', 'Masses']:
#         key0 = '0_' + key
#         if key0 in snap.P_attrs:
#             if len(snap.P_attrs[key0]) > 0:
#                 reference_value = snap.P_attrs[key0]['to_cgs']
#                 unit_quant = snap[key0].unit_quantity
#                 paicos_value = unit_quant.cgs.value

#                 np.testing.assert_allclose(reference_value, paicos_value)

# # Test the conversion to physical values,
# # comparing with code from arepo-snap-util

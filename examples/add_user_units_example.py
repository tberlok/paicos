import paicos as pa
pa.use_units(True)


# Add new units before loading your data (or reload the data afterwards)

# Define a unitless tracer
pa.add_user_unit('voronoi_cells', 'JetTracer', '')

# Load a snapshot
snap = pa.Snapshot(pa.data_dir, 247)

# Add a unit which is already there (not allowed and fails!)
pa.add_user_unit('voronoi_cells', 'Coordinates', 'arepo_mass')

try:
    snap = pa.Snapshot(pa.data_dir, 247)
except RuntimeError as err:
    print('\nExpected error occurred:')
    print(err)

import pytest


@pytest.fixture  # (helps run this test independently from the others)
def test_non_comoving():

    import paicos as pa
    snap = pa.Snapshot(pa.root_dir + '/data/', 7,
                       basename='small_non_comoving')

    snap['0_Density']
    snap['0_Masses']

    snap['0_Masses'].no_small_h

    snap['0_Density'].to_physical


if __name__ == '__main__':
    test_non_comoving()

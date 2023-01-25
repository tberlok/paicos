

def test_non_comoving():

    import paicos as pa
    snap = pa.Snapshot(pa.root_dir + '/data/', 7)

    snap['0_Density']
    snap['0_Masses']

    snap['0_Masses'].no_small_h

    snap['0_Density'].to_physical


if __name__ == '__main__':
    test_non_comoving()

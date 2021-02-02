import dataset

if __name__ == "__main__":
    test_dataset = dataset.DataSet('result_test/network1/')
    test_dataset.create_subset('train', frame_len=44100)
    test_dataset.create_subset('val')
    test_dataset.create_subset('test')
    test_dataset.load_file('KDonnerFlangerra12c12rg9Singles1', ['train', 'val', 'test'], [0.75, 0.125, 0.125])
    test_dataset.load_file('KDonnerFlangerra12c12rg9Singles1')


    test_dataset2 = dataset.DataSet('result_test/network1/', extensions=None)
    test_dataset2.create_subset('train', frame_len=44100)
    test_dataset2.create_subset('val')
    test_dataset2.create_subset('test')
    test_dataset2.load_file('KDonnerFlangerra12c12rg9Singles1-input')
    test_dataset2.load_file('KDonnerFlangerra12c12rg9Singles1-input', ['train', 'val', 'test'], [0.75, 0.125, 0.125])

    test_dataset3 = dataset.DataSet('result_test/network1/', extensions=None)
    test_dataset3.create_subset('train', frame_len=44100)
    test_dataset3.create_subset('val')
    test_dataset3.create_subset('test')
    test_dataset3.load_file('KDonnerFlangerra12c12rg9Singles1-input', cond_val=0)
    test_dataset3.load_file('KDonnerFlangerra12c12rg9Singles1-input', ['train', 'val', 'test'], [0.75, 0.125, 0.125],
                            cond_val=0.25)
    test_dataset3.load_file('KDonnerFlangerra12c12rg9Singles1-input', cond_val=0.5)

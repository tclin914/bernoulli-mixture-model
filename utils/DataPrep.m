function [ train, test ] = DataPrep( path_to_dataset )

    [train.images, train.labels] = readMNIST ...
        ([path_to_dataset, '/train-images.idx3-ubyte'], ...
         [path_to_dataset, '/train-labels.idx1-ubyte'], 50000, 0);
    [test.images, test.labels] = readMNIST ...
        ([path_to_dataset, '/t10k-images.idx3-ubyte'], ...
         [path_to_dataset, '/t10k-labels.idx1-ubyte'], 10000, 0);

    train.images = reshape(train.images, 400, size(train.images, 3)) > 0.5;
    test.images = reshape(test.images, 400, size(test.images, 3)) > 0.5;

end


import tqdm
import glob
import tensorflow as tf
import numpy as np
import pandas as pd


source_path = "C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/"


def load_dataset(path="C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results/DataEmoda.npy"):
    df = pd.read_pickle(path)
    return df


def create_input_output(df, input_labels, output_labels):
    inputImage = []
    outputImage = []
    for i in df.index:
        inputImage.append(df[input_labels][i].reshape(256, 256, 1))
        outputImage.append(df[output_labels][i].reshape(256, 256, 1))
    return np.array(inputImage), np.array(outputImage)


def create_input(df, input_labels):
    return df[input_labels].values


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        # BytesList won't unpack a string from an EagerTensor.
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


def parse_combined_data(bathy, hs, eta, zeta, theta):

    # define the dictionary -- the structure -- of our single example
    # bathy and hs have the same shape
    data = {
        'height': _int64_feature(bathy.shape[0]),
        'width': _int64_feature(bathy.shape[1]),
        'depth': _int64_feature(bathy.shape[2]),
        'bathy': _bytes_feature(serialize_array(bathy)),
        'hs': _bytes_feature(serialize_array(hs)),
        'eta': _float_feature(eta),
        'zeta': _float_feature(zeta),
        'theta': _float_feature(theta)
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def write_data(bathy, hs, eta, zeta, theta,
               filename: str = 'combined_data', max_files: int = 10,
               out_dir=source_path+"combined_data/"):

    splits = (len(bathy)//max_files) + 1
    if len(bathy) % max_files == 0:
        splits -= 1

    print(
        f"\nUsing {splits} shard(s) for {len(bathy)} files,\
            with up to {max_files} samples per shard")

    file_count = 0

    for i in tqdm.tqdm(range(splits)):
        current_shard_name = "{}{}_{}{}.tfrecords".format(
            out_dir, i+1, splits, filename)
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files:
            index = i*max_files + current_shard_count
            if index == len(bathy):
                break

            current_bathy = bathy[index]
            current_hs = hs[index]

            current_eta = eta[index]
            current_zeta = zeta[index]
            current_theta = theta[index]

            out = parse_combined_data(bathy=current_bathy, hs=current_hs,
                                      eta=current_eta, zeta=current_zeta,
                                      theta=current_theta)

            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_count += 1

        writer.close()

    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count


def get_dataset_large(tfr_dir=source_path+'combined_data/', 
                      pattern: str = "*combined_data.tfrecords"):
    files = glob.glob(tfr_dir+pattern, recursive=False)

    dataset = tf.data.TFRecordDataset(files)

    dataset = dataset.map(
        tf_parse)

    return dataset


def tf_parse(eg):
    """parse an example (or batch of examples, not quite sure...)"""

    # here we re-specify our format
    # you can also infer the format from the data using tf.train.Example.FromString
    # but that did not work
    example = tf.io.parse_example(
        eg[tf.newaxis],
        {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'bathy': tf.io.FixedLenFeature([], tf.string),
            'hs': tf.io.FixedLenFeature([], tf.string),
            'eta': tf.io.FixedLenFeature([], tf.float32),
            'zeta': tf.io.FixedLenFeature([], tf.float32),
            'theta': tf.io.FixedLenFeature([], tf.float32),
        },
    )
    bathy = tf.io.parse_tensor(example["bathy"][0], out_type="float32")
    hs = tf.io.parse_tensor(example["hs"][0], out_type="float32")
    eta = example["eta"]
    zeta = example["zeta"]
    theta = example["theta"]
    attr = tf.stack([eta, zeta, theta], axis=1)
    return (bathy, attr), hs


df = load_dataset()
(inputImages, outputImages) = create_input_output(df, "bathy", "hs")
inputAttr = create_input(df, ['$\eta$', '$\zeta$', '$\theta_{wave}$'])

inputImages = (inputImages - np.nanmean(inputImages))/np.nanstd(inputImages)
(inputImages, outputImages) = (np.nan_to_num(
    inputImages, nan=-2.), np.nan_to_num(outputImages, nan=-2.))

inputAttr[:, 0] = (inputAttr[:, 0] - np.mean(inputAttr[:, 0])
                   ) / np.std(inputAttr[:, 0])
inputAttr[:, 1] = (inputAttr[:, 1] - np.mean(inputAttr[:, 1])
                   ) / np.std(inputAttr[:, 1])
inputAttr[:, 2] = inputAttr[:, 2] / (2*np.pi)
#count = write_images_to_tfr_short(i, o)

#dataset = get_dataset_small(r'C:\Users\hemert\OneDrive - Stichting Deltares\Programmas\Data\images.tfrecords')

write_data(inputImages, outputImages,
           inputAttr[:, 0], inputAttr[:, 1], inputAttr[:, 2], max_files=10)


dataset = get_dataset_large()

for sample in dataset.take(1):
    print(repr(sample))
    print(sample[0].shape)
    print(sample[1].shape)

train_size = int(0.7*1016)
val_size = int(0.15*1016)
test_size = int(0.15*1016)

dataset = dataset.shuffle(buffer_size=30)
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)
val_dataset = test_dataset.skip(test_size)
test_dataset = test_dataset.take(test_size)

shape = []
for sample in val_dataset.take(val_size):
    # print(sample[0].shape)
    # print(sample[1].shape)
    shape.append(sample[1].shape)

examples = dataset.take(10)
example_bytes = list(examples)[0].numpy()
parsed = tf.train.Example.FromString(example_bytes)
parsed.features.feature('height')
parsed.features.feature('width')
list(parsed.features.feature.keys())

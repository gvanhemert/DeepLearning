#%% Imports
import tqdm
import glob
import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from pathlib import Path

#%% Paths
source_path = Path('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/Analysis Results')
source_path2 = Path('C:/Users/hemert/OneDrive - Stichting Deltares/Programmas/Data/')

#%% Functions
def create_input_images(df, input_label):
  '''Input a dataframe and an label for the column to convert
     returns an array containing the reshaped arrays from the chosen column containing arrays'''
    input_image = []
    for i in df.index:
        input_image.append(df[input_labels][i].reshape(256, 256, 1))
    return np.array(input_image)

def create_output(df, output_label):
  '''Input a dataframe and an label for the column to convert
     returns an array containing the reshaped arrays from the chosen column containing arrays'''
    output_image = []
    for i in df.index:
        output_image.append(df[output_label][i].reshape(256, 256, 1))
    return np.array(output_image)


def create_input_attributes(df, input_labels):
  '''Input a dataframe and multiple labels for the columns containing scalars to convert
     returns a stacked array of the chosen columns'''
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


def parse_combined_data(bathy, hs, tm01, theta0x, theta0y, mask,
                        eta, zeta, theta_wavex, theta_wavey):

    # define the dictionary -- the structure -- of our single example
    # bathy and hs have the same shape
    data = {
        'height': _int64_feature(bathy.shape[0]),
        'width': _int64_feature(bathy.shape[1]),
        'depth': _int64_feature(bathy.shape[2]),
        'bathy': _bytes_feature(serialize_array(bathy)),
        'hs': _bytes_feature(serialize_array(hs)),
        'tm01': _bytes_feature(serialize_array(tm01)),
        'theta0x': _bytes_feature(serialize_array(theta0x)),
        'theta0y': _bytes_feature(serialize_array(theta0y)),
        'mask': _bytes_feature(serialize_array(mask)),
        'eta': _float_feature(eta),
        'zeta': _float_feature(zeta),
        'theta_wavex': _float_feature(theta_wavex),
        'theta_wavey': _float_feature(theta_wavey)
    }

    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def write_data(bathy, hs, tm01, theta0x, theta0y, mask,
               eta, zeta, theta_wavex, theta_wavey,
               filename, max_files,
               out_dir"):
    '''Writes the data to multiple tfrecord files each containing max_files examples'''
    splits = (len(bathy)//max_files) + 1
    if len(bathy) % max_files == 0:
        splits -= 1

    print(
        f"\nUsing {splits} shard(s) for {len(bathy)} files,\
            with up to {max_files} samples per shard")

    file_count = 0

    for i in tqdm.tqdm(range(splits)):
        current_shard_name = "{}{}_{}{}_{}.tfrecords".format(
            out_dir, i+1, splits, filename, max_files)
        writer = tf.io.TFRecordWriter(current_shard_name)

        current_shard_count = 0
        while current_shard_count < max_files:
            index = i*max_files + current_shard_count
            if index == len(bathy):
                break

            current_bathy = bathy[index]
            current_hs = hs[index]
            current_tm01 = tm01[index]
            current_theta0x = theta0x[index]
            current_theta0y = theta0y[index]
            current_mask = mask[index]

            current_eta = eta[index]
            current_zeta = zeta[index]
            current_theta_wavex = theta_wavex[index]
            current_theta_wavey = theta_wavey[index]

            out = parse_combined_data(bathy=current_bathy, hs=current_hs,
                                      tm01 = current_tm01,
                                      theta0x = current_theta0x,
                                      theta0y = current_theta0y,
                                      mask = current_mask,
                                      eta=current_eta, zeta=current_zeta,
                                      theta_wavex=current_theta_wavex,
                                      theta_wavey=current_theta_wavey)

            writer.write(out.SerializeToString())
            current_shard_count += 1
            file_count += 1

        writer.close()

    print(f"\nWrote {file_count} elements to TFRecord")
    return file_count


def get_dataset_large(tfr_dir=source_path2+'train_data_mask/', 
                      pattern: str = "*train_data_mask.tfrecords"):
    '''Loads the tfrecord files and returns a tfrecord dataset''' 
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
            'tm01': tf.io.FixedLenFeature([], tf.string),
            'theta0x': tf.io.FixedLenFeature([], tf.string),
            'theta0y': tf.io.FixedLenFeature([], tf.string),
            'mask': tf.io.FixedLenFeature([], tf.string),
            'eta': tf.io.FixedLenFeature([], tf.float32),
            'zeta': tf.io.FixedLenFeature([], tf.float32),
            'theta_wavex': tf.io.FixedLenFeature([], tf.float32),
            'theta_wavey': tf.io.FixedLenFeature([], tf.float32),
        },
    )
    bathy = tf.io.parse_tensor(example["bathy"][0], out_type="float32")
    hs = tf.io.parse_tensor(example["hs"][0], out_type="float32")
    tm01 = tf.io.parse_tensor(example["tm01"][0], out_type="float32")
    theta0x = tf.io.parse_tensor(example["theta0x"][0], out_type="float32")
    theta0y = tf.io.parse_tensor(example["theta0y"][0], out_type="float32")
    mask = tf.io.parse_tensor(example["mask"][0], out_type="int32")
    mask = tf.ensure_shape(mask, (256, 256, 1))
    eta = example["eta"]
    zeta = example["zeta"]
    theta_wavex = example["theta_wavex"]
    theta_wavey = example["theta_wavey"]
    attr = tf.stack([eta, zeta, theta_wavex, theta_wavey], axis=1)
    attr = tf.reshape(attr,shape=[-1])
    output = (hs, tm01, theta0x, theta0y, mask)
    #output = tf.concat([hs, tm01, theta0x, theta0y], axis=-1)
    return (bathy, attr), output

#%% Loading
combined_dataframe = pd.read_pickle(source_path2.joinpath('combined_dataframe.npy'))
               
#%% Computing mean and std of inpu data for standardization
'''The bathy column gets converted to an array containing the arrays to be able to get one number for the mean'''
bathy_full_array = np.array(combined_dataframe['bathy'].values.tolist()) 

bathy_mean = bathy_full_array.mean()

'''Due to memory constraints, the std of the bathy inpu needs to be calculated manually and in parts'''
bathy_std = {}
for i in range(10):
    bathy_std[i] = ((bathy_full_array[i*1000:(i+1)*1000]- bathy_mean)**2).sum()
bathy_std[11] = ((bathy_full_array[10000:10275] - bathy_mean)**2).sum()

bathy_std_sum = 0
for i in bathy_std:
    bathy_std_sum += bathy_std[i]
    
bathy_std = np.sqrt(bathy_std_sum/(10275*256*256))

eta_mean = combined_dataframe['$\eta$'].mean()
eta_std = combined_dataframe['$\eta$'].std()

zeta_mean = combined_dataframe['$\zeta$'].mean()
zeta_std = combined_dataframe['$\zeta$'].std()

theta_wavex_mean = combined_dataframe['theta_wavex'].mean()
theta_wavex_std = combined_dataframe['theta_wavex'].std()

theta_wavey_mean = combined_dataframe['theta_wavey'].mean()
theta_wavey_std = combined_dataframe['theta_wavey'].std()

'''Storing the mean and std to be able to go back to the original input'''
input_attributes = {'bathy':{'mean':bathy_mean, 'std':bathy_std},
                    'eta':{'mean':eta_mean, 'std':eta_std},
                    'zeta':{'mean':zeta_mean, 'std':zeta_std},
                    'theta_wavex':{'mean':theta_wavex_mean, 'std':theta_wavex_std},
                    'theta_wavey':{'mean':theta_wavey_mean, 'std':theta_wavey_std}
                    }

np.save(source_path2.joinpath('input_attributes.npy'), input_attributes)



#%% Standardizing the input data

'''Since the bathy part of the dataframe is much larger, it has to be done in parts due to memory constraints'''
for i in range(10):
    combined_dataframe['bathy'][i*1000:(i+1)*1000] = (combined_dataframe['bathy'][i*1000:(i+1)*1000] - bathy_mean) / bathy_std
combined_dataframe['bathy'][10000:10275] = (combined_dataframe['bathy'][10000:10275] - bathy_mean) / bathy_std
    
combined_dataframe['$\eta$'] = (combined_dataframe['$\eta$'] - eta_mean) / eta_std
combined_dataframe['$\zeta$'] = (combined_dataframe['$\zeta$'] - zeta_mean) / zeta_std
#combined_dataframe['theta_wavex'] = (combined_dataframe['theta_wavex'] - theta_wavex_mean) / theta_wavex_std
#combined_dataframe['theta_wavey'] = (combined_dataframe['theta_wavey'] - theta_wavey_mean) / theta_wavey_std

#%%
'''Splitting the dataframe in train and test data'''
(train_data, test_data) = train_test_split(combined_dataframe, test_size=0.3,
                                           random_state=0)


input_attr_test = create_input(test_data, ['$\eta$', '$\zeta$',
                                           'theta_wavex', 'theta_wavey'])
input_attr_train = create_input(train_data, ['$\eta$', '$\zeta$',
                                           'theta_wavex', 'theta_wavey'])
bathy_img_test = create_input_output(test_data, 'bathy')
bathy_img_train = create_input_output(train_data, 'bathy')

hs_img_test = create_output(test_data, 'hs')
tm01_img_test = create_output(test_data, 'tm01')
theta0x_img_test = create_output(test_data, 'theta0x')
theta0y_img_test = create_output(test_data, 'theta0y')

hs_img_train = create_output(train_data, 'hs')
tm01_img_train = create_output(train_data, 'tm01')
theta0x_img_train = create_output(train_data, 'theta0x')
theta0y_img_train = create_output(train_data, 'theta0y')

#%% Removing nan and creating a mask for the nan values
mask_test = np.zeros(tm01_img_train.shape)
mask_test = np.logical_or(mask_test, np.isnan(tm01_img_train))

hs_img_test = np.nan_to_num(hs_img_test, nan=0)
tm01_img_test = np.nan_to_num(tm01_img_test, nan=0)
theta0x_img_test = np.nan_to_num(theta0x_img_test, nan=0)
theta0y_img_test = np.nan_to_num(theta0y_img_test, nan=0)

hs_img_train = np.nan_to_num(hs_img_train, nan=0)
tm01_img_train = np.nan_to_num(tm01_img_train, nan=0)
theta0x_img_train = np.nan_to_num(theta0x_img_train, nan=0)
theta0y_img_train = np.nan_to_num(theta0y_img_train, nan=0)

#%% Writing data to tfrecord files
write_data(bathy_img_test, hs_img_test, tm01_img_test,
           theta0x_img_test, theta0y_img_test, mask_test,
           input_attr_test[:, 0], input_attr_test[:, 1], input_attr_test[:, 2],
           input_attr_test[:,3],max_files=100,
           file_name='test_data_mask', out_dir=source_path2+"test_data_mask/")

write_data(bathy_img_train, hs_img_train, tm01_img_train,
           theta0x_img_train, theta0y_img_train, mask_train,
           input_attr_train[:, 0], input_attr_train[:, 1], input_attr_train[:, 2],
           input_attr_train[:,3],max_files=100,
           file_name='train_data_mask', out_dir=source_path2+"train_data_mask/"))

#%% Testing
dataset = get_dataset_large()

for sample in dataset.take(1):
    print(repr(sample))
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)
    

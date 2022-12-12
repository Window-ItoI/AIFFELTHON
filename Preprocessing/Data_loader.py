import tensorflow as tf

def _parse_tfrecord():
    def parse_tfrecord(tfrecord):
        features = {'image': tf.io.FixedLenFeature([], tf.string),
                    'label': tf.io.FixedLenFeature([], tf.string)}
        parsed_example = tf.io.parse_single_example(tfrecord, features)
        img = tf.image.decode_jpeg(parsed_example['image'], channels=3)
        labels = tf.io.decode_raw(parsed_example['label'], tf.float64)
        labels = tf.reshape(labels, [8])
        labels = tf.cast(labels, tf.float32)
        # labels = tf.sparse.to_dense(parsed_example['image/source_id'])
        # labels = tf.cast(labels, tf.float32)
        img = float(img)
        
        img = _transform_images()(img)
        labels = _transform_targets(labels)
        return img, labels
    return parse_tfrecord


def _transform_images():
    def transform_images(img):
        img = tf.image.resize_with_crop_or_pad(img, 256,256)
        img = tf.clip_by_value(img / 255.0, 0.0, 1.0)
        return img
    return transform_images


def _transform_targets(y_train):
    # print(y_train)
    return y_train

def load_tfrecord_dataset(tfrecord_name, batch_size, shuffle=True, buffer_size=1024):
    raw_dataset = tf.data.TFRecordDataset(tfrecord_name)
    # raw_dataset = raw_dataset.repeat()
    if shuffle:
        raw_dataset = raw_dataset.shuffle(buffer_size=buffer_size)
    dataset = raw_dataset.map(
        _parse_tfrecord(),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
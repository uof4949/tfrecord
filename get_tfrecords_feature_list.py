import tensorflow as tf
import sys

def get_tfrecords_feature_list(tfrecords_filename):
    ptr = 0
    record_iterator = tf.python_io.tf_record_iterator(path=tfrecords_filename)

    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)clmpt
        return example.features.feature.keys()

    return []

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python3 get_tfrecords_feature_list.py <tfrecord_file_name>")
    else:
        tfrecords_filename = sys.argv[1]
        feature_list = get_tfrecords_feature_list(tfrecords_filename)
        print(feature_list)
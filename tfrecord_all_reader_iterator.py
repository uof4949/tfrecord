import tensorflow as tf
from PIL import Image
import io
import sys
import numpy as np

#tfrecord_filename = 'example_cat.tfrecord'

def readRecordAll(filename):
    features = []

    count = 0

    record_iterator = tf.python_io.tf_record_iterator(path=filename)

    for record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(record)

        height = int(example.features.feature['image/height'].int64_list.value[0])
        width = int(example.features.feature['image/width'].int64_list.value[0])
        colorspace = (example.features.feature['image/colorspace'].bytes_list.value[0])
        channels = int(example.features.feature['image/channels'].int64_list.value[0])
        label = int(example.features.feature['image/class/label'].int64_list.value[0])
        text = (example.features.feature['image/class/text'].bytes_list.value[0])
        format = (example.features.feature['image/format'].bytes_list.value[0])
        filename = (example.features.feature['image/filename'].bytes_list.value[0])
        encoded = (example.features.feature['image/encoded'].bytes_list.value[0])

        features.append((height, width, colorspace, channels, label, text, format, filename, encoded))
        if count % 100 == 0:
            print("count : %d" %count)
        count = count + 1

    return features
    
def main():
    if len(sys.argv) != 2:
        print("Usage : python3 tfrecord_reader.py <tfrecord_file_name>")
    else:
        tfrecord_filename = sys.argv[1]
        count = 0
        features = readRecordAll(tfrecord_filename)

        for feature in features:
            print("height : %d" %feature[0])
            print("width : %d" %feature[1])
            print("colorspace : %s" %feature[2])
            print("channels : %d" %feature[3])
            print("label : %d" %feature[4])
            print("text : %s" %feature[5])
            print("format : %s" %feature[6])
            print("filename : %s" %feature[7])
            print("count = %d" %count)
            count = count + 1

         
if __name__ == "__main__":
    main()
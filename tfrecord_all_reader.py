import tensorflow as tf
from PIL import Image
import io
import sys

#tfrecord_filename = 'example_cat.tfrecord'

def readRecordAll(filename):
    keys_to_features = {
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/colorspace': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/channels': tf.FixedLenFeature((), tf.int64, 1),
        'image/class/label': tf.FixedLenFeature((), tf.int64, 1),
        'image/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    
    features = []

    count = 0
    for example in tf.python_io.tf_record_iterator(filename):
        feature = tf.parse_single_example(example,features= keys_to_features)
        features.append(feature)
        count = count + 1
        if count % 100 == 0:
            print("count : %d" %count)
    
    return features
    
def main():
    keys_to_features = {
        'image/height': tf.FixedLenFeature((), tf.int64, 1),
        'image/width': tf.FixedLenFeature((), tf.int64, 1),
        'image/colorspace': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/channels': tf.FixedLenFeature((), tf.int64, 1),
        'image/class/label': tf.FixedLenFeature((), tf.int64, 1),
        'image/class/text': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='JPEG'),
        'image/filename': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    }
    if len(sys.argv) != 2:
        print("Usage : python3 tfrecord_reader.py <tfrecord_file_name>")
    else:
        tfrecord_filename = sys.argv[1]
        count = 0
        features = readRecordAll(tfrecord_filename)
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            vfeatures = sess.run(features)

            for _feature in vfeatures:
                feature = tf.parse_single_example(_feature,features= keys_to_features)
    
                height = tf.cast(feature['image/height'],tf.int64)
                width = tf.cast(feature['image/width'],tf.int64)
                colorspace = tf.cast(feature['image/colorspace'], tf.string)
                channels = tf.cast(feature['image/channels'], tf.int64)
                label = tf.cast(feature['image/class/label'], tf.int64)
                text = tf.cast(feature['image/class/text'], tf.string)
                format = tf.cast(feature['image/format'],tf.string)
                filename = tf.cast(feature['image/filename'],tf.string)
                encoded = tf.cast(feature['image/encoded'],tf.string)

                print("height : %d" %height)
                print("width : %d" %width)
                print("colorspace : %s" %colorspace)
                print("channels : %d" %channels)
                print("label : %d" %label)
                print("text : %s" %text)
                print("format : %s" %format)
                print("filename : %s" %filename)
                print("count = %d" %count)
                count = count + 1

         
if __name__ == "__main__":
    main()
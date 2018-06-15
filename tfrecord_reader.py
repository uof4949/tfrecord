import tensorflow as tf
from PIL import Image
import io
import sys

#tfrecord_filename = 'example_cat.tfrecord'

def readRecord(filename_queue):
    reader = tf.TFRecordReader()
    _,serialized_example = reader.read(filename_queue)
    
    #'''
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
    
    features = tf.parse_single_example(serialized_example,features= keys_to_features)
    
    height = tf.cast(features['image/height'],tf.int64)
    width = tf.cast(features['image/width'],tf.int64)
    colorspace = tf.cast(features['image/colorspace'], tf.string)
    channels = tf.cast(features['image/channels'], tf.int64)
    label = tf.cast(features['image/class/label'], tf.int64)
    text = tf.cast(features['image/class/text'], tf.string)
    format = tf.cast(features['image/format'],tf.string)
    filename = tf.cast(features['image/filename'],tf.string)
    encoded = tf.cast(features['image/encoded'],tf.string)


    return height,width,colorspace,channels,label,text,format,filename,encoded
    
def main():
    if len(sys.argv) != 2:
        print("Usage : python3 tfrecord_reader.py <tfrecord_file_name>")
    else:
        tfrecord_filename = sys.argv[1]
        count = 0
        with tf.Session() as sess:
            filename_queue = tf.train.string_input_producer([tfrecord_filename])
            height,width,colorspace,channels,label,text,format,filename,encoded = readRecord(filename_queue)
     
            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                while True:
                    vheight,vwidth,vcolorspace,vchannels,vlabel,vtext,vformat,vfilename,vencoded = \
                    sess.run([height,width,colorspace,channels,label,text,format,filename,encoded])
                    print("vheight : %d" %vheight)
                    print("vwidth : %d" %vwidth)
                    print("vcolorspace : %s" %vcolorspace)
                    print("vchannels : %d" %vchannels)
                    print("vlabel : %d" %vlabel)
                    print("vtext : %s" %vtext)
                    print("vformat : %s" %vformat)
                    print("vfilename : %s" %vfilename)
                    print("count = %d" %count)
                    count = count + 1
            except tf.errors.OutOfRangeError:
                coord.request_stop()
            finally:
                coord.request_stop()
                coord.join(threads)
        """
        filename_queue = tf.train.string_input_producer([tfrecord_filename])
        height,width,colorspace,channels,label,text,format,filename,encoded = readRecord(filename_queue)
     
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init_op)
    
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
         
            vheight,vwidth,vcolorspace,vchannels,vlabel,vtext,vformat,vfilename,vencoded = \
            sess.run([height,width,colorspace,channels,label,text,format,filename,encoded])
            print("vheight : %d" %vheight)
            print("vwidth : %d" %vwidth)
            print("vcolorspace : %s" %vcolorspace)
            print("vchannels : %d" %vchannels)
            print("vlabel : %d" %vlabel)
            print("vtext : %s" %vtext)
            print("vformat : %s" %vformat)
            print("vfilename : %s" %vfilename)
            image = Image.open(io.BytesIO(vencoded))
            image.show()

            coord.request_stop()
            coord.join(threads)
            """
         
if __name__ == "__main__":
    main()
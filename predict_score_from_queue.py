import json
import os
import numpy as np

from vsum_tools import *
from vasnet import *
from datetime import datetime
from log_helper import LogHelper as logger
from clickhouse_driver import Client

segment_length = 2 #in seconds
sampling_rate = 2 #in frame per second
model_root_dir = '/working'
model_file_path = model_root_dir + '/model.pth.tar'

def extract_features_from_images(model_func, preprocess_func, target_size, images):
  import cv2
  from tensorflow.keras.preprocessing import image
  import numpy as np
  import base64

  features = []
  converted_frame = []
  count = 0
  for frame in images:
    # print(base64.b64decode(frame['Image']['Data']))
    a = bytearray(base64.b64decode(frame['Image']['Data']))
    # break
    frame = cv2.imdecode(np.asarray(a), cv2.IMREAD_UNCHANGED)
    count += 1
    if(count % 50 == 0):
      print("-- Processing frame ", count, "/", len(images))
    # convert it from BGR to RGB
    # ordering, resize the frame to a fixed target_size, and then
    # perform mean subtraction
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, target_size).astype("float32")
    img_array = image.img_to_array(frame)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_func(expanded_img_array)
    converted_frame.append(preprocessed_img)

  converted_frame = np.vstack(converted_frame)
  print("pick frames ", len(converted_frame))
  # xử lý theo batch
  print("-- Extracting features: ...")
  features = model_func.predict(converted_frame, batch_size = 32)
  print("-- Done. Feature size: ", features.shape)

  return features


# main program

model_func, preprocess_func, target_size = model_picker('inceptionv3', 'max')

# get message from queue here
uri = os.environ.get('RABBITMQ_URI') or "amqp://vinhtk:9514753@vinhtk.vietbando.net:5672/"
queue = os.environ.get('RABBITMQ_QUEUE') or "events"
# get db env
db_host = os.environ.get('DB_HOST') or "vinhtk.vietbando.net"
db_database = os.environ.get('DB_DATABASE') or "aacs"
db_user = os.environ.get('DB_USER') or "vinhtk"
db_password = os.environ.get('DB_PASSWORD') or "91ZEAtafoX7Q"

# connect to clickhouse
client = Client(host=db_host, database=db_database,
                user=db_user, password=db_password)

def pushScore(camId, score):
    args = {'camId':camId, 'score':score}
    sql = """
    INSERT INTO Event(eventId, cameraId, eventName, eventTime, score)
    SELECT
        toString(generateUUIDv4()) AS eventId,
        '{camId}' AS cameraId,
        'VSUM' AS eventName,
        now() AS eventTime,
        {score} AS score
    FROM system.numbers
    LIMIT 1
    SETTINGS async_insert=1, wait_for_async_insert=0;;
    """.format(**args)
    client.execute(sql)

def process_message(ch, method, props, body):
    # print(ch, method, properties, body)
    try:
        # print(body)
        # outputJson(json.dumps())
        # with open(argument.file_name, 'r') as f:  inp = json.load(f)
        inp = json.loads(body)
        camId = inp["Attributes"]["CapturedBy"]
        # ai function here
        features = extract_features_from_images(model_func, preprocess_func, target_size, inp['Images'])
        aonet = AONet()
        aonet.initialize(f_len = len(features[0]))
        aonet.load_model(model_file_path)
        predict = aonet.eval(features)
        mean = np.mean(predict)
        # print(predict)
        print("CAM: {0} - {1}".format(camId, mean))
        pushScore(camId, mean)

    except:
        logger.error()


if __name__ == '__main__':
    from rabbitmq_manager import RabbitMQManager, QueueDefinition
    definitions = [
        QueueDefinition(queue, process_message, auto_ack=True)
    ]
    manager = RabbitMQManager(uri)
    try:
        manager.run(definitions)
    except:
        logger.error()
    finally:
        manager.stop()

    logger.info('STOP')
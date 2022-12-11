import json
import os
import numpy as np
import uuid
import threading

from vsum_tools import *
from vasnet import *
from datetime import datetime
from log_helper import LogHelper as logger
from clickhouse_driver import Client
import sched, time

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
settings = {'async_insert': 1}
client = Client(host=db_host, database=db_database,
                user=db_user, password=db_password, settings=settings)

batchData = []

def pushScore(camId, score):
    batchData.append({
        'eventId': str(uuid.uuid4()),
        'cameraId': camId,
        'eventName': 'VSUM',
        'eventTime': datetime.now(),
        'score': score
    })

batchFrame = []

def process_message(ch, method, props, body):
    # print(ch, method, properties, body)
    try:
        # print(body)
        # outputJson(json.dumps())
        # with open(argument.file_name, 'r') as f:  inp = json.load(f)
        inp = json.loads(body)
        camId = inp["Attributes"]["CapturedBy"]
        batchFrame.append({
            "camId": camId,
            "frames": inp['Images']
        })
        # mean = np.mean(predict)
        # print(predict)
        # print("CAM: {0} - {1}".format(camId, mean))
        # pushScore(camId, mean)

    except:
        logger.error()

def extractFeaturesFromFrames(inp):
    # ai function here
    features = extract_features_from_images(model_func, preprocess_func, target_size, inp)
    aonet = AONet()
    aonet.initialize(f_len = len(features[0]))
    aonet.load_model(model_file_path)
    predict = aonet.eval(features)
    return predict

# init schedule to run insert
s = sched.scheduler(time.time, time.sleep)

# delay insert every 5 seconds
delayInsert = 5

def insertToDb(sc): 
    # run sql insert
    print("BEGIN INSERT:" + str(len(batchData)))
    print(batchData)
    
    if (len(batchData) > 0):
        client.execute('INSERT INTO Event(eventId, cameraId, eventName, eventTime, score) VALUES', batchData)
        batchData.clear()

    print("END INSERT")
    # re-schedule next run
    sc.enter(delayInsert, 1, insertToDb, (sc,))


def startQueue():
    print("START QUEUE")
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

def startScheduleInsert():
    # start schedule task
    print("START SCHEDULE")
    s.enter(delayInsert, 1, insertToDb, (s,))
    s.run()    

delayExtract = 5
def extractFeatures(sc): 
    # run sql insert
    print("BEGIN EXTRACT:" + str(len(batchFrame)))
    
    if (len(batchFrame) > 0):
        print("EXTRACTING: " + str(len(batchFrame)))
        extractFrames = []
        camera = []

        for frames in batchFrame:
            for frame in frames['frames']:
                extractFrames.append(frame)
                camera.append(frames['camId'])
        
        # print(camera)
        print("EXTRACTING FRAME: " + str(len(extractFrames)))
        scores = extractFeaturesFromFrames(extractFrames)
        # print(scores)
        
        mergeCamera = zip(camera, scores)
        result = dict([])

        for cam in mergeCamera:
            if cam[0] not in result:
                result[cam[0]] = []
            result[cam[0]].append(cam[1])

        # print(result)

        finalResult = dict([])
        for res in result:
            finalResult[res] = np.mean(result[res])
            pushScore(res, finalResult[res])
            
        print(finalResult)

        batchFrame.clear()

    print("END EXTRACT")
    # re-schedule next run
    sc.enter(delayExtract, 1, extractFeatures, (sc,))

def startExtractFeatures(): 
    # start extract features task
    print("START EXTRACT")
    s.enter(delayExtract, 1, extractFeatures, (s,))
    s.run()  

if __name__ == '__main__':
    try:
        # creating thread
        t1 = threading.Thread(target=startQueue)
        t2 = threading.Thread(target=startExtractFeatures)
        t3 = threading.Thread(target=startScheduleInsert)
    
        # starting thread 1
        t1.start()
        # starting thread 2
        t2.start()
        # starting thread 3
        t3.start()
    except:
        print("Error: unable to start thread")
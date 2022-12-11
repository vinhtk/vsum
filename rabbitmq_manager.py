import pika
import pika.exceptions
import time
import logging
import traceback
from log_helper import LogHelper as logger


class QueueDefinition(object):
    def __init__(self, queue_name, callback, auto_ack=True):
        self.queue_name = queue_name
        self.callback = callback
        self.auto_ack = auto_ack


class RabbitMQManager(object):
    def __init__(self, uri, try_again_in_seconds=1, prefetch_count=0):
        self.uri = uri
        self.try_again_in_seconds = try_again_in_seconds
        self.prefetch_count = prefetch_count

        self.__allow_run = True
        self.__connection = None
        self.__channel = None

    def run(self, queue_definitions):
        logger.info(f'RUN uri={self.uri} queues={",".join(q.queue_name for q in queue_definitions)}')
        while self.__allow_run:
            try:
                self.__consume(queue_definitions)
            except (pika.exceptions.ConnectionClosed, pika.exceptions.ChannelClosed, pika.exceptions.StreamLostError, pika.exceptions.AMQPConnectionError):
                logger.error()
                logger.info(f'reconnect in {self.try_again_in_seconds}s ...')
                time.sleep(self.try_again_in_seconds)

    def stop(self):
        logger.info(f'STOP uri={self.uri}')
        self.__allow_run = False
        if self.__channel:
            self.__channel.stop_consuming()

        if self.__connection:
            self.__connection.close()

    def __consume(self, queue_definitions):
        self.__connection = pika.BlockingConnection(pika.URLParameters(self.uri))
        self.__channel = self.__connection.channel()
        if self.prefetch_count > 0:
            self.__channel.basic_qos(prefetch_count=self.prefetch_count)

        for queue_definition in queue_definitions:
            self.__channel.basic_consume(queue=queue_definition.queue_name, on_message_callback=queue_definition.callback, auto_ack=queue_definition.auto_ack)
        self.__channel.start_consuming()

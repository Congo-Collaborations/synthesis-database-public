BROKER_URL = 'amqp://'
RESULT_BACKEND = 'rpc://'
CELERY_TASK_SERIALIZER = 'json'
CELERY_TASK_RESULT_EXPIRES = 3600
CELERYD_TASK_TIME_LIMIT = 18000
CELERY_IMPORTS = ("parallel_worker", )
CELERY_RESULT_SERIALIZER = 'json'
CELERY_ACCEPT_CONTENT=['json']
CELERY_TIMEZONE = 'US/Eastern'
CELERY_ENABLE_UTC = True

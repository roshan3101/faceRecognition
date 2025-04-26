import multiprocessing

# Gunicorn config variables
bind = "0.0.0.0:" + str(5000)
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gevent"
timeout = 120
keepalive = 5
max_requests = 1000
max_requests_jitter = 50 
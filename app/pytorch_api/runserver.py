
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those libraries directly.
from time import sleep
import pytorch_detector
from flask import Flask, request, abort
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import APIService
from io import BytesIO

print('Creating Application')
ACCEPTED_CONTENT_TYPES = ['image/png', 'application/octet-stream', 'image/jpeg', 'image/tiff']

app = Flask(__name__)
app.debug = True
# Use the AI4EAppInsights library to send log messages.
log = AI4EAppInsights()

# Use the APIService to executes your functions within a logging trace, supports long-running/async functions,
# handles SIGTERM signals from AKS, etc., and handles concurrent requests.
with app.app_context():
    ai4e_service = APIService(app, log)

# Define a function for processing request data, if applicable.  This function loads data or files into
# a dictionary for access in your API function.  We pass this function as a parameter to your API setup.
def process_request_data(request):
    print('Processing data...')
    return_values = {'image_bytes': None}
    try:
        # Attempt to load the body
        return_values['image_bytes'] = BytesIO(request.data)
    except:
        log.log_error('Unable to load the request data')   # Log to Application Insights
    return return_values

# POST, long-running/async API endpoint example
@ai4e_service.api_async_func(
    api_path = '/detect', 
    methods = ['POST'], 
    request_processing_function = process_request_data, # This is the data process function that you created above.
    maximum_concurrent_requests = 3, # If the number of requests exceed this limit, a 503 is returned to the caller.
    content_types = ACCEPTED_CONTENT_TYPES,
    content_max_length = 1000, # In bytes
    trace_name = 'post:detect')

def detect(*args, **kwargs):
    # Since this is an async function, we need to keep the task updated.
    taskId = kwargs.get('taskId')
    log.log_debug('Started task', taskId) # Log to Application Insights

    # Get the data from the dictionary key that you assigned in your process_request_data function.
    request_json = kwargs.get('image_bytes')

    if not request_json:
        ai4e_service.api_task_manager.FailTask(taskId, 'Task failed - Body was empty or could not be parsed.')
        return -1

    # Run your model function
    cfg = pytorch_detector.load_and_edit_cfg_for_inference()
    model = pytorch_detector.load_model_for_inference(cfg)
    # Update the task status, so the caller knows it has been accepted and is running.
    ai4e_service.api_task_manager.UpdateTaskStatus(taskId, 'running model')
    log.log_debug('Running model', taskId) # Log to Application Insights
    predictions = pytorch_detector.run_model_single_image(request_json, model)

    # Once complete, ensure the status is updated.
    log.log_debug('Completed task', taskId) # Log to Application Insights
    # Update the task with a completion event.
    ai4e_service.api_task_manager.CompleteTask(taskId, 'completed')

# GET, sync API endpoint example
@ai4e_service.api_sync_func(api_path = '/echo/<string:text>', methods = ['GET'], maximum_concurrent_requests = 1000, trace_name = 'get:echo', kwargs = {'text'})
def echo(*args, **kwargs):
    return 'Echo: ' + kwargs['text']

if __name__ == '__main__':
    app.run()

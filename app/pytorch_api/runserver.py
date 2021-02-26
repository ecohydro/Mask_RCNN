# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# # /ai4e_api_tools has been added to the PYTHONPATH, so we can reference those libraries directly.
from time import sleep
import pytorch_detector
from flask import Flask, request, abort
from ai4e_app_insights_wrapper import AI4EAppInsights
from ai4e_service import APIService
import base64

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
    return_values = {'image_bytes': None,
                    'outname': None,
                    "score_threshold" :None
                    }
    try:
        # Attempt to load the body
        return_values['image_bytes'] = base64.b64decode(request.json['data']) # b64 encoded string
        return_values['outname'] = request.json['outname'] # landsat filename string with data metadata
        return_values['score_thresh'] = request.json['score_threshold'] # to filter predictions by confidence
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
    request_bytes = kwargs.get('image_bytes')

    if not request_bytes:
        ai4e_service.api_task_manager.FailTask(taskId, 'Task failed - Body was empty or could not be parsed.')
        return -1

    # Run your model function
    cfg = pytorch_detector.load_and_edit_cfg_for_inference()
    model = pytorch_detector.load_model_for_inference(cfg)
    # Update the task status, so the caller knows it has been accepted and is running.
    ai4e_service.api_task_manager.UpdateTaskStatus(taskId, 'running model')
    log.log_debug('Running model', taskId) # Log to Application Insights
    predictions, rgb_img = pytorch_detector.run_model_single_image(request_bytes, model, cfg)

    # Once complete, ensure the status is updated.
    log.log_debug('Completed task', taskId) # Log to Application Insights

    print("Saving Image")

    from detectron2.utils.visualizer import ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.utils.visualizer import Visualizer
    import matplotlib.pyplot as plt
    metadata = MetadataCatalog.get("test")
    vis_p = Visualizer(rgb_img, metadata, instance_mode=ColorMode.SEGMENTATION)

    # move to cpu
    # instances = result['instances']
    print("score threshold")
    print(type(kwargs.get("score_thresh")))
    predictions = predictions[predictions.scores >  kwargs.get("score_thresh")]
    vis_pred_im = vis_p.draw_instance_predictions(predictions).get_image()

    def show_im(image, ax, figname):
        # Show area outside image boundaries.
        ax.axis('off')
        ax.imshow(image)
        plt.savefig(figname)
        return ax
    input_name = kwargs.get('outname')
    file_id = input_name.split(".")[0]
    figname = file_id + ".png"
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 2
    plt.style.use("seaborn")
    fig,ax = plt.subplots(figsize=(10,10))
    show_im(vis_pred_im,ax, figname)

    vector_name = file_id + "_predictions.gpkg"

    print(f"Task Completed {figname} saved")

    # Update the task with a completion event.
    ai4e_service.api_task_manager.CompleteTask(taskId, 'completed')

# GET, sync API endpoint example
@ai4e_service.api_sync_func(api_path = '/echo/<string:text>', methods = ['GET'], maximum_concurrent_requests = 1000, trace_name = 'get:echo', kwargs = {'text'})
def echo(*args, **kwargs):
    return 'Echo: ' + kwargs['text']

if __name__ == '__main__':
    app.run(debug=True)

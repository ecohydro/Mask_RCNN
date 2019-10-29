import requests
import getpass
import os
import cgi

inDir = "/scratch/rave/"  # Set input directory to the current working directory
os.chdir(inDir)      

# Enter Earthdata login credentials
username = getpass.getpass('Earthdata Username:')
password = getpass.getpass('Earthdata Password:')

API = 'https://lpdaacsvc.cr.usgs.gov/appeears/api/'  # Set the AρρEEARS API to a variable

# Insert API URL, call login service, provide credentials & return json
login_response = requests.post(f"{API}/login", auth=(username, password)).json() 
del username, password
print(login_response)

# Assign the token to a variable
token = login_response['token']
head = {'Authorization': f"Bearer {token}"}
print(head)

response = requests.get(
    'https://lpdaacsvc.cr.usgs.gov/appeears/api/task', 
    headers=head)
task_response = response.json()
print(task_response)

def done_task_ids(task_response):
    done_task_ids = []
    for task in task_response:
        try:
            if task['error'] != None:
                print(f"Task {task['task_id']} had an error \n {task}")
        except Exception as e:
            print("Task doesn't have error key, probably still in pending state")
        if task['status'] == 'done':
            done_task_ids.append(task['task_id'])
        else:
            print(f"{task['task_id']} is in status {task['status']}")
    return done_task_ids

def get_bundle_size_gb(bundle):
    filesizes_gb = [i['file_size']/1e9 for i in bundle['files']]
    return sum(filesizes_gb)

def get_bundles_and_sizes(task_ids):
    bundles = []
    for task_id in task_ids:
        bundle = requests.get(f"{API}/bundle/{task_id}").json()    # Call API and return bundle contents for the task_id as json
        print(f"Size of bundle for {task_id} is {get_bundle_size_gb(bundle)} Gb")
        bundles.append(bundle)
    return bundles

def download_bundle(bundle, root_dir):
    files = {}
    for f in bundle['files']: 
        files[f['file_id']] = f['file_name']    # Fill dictionary with file_id as keys and file_name as values
    # Set up output directory on local machine
    outDir = f"{root_dir}taskid-{bundle['task_id']}"
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print(f"made directory at {outDir}")
    print(f"Downloading files for {bundle['task_id']}")
    for file in files:
        download_response = requests.get(f"{API}/bundle/{bundle['task_id']}/{file}", stream=True)                                   # Get a stream to the bundle file
        filename = os.path.basename(cgi.parse_header(download_response.headers['Content-Disposition'])[1]['filename'])    # Parse the name from Content-Disposition header 
        filepath = os.path.join(outDir, filename)                                                                         # Create output file path
        with open(filepath, 'wb') as file:                                                                                # Write file to dest dir
            for data in download_response.iter_content(chunk_size=8192): 
                file.write(data)
    print(f"Downloading {bundle['task_id']} complete!")
                                         
def download_bundles(bundles, root_dir):
    for bundle in bundles:
        download_bundle(bundle, root_dir)


done_task_ids = done_task_ids(task_response)
bundles = get_bundles_and_sizes(done_task_ids)
download_bundles(bundles, inDir)
print("Done downloading all completed bundles")


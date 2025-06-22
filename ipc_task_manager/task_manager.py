import os
import shutil
import tempfile
import time
import pickle
import fcntl
import random
import string
from collections import deque


class TaskManager():
    """A class to manage tasks in a multiprocessing environment.
       This class is responsible for creating and managing queues for different tasks
       It create a temporary directory for storing task results and handles the lifecycle of task processes.
       The tempoary directory is created using the `tempfile` module and is cleaned up after the tasks are completed.
       There is a pushTask method to push tasks, it creates a temporary file in the temporary directory with the name,
       the file stores the task data as a python list. It returns a 
       
       """
    
    def __init__(self, temp_dir=None):
        # Check if temp_dir already exists (from unpickling)
        if hasattr(self, 'temp_dir') and self.temp_dir:
            return  # Don't recreate if already set
            
        if temp_dir:
            self.temp_dir = temp_dir
            os.makedirs(temp_dir, exist_ok=True)
        else:
            self.temp_dir = tempfile.mkdtemp()


    
    def push_task(self, task):
        """ Task should be a dictionary with 'name' and 'data' keys.
            This method creates a temporary file in the temporary directory with the task name
            and stores the task data as a Python queue in that file.
        """
        if not isinstance(task, dict) or 'name' not in task or 'data' not in task:
            raise ValueError("Task must be a dictionary with 'name' and 'data' keys.")

        task_name = task['name']
        # add a task number field that is a unique number for each task
        # Unique task code consists of the name of the task _ based on current time in milliseconds _ plus some random letters


        number_part = int(time.time() * 1000)  
        random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        unique_task_code = task_name + "^|^" + str(number_part) + "^|^" + random_part
        task['task_code'] = unique_task_code

        # Check if the task name already exists in the temporary directory
        task_file_path = os.path.join(self.temp_dir, f"{task_name}.task")


        # Create the task file and write the task data
        if os.path.exists(task_file_path):
            # get the lock of the file
            # this part is the tasks part
            with open(task_file_path, 'rb+') as task_file:
                fcntl.flock(task_file, fcntl.LOCK_EX)
                task_file.seek(0)
                # read existing data
                existing_queue = pickle.load(task_file)
                # push the new task data to the existing queue 
                if isinstance(existing_queue,  deque):
                    # check if the queue has the end signal
                    # pop the last task if it is the end signal
                    # remove it if not put it back
                    if existing_queue and existing_queue[-1].get('data', {}).get('__end__'):
                        existing_queue.pop()
                    existing_queue.append(task)
                else:
                    # Through an error if the existing data is not a queue
                    raise ValueError("Existing data in the task file is not a queue.")
                # move the file pointer to the beginning of the file
                task_file.truncate()  # Clear the file content
                # write the updated data back to the file
                task_file.seek(0)
                pickle.dump(existing_queue, task_file)
                fcntl.flock(task_file, fcntl.LOCK_UN)
        else:
            # Create a new task file and write the task data
            with open(task_file_path, 'wb') as task_file:
                fcntl.flock(task_file, fcntl.LOCK_EX)
                # create a queue with the task data
                task_queue = deque()
                task_queue.append(task)
                pickle.dump(task_queue, task_file)
                fcntl.flock(task_file, fcntl.LOCK_UN)

        # results part 
        result_file_path = os.path.join(self.temp_dir, f"{task_name}.res")

        # the result file stores a dictionary with task numbers as keys and results as values
        if not os.path.exists(result_file_path):
            with open(result_file_path, 'wb') as result_file:
                fcntl.flock(result_file, fcntl.LOCK_EX)
                # empty dictionary for results
                pickle.dump({}, result_file)
                fcntl.flock(result_file, fcntl.LOCK_UN)


        # return the unique task number
        return unique_task_code
    

    def pull_task(self, task_name):
        """Pulls a task from the temporary directory by its name.
           Returns a single task,
           if there is no task with the given name, it keeps waiting until a task is available.
           The task is expected to be a dictionary with 'name' and 'data' keys.
           The task data is stored in a queue in a temporary file.
        """
        ## call  the pull_task_timeout with a timeout of None
        return self.pull_task_timeout(task_name, timeout=None)

    def pull_task_timeout(self, task_name, timeout=10):
        """Pulls a task from the temporary directory by its name with a timeout.
           Returns a single task, if there is no task with the given name within the timeout,
           it returns None.
           The task is expected to be a dictionary with 'name' and 'data' keys.
           The task data is stored in a queue in a temporary file.
           If the end signal is received, it returns None.
        """
        task_file_path = os.path.join(self.temp_dir, f"{task_name}.task")
        start_time = time.time()

        while True:
            if os.path.exists(task_file_path):
                with open(task_file_path, 'rb+') as task_file:
                    fcntl.flock(task_file, fcntl.LOCK_EX)
                    try:
                        tasks = pickle.load(task_file)
                        if isinstance(tasks, deque) and tasks:
                            # Pop the first task from the list
                            task = tasks.popleft()


                            # Check if the task is the end signal throw an exception
                            if '__end__' in task.get('data', {}):
                                fcntl.flock(task_file, fcntl.LOCK_UN)
                                return None
                            else:
                                # Write the updated list back to the file
                                task_file.seek(0)
                                pickle.dump(tasks, task_file)
                                # Truncate the file to remove any leftover data
                                task_file.truncate()
                                fcntl.flock(task_file, fcntl.LOCK_UN)
                            return task
                    except IndexError:
                        # If the file is empty, wait for a new task
                        fcntl.flock(task_file, fcntl.LOCK_UN)
                        time.sleep(1)
                        continue

            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                return None

            time.sleep(1)

    

    def put_result(self, task_code, result):
        """" Store the result of a task in the result file.
            The result file is a temporary file in the temporary directory with the name of the task.
            The result is stored as a dictionary with task codes as keys and results as values.
            If the result file does not exist, it creates a new one.
            """


        result_file_path = os.path.join(self.temp_dir, f"{task_code}.res")


        
        with open(result_file_path, 'wb') as result_file:
            fcntl.flock(result_file, fcntl.LOCK_EX)
            # Move the file pointer to the beginning of the file
            result_file.seek(0)
            # Write the updated results dictionary back to the file
            pickle.dump(result, result_file)
            fcntl.flock(result_file, fcntl.LOCK_UN)


    def get_result(self, task_code):
        """ Get the results until it is available, if not available, it keeps waiting until the result is available."""

        # call the get_results_timeout with a timeout of None
        return self.get_results_timeout(task_code, timeout=None)

    def get_results_timeout(self, task_name, timeout=10):
        """Get results for a specific task name with a timeout.
           Returns the results as a dictionary with task codes as keys and results as values.
           If the result is not available within the timeout, it returns Null.
        """

        result_file_path = os.path.join(self.temp_dir, f"{task_name}.res")
        start_time = time.time()

        while True:
            if os.path.exists(result_file_path):
                with open(result_file_path, 'rb') as result_file:
                    fcntl.flock(result_file, fcntl.LOCK_EX)
                    try:
                        results = pickle.load(result_file)
                        return results
                    except EOFError:
                        # If the file is empty, wait for a new result
                        fcntl.flock(result_file, fcntl.LOCK_UN)
                        time.sleep(1)

            # Check if timeout has been reached
            if timeout is not None and time.time() - start_time > timeout:
                return None

            time.sleep(1)

        
    def cleanup(self):
        """Clean up the temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()



    def get_temp_dir(self):
        """Get the temporary directory path."""
        return self.temp_dir
    

    def send_stop_signal(self, task_name):
        """Send a stop signal to the task manager.
           This method creates a stop task with the name 'task_name' and data {'__end__': True}.
           It pushes the stop task to the task manager.
        """
        stop_task = {"name": task_name, "data": {"__end__": True}}
        self.push_task(stop_task)
    
    
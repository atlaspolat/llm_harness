from ipc_task_manager import TaskManager
import torch

from multiprocessing import Process


def produce_tasks(task_manager, task_name):
    """Produce tasks for the TaskManager."""

    for i in range(5):
        

        # print the temp dir

        print(task_manager.temp_dir)

        # Example tasks
        task = {"name": task_name,
             "data": {"info": "data1"}}

        key = task_manager.push_task(task)

        # wait for the result

        print(f"Produced task {i+1} with key: {key}")

        # get the result
        result = task_manager.get_result(key)
        print(f"Result for task {i+1}: {result}")


def produce_results(task_manager, task_name, device='cpu'):
    """Produce results for the TaskManager."""

    while True:
        # check if there are tasks to process
        task = task_manager.pull_task(task_name)

        if '__end__' in task:
            print("Ending...")
            break

        # get the data from the task
        data = task.get('data', {})
        #get the task code
        task_code = task.get('task_code', '')


        if task_code == '':
            # Throw an exception if the task code is not found
            raise ValueError("Task code not found in the task.")
        
        print(f"Processing task: {task_code} with data: {data}")

        # Simulate processing the task

        # normalize a random tensor
        tensor1 = torch.randn(2000, 2000).to(device)

        tensor1 = tensor1 / torch.norm(tensor1)

        tensor2 = torch.randn(2000, 2000).to(device)

        tensor2 = tensor2 / torch.norm(tensor2)

        # Perform a matrix multiplication
        result_tensor = torch.matmul(tensor1, tensor2)
        # Get the determinant of the result tensor

        det_result = torch.det(result_tensor)

        # Create the result dictionary

        result = {
            "task_code": task_code,
            "task_type": task_name,
            "data": {
                "det_result": det_result.item(),
            }
        }


        # Store the result in the TaskManager
        task_manager.put_result(task_code, result)

        print(f"Processed task: {task_code} with result: {result}")







        
        
if __name__ == "__main__":
    # Create a TaskManager instance
    task_manager = TaskManager()

    # Define the task name
    task_name = "example_task"


    # Get all the cuda devices available
    cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    print(f"Available CUDA devices: {cuda_devices}")

    # Create processes for each cuda device that produces results
    workers = []
    for device in cuda_devices:
        p = Process(target=produce_results, args=(task_manager, task_name, device))
        p.start()
        workers.append(p)

    bosses = []
    # Create ten processes that produce tasks
    for i in range(5):
        p = Process(target=produce_tasks, args=(task_manager, task_name))
        p.start()
        bosses.append(p)

    # Wait for all producer processes to finish
    try:
        for p in bosses:
                p.join()
        
        # Send stop signals to workers
        for device in cuda_devices:
            stop_task = {"name": task_name, "data": {"__end__": True}}
            task_manager.push_task(stop_task)
        
        # Wait for workers to finish
        for p in workers:
                p.join()
                
    except KeyboardInterrupt:
        print("Interrupted by user")
        for p in workers:
            p.terminate()
    
    finally:
        task_manager.cleanup()


    
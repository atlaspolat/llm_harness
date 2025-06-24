import torch
from multiprocessing import Process, Queue

# task scheme
# task = {
#     "task_type": "ocr" | "imageqa" | "geo",

#     "data": {
#         "image": "path/to/image.jpg",
#         "question": "What is in the image?",
#     }
#      "return_queue": result_queue
# }




def geo_process(task_queue, return_queue, device="cuda:0"):
    
    print(f"Geo process started on {device}")

    while True:

        if task_queue.empty():
            continue
    
        task = task_queue.get()

        if task == "STOP":
            print(f"Geo process on {device} stopping.")
            break
    
        # Process the task

        # do a big matrix multiplication to simulate a long task

        print(f"Geo process on {device} processing task: {task["data"]["question"]}")


        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)
        c = torch.matmul(a, b)

        # find the determinant of the matrix a and b and c
        det_a = torch.det(a)
        det_b = torch.det(b)
        det_c = torch.det(c)

        result = {
            "task_type": "geo",
            "data": {
                "det_a": det_a.item(),
                "det_b": det_b.item(),
                "det_c": det_c.item(),
            }
        }

        # put the result in the return queue
        return_queue.put(result)

        print(f"Geo process on {device} finished task: {task['data']['question']}")







def ocr_process(task_queue, return_queue, device="cuda:0"):
    print(f"OCR process started on {device}")


    while True:
        if task_queue.empty():
            continue
        task = task_queue.get()

        if task == "STOP":
            print(f"OCR process on {device} stopping.")
            break
        # Process the task
        print(f"OCR process on {device} processing task: {task['data']['question']}")
        # do a big matrix multiplication to simulate a long task

        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)


        # normalize the matrix a and b
        # 
        a = a / torch.norm(a)
        b = b / torch.norm(b)

        c = torch.matmul(a, b)

        # return the determinant of the matrix c
        det_c = torch.det(c)

        result = {
            "task_type": "ocr",
            "data": {
                "det_c": det_c.item(),
            }
        }

        # put the result in the return queue
        return_queue.put(result)

        print(f"OCR process on {device} finished task: {task['data']['question']}")



def imageqa_process(task_queue, return_queue, device="cuda:0"):
    print(f"ImageQA process started on {device}")

    while True:
        if task_queue.empty():
            continue
        task = task_queue.get()

        if task == "STOP":
            print(f"ImageQA process on {device} stopping.")
            break
        # Process the task
        print(f"ImageQA process on {device} processing task: {task['data']['question']}")
        # do a big matrix multiplication to simulate a long task

        a = torch.randn(2000, 2000, device=device)
        b = torch.randn(2000, 2000, device=device)

        ## apply a ReLU activation function to the matrix a and b
        a = torch.relu(a)
        b = torch.relu(b)
        c = torch.matmul(a, b)

        # return the determinant of the matrix c
        det_c = torch.det(c)
        result = {
            "task_type": "imageqa",
            "data": {
                "det_c": det_c.item(),
            }
        }

        # put the result in the return queue
        return_queue.put(result)
        print(f"ImageQA process on {device} finished task: {task['data']['question']}")





def llm_process(ocr_queue, ocr_result_queue, imageqa_queue, imageqa_result_queue, geo_queue, geo_result_queue, device="cuda:0"):

    print(f"LLM process started on {device}")


    # Set up a timer and work for 15 seconds
    import time
    start_time = time.time()

    while True:
        if time.time() - start_time > 25:
            print(f"LLM process on {device} stopping after 15 seconds.")
            break

        # add some tasks to the queues

        task = {
            "task_type": "ocr",
            "data": {
                "image": "path/to/image.jpg",
                "question": "What is written in the image?",
            },
        }

        ocr_queue.put(task)

        task = {
            "task_type": "imageqa",
            "data": {
                "image": "path/to/image.jpg",
                "question": "What is in the image?",
            },
        }
        imageqa_queue.put(task)

        task = {
            "task_type": "geo",
            "data": {
                "image": "path/to/image.jpg",
                "question": "What is the location in the image?",
            },

        }

        geo_queue.put(task)


        # collect results from the queues

        # wait for results from ocr

        ocr_result = ocr_result_queue.get(timeout=15)
        if not ocr_result:
            print("No OCR result received.")
        else:
            print(f"OCR result received: {ocr_result}")
        
        imageqa_result = imageqa_result_queue.get(timeout=15)
        if not imageqa_result:
            print("No ImageQA result received.")
        else:
            print(f"ImageQA result received: {imageqa_result}")    
        
        geo_result = geo_result_queue.get(timeout=15)
        if not geo_result:
            print("No Geo result received.")
        else:
            print(f"Geo result received: {geo_result}")
        

        #wait for a second
        time.sleep(1)
    


    # send the stop signal to all processes
    ocr_queue.put("STOP")
    imageqa_queue.put("STOP")
    geo_queue.put("STOP")

    print("LLM process stopping and sending stop signals to all processes.")

  

    



if __name__ == "__main__":

    #ocr queue
    #set up a queue for OCR tasks
    ocr_queue = Queue()

    # imageqa queue
    imageqa_queue = Queue()


    geo_queue = Queue()

    # create result  queues for each task type
    ocr_result_queue = Queue()
    imageqa_result_queue = Queue()
    geo_result_queue = Queue()




    # get all the available devices

    print("Available devices:")
    print(torch.cuda.device_count(), "GPUs available")

    # make a list of all available devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]


    # if there are less than 4 devices, exception
    if len(devices) < 4:
        raise Exception("Not enough devices available. At least 4 GPUs are required.")
    # create a process for each device
    processes = []

    for i, device in enumerate(devices):
        if i == 0:
            # imageqa process
            p = Process(target=imageqa_process, args=(imageqa_queue, imageqa_result_queue, device))
        elif i == 1:
            # geo process
            p = Process(target=geo_process, args=(geo_queue, geo_result_queue, device))
        elif i == 2:
            # ocr process
            p = Process(target=ocr_process, args=(ocr_queue, ocr_result_queue, device))
        else:
            # owner process
            p = Process(target=llm_process, args=(ocr_queue, ocr_result_queue, imageqa_queue, imageqa_result_queue, geo_queue, geo_result_queue, device))
        processes.append(p)
        p.start()


    # check if more than one device is available
    # make an ownner process and the rest of the processes are workers

    for p in processes:
        p.join()
        print(f"Process {p.name} stopped.")

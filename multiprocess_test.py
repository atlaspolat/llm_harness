import torch

def proc1(device="cuda:0"):
    # do a giant matrix multiplication
    print("Process 1 is starting")
    a = torch.randn(10000, 10000, device=device)
    b = torch.randn(10000, 10000, device=device)
    c = torch.matmul(a, b)
    print("Process 1 finished matrix multiplication on", device)






from multiprocessing import Process


if __name__ == "__main__":


    # get all the available devices

    print("Available devices:")
    print(torch.cuda.device_count(), "GPUs available")

    # make a list of all available devices
    devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]


    # allocate a process for each device
    processes = []
    for device in devices:
        p = Process(target=proc1, args=(device,))
        processes.append(p)
        p.start()
    # wait for all processes to finish
    for p in processes:
        p.join()
    print("All processes finished")
    print("Exiting main process")
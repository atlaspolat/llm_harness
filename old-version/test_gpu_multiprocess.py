"""
Simple test script to verify that multiprocessing with GPUs works correctly.
This will help debug any issues before running the full harness.
"""

import torch
import os
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def test_gpu_worker(gpu_id):
    """Simple worker function to test GPU allocation in separate processes"""
    
    print(f"[Process {os.getpid()}] Starting on GPU {gpu_id}")
    
    # Set the GPU for this process
    torch.cuda.set_device(gpu_id)
    
    # Print GPU info for this process
    print(f"[Process {os.getpid()}] CUDA available: {torch.cuda.is_available()}")
    print(f"[Process {os.getpid()}] Current device: {torch.cuda.current_device()}")
    print(f"[Process {os.getpid()}] Device count: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        # Create a simple tensor on the GPU to verify it's working
        device = f"cuda:{gpu_id}"
        test_tensor = torch.randn(1000, 1000, device=device)
        
        # Do some computation
        for i in range(5):
            result = torch.matmul(test_tensor, test_tensor.T)
            print(f"[Process {os.getpid()}] GPU {gpu_id} - Iteration {i+1} completed")
            time.sleep(1)  # Simulate work
        
        # Check memory usage
        allocated = torch.cuda.memory_allocated(device) / 1e6  # MB
        cached = torch.cuda.memory_reserved(device) / 1e6  # MB
        print(f"[Process {os.getpid()}] GPU {gpu_id} Memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
        
        # Clean up
        del test_tensor, result
        torch.cuda.empty_cache()
        
        print(f"[Process {os.getpid()}] GPU {gpu_id} test completed successfully!")
        return f"GPU {gpu_id} - Process {os.getpid()} - SUCCESS"
    else:
        print(f"[Process {os.getpid()}] CUDA not available!")
        return f"GPU {gpu_id} - Process {os.getpid()} - FAILED (No CUDA)"

def main():
    # Set multiprocessing start method to 'spawn' for CUDA compatibility
    mp.set_start_method('spawn', force=True)
    
    print("="*60)
    print("GPU MULTIPROCESSING TEST")
    print("="*60)
    
    # Print main process GPU info
    print(f"Main process ID: {os.getpid()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus == 0:
        print("No GPUs available. Exiting.")
        return
    
    print(f"\nStarting test with {num_gpus} GPUs using ProcessPoolExecutor...")
    start_time = time.time()
    
    # Test with ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        # Submit all GPU tasks
        futures = []
        for i in range(num_gpus):
            print(f"Submitting task for GPU {i}")
            future = executor.submit(test_gpu_worker, i)
            futures.append(future)
        
        # Wait for all tasks to complete
        results = []
        for i, future in enumerate(futures):
            try:
                result = future.result()
                results.append(result)
                print(f"GPU {i} task completed: {result}")
            except Exception as e:
                print(f"GPU {i} task failed: {e}")
                import traceback
                traceback.print_exc()
    
    end_time = time.time()
    print(f"\nTest completed! Total time: {end_time - start_time:.2f} seconds")
    print(f"Results: {results}")
    
    # Verify all GPUs were used
    if len(results) == num_gpus:
        print("✅ SUCCESS: All GPUs were utilized in parallel!")
    else:
        print("❌ FAILURE: Not all GPUs were utilized!")

if __name__ == "__main__":
    main()

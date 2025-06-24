from ipc_task_manager import TaskManager


## download the dataset

def create_question_tasks(task_manager):

    task= {
        "name": "question",
        "data": {
            "question": "What is the capital of France?",
            "context": "Paris is the capital of France."
        }
    }



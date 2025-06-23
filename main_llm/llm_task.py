from transformers import AutoModelForCausalLM, AutoTokenizer






def  llm_task(task_manager, task_name, config):
    """
    Executes a task using the task manager.

    It find and downloands the model in the yaml config file, then executes the task

    Args:
        task_manager: An instance of TaskManager to handle the task.
        task_name (str): The name of the task to execute.

    Returns:
        str: The result of the executed task.

    """


    pass


def download_model(config):
    """
    Downloads the model specified in the config file.

    Args:
        model_name (str): The name of the model to download.
        config (dict): Configuration dictionary containing model details.

    Returns:
        str: Path to the downloaded model.
    """
    
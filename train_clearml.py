import os
from pathlib import Path
from importlib.util import spec_from_file_location, module_from_spec
from clearml import Task

yolov5_repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
print('repo_path',yolov5_repo_path)
yolov5_train_path = Path(yolov5_repo_path) / "train.py"
spec = spec_from_file_location("train", yolov5_train_path)
train_module = module_from_spec(spec)
spec.loader.exec_module(train_module)

train = train_module.train

task = Task.init(project_name="coverDataset", task_name="remote-test")
task.set_base_docker("ultralytics/yolov5:latest")

task.execute_remotely(queue_name="god")

if __name__ == "__main__":
    train()

from pathlib import Path

from ddiffpg.utils.common import list_class_names

cur_path = Path(__file__).resolve().parent
model_name_to_path = list_class_names(cur_path)

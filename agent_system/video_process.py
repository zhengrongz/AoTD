import inspect
import vision_models
from configs import config
import torch


def make_fn(model_class, process_name, counter):
    """
    model_class.name and process_name will be the same unless the same model is used in multiple processes, for
    different tasks
    """
    # We initialize each one on a separate GPU, to make sure there are no out of memory errors
    num_gpus = torch.cuda.device_count()
    gpu_number = counter % num_gpus

    model_instance = model_class(gpu_number=gpu_number)

    def _function(*args, **kwargs):
        if process_name != model_class.name:
            kwargs['process_name'] = process_name

        if model_class.to_batch:
            # Batchify the input. Model expects a batch. And later un-batchify the output.
            args = [[arg] for arg in args]
            kwargs = {k: [v] for k, v in kwargs.items()}

            # The defaults that are not in args or kwargs, also need to listify
            full_arg_spec = inspect.getfullargspec(model_instance.forward)
            if full_arg_spec.defaults is None:
                default_dict = {}
            else:
                default_dict = dict(zip(full_arg_spec.args[-len(full_arg_spec.defaults):], full_arg_spec.defaults))
            non_given_args = full_arg_spec.args[1:][len(args):]
            non_given_args = set(non_given_args) - set(kwargs.keys())
            for arg_name in non_given_args:
                kwargs[arg_name] = [default_dict[arg_name]]

        try:
            out = model_instance.forward(*args, **kwargs)
            if model_class.to_batch:
                out = out[0]
        except Exception as e:
            print(f'Error in {process_name} model:', e)
            out = None
        return out

    return _function

list_models = [m[1] for m in inspect.getmembers(vision_models, inspect.isclass)
            if issubclass(m[1], vision_models.BaseModel) and m[1] != vision_models.BaseModel]
# Sort by attribute "load_order"
list_models.sort(key=lambda x: x.load_order)

consumers = dict()

counter_ = 0
for model_class_ in list_models:
    for process_name_ in model_class_.list_processes():
        if process_name_ in config.load_models and config.load_models[process_name_]:
            consumers[process_name_] = make_fn(model_class_, process_name_, counter_)
            counter_ += 1

queues_in = None
print(consumers.keys())


def forward(model_name, *args, queues=None, **kwargs):
    out = consumers[model_name](*args, **kwargs)

    return out
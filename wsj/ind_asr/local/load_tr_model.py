from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.asr.asr_utils import get_model_conf
from espnet.utils.dynamic_import import dynamic_import
from espnet.asr.asr_utils import torch_load
import os
import logging

def load_trained_model(model_path, training=True):
    """Load the trained model for recognition.

    Args:
        model_path (str): Path to model.***.best
        training (bool): Training mode specification for transducer model.

    Returns:
        model (torch.nn.Module): Trained model.
        train_args (Namespace): Trained model arguments.

    """
    idim, odim, train_args = get_model_conf(
        model_path, os.path.join(os.path.dirname(model_path), "model.json")
    )

    logging.info(f"Reading model parameters from {model_path}")

    if hasattr(train_args, "model_module"):
        model_module = train_args.model_module
    else:
        model_module = "espnet.nets.pytorch_backend.e2e_asr:E2E"

    # CTC Loss is not needed, default to builtin to prevent import errors
    if hasattr(train_args, "ctc_type"):
        train_args.ctc_type = "builtin"

    model_class = dynamic_import(model_module)

    if "transducer" in model_module:
        model = model_class(idim, odim, train_args, training=training)
        custom_torch_load(model_path, model, training=training)
    else:
        model = model_class(idim, odim, train_args)
        from acc_classifier import AccentClassifier
        classifier = AccentClassifier(5)
        from DANN_KMeans import GRL,DANN_ASR
        mdl = DANN_ASR(model,classifier,train_args) 
        
        torch_load(model_path, mdl)

    return mdl, train_args

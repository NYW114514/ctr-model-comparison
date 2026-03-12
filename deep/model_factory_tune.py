# model_factory_tune.py
import inspect
from deepctr_torch.models import WDL, DeepFM, DCN, xDeepFM, AutoInt, PNN, FiBiNET
from config import (
    DNN_HIDDEN_UNITS, DROPOUT,
    L2_REG_LINEAR, L2_REG_EMBEDDING, L2_REG_DNN, L2_REG_CIN,
)

MODEL_ZOO = {
    "WDL": WDL,
    "DeepFM": DeepFM,
    "DCN": DCN,
    "xDeepFM": xDeepFM,
    "AutoInt": AutoInt,
    "PNN": PNN,
    "FiBiNET": FiBiNET,
}

def _filter_kwargs_by_signature(cls, kwargs: dict) -> dict:
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}

def create_model(model_name: str, feature_columns, device="cuda",
                 dnn_hidden_units=None,
                 dropout=None,
                 l2_reg_embedding=None,
                 l2_reg_dnn=None,
                 cross_num=None,
                 cin_layer_size=None,
                 att_layer_num=None,
                 att_head_num=None,
                 bilinear_type=None):
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_ZOO.keys())}")

    ModelCls = MODEL_ZOO[model_name]

    _dnn_hidden_units = dnn_hidden_units if dnn_hidden_units is not None else DNN_HIDDEN_UNITS
    _dropout          = dropout          if dropout          is not None else DROPOUT
    _l2_reg_embedding = l2_reg_embedding if l2_reg_embedding is not None else L2_REG_EMBEDDING
    _l2_reg_dnn       = l2_reg_dnn       if l2_reg_dnn       is not None else L2_REG_DNN

    common_kwargs = dict(
        dnn_feature_columns=feature_columns,
        linear_feature_columns=feature_columns,
        dnn_hidden_units=_dnn_hidden_units,
        dnn_dropout=_dropout,
        l2_reg_linear=L2_REG_LINEAR,
        l2_reg_embedding=_l2_reg_embedding,
        l2_reg_dnn=_l2_reg_dnn,
        task="binary",
        device=device,
    )

    if cross_num is not None and model_name == "DCN":
        common_kwargs["cross_num"] = cross_num

    if cin_layer_size is not None and model_name == "xDeepFM":
        common_kwargs["cin_layer_size"] = cin_layer_size

    if att_layer_num is not None and model_name == "AutoInt":
        common_kwargs["att_layer_num"] = att_layer_num

    if att_head_num is not None and model_name == "AutoInt":
        common_kwargs["att_head_num"] = att_head_num

    if bilinear_type is not None and model_name == "FiBiNET":
        common_kwargs["bilinear_type"] = bilinear_type

    if model_name == "xDeepFM":
        common_kwargs["l2_reg_cin"] = L2_REG_CIN

    safe_kwargs = _filter_kwargs_by_signature(ModelCls, common_kwargs)
    return ModelCls(**safe_kwargs)
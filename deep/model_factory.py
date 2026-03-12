# model_factory.py
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
    # 去掉 self
    allowed.discard("self")
    return {k: v for k, v in kwargs.items() if k in allowed}

def create_model(model_name: str, feature_columns, device="cuda"):
    if model_name not in MODEL_ZOO:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_ZOO.keys())}")

    ModelCls = MODEL_ZOO[model_name]

    common_kwargs = dict(
        dnn_feature_columns=feature_columns,

        linear_feature_columns=feature_columns,

        dnn_hidden_units=DNN_HIDDEN_UNITS,
        dnn_dropout=DROPOUT,

        l2_reg_linear=L2_REG_LINEAR,
        l2_reg_embedding=L2_REG_EMBEDDING,
        l2_reg_dnn=L2_REG_DNN,

        task="binary",
        device=device,
    )

    # ====== 个别模型的结构差异 ======
    if model_name == "xDeepFM":
        # xDeepFM 有 CIN 正则参数
        common_kwargs["l2_reg_cin"] = L2_REG_CIN

    if model_name == "PNN":
        # PNN 没有 linear_feature_columns；只用 dnn_feature_columns
        common_kwargs.pop("linear_feature_columns", None)

    if model_name == "AutoInt":
        # AutoInt 不吃 l2_reg_linear
        common_kwargs.pop("l2_reg_linear", None)

    # 按签名过滤参数
    safe_kwargs = _filter_kwargs_by_signature(ModelCls, common_kwargs)

    return ModelCls(**safe_kwargs)

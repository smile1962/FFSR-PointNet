def build_model(model_name, input_channels=4, output_channels=2, **kwargs):
    """Create either the SR-Unet or the FLO-SR CNN model."""
    name = str(model_name).lower()
    if name in ("srunet", "sr_unet", "sr-unet", "unet"):
        from model.SRUnet import SuperResolutionModel

        return SuperResolutionModel(
            input_channels=input_channels,
            output_channels=output_channels,
            **kwargs,
        )
    if name in ("flosr", "flo-sr", "flo_sr", "edsr"):
        from model.FLO_SR import FLOSRModel

        return FLOSRModel(
            input_channels=input_channels,
            output_channels=output_channels,
            **kwargs,
        )
    raise ValueError(f"Unknown model name: {model_name}")


"""
This file is based on Transformer for Partial Differential Equations' Operator Learning defined in paper [https://arxiv.org/abs/2205.13671.pdf].

"""


from .encoder import SpatialEncoder2D, Encoder1D, SpatialTemporalEncoder2D
from .decoder import PointWiseDecoder2DSimple, PointWiseDecoder2DComplex, PointWiseDecoder1D

def build_model_2d(res, in_channels=3) -> (SpatialEncoder2D, PointWiseDecoder2DSimple):
    # currently they are hard coded
    encoder = SpatialEncoder2D(
        in_channels,   # a + xy coordinates
        96,
        256,
        4,
        6,
        res=res,
        use_ln=True
    )

    decoder = PointWiseDecoder2DSimple(
        latent_channels=256,
        out_channels=1,
        res=res,
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder

def build_model_2d_mech(res, in_channels=3) -> (SpatialEncoder2D, PointWiseDecoder2DComplex):
    # currently they are hard coded
    encoder = SpatialEncoder2D(
        in_channels,   # a + xy coordinates
        384,
        256,
        4,
        8,
        res=res,
        use_ln=True
    )

    decoder = PointWiseDecoder2DComplex(
        latent_channels=256,
        out_channels=1,
        res=res,
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder

def build_model_2d_mech_mnist(res, in_channels=3) -> (SpatialEncoder2D, PointWiseDecoder2DComplex):
    # currently they are hard coded
    encoder = SpatialEncoder2D(
        in_channels,   # a + xy coordinates
        96,
        256,
        4,
        6,
        res=res,
        use_ln=True
    )

    decoder = PointWiseDecoder2DComplex(
        latent_channels=256,
        out_channels=1,
        res=res,
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder

def build_model_1d(res) -> (Encoder1D, PointWiseDecoder1D):
    # currently they are hard coded
    encoder = Encoder1D(
        2,   # u + x coordinates
        96,
        96,
        4,
        res=res
    )

    decoder = PointWiseDecoder1D(
        96,
        1,
        3,
        scale=2,
        res=res
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder



def build_model_2d_time(in_channels) -> (SpatialTemporalEncoder2D, PointWiseDecoder2DComplex):

    encoder = SpatialTemporalEncoder2D(
        in_channels,
        96,
        384,
        1,
        5,
    )

    decoder = PointWiseDecoder2DSimple(
        latent_channels=384,
        out_channels=1,
        scale=8,
        res=64,
    )

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad) + \
                   sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total_params}')
    return encoder, decoder

def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
        feat: feature name
        feat_num: the total number of sparse features that do not repeat
        embed_dim: embedding dimension
    """
    return {'feat_name': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
        feat: dense feature name
    """
    return {'feat_name': feat}
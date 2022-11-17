class BoxCoder(object):
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type:(Tuple[float,float,float,float],float)->None
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

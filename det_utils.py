import math


class BoxCoder(object):
    def __init__(self, weights, bbox_xform_clip=math.log(1000. / 16)):
        # type:(Tuple[float,float,float,float],float)->None
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip


class Matcher(object):

    def __init__(self, high_threshold, low_threshold):
        self.BELOW_LOW_THRESHHOLD = -1
        self.BETWEEN_THRESHOLD = -2
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

    def __call__(self):
        pass

class BalancedPositiveNegativeSampler(object):
    def __init__(self,batch_size_per_image,positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
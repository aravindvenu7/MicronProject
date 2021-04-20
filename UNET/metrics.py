
import numpy as np
import keras.backend as K
from tensorflow import ones_like, equal   #, log
#from tensorflow.python import mul
import tensorflow as tf
from tensorflow.python.keras import losses

def sp(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Specificity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity
def sn(y_true, y_pred):
    """
    param:
    y_pred - Predicted labels
    y_true - True labels 
    Returns:
    Sensitivity score
    """
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    tp = K.sum(y_true * y_pred)
    fn = K.sum(y_true * neg_y_pred )
    sensitivity = tp / (tp + fn + K.epsilon())
    return sensitivity  
def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score
#returns DSC     
 
def dice_loss(y_true,y_pred):
  loss =1- dice_coeff(y_true,y_pred)
  return loss 
def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + K.math.log(dice_loss(y_true, y_pred))
    return loss 
def soft_dice_loss(y_true, y_pred ): 
    ''' 
    Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    
    # References
        V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation 
        https://arxiv.org/abs/1606.04797
        More details on Dice loss formulation 
        https://mediatum.ub.tum.de/doc/1395260/1395260.pdf (page 72)
        
        Adapted from https://github.com/Lasagne/Recipes/issues/99#issuecomment-347775022
    '''
    epsilon=1e-6
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(0, len(y_pred.shape))) 
    numerator = 2. * K.sum(y_pred * y_true)
    denominator = K.sum(K.square(y_pred) + K.square(y_true))
    
    return 1 - K.mean(numerator / (denominator + epsilon))


def weighted_binary_crossentropy(pos_weight=1):
   
    
    def _to_tensor(x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)
  
  
    def _calculate_weighted_binary_crossentropy(target, output, from_logits=False):
        """Calculate weighted binary crossentropy between an output tensor and a target tensor.
        # Arguments
            target: A tensor with the same shape as `output`.
            output: A tensor.
            from_logits: Whether `output` is expected to be a logits tensor.
                By default, we consider that `output`
                encodes a probability distribution.
        # Returns
            A tensor.
        """
        # Note: tf.nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            _epsilon = _to_tensor(K.common.epsilon(), output.dtype.base_dtype)
            output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
            output = tf.log(output / (1 - output))
        return tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=output, pos_weight=pos_weight)


    def _weighted_binary_crossentropy(y_true, y_pred):
        return K.mean(_calculate_weighted_binary_crossentropy(y_true, y_pred), axis=-1)
    
    return _weighted_binary_crossentropy

from network import Network
import tensorflow as tf

class ResNet50(Network):
    def setup(self, is_training, num_classes):
        '''Network definition.
        Args:
          is_training: whether to update the running mean and variance of the batch normalisation layer.
                       If the batch size is small, it is better to keep the running mean and variance of
                       the-pretrained model frozen.
          num_classes: number of classes to predict (including background).
        '''
        (self.feed('data')
             .conv(3, 3, 64, 2, 2, biased=False, relu=False, padding='SAME', name='conv1_1_3x3_s2')
             .batch_normalization(relu=False, name='conv1_1_3x3_s2_bn')
             .relu(name='conv1_1_3x3_s2_bn_relu')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_2_3x3')
             .batch_normalization(relu=True, name='conv1_2_3x3_bn')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, padding='SAME', name='conv1_3_3x3')
             .batch_normalization(relu=True, name='conv1_3_3x3_bn')
             .max_pool(3, 3, 2, 2, padding='SAME', name='pool1_3x3_s2')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_proj')
             .batch_normalization(relu=False, name='conv2_1_1x1_proj_bn'))

        (self.feed('pool1_3x3_s2')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding1')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_1_3x3')
             .batch_normalization(relu=True, name='conv2_1_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_1_1x1_increase')
             .batch_normalization(relu=False, name='conv2_1_1x1_increase_bn'))

        (self.feed('conv2_1_1x1_proj_bn',
                   'conv2_1_1x1_increase_bn')
             .add(name='conv2_1')
             .relu(name='conv2_1/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding2')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_2_3x3')
             .batch_normalization(relu=True, name='conv2_2_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_2_1x1_increase')
             .batch_normalization(relu=False, name='conv2_2_1x1_increase_bn'))

        (self.feed('conv2_1/relu',
                   'conv2_2_1x1_increase_bn')
             .add(name='conv2_2')
             .relu(name='conv2_2/relu')
             .conv(1, 1, 64, 1, 1, biased=False, relu=False, name='conv2_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv2_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding3')
             .conv(3, 3, 64, 1, 1, biased=False, relu=False, name='conv2_3_3x3')
             .batch_normalization(relu=True, name='conv2_3_3x3_bn')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv2_3_1x1_increase')
             .batch_normalization(relu=False, name='conv2_3_1x1_increase_bn'))

        (self.feed('conv2_2/relu',
                   'conv2_3_1x1_increase_bn')
             .add(name='conv2_3')
             .relu(name='conv2_3/relu')
             .conv(1, 1, 512, 2, 2, biased=False, relu=False, name='conv3_1_1x1_proj')
             .batch_normalization(relu=False, name='conv3_1_1x1_proj_bn'))

        (self.feed('conv2_3/relu')
             .conv(1, 1, 128, 2, 2, biased=False, relu=False, name='conv3_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_1_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding4')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_1_3x3')
             .batch_normalization(relu=True, name='conv3_1_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_1_1x1_increase')
             .batch_normalization(relu=False, name='conv3_1_1x1_increase_bn'))

        (self.feed('conv3_1_1x1_proj_bn',
                   'conv3_1_1x1_increase_bn')
             .add(name='conv3_1')
             .relu(name='conv3_1/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_2_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding5')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_2_3x3')
             .batch_normalization(relu=True, name='conv3_2_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_2_1x1_increase')
             .batch_normalization(relu=False, name='conv3_2_1x1_increase_bn'))

        (self.feed('conv3_1/relu',
                   'conv3_2_1x1_increase_bn')
             .add(name='conv3_2')
             .relu(name='conv3_2/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_3_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding6')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_3_3x3')
             .batch_normalization(relu=True, name='conv3_3_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_3_1x1_increase')
             .batch_normalization(relu=False, name='conv3_3_1x1_increase_bn'))

        (self.feed('conv3_2/relu',
                   'conv3_3_1x1_increase_bn')
             .add(name='conv3_3')
             .relu(name='conv3_3/relu')
             .conv(1, 1, 128, 1, 1, biased=False, relu=False, name='conv3_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv3_4_1x1_reduce_bn')
             .zero_padding(paddings=1, name='padding7')
             .conv(3, 3, 128, 1, 1, biased=False, relu=False, name='conv3_4_3x3')
             .batch_normalization(relu=True, name='conv3_4_3x3_bn')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv3_4_1x1_increase')
             .batch_normalization(relu=False, name='conv3_4_1x1_increase_bn'))

        (self.feed('conv3_3/relu',
                   'conv3_4_1x1_increase_bn')
             .add(name='conv3_4')
             .relu(name='conv3_4/relu')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_proj')
             .batch_normalization(relu=False, name='conv4_1_1x1_proj_bn'))

        (self.feed('conv3_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_1_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding8')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_1_3x3')
             .batch_normalization(relu=True, name='conv4_1_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_1_1x1_increase')
             .batch_normalization(relu=False, name='conv4_1_1x1_increase_bn'))

        (self.feed('conv4_1_1x1_proj_bn',
                   'conv4_1_1x1_increase_bn')
             .add(name='conv4_1')
             .relu(name='conv4_1/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_2_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding9')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_2_3x3')
             .batch_normalization(relu=True, name='conv4_2_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_2_1x1_increase')
             .batch_normalization(relu=False, name='conv4_2_1x1_increase_bn'))

        (self.feed('conv4_1/relu',
                   'conv4_2_1x1_increase_bn')
             .add(name='conv4_2')
             .relu(name='conv4_2/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_3_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding10')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_3_3x3')
             .batch_normalization(relu=True, name='conv4_3_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_3_1x1_increase')
             .batch_normalization(relu=False, name='conv4_3_1x1_increase_bn'))

        (self.feed('conv4_2/relu',
                   'conv4_3_1x1_increase_bn')
             .add(name='conv4_3')
             .relu(name='conv4_3/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_4_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_4_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding11')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_4_3x3')
             .batch_normalization(relu=True, name='conv4_4_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_4_1x1_increase')
             .batch_normalization(relu=False, name='conv4_4_1x1_increase_bn'))

        (self.feed('conv4_3/relu',
                   'conv4_4_1x1_increase_bn')
             .add(name='conv4_4')
             .relu(name='conv4_4/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_5_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_5_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding12')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_5_3x3')
             .batch_normalization(relu=True, name='conv4_5_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_5_1x1_increase')
             .batch_normalization(relu=False, name='conv4_5_1x1_increase_bn'))

        (self.feed('conv4_4/relu',
                   'conv4_5_1x1_increase_bn')
             .add(name='conv4_5')
             .relu(name='conv4_5/relu')
             .conv(1, 1, 256, 1, 1, biased=False, relu=False, name='conv4_6_1x1_reduce')
             .batch_normalization(relu=True, name='conv4_6_1x1_reduce_bn')
             .zero_padding(paddings=2, name='padding13')
             .atrous_conv(3, 3, 256, 2, biased=False, relu=False, name='conv4_6_3x3')
             .batch_normalization(relu=True, name='conv4_6_3x3_bn')
             .conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='conv4_6_1x1_increase')
             .batch_normalization(relu=False, name='conv4_6_1x1_increase_bn'))

        (self.feed('conv4_5/relu',
                   'conv4_6_1x1_increase_bn')
             .add(name='conv4_6')
             .relu(name='conv4_6/relu')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_proj')
             .batch_normalization(relu=False, name='conv5_1_1x1_proj_bn'))

        (self.feed('conv4_6/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_1_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_1_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding31')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_1_3x3')
             .batch_normalization(relu=True, name='conv5_1_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_1_1x1_increase')
             .batch_normalization(relu=False, name='conv5_1_1x1_increase_bn'))

        (self.feed('conv5_1_1x1_proj_bn',
                   'conv5_1_1x1_increase_bn')
             .add(name='conv5_1')
             .relu(name='conv5_1/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_2_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_2_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding32')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_2_3x3')
             .batch_normalization(relu=True, name='conv5_2_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_2_1x1_increase')
             .batch_normalization(relu=False, name='conv5_2_1x1_increase_bn'))

        (self.feed('conv5_1/relu',
                   'conv5_2_1x1_increase_bn')
             .add(name='conv5_2')
             .relu(name='conv5_2/relu')
             .conv(1, 1, 512, 1, 1, biased=False, relu=False, name='conv5_3_1x1_reduce')
             .batch_normalization(relu=True, name='conv5_3_1x1_reduce_bn')
             .zero_padding(paddings=4, name='padding33')
             .atrous_conv(3, 3, 512, 4, biased=False, relu=False, name='conv5_3_3x3')
             .batch_normalization(relu=True, name='conv5_3_3x3_bn')
             .conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='conv5_3_1x1_increase')
             .batch_normalization(relu=False, name='conv5_3_1x1_increase_bn'))

        (self.feed('conv5_2/relu',
                   'conv5_3_1x1_increase_bn')
             .add(name='conv5_3')
             .relu(name='conv5_3/relu'))

        (self.feed('conv5_3/relu')
             .conv(3, 3, 512, 1, 1, biased=False, relu=False, padding='SAME', name='conv5_4')
             .batch_normalization(relu=True, name='conv5_4_bn'))

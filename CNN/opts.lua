--
-- User: peyman
-- Date: 11/14/16
-- Time: 9:50 PM
--

require 'image'
require 'dp'

local M = { }

function M.parse(arg)
    cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Train a CNN for float images')
    cmd:text('Example:')
    cmd:text("th -i main.lua -gpu -batch_size 100")
    cmd:text('Options:')

    -- Dataset options
    cmd:option('-dataset', '../data/plantPhenotype/Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets', 'path to training data')
    cmd:option('-bceSerializedData', '../data/BCE_trainData.t7', 'path to pre-serialized training data for BCE loss')
    cmd:option('-nllSerializedData', '../data/NLL_trainData.t7', 'path to pre-serialized training data for NLL loss')
    cmd:option('-genDataOnly', false, 'Whether to just generate dataset and not do any learning. Default: False')
    cmd:option('-portionTrain', 0.8, 'portion of data to train on: Default=0.8')
    cmd:option('-inputPixels', '{3,256,256}', 'number of pixels in c,h,w  of image: Default: 3,256,25')
    cmd:option('-labelPixels', '{64,64}', 'number of pixels in c,w of label image: Default: 1,4096')
    cmd:option('-batchSize', 33, 'number of examples per batch')

    -- data transforms
    cmd:option('-transforms', '{None,HorizontalFlip,VerticallFlip,Rotation,RandomScaleAndCrop}', 'What transformation to perform for data augmentation {HorizontalFlip | Rotation} ?')
    cmd:option('-hflip', 0.7, 'Probability of Horizontal flip? Default: 0.7')
    cmd:option('-vflip', 0.7, 'Probability of Vertical flip? Default: 0.7')
    cmd:option('-rotate', 80, 'Degree to rotate the image (plus some random amount) Default: 90')
    cmd:option('-brightness', 0.4, 'Brightness Default: 0.4')
    cmd:option('-contrast', 0.4, 'Contrast. Default: 0.4')
    cmd:option('-saturation', 0.4, 'Saturation Default: 0.4')
    cmd:option('-minscale', 256, 'Minimium scaling. Has to be >= inputPixels w and h. Default: 256')
    cmd:option('-maxscale', 4000, 'Minimium scaling.  Default: 4000')

    -- Model options
    cmd:option('-network', 'model.net', 'pretrained network')
    cmd:option('-padding', 1, 'add math.floor(kernelSize/2) padding to the input of each convolution')
    cmd:option('-channels', '{r,g,b}', 'Number of input image channels: Default: {r,g,b}')
    cmd:option('-channelSize', '{8,16}', 'Number of output channels for each convolution layer: Default: {8,16}')
    cmd:option('-kernelSize', '{3,3,3,3}', 'kernel size of each convolution layer. Height = Width')
    cmd:option('-kernelStride', '{1,1,1,1}', 'kernel stride of each convolution layer. Height = Width')
    cmd:option('-poolSize', '{2,2,2,2}', 'size of the max pooling of each convolution layer. Height = Width')
    cmd:option('-poolStride', '{2,2,2,2}', 'stride of the max pooling of each convolution layer. Height = Width')
    cmd:option('-dropout', false, 'use dropout')
    cmd:option('-dropoutProb', '{0.5,0.5}', 'dropout probabilities')
    cmd:option('-batchNorm', true, 'use batch normalization. dropout is mostly redundant with this')
    cmd:option('-activation', 'ReLU', 'transfer function (ReLU | Tanh | Sigmoid). Default: ReLU')
    cmd:option('-loss', 'NLL', 'Loss (BCE | NLL). Default: BCE')
    cmd:option('-bceThresh', 0.5, 'Classification threshold for BCE')
    cmd:option('-bceOffset', 0, 'BCE offset for clipping the sigmoid.')
    cmd:option('-labelThresh', 0.5, 'Classification threshold for leaf scores.')

    -- Optimization options
    cmd:option('-optimization', 'SGD')
    cmd:option('-maxEpochs', 1000, 'maximum number of epochs to try to find a better local minima for early-stopping. Default: 1000')
    cmd:option('-learningRate', 0.01, 'learning rate at t=0')
    cmd:option('-momentum', 0.6, 'momentum')
    cmd:option('-weightDecay', 1e-5, 'L2 penalty on the weights. Default=')
    cmd:option('-lrDecay', 1e-7, 'learning rate decay (in # samples) : Default: 1e-7')
    cmd:option('-earlystop', 40, 'Early stopping patience. Default: 20')

    -- Output options
    cmd:option('-showWs', true, 'Dump image of learnt Weights. Default: false')
    cmd:option('-visualize', false, 'visualize sample of input images. Default: true')
    cmd:option('-plot', true, 'plot learning errors. Default: true')
    cmd:option('-save', 'results', 'save directory')
    cmd:option('-silent', true, 'don\'t print anything to stdout')

    -- Backend options
    cmd:option('-threads', 8, 'number of threads. Default: 8')
    cmd:option('-gpu', false, 'use gpus. Default: False')
    cmd:option('-device', 1, 'sets the device (GPU) to use')

    cmd:text()

    opt = cmd:parse(arg or {})
    if not opt.silent then table.print(opt) end

    opt.portionTrain = tonumber(opt.portionTrain)
    opt.transforms = opt.transforms:gsub('[{}]',''):split(',')
    opt.channels = opt.channels:gsub('[{}]',''):split(',')
    opt.channelSize = table.fromString(opt.channelSize)
    opt.inputPixels = table.fromString(opt.inputPixels)
    opt.labelPixels = table.fromString(opt.labelPixels)

    opt.kernelSize = table.fromString(opt.kernelSize)
    opt.kernelStride = table.fromString(opt.kernelStride)
    opt.poolSize = table.fromString(opt.poolSize)
    opt.poolStride = table.fromString(opt.poolStride)
    opt.dropoutProb = table.fromString(opt.dropoutProb)

    return opt
end

return M

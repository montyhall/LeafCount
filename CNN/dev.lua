--
-- User: peyman
-- Date: 11/8/16
-- Time: 1:19 PM
-- To change this template use File | Settings | File Templates.
--

--
-- User: peyman
-- Date: 11/2/16
-- Time: 11:58 AM
-- To change this template use File | Settings | File Templates.
--

--
-- Ara2012 and Ara2013(Cannon) 7 megapixel camera -> width × height: 3108×2324 pixels
-- Ara2013 (Rasp. Pi) Raspberry Pi7 with 5 megapixel camera -> width × height: 2592×1944 pixels

-- 2655 files

require 'image'
require 'paths'
require 'xlua'
require 'pl.stringx'
require 'lfs'
posix = require 'posix'

require 'nn'

opt = dofile('opts.lua').parse(arg)

function run()
    --[[
    --nn.Sequential {
          [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
          (1): nn.SpatialConvolutionMM(3 -> 8, 3x3, 1,1, 1,1)
          (2): nn.SpatialBatchNormalization
          (3): nn.ReLU
          (4): nn.SpatialMaxPooling(2x2, 2,2)
          (5): nn.SpatialConvolutionMM(8 -> 16, 3x3, 1,1, 1,1)
          (6): nn.SpatialBatchNormalization
          (7): nn.ReLU
          (8): nn.SpatialMaxPooling(2x2, 2,2)
          (9): nn.SpatialConvolutionMM(16 -> 2, 7x5, 1,1, 3,2)
          (10): nn.Transpose
          (11): nn.Reshape(-1x2)
          (12): nn.LogSoftMax
        }
     ]]
    input = torch.randn(10,3,256,256)

    model = nn.Sequential()
    model:add(nn.SpatialConvolution(3, 8, 3,3, 1,1, 1,1)) --10,8,256,256
    model:add(nn.SpatialBatchNormalization(8)) --10,8,256,256
    model:add(nn.ReLU()) --10,8,256,256
    model:add(nn.SpatialMaxPooling(2,2,2,2)) --10,8,128,128
    model:add(nn.SpatialConvolution(8, 16, 3,3, 1,1, 1,1)) --10,16,128,128
    model:add(nn.SpatialBatchNormalization(16)) --10,16,128,128
    model:add(nn.ReLU()) --10,16,128,128
    model:add(nn.SpatialMaxPooling(2,2,2,2)) --10,16,64,64

    --if NLL
    model:add(nn.SpatialConvolution(16, 2, 7, 5, 1, 1, 3, 2)) --10,2,64,64
    model:add(nn.Transpose({ 2, 3 }, { 3, 4 })) --10,64,64,2
    model:add(nn.Reshape(-1,2,false)) -- 4096,2
    model:add(nn.LogSoftMax()) -- 4096,2

    model:forward(input):size()
end

function inference()
    --[[
    --nn.Sequential {
          [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
          (1): nn.SpatialConvolutionMM(3 -> 8, 3x3, 1,1, 1,1)
          (2): nn.SpatialBatchNormalization
          (3): nn.ReLU
          (4): nn.SpatialMaxPooling(2x2, 2,2)
          (5): nn.SpatialConvolutionMM(8 -> 16, 3x3, 1,1, 1,1)
          (6): nn.SpatialBatchNormalization
          (7): nn.ReLU
          (8): nn.SpatialMaxPooling(2x2, 2,2)
          (9): nn.SpatialConvolutionMM(16 -> 2, 7x5, 1,1, 3,2)
          (10): nn.Transpose
          (11): nn.Reshape(-1x2)
          (12): nn.LogSoftMax
        }
     ]]
    inputSize = opt.inputPixels[1]
    h = opt.inputPixels[2]
    w = opt.inputPixels[3]

    img  = image.load('../data/test/raft_430_top_1.jpg')
    img = image.scale(img,w,h)

    model = nn.Sequential()
    model:add(nn.SpatialConvolution(3, 8, 3,3, 1,1, 1,1)) --8,256,256
    model:add(nn.ReLU()) --8,256,256
    model:add(nn.SpatialMaxPooling(2,2,2,2)) --8,128,128
    model:add(nn.SpatialConvolution(8, 16, 3,3, 1,1, 1,1)) --16,128,128
    model:add(nn.ReLU()) --16,128,128
    model:add(nn.SpatialMaxPooling(2,2,2,2)) --16,64,64

    --if NLL
    model:add(nn.SpatialConvolution(16, 2, 7, 5, 1, 1, 3, 2)) --2,64,64
    model:add(nn.Transpose({ 1, 2 }, { 2, 3 })) --64,64,2
    model:add(nn.Reshape(-1,2,false)) -- 40960,2
    model:add(nn.LogSoftMax()) -- 40960,2

    model:forward(img):size()
end

function activations()
    ---input = torch.randn(10,3,256,256)
    img  = image.load('../data/test/raft_430_top_1.jpg')

    inputSize = opt.inputPixels[1]
    h = opt.inputPixels[2]
    w = opt.inputPixels[3]

    img = image.scale(img,w,h)

    input = torch.Tensor(1,inputSize,h,w) --leaf data

    input[1]:copy(img)

    -- target dimensions (64x64):
    target_h = opt.labelPixels[1]
    target_w = opt.labelPixels[2]

    cnn = nn.Sequential()
    -- convolutional and pooling layers
    depth = 1
    for i = 1, #opt.channelSize do

        if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
            -- dropout can be useful for regularization
            model:add(nn.SpatialDropout(opt.dropoutProb[depth]))
        end
        --SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
        --
        model:add(nn.SpatialConvolution(inputSize, opt.channelSize[i],
            opt.kernelSize[i], opt.kernelSize[i],
            opt.kernelStride[i], opt.kernelStride[i],
            opt.padding, math.floor(opt.kernelSize[i] / 2) or 0))

        if opt.batchNorm then
            model:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
        end

        model:add(nn[opt.activation]())

        output = cnn:forward(input)

        print('layer: ' .. i .. ' activation dim: ' ..
                output:size(1) .. ' x ' ..
                output:size(2) .. ' x ' ..
                output:size(3) .. ' x ' ..
                output:size(4))

        if opt.poolSize[i] and opt.poolSize[i] > 0 then
            cnn:add(nn.SpatialMaxPooling(opt.poolSize[i], opt.poolSize[i],
                opt.poolStride[i] or opt.poolSize[i],
                opt.poolStride[i] or opt.poolSize[i]))
        end
        inputSize = opt.channelSize[i]
        depth = depth + 1
    end
    if opt.useBce then
        -- Convo trick
        cnn:add(nn.SpatialConvolutionMM(opt.channelSize[#opt.channelSize], 1, 7, 5, 1, 1, 3, 2))
        cnn:add(nn.Reshape(target_h * target_w))
        cnn:add(nn.Sigmoid())

        loss = nn.BCECriterion()
    else
        -- we have 2 feature maps
        cnn:add(nn.SpatialConvolutionMM(opt.channelSize[#opt.channelSize], 2, 7, 5, 1, 1, 3, 2))
        cnn:add(nn.Transpose({2,3},{3,4})) --transpose dims 2 and 3, then 3 and 4...
        cnn:add(nn.Reshape(-1,2,false))
        cnn:add(nn.LogSoftMax())

        loss = nn.ClassNLLCriterion()
    end

    output = cnn:forward(input)
    print('FC layer: ' .. ' activation dim: ' .. output:size(1) .. ' x ' .. output:size(2))

    print(cnn)

end

inference()


--
-- User: peyman
-- Date: 11/2/16
-- Time: 10:10 AM
--


require 'nn'
require 'image'

-- number of channels in input (RGB=3):
inputSize = opt.inputPixels[1]

-- target dimensions (64x64):
target_h = opt.labelPixels[1]
target_w = opt.labelPixels[2]

local ok,savedNet = pcall(torch.load, paths.concat(opt.save, opt.network))

if not ok then

    print(sys.COLORS.blue ..  '==> building model')

    cnn = nn.Sequential()

    -- convolutional and pooling layers
    depth = 1
    for i = 1, #opt.channelSize do
        if opt.dropout and (opt.dropoutProb[depth] or 0) > 0 then
            -- dropout can be useful for regularization
            cnn:add(nn.SpatialDropout(opt.dropoutProb[depth]))
        end
        --SpatialConvolution(nInputPlane, nOutputPlane, kW, kH, [dW], [dH], [padW], [padH])
        cnn:add(nn.SpatialConvolution(inputSize, opt.channelSize[i],
            opt.kernelSize[i], opt.kernelSize[i],
            opt.kernelStride[i], opt.kernelStride[i],
            math.floor(opt.kernelSize[i]/2), math.floor(opt.kernelSize[i]/2)))

        if opt.batchNorm then
            cnn:add(nn.SpatialBatchNormalization(opt.channelSize[i]))
        end

        cnn:add(nn[opt.activation]())

        if opt.poolSize[i] and opt.poolSize[i] > 0 then
            cnn:add(nn.SpatialMaxPooling(opt.poolSize[i], opt.poolSize[i],
                opt.poolStride[i] or opt.poolSize[i],
                opt.poolStride[i] or opt.poolSize[i]))
        end
        inputSize = opt.channelSize[i]
        depth = depth + 1
    end

    if opt.loss == 'BCE' then
        -- Convo trick
        cnn:add(nn.SpatialConvolution(opt.channelSize[#opt.channelSize], 1, 7, 5, 1, 1, 3, 2))
        cnn:add(nn.Sigmoid())
        cnn:add(nn.Reshape(target_h * target_w))

    elseif opt.loss == 'NLL' then
        -- NOTE: we have 2 feature maps
        cnn:add(nn.SpatialConvolution(opt.channelSize[#opt.channelSize], 2, 7, 5, 1, 1, 3, 2))
        cnn:add(nn.Transpose({ 2, 3 }, { 3, 4 })) --transpose dims 2 and 3, then 3 and 4...
        cnn:add(nn.Reshape(-1,2,false))
        cnn:add(nn.LogSoftMax())
    end

    -- save model
    net={
        model = cnn,
        inputPixels = opt.inputPixels,
        labelPixels = opt.labelPixels,
        channels = opt.channels,
        channelSize = opt.channelSize,
        kernelSize = opt.kernelSize,
        kernelStride = opt.kernelStride,
        poolSize = opt.poolSize,
        poolStride = opt.poolStride,
        dropout = opt.dropout,
        dropoutProb = opt.dropoutProb,
        batchNorm = opt.batchNorm,
        activation =opt.activation,
        loss = opt.loss,
        bceThresh = opt.bceThresh,
        labelThresh = opt.labelThresh
    }
else
    print(sys.COLORS.blue ..  '==> loading model')
    net = savedNet
end

if not opt.silent then
    print(net.model)
end

if opt.gpu then
    net.model:cuda()
end
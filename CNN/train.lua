--
-- User: peyman
-- Date: 11/1/16
-- Time: 11:33 PM

----------------------------------------------------------------------
-- Training
----------------------------------------------------------------------

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

-- Save light network tools:
function nilling(module)
    module.gradBias   = nil
    if module.finput then module.finput = torch.Tensor() end
    module.gradWeight = nil
    module.output     = torch.Tensor()
    if module.fgradInput then module.fgradInput = torch.Tensor() end
    module.gradInput  = nil
end

function netLighter(network)
    nilling(network)
    if network.modules then
        for _,a in ipairs(network.modules) do
            netLighter(a)
        end
    end
end
----------------------------------------------------------------------
-- Log results to files
local trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

----------------------------------------------------------------------
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = net.model:getParameters()

----------------------------------------------------------------------
local optimState = {
    learningRate = opt.learningRate,
    momentum = opt.momentum,
    weightDecay = opt.weightDecay,
    learningRateDecay = opt.lrDecay
}
----------------------------------------------------------------------

local function train(datasource)

    print(sys.COLORS.red ..  '==> starting training procedure')

    -- This matrix records the current confusion across classes
    local confusion = optim.ConfusionMatrix(datasource.classes)

    local trainData = datasource.trainData
    local mean = datasource.mean
    local std = datasource.std
    local minH = datasource.minH
    local maxH = datasource.maxH
    local minW = datasource.minW
    local maxW = datasource.maxW

    local epoch
    local confusionPreds
    local confusionTargets

    local bceThresh = opt.bceThresh or 0.5
    local target_h=opt.labelPixels[1]
    local target_w=opt.labelPixels[2]

    local min_loss=math.huge
    local max_accuracy=0
    local best_valid_epoch=0
    local ntrial = 0

    local cy, cyt ,bestModel

    local x = torch.Tensor(opt.batchSize,trainData.data:size(2),
        trainData.data:size(3), trainData.data:size(4)) --leaf data

    local yt = torch.Tensor()
    local ytt = torch.Tensor(opt.batchSize,trainData.labels:size(2))

    if opt.gpu then
        x = x:cuda()
        yt = yt:cuda()
        ytt = ytt:cuda()
    end

    for epoch=1,opt.maxEpochs do

        net.model:training()

        local debug = true
        local randImgIdx = torch.random(1,trainData.size())

        -- next epoch
        current_loss = 0
        current_accuracy = 0
        confusion:zero()
        ntrial = ntrial + 1

        -- local vars
        local time = sys.clock()

        -- shuffle at each epoch
        local shuffle = torch.randperm(trainData:size())

        -- do one epoch
        print(sys.COLORS.green .. '==> doing epoch on training data:')
        print("==> online epoch # " .. epoch .. '/' .. opt.maxEpochs .. ' [batchSize = ' .. opt.batchSize .. ']')

        for t = 1,trainData:size(),opt.batchSize do
            -- disp progress
            xlua.progress(t, trainData:size())
            collectgarbage()

            -- create mini batch
            local idx = 1
            --for i = t,t+opt.batchSize-1 do
            for i = t,math.min(t+opt.batchSize-1,trainData:size()) do

                local img = trainData.data[shuffle[i]]
                local lab = trainData.labels[shuffle[i]]

                -- augment image and label
                img,lab = transform(i,randImgIdx,img,lab,target_h,target_w)

                x[idx]:copy(img)
                ytt[idx]:copy(lab)

                idx = idx + 1
            end

--            for i = t,math.min(t+opt.batchSize-1,trainData:size()) do
--
--                img = trainData.data[shuffle[i]]
--                lab = trainData.labels[shuffle[i]]
--
--                x[idx]:copy(img)
--                ytt[idx]:copy(lab)
--
--                -- augment image and label
--                idx,x,ytt = transform(i,idx,randImgIdx,img,lab,target_h,target_w,x,ytt)
--
--                idx = idx + 1
--            end

            if opt.loss == 'NLL' then
                --nn.ClassNLLCriterion expects target to be a 1D tensor of size batch_size or scalar
                yt = ytt:view(-1)
            else
                yt = ytt
            end

            -- create closure to evaluate f(X) and df/dX
            local eval_E = function(w)
                -- reset gradients
                dE_dw:zero()

                -- evaluate function for complete mini batch
                local y = net.model:forward(x)
                local E = loss:forward(y,yt)

                -- estimate df/dW
                local dE_dy = loss:backward(y,yt)

                net.model:backward(x,dE_dy)

                if opt.gpu then
                    cy = y:float()
                    cyt = yt:float()
                else
                    cy = y
                    cyt = yt
                end

                if opt.loss == 'BCE' then
                    cy = cy or y.new() -- confusion for predicted y
                    cyt = cyt or yt.new() -- confusion for actual y

                    --cy:gt(y, opt.bceThresh):add(1)
                    cy:gt(cy, bceThresh):add(1)
                    cyt:resize(yt:size()):gt(cyt:clone(), bceThresh):add(1)

                    -- update confusion
                    for i = 1,opt.batchSize do
                        confusion:batchAdd(cy[i], cyt[i])
                    end
                else
                    -- update confusion
                    confusion:batchAdd(cy, cyt)
                end

                -- return f and df/dX
                return E,dE_dw
            end

            -- optimize on current mini-batch
            _,fs = optim.sgd(eval_E, w, optimState)
            current_loss = current_loss + fs[1] -- fs is return value of eval_E
        end

        -- time taken
        time = sys.clock() - time
        time = time / trainData:size()
        print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

        -- print confusion matrix
        confusion:updateValids()
        print(confusion)

        -- update accuracy and average loss
        current_accuracy = confusion.totalValid * 100
        current_loss = current_loss / trainData:size()

        print("ntrial           = " .. ntrial)
        print("current accuracy = " .. current_accuracy)
        print("max accuracy     = " .. max_accuracy)
        print("current loss     = " .. current_loss)
        print("min average loss = " .. min_loss)
        print("learning rate    = " .. optimState.learningRate)

        -- update logger/plot
        trainLogger:add{['% mean class accuracy (train set)'] = current_loss}
        if opt.plot then
            trainLogger:style{['% mean class accuracy (train set)'] = '-'}
            trainLogger:plot()
        end

        -- early stopping
        if current_accuracy > max_accuracy then
            max_accuracy = current_accuracy
            best_valid_epoch = epoch

            bestModel = net.model:clone()
            netLighter(bestModel)

            if current_loss < min_loss then
                min_loss = current_loss
            end
            --local filename = paths.concat(opt.save, 'model.net')
            --os.execute('mkdir -p ' .. sys.dirname(filename))
            --print('==> saving model to '..filename)
            --netLighter(net.model1)
            --torch.save(filename, net.model1)

            ntrial = 0

        elseif ntrial >= opt.earlystop  then
            print(sys.COLORS.red ..  "No new minima found after "..ntrial.." epochs.")
            print(sys.COLORS.red ..  'Best accuracy was: ' ..  max_accuracy .. 'at epoch: ' .. best_valid_epoch)
            --serializeBestModel()
            return bestModel
        end
    end
    --serializeBestModel()
    return bestModel
end

-- Export:
return train

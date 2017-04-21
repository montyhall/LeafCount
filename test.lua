--
-- User: peyman
-- Date: 11/3/16
-- Time: 2:23 PM
-- To change this template use File | Settings | File Templates.
--

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

----------------------------------------------------------------------
-- Logger:
local testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

----------------------------------------------------------------------

-- test function
local cy, cyt
function test(datasource)
    print(sys.COLORS.red .. '==> starting testing procedure')

    -- This matrix records the current confusion across classes
    local confusion = optim.ConfusionMatrix(datasource.classes)

    -- local vars
    local testData = datasource.testData

    -- Batch test:
    local x = torch.Tensor(opt.batchSize,testData.data:size(2),
        testData.data:size(3), testData.data:size(4)) --leaf data

    --local yt = torch.Tensor(opt.batchSize,trainData.labels:size(2))
    local yt = torch.Tensor()

    local ytt = torch.Tensor(opt.batchSize,testData.labels:size(2))

    if opt.gpu then
        x = x:cuda()
        yt = yt:cuda()
        ytt = ytt:cuda()
    end

    local bceThresh = opt.bceThresh or 0.5
    local target_h=opt.labelPixels[1]
    local target_w=opt.labelPixels[2]

    local time = sys.clock()

    local debug = true
    --torch.manualSeed(15)
    local randImgIdx = torch.random(1,testData:size())

    local leaf, nonLeaf, expSum

    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    net.model:evaluate()

    -- test over test data
    print(sys.COLORS.red .. '==> testing on test set:')

    for t = 1,testData:size(),opt.batchSize do
        --xlua.progress(t, testData:size())

        -- create mini batch
        local idx = 1
        for i = t,math.min(t+opt.batchSize-1,testData:size()) do
            local img = testData.data[i]
            local lab = testData.labels[i]

            if debug and i == randImgIdx then
                image.save('results/actual.png', img)

                if opt.gpu then
                    prediction = net.model:forward(img:cuda())
                else
                    prediction = net.model:forward(img)
                end

                if opt.loss == 'BCE' then
                    --take output from last sigmoid layer
                    leaf = net.model.modules[#model.modules-1].output[1]
                else
                    --take output from last SpatialConvolution layer
                    prediction = torch.exp(net.model.modules[#net.model.modules-3].output)
                    nonLeaf = prediction[1][1]
                    leaf = leaf or prediction[1][2].new()
                    leaf:resizeAs(prediction[1][2]):copy(prediction[1][2])
                    expSum = expSum or leaf.new()
                    expSum:resizeAs(leaf):copy(leaf):add(nonLeaf)
                    leaf:cdiv(expSum)
                end
                if opt.gpu then
                   torch.save('results/prediction.t7', leaf:float())
                else
                   torch.save('results/prediction.t7', leaf)
                end
                leaf:ge(leaf, opt.labelThresh)
                image.save('results/prediction.png', leaf)
            end
            -- augment image and label
            --idx,x,ytt = transform(i,idx,randImgIdx,img,lab,target_h,target_w,x,ytt)

            img,lab = transform(i,randImgIdx,img,lab,target_h,target_w)

            x[idx] = img
            ytt[idx] = lab

            idx = idx + 1
        end

        if opt.loss == 'NLL' then
            yt = ytt:view(-1)
        else
            yt = ytt
        end

        -- test sample
        local y = net.model:forward(x)

        if opt.loss == 'BCE' then
            cy = cy or y.new() -- confusion for predicted y
            cyt = cyt or yt.new() -- confusion for actual y

            cy:gt(y, bceThresh):add(1)
            cyt:resize(yt:size()):gt(yt:clone(),bceThresh):add(1)

            -- update confusion
            for i = 1,opt.batchSize do
                confusion:batchAdd(cy[i], cyt[i])
            end
        else
            -- update confusion
            confusion:batchAdd(y, yt)
        end

    end

    -- timing
    time = sys.clock() - time
    time = time / testData:size()
    print("\n==> time to test 1 sample = " .. (time*1000) .. 'ms')

    confusion:updateValids()
    --confusion:totalValid()
    print(confusion)

    -- update log/plot
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
    if opt.plot then
        testLogger:style{['% mean class accuracy (test set)'] = '-'}
        testLogger:plot()
    end
end

-- Export:
return test

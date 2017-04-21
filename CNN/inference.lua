--
-- User: peyman
-- Date: 11/18/16
-- Time: 3:02 PM
--

require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'paths'
require 'nn'
require 'image'

--local hasCuda = pcall(function() require 'cunn' end)
--if hasCuda then
--   require 'cunn'
--end

----------------------------------------------------------
cmd = torch.CmdLine()
cmd:text()
cmd:text('Use trained CNN to infer leaves from an image')
cmd:text('Example:')
cmd:text("th -i inference.lua -gpu -batch_size 100")
cmd:text('Options:')
-- Dataset options
cmd:option('-serializedModel', 'results/model.net', 'path to trained model')
cmd:option('-serializedData', '../data/NLL_trainData.t7', 'path to training data')
cmd:option('-testPath', '../data/test', 'path to inference images')
cmd:option('-batch', false, 'batch inference. Default: false')
cmd:option('-labelThresh', 0.5, 'Threshold probability above which a pixel is regarded as leaf. Default: 0.5')
-- Backend options
cmd:option('-threads', 8, 'number of threads. Default: 8')
cmd:option('-gpu', false, 'use gpus. Default: False')
cmd:option('-device', 1, 'sets the device (GPU) to use')
-- Debug
cmd:option('-verbose', true, 'Print to console. Default: true')
cmd:text()

opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')


----------------------------------------------------------
local function loadDataAndModel()
    -- load data for mean and sds
    ok,data4Learning = pcall(torch.load, opt.serializedData)
    if not ok then
        print(sys.COLORS.red .. '... could not load mean+sds-> ',opt.serializedData)
        return
    end

    -- load learnt model
    torch.load(opt.serializedModel)
    ok,model = pcall(torch.load, opt.serializedModel)
    if not ok then
        print(sys.COLORS.red .. '... could not load model -> ',opt.serializedModel)
        return
    end
    return data4Learning,model
end
----------------------------------------------------------
data, net  = loadDataAndModel()

net.model:evaluate()

loss=net.loss
channels=net.channels
c=net.inputPixels[1]
h=net.inputPixels[2]
w=net.inputPixels[3]
target_h=net.labelPixels[1]
target_w=net.labelPixels[2]

-- load images
local images={}
local leaf, nonLeaf, expSum

--raft_430_top_3
-- x=16+18+18+18+18+18+17+18+16+18+16+18+18+18+18+18+18+16
-- x=315

local i=0
for file in paths.files(opt.testPath) do
    if file:match(".*[jpg | png]$") then
        i=i+1
        img  = image.load(opt.testPath ..'/'..file)
        if img:size()[1] == #channels then
            --img = image.scale(img,w,h)
            --img = image.scale(img, 5000)
   
            for c,channel in ipairs(channels) do
                -- normalize each channel globally:
                img[{ c,{},{} }]:add(-data.mean[c])
                img[{ c,{},{} }]:div(data.std[c])
            end

            if opt.batch then
               table.insert(images, img)
            else
                if opt.gpu then
                   prediction = net.model:forward(img:cuda())
                else
                   prediction = net.model:forward(img)
                end

                if loss == 'BCE' then
                    leaf = net.model.modules[#net.model.modules-1].output
                else
                    prediction = net.model.modules[#net.model.modules-3].output
                    prediction:exp()
                    nonLeaf = prediction[1][1]
                    leaf = leaf or prediction[1][2].new()
                    leaf:resizeAs(prediction[1][2]):copy(prediction[1][2])
                    expSum = expSum or leaf.new()
                    expSum:resizeAs(leaf):copy(leaf):add(nonLeaf)
                    leaf:cdiv(expSum)
                end

                leaf:ge(leaf, opt.labelThresh)
                numLeaves = leaf:sum()

                -- visualization
                print(sys.COLORS.blue .. '...' .. file ..
                        ' (' .. leaf:size(1) .. 'x' .. leaf:size(2) ..
                        ') num leaves = ' .. numLeaves)
                image.display{image=img, legend=file }
                image.display{image=leaf, legend='prediction' }
            end
        end
    end
end

-- batch inference
if opt.batch then
    print(sys.COLORS.blue .. '... batch inference')
    --setup batch inference tensors
    local x = torch.Tensor(#images,c,h,w) --leaf data

    -- run predictions
    if opt.gpu then
        prediction = net.model:forward(x:cuda())
    else
        prediction = net.model:forward(x)
    end

    if loss == 'NLL' then
        print("Model with NLL")
        prediction = net.model.modules[#net.model.modules-3].output
        print(prediction:size())
        prediction:exp()
        nonLeaf = prediction[{{},{1,1}, {}, {}}]
        local tempLeaf = prediction[{{},{2,2}, {}, {}}]
        leaf = leaf or tempLeaf.new()
        leaf:resizeAs(tempLeaf):copy(tempLeaf)
        expSum = expSum or leaf.new()
        expSum:resizeAs(leaf):copy(leaf):add(nonLeaf)
        leaf:cdiv(expSum)
        
        leaf:ge(leaf, opt.labelThresh)
        numLeaves = leaf:sum(4):sum(3):squeeze()
    else
        print("Model with BCE")
        leaf = net.model.modules[#net.model.modules-1].output
        print(leaf:size())
        numLeaves = leaf:sum(4):sum(3):squeeze()
    end
    print(numLeaves:size())
end

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
stringx = require 'pl.stringx'
require 'lfs'
posix = require 'posix'

local function imagestats(trainData,testData)
    local minH=0
    local maxH=0
    local minW=0
    local maxW=0

    for i=1, trainData.size() do
        local h = trainData.data[i]:size(2)
        local w = trainData.data[i]:size(3)
        if minH == 0 then minH = h end
        if minW == 0 then minW = w end

        minH = math.min(minH,h)
        maxH = math.max(maxH,h)
        minW = math.min(minW,w)
        maxW = math.max(maxW,w)
    end
    for i=1, testData.size() do
        local h = testData.data[i]:size(2)
        local w = testData.data[i]:size(3)

        minH = math.min(minH,h)
        maxH = math.max(maxH,h)
        minW = math.min(minW,w)
        maxW = math.max(maxW,w)
    end
    print('---------')
    print('image stats')
    print(string.format('Height min: %d max: %d',minH,maxH))
    print(string.format('Width min: %d max: %d',minW,maxW))

    return minH,maxH,minW,maxW
end

local n=0
local m=0
local numFiles=0

if opt.loss == 'NLL' then
    ok,data = pcall(torch.load, opt.nllSerializedData)
else
    ok,data = pcall(torch.load, opt.bceSerializedData)
end

if ok then
    --imagestats(data)
    return data
else
    -- classes: GLOBAL var!
    classes = {'noleaf','leaf'}

    --[[ Recurseively walk down directories and find
    -- training and label data
     ]]
    local function walk (path,images,targets)
        for file in lfs.dir(path) do
            if file ~= "." and file ~= ".." then
                local f = path..'/'..file
                if lfs.attributes(f).mode == "directory" then
                    walk (f,images,targets)
                else
                    numFiles = numFiles+1

                    if string.find(f, "_rgb.png") then

                        imageBaseName,b = stringx.splitv(f,'_rgb.png')
                        imageBaseName = imageBaseName:gsub("%s+", "")

                        labelName=imageBaseName .. "_label.png"

                        if posix.stat(labelName) then
                            if  lfs.attributes(labelName).mode == "file" then
                                --print(sys.COLORS.blue .. '... match ',f,labelName)
                                n = n+1
                                images[n]=f
                                targets[n]=labelName
                            end
                        else
                            labelName=imageBaseName .. "_fg.png"
                            if posix.stat(labelName) then
                                if  lfs.attributes(labelName).mode == "file" then
                                    --print(sys.COLORS.blue .. '... match ',f,labelName)
                                    n = n+1
                                    images[n]=f
                                    targets[n]=labelName
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    -----------------------------------
    -- normalize the data
    -----------------------------------
    local function normalize(trainData,testData)
        -- Normalize each channel, and store mean/std
        -- per channel. These values are important, as they are part of
        -- the trainable parameters. At test time, test data will be normalized
        -- using these values.
        print(sys.COLORS.blue ..  '==> preprocessing data: normalize each feature (channel) globally')

        -- Name channels for convenience
        --local channels = {'y'}--,'u','v'}

        local mean = {}
        local std = {}
        for i,channel in ipairs(opt.channels) do --channels = r,g,b
            -- normalize each channel globally:
            mean[i] = trainData.data[{ {},i,{},{} }]:mean()
            std[i] = trainData.data[{ {},i,{},{} }]:std()
            trainData.data[{ {},i,{},{} }]:add(-mean[i])
            trainData.data[{ {},i,{},{} }]:div(std[i])
        end

        -- Normalize test data, using the training means/stds
        for i,channel in ipairs(opt.channels) do
            -- normalize each channel globally:
            testData.data[{ {},i,{},{} }]:add(-mean[i])
            testData.data[{ {},i,{},{} }]:div(std[i])
        end

        -- data augmentation
        -- https://github.com/eladhoffer/ImageNet-Training/blob/ffd48953a34ed41f1407e35032bd3e0a71740af9/Data.lua
        -- http://benanne.github.io/2015/03/17/plankton.html
        -- http://stackoverflow.com/questions/36144993/data-augmentation-techniques-for-small-image-datasets
        -- https://github.com/facebook/fb.resnet.torch/blob/e8fb31378fd8dc188836cf1a7c62b609eb4fd50a/datasets/transforms.lua

        ----------------------------------------------------------------------
        print(sys.COLORS.blue ..  '==> verify statistics')

        -- verify that data is properly normalized.
        for i,channel in ipairs(opt.channels) do
            local trainMean = trainData.data[{ {},i }]:mean()
            local trainStd = trainData.data[{ {},i }]:std()

            local testMean = testData.data[{ {},i }]:mean()
            local testStd = testData.data[{ {},i }]:std()

            print('training data, '..channel..'-channel, mean: ' .. trainMean)
            print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

            print('test data, '..channel..'-channel, mean: ' .. testMean)
            print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
        end

        return mean, std
    end

    -----------------------------------
    -- build the training and test data
    -----------------------------------
        --
        -- NOTE: image module handles images as: nChannel x height x width
        -- --

        -----------------------------
        -- find training data
        ----------------------------
        local images={}
        local targets={}
        walk(opt.dataset,images,targets)

        print(sys.COLORS.blue .. '... number of files seen -> ',numFiles)
        print(sys.COLORS.blue .. '... Total number of images found on disc -> ',#images)
        print(sys.COLORS.blue .. '... Total number of labels found on disc -> ',#targets)

        -- if images and lables are not balanced
        if #images ~= #targets then
            print(sys.COLORS.red .. '...ERROR:  number of images and labels do not match -> ',#images,#targets)
            if #images > #targets then
                for i=1, #images do
                    match=false
                    imageBaseName,b = stringx.splitv(images[i],'_rgb.png')
                    imageBaseName = imageBaseName:gsub("%s+", "")

                    for j=1, #targets do
                        targetBaseName,c = stringx.splitv(targets[j],'_label.png')
                        targetBaseName = targetBaseName:gsub("%s+", "")

                        if imageBaseName == targetBaseName then
                            match = true
                            break
                        end
                    end
                    if not match then
                        print(sys.COLORS.red .. '...has no label -> ',i, images[i])
                    end
                end
            end
        end

        -----------------------------
        -- DO the actual loading now
        ----------------------------
        print(sys.COLORS.blue .. '... loading training data')

        local c=opt.inputPixels[1]
        local h=opt.inputPixels[2]
        local w=opt.inputPixels[3]

        local target_h=opt.labelPixels[1]
        local target_w=opt.labelPixels[2]

        local imgC,imgH,imgW,minH,maxH,minW,maxW=0,0,0,0,0,0

        print(sys.COLORS.blue .. '... loading dataset from: ', opt.dataset)

        local imagesAll = torch.Tensor(#images, c, h, w)
        local targetAll = torch.Tensor(#targets, target_h*target_w)

        local idx = 0
        local b=0

        for i = 1, #images do
            xlua.progress(i, #images)

            --image
            img  = image.load(images[i])
            imgC = img:size()[1]
            imgH = img:size()[2]
            imgW = img:size()[3]

            if imgC == opt.inputPixels[1] then

                -- get image scales and update scale ranges
                if minH == 0 then minH = imgH end
                if minW == 0 then minW = imgW end

                minH = math.min(minH,imgH)
                maxH = math.max(maxH,imgH)
                minW = math.min(minW,imgW)
                maxW = math.max(maxW,imgW)

                -- scale image to desired square
                img = image.scale(img,w,h,'bicubic')

                --load target
                imgT = image.load(targets[i])
                imgT = image.rgb2y(imgT)
                imgT = image.scale(imgT,target_w,target_h,'bicubic')
                imgT = torch.reshape(imgT,target_h*target_w)
                imgT:resize(imgT:nElement())

                imagesAll[i]:copy(img)
                targetAll[i]:copy(imgT)
            else
                --TODO: RGBA -> RGB
                b = b+1
                --print(b,i, img:size()[1]..img:size()[2]..img:size()[3],images[i], '-> ',targets[i])
            end
        end

        --Height min: 50 max: 2324
        --Width min: 48 max: 3108

        print(sys.COLORS.blue .. '... Total number of images after data augmentation -> ',imagesAll:size(1))
        print(sys.COLORS.blue .. '... Total number of labels after data augmentation -> ',targetAll:size(1))
        print(sys.COLORS.blue .. 'image stats')
        print(sys.COLORS.blue .. string.format('Height min: %d max: %d',minH,maxH))
        print(sys.COLORS.blue .. string.format('Width min: %d max: %d',minW,maxW))

        -----------------------------
        -- split into train and test
        ----------------------------
        -- shuffle dataset: get shuffled indices in this variable:
        local imgsShuffle = torch.randperm((#imagesAll)[1])
        local trsize = torch.floor(imgsShuffle:size(1)*opt.portionTrain)
        local tesize = imgsShuffle:size(1) - trsize

        print(sys.COLORS.blue .. '... portion of training -> ',opt.portionTrain)
        print(sys.COLORS.blue .. '... training size -> ',trsize)
        print(sys.COLORS.blue .. '... test size -> ',tesize)

        -- create train set:
        trainData = {
            data = torch.Tensor(trsize, c, h, w),
            labels = torch.Tensor(trsize,target_h*target_w),
            size = function() return trsize end
        }
        --create test set:
        testData = {
            data = torch.Tensor(tesize, c, h, w),
            labels = torch.Tensor(tesize,target_h*target_w),
            size = function() return tesize end
        }

        for i=1,trsize do
            trainData.data[i]:copy(imagesAll[imgsShuffle[i]])
            trainData.labels[i]:copy(targetAll[imgsShuffle[i]])
        end
        for i=trsize+1,tesize+trsize do
            testData.data[i-trsize]:copy(imagesAll[imgsShuffle[i]])
            testData.labels[i-trsize]:copy(targetAll[imgsShuffle[i]])
        end

         --remove from memory temp image files:
        imagesAll = nil
        labelsAll = nil

        mean,std = normalize(trainData,testData)

        if opt.visualize then
            -- save/display sample traindata
            local first256Samples_y = trainData.data[{ {1,27}}]
            image.display{image=first256Samples_y, nrow=3, legend='first 27 training samples'}

            local first256Samples_y = trainData.labels[{ {1,1} }]
            first256Samples_y = torch.reshape(first256Samples_y,target_h,target_w)
            image.save('../data/trainLabels.jpg',first256Samples_y)
            image.display{image=first256Samples_y, nrow=3, legend='Some training labels examples: Y channel' }

            -- save/display sample testdata
            local first256Samples_y = testData.data[{ {1,27}}]
            image.display{image=first256Samples_y, nrow=3, legend='Some testing examples: Y channel' }

            local first256Samples_y = testData.labels[{ {1,1} }]
            first256Samples_y = torch.reshape(first256Samples_y,target_h,target_w)
            image.save('../data/testLabels.jpg',first256Samples_y)
            image.display{image=first256Samples_y, nrow=3, legend='Some testing labels examples: Y channel' }
        end

        -- Updating the targets to match lua indexing
        -- for classes (1,2) == (noleaf, leaf)
        -- image scaling means values are not 0,1... make them
        trainData.labels:gt(trainData.labels, 0.5)
        testData.labels:gt(testData.labels, 0.5)

        if opt.loss == 'NLL' then
            trainData.labels:add(1)
            testData.labels:add(1)

        elseif opt.bceOffset ~= 0 then
            local mask = torch.ByteTensor()
            -- 0 --> 0.1
            trainData.labels.eq(mask, trainData.labels, 0)
            trainData.labels[mask] = opt.bceOffset
            -- 1 --> 0.9
            trainData.labels.eq(mask, trainData.labels, 1)
            trainData.labels[mask] = 1 - opt.bceOffset

            -- 0 --> 0.1
            testData.labels.eq(mask, testData.labels, 0)
            testData.labels[mask] = opt.bceOffset
            -- 1 --> 0.9
            testData.labels.eq(mask, testData.labels, 1)
            testData.labels[mask] = 1 - opt.bceOffset
        end

        alldata = {
            trainData=trainData,
            testData=testData,
            mean=mean,
            std=std,
            minH = minH,
            maxH = maxH,
            minW = minW,
            maxW = maxW,
            classes = classes }

        if opt.loss == 'NLL' then
            print('==> saving model to ' .. opt.nllSerializedData)
            torch.save(opt.nllSerializedData, alldata)
        else
            print('==> saving model to ' .. opt.bceSerializedData)
            torch.save(opt.bceSerializedData, alldata)
        end

        return alldata
end

--
-- User: peyman
-- Date: 11/12/16
-- Time: 10:56 AM
--

require 'image'

--[[
-- Perform data augmentation on image,label pair
 ]]
transform = function (i,randImgIdx,img,label,target_h,target_w)

    labT = torch.reshape(label,target_h,target_w)

    if opt.loss == 'NLL' then labT:csub(1) end

    if opt.visualize and i == randImgIdx then
        image.display{image=img, legend='Original image' }
        image.display{image=labT, legend='Original label' }
    end

    -- pick a transformation randomly
    local randTransform = opt.transforms[torch.random(#opt.transforms)]

    if randTransform == 'Rotation' then
        imgTrans,labelTrans = Rotation(img,labT,opt.rotate)
        if opt.visualize and i == randImgIdx then
            image.display{image=imgTrans, legend='Rotated image' }
            image.display{image=labelTrans, legend='Rotated label' }
        end

    elseif randTransform == 'HorizontalFlip' then
        imgTrans,labelTrans = HorizontalFlip(img,labT,opt.hflip)
        if opt.visualize and i == randImgIdx and not torch.all(torch.eq(img,imgTrans)) then
            image.display{image=imgTrans, legend='HorizontalFlip image' }
            image.display{image=labelTrans, legend='HorizontalFlip label' }
        end

    elseif  randTransform == 'VerticallFlip' then
        imgTrans,labelTrans = VerticallFlip(img,labT,opt.vflip)
        if opt.visualize and i == randImgIdx and not torch.all(torch.eq(img,imgTrans)) then
            image.display{image=imgTrans, legend='VerticallFlip image' }
            image.display{image=labelTrans, legend='VerticallFlip label' }
        end

    elseif randTransform == 'RandomScaleAndCrop' then
        imgTrans,labelTrans = RandomScaleAndCrop(img,labT,opt.minscale,opt.maxscale)
        if opt.visualize and i == randImgIdx and not torch.all(torch.eq(img,imgTrans)) then
            image.display{image=imgTrans, legend='Scaled and cropped image' }
            image.display{image=labelTrans, legend='Scaled and cropped label' }
        end

    elseif randTransform == 'None' then
        return img,label
    end

    if not torch.all(torch.eq(img,imgTrans)) then
        targetTrans = torch.reshape(labelTrans,target_h*target_w)
        targetTrans:resize(targetTrans:nElement())

        targetTrans:gt(targetTrans, opt.bceThresh)

        if opt.loss == 'NLL' then
            targetTrans:gt(targetTrans, opt.bceThresh)
            targetTrans:add(1)

        elseif opt.bceOffset ~= 0 then
            local mask
            if torch.type(targetTrans) == 'torch.CudaTensor' then
                mask = torch.CudaTensor()
            else
                mask = torch.ByteTensor()
            end
            -- 0 --> opt.bceOffset
            targetTrans.eq(mask, targetTrans, 0)
            targetTrans[mask] = opt.bceOffset
            -- 1 --> 1 - opt.bceOffset
            targetTrans.eq(mask, targetTrans, 1)
            targetTrans[mask] = 1 - opt.bceOffset
        end
        return imgTrans,targetTrans -- 99 x 4096
    else
        return img,label -- 99 x 4096
    end
end

transform2 = function (i,idx,randImgIdx,img,label,target_h,target_w,x,ytt)

    labT = torch.reshape(label,target_h,target_w)
    if opt.loss == 'NLL' then labT:csub(1) end

    if opt.visualize and i == randImgIdx then
        image.display{image=img, legend='Original image' }
        image.display{image=labT, legend='Original label' }
    end

    for j=1, #opt.transforms do
        idx = idx +1
        if opt.transforms[j] == 'Rotation' then
            imgTrans,labelTrans = Rotation(img,labT,opt.rotate)
            if opt.visualize and i == randImgIdx then
                image.display{image=imgTrans, legend='Rotated image' }
                image.display{image=labelTrans, legend='Rotated label' }
            end

        elseif opt.transforms[j] == 'HorizontalFlip' then
            imgTrans,labelTrans = HorizontalFlip(img,labT,opt.hflip)
            if opt.visualize and i == randImgIdx and not torch.all(torch.eq(img,imgTrans)) then
                image.display{image=imgTrans, legend='HorizontalFlip image' }
                image.display{image=labelTrans, legend='HorizontalFlip label' }
            end

        elseif  opt.transforms[j] == 'VerticallFlip' then
            imgTrans,labelTrans = VerticallFlip(img,labT,opt.vflip)
            if opt.visualize and i == randImgIdx and not torch.all(torch.eq(img,imgTrans)) then
                image.display{image=imgTrans, legend='VerticallFlip image' }
                image.display{image=labelTrans, legend='VerticallFlip label' }
            end

        elseif opt.transforms[j] == 'RandomScaleAndCrop' then
            imgTrans,labelTrans = RandomScaleAndCrop(img,labT,opt.minscale,opt.maxscale)
            if opt.visualize and i == randImgIdx and not torch.all(torch.eq(img,imgTrans)) then
                image.display{image=imgTrans, legend='Scaled image' }
                image.display{image=labelTrans, legend='Scaled label' }
            end
        end

        if not torch.all(torch.eq(img,imgTrans)) then
            targetTrans = torch.reshape(labelTrans,target_h*target_w)
            targetTrans:resize(targetTrans:nElement())

            targetTrans:gt(targetTrans, opt.bceThresh)
            if opt.loss == 'NLL' then
                targetTrans:gt(targetTrans, opt.bceThresh)
                targetTrans:add(1)
            elseif opt.bceOffset ~= 0 then
                local mask
                if torch.type(targetTrans) == 'torch.CudaTensor' then
                    mask = torch.CudaTensor()
                else
                    mask = torch.ByteTensor()
                end
                -- 0 --> opt.bceOffset
                targetTrans.eq(mask, targetTrans, 0)
                targetTrans[mask] = opt.bceOffset
                -- 1 --> 1 - opt.bceOffset
                targetTrans.eq(mask, targetTrans, 1)
                targetTrans[mask] = 1 - opt.bceOffset
            end

            x[idx]:copy(imgTrans)
            ytt[idx]:copy(targetTrans) -- 99 x 4096

            -- TODO: fix below
        else
            x[idx]:copy(img)
            ytt[idx]:copy(label) -- 99 x 4096
        end
    end
    return idx,x,ytt
end


------------
-- flip
------------
function hflip(x)
    return torch.random(0,1) == 1 and x or image.hflip(x)
end
------------
-- rotate
------------
function rotate(x)
    -- rotate image slightly, in degrees
    degrees = math.rad(math.floor(torch.uniform(0, 360)))
    return image.rotate(x,degrees)
end
ColorNormalize = function (img,meanstd)
    img = img:clone()
    for i=1,3 do
        img[i]:add(-meanstd.mean[i])
        img[i]:div(meanstd.std[i])
    end
    return img
end

-- Scales the smaller edge to size
Scale = function (input,size, interpolation)
    interpolation = interpolation or 'bicubic'
    local w, h = input:size(3), input:size(2)
    if (w <= h and w == size) or (h <= w and h == size) then
        return input
    end
    if w < h then
        return image.scale(input, size, h/w * size, interpolation)
    else
        return image.scale(input, w/h * size, size, interpolation)
    end
end


------------
-- crop
------------

-- Crop to centered rectangle
-- lower the size the more of image is cropped
CenterCrop = function (input,size)
    local w1 = math.ceil((input:size(3) - size)/2)
    local h1 = math.ceil((input:size(2) - size)/2)
    return image.crop(input, w1, h1, w1 + size, h1 + size)
end

-- Random crop form larger image with optional zero padding
RandomCrop = function (input,label,size, padding)
    padding = padding or 0

    if padding > 0 then
        local temp = input.new(3, input:size(2) + 2*padding, input:size(3) + 2*padding)
        temp:zero()
        :narrow(2, padding+1, input:size(2))
        :narrow(3, padding+1, input:size(3))
        :copy(input)
        input = temp
    end

    local w, h = input:size(3), input:size(2)
    if w == size and h == size then
        return input,label
    end

    local x1, y1 = torch.random(0, w - size), torch.random(0, h - size)
    local img = image.crop(input, x1, y1, x1 + size, y1 + size)
    local lbl = image.crop(label, x1, y1, x1 + size, y1 + size)
    assert(img:size(2) == size and img:size(3) == size, 'wrong crop size')
    return img,lbl
end

-- Four corner patches and center crop from image and its horizontal reflection
TenCrop = function (input, size)
    local w, h = input:size(3), input:size(2)

    local output = {}
    for _, img in ipairs{input, image.hflip(input)} do
        table.insert(output, CenterCrop(img,size))
        table.insert(output, image.crop(img, 0, 0, size, size))
        table.insert(output, image.crop(img, w-size, 0, w, size))
        table.insert(output, image.crop(img, 0, h-size, size, h))
        table.insert(output, image.crop(img, w-size, h-size, w, h))
    end

    -- View as mini-batch
    for i, img in ipairs(output) do
        output[i] = img:view(1, img:size(1), img:size(2), img:size(3))
    end

    return input.cat(output, 1)
end

-- Resized with shorter side randomly sampled from [minSize, maxSize]
-- (ResNet-style)
RandomScale = function (img,label,minSize, maxSize)
    local w, h = img:size(3), img:size(2)

    local targetSz = torch.random(minSize, maxSize)
    local targetW, targetH = targetSz, targetSz
    if w < h then
        targetH = torch.round(h / w * targetW)
    else
        targetW = torch.round(w / h * targetH)
    end
    imgscaled = image.scale(img, targetW, targetH, 'bicubic')
    labelscaled = image.scale(label, targetW, targetH, 'bicubic')

    return imgscaled,labelscaled
end

RandomScaleAndCrop = function (img,label,minSize, maxSize)
    local wImg      = img:size(3)
    local wLabel    = label:size(1)
    local hLabel    = label:size(2)
    local imgscaled,labelscaled =  RandomScale(img,label,minSize, maxSize)
    local img2,label2 =  RandomCrop(imgscaled,labelscaled ,wImg)
    if wImg ~= label2 then
        label2 = image.scale(label2, wLabel, hLabel, 'bicubic')
    end
    return img2,label2
end

-- Random crop with size 8%-100% and aspect ratio 3/4 - 4/3 (Inception-style)
RandomSizedCrop = function (input,size)
    local scale = Scale(input,size)
    local crop = CenterCrop(input,size)

    local attempt = 0
    repeat
        local area = input:size(2) * input:size(3)
        local targetArea = torch.uniform(0.08, 1.0) * area

        local aspectRatio = torch.uniform(3/4, 4/3)
        local w = torch.round(math.sqrt(targetArea * aspectRatio))
        local h = torch.round(math.sqrt(targetArea / aspectRatio))

        if torch.uniform() < 0.5 then
            w, h = h, w
        end

        if h <= input:size(2) and w <= input:size(3) then
            local y1 = torch.random(0, input:size(2) - h)
            local x1 = torch.random(0, input:size(3) - w)

            local out = image.crop(input, x1, y1, x1 + w, y1 + h)
            assert(out:size(2) == h and out:size(3) == w, 'wrong crop size')

            return image.scale(out, size, size, 'bicubic')
        end
        attempt = attempt + 1
    until attempt >= 10

    -- fallback
    return crop(scale(input))
end

HorizontalFlip = function (img,label,prob)
    if torch.uniform() < prob then
        imgFlipped = image.hflip(img)
        labelFlipped = image.hflip(label)
        return imgFlipped,labelFlipped
    else
        return img,label
    end
end

VerticallFlip = function (img,label,prob)
    if torch.uniform() < prob then
        imgFlipped = image.vflip(img)
        labelFlipped = image.hflip(label)
        return imgFlipped,labelFlipped
    else
        return img,label
    end
end

Rotation = function (input, target, deg)
    if deg ~= 0 then
        rand = (torch.uniform() - 0.5) * deg * math.pi / 18
        imgFlipped = image.rotate(input, rand, 'bilinear')
        labelFlipped = image.rotate(target, rand, 'bilinear')
        return imgFlipped,labelFlipped
    else
        return input, target
    end
end

-- Lighting noise (AlexNet-style PCA-based noise)
Lighting = function (input, alphastd, eigval, eigvec)
    if alphastd == 0 then
        return input
    end

    local alpha = torch.Tensor(3):normal(0, alphastd)
    local rgb = eigvec:clone()
    :cmul(alpha:view(1, 3):expand(3, 3))
    :cmul(eigval:view(1, 3):expand(3, 3))
    :sum(2)
    :squeeze()

    input = input:clone()
    for i=1,3 do
        input[i]:add(rgb[i])
    end
    return input
end

local function blend(img1, img2, alpha)
    return img1:mul(alpha):add(1 - alpha, img2)
end

local function grayscale(dst, img)
    dst:resizeAs(img)
    dst[1]:zero()
    dst[1]:add(0.299, img[1]):add(0.587, img[2]):add(0.114, img[3])
    dst[2]:copy(dst[1])
    dst[3]:copy(dst[1])
    return dst
end

Saturation = function (input,var)
    local gs
    gs = gs or input.new()
    grayscale(gs, input)

    local alpha = 1.0 + torch.uniform(-var, var)
    blend(input, gs, alpha)
    return input
end

Brightness = function (input,var)
    local gs

    gs = gs or input.new()
    gs:resizeAs(input):zero()

    local alpha = 1.0 + torch.uniform(-var, var)
    blend(input, gs, alpha)
    return input
end

Contrast = function (input,var)
    local gs

    gs = gs or input.new()
    grayscale(gs, input)
    gs:fill(gs[1]:mean())

    local alpha = 1.0 + torch.uniform(-var, var)
    blend(input, gs, alpha)
    return input
end

RandomOrder = function (input,ts)
    local img = input.img or input
    local order = torch.randperm(#ts)
    for i=1,#ts do
        img = ts[order[i]](img)
    end
    return input
end

ColorJitter = function (input,opt)
    local brightness = opt.brightness or 0
    local contrast = opt.contrast or 0
    local saturation = opt.saturation or 0

    local ts = {}
    if brightness ~= 0 then
        table.insert(ts, Brightness(input,brightness))
    end
    if contrast ~= 0 then
        table.insert(ts, Contrast(input,contrast))
    end
    if saturation ~= 0 then
        table.insert(ts, Saturation(input,saturation))
    end

    if #ts == 0 then
        return function(input) return input end
    end

    return RandomOrder(input,ts)
end



--
-- User: peyman
-- Date: 11/2/16
-- Time: 10:58 AM
-- To change this template use File | Settings | File Templates.
--
local pl = require 'pl.import_into'() -- see https://stevedonovan.github.io/Penlight/api/manual/01-introduction.md.html#To_Inject_or_not_to_Inject_

----------------------------------------------------------------------
-- nb of threads and fixed seed (for repeatable experiments)
torch.setnumthreads(opt.threads)
gen = torch.Generator()
torch.manualSeed(gen, 0)

-- use floats, for SGD
if opt.optimization == 'SGD' then
    torch.setdefaulttensortype('torch.FloatTensor')
end
-- type:
if opt.gpu then
    print(sys.COLORS.red ..  '==> switching to CUDA')
    local ok,cunn = pcall(require, 'cunn')
    if not ok then
        print(sys.COLORS.red ..  '==> Could not find CUNN')
    else
        count = cutorch.getDeviceCount()
        print(string.format("num GPUs: %d", count))

        cutorch.setDevice(opt.device)
        deviceParams = cutorch.getDeviceProperties(1)
        print(string.format("deviceParams: %s", pl.pretty.write(deviceParams)))
        --cudaComputeCapability = deviceParams.major + deviceParams.minor/10
    end

end

showWs = function()
    -- show weights
    if opt.showWs then
        -- https://github.com/nicholas-leonard/dp/blob/master/scripts/showfilters.lua
        for i,layer in pairs(net.model:findModules('nn.SpatialConvolution')) do
            wi = layer.weight

            if wi:size(2) == 3 then
                wi = wi:view(layer.nOutputPlane, layer.nInputPlane, layer.kW, layer.kH)
            else
                wi = wi:view(-1, layer.kW, layer.kH) --filters grey
            end
            local filters = image.toDisplayTensor{input=wi, nrow=layer.nOutputPlane,padding=1, scaleeach=false }
            image.save('results/layer_' .. i .. '_weights.png', filters)

            --oi = layer.output
            --oi = oi:view(-1, oi:size(3),oi:size(4)) --filters grey
            --image.save('results/layer_' .. i .. '_output.png', oi)
            if opt.visualize then
                image.display{image=filters, nrow=layer.nOutputPlane, gui=true, legend='filters at layer: ' .. i }
                image.display{image=oi, nrow=oi:size(1), gui=true, legend='outputs at layer: ' .. i }
            end
        end
    end
end




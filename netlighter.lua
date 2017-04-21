--
-- User: peyman
-- Date: 11/21/16
-- Time: 9:14 AM
--


----------------------------------------------------------------------
-- Save light network tools:
nilling = function(module)
    module.gradBias   = nil
    if module.finput then module.finput = torch.Tensor() end
    module.gradWeight = nil
    module.output     = torch.Tensor()
    if module.fgradInput then module.fgradInput = torch.Tensor() end
    module.gradInput  = nil
end

netLighter = function(network)
    print('reducing: ',network)
    nilling(network)
    if network.modules then
        for _,a in ipairs(network.modules) do
            netLighter(a)
        end
    end
end


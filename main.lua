
--[[
-- User: peyman
-- Date: 11/2/16
-- Time: 10:09 AM
--

 To start anew:
    - delete content of results/
    - delete any dataset data/NLL_trainData.t7 and/or data/BCE_trainData.t7
    - th main

 To generate data only:
    - delete content of results/
    - delete any dataset data/NLL_trainData.t7 and/or data/BCE_trainData.t7
    - th main -genDataOnly

-- to see datasets:
    - delete any dataset data/NLL_trainData.t7 and/or data/BCE_trainData.t7
    -- qlua main.lua
--]]

----------------------------------------------------------------------
opt = dofile('opts.lua').parse(arg)
misc = dofile('misc.lua')
augment = dofile('dataAugment.lua')
data = dofile('datasource.lua')

if opt.genDataOnly then
    os.exit()
end

--classes = data.classes

dofile('model.lua')
dofile('criteria.lua')
train = dofile('train.lua')
test = dofile('test.lua')

-- train
optModel = train(data)

--save layer weights
showWs(net)

-- run test
test(data)

if opt.gpu then
    net.model=optModel:float()
end

local filename = paths.concat(opt.save, opt.network)
print('saving final model to ' .. filename)
os.execute('mkdir -p ' .. sys.dirname(filename))
torch.save(filename, net)

print('----------')
print('done')
print('----------')